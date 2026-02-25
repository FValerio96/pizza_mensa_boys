"""
predict_risk_bari.py

Applica il modello di rischio addestrato su Chicago (models/risk_model.pkl)
al grafo stradale di Bari, Italia.

Pipeline:
1. Scarica il grafo pedonale OSM di Bari (centro storico + Municipio I)
2. Recupera lampioni via Overpass API (con cache)
3. Arricchisce gli archi con lamp_count, lamp_density, lighting_score
4. Prepara le stesse feature usate in training
5. Applica il modello → risk_score per ogni arco
6. Esporta CSV + mappa HTML

Uso:
    python notebooks/predict_risk_bari.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
from shapely.geometry import Point

# Riutilizza funzioni condivise
from train_risk_model import prepare_features

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
MODEL_PATH = Path("models/risk_model.pkl")
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"
LAMP_BUFFER_METERS = 30
CACHE_DIR = Path("cache")

# Luogo da analizzare
PLACE_NAME = "Bari, Italy"


# ---------------------------------------------------------------------------
# Funzioni riutilizzate / adattate da chicago_crime_graph.py
# ---------------------------------------------------------------------------

def build_city_graph(place: str) -> tuple[nx.MultiDiGraph, object]:
    """Costruisce il grafo pedonale OSM di qualsiasi città/quartiere."""
    print(f"[build_city_graph] Geocoding '{place}'...")
    gdf_place = ox.geocode_to_gdf(place)
    polygon = gdf_place.geometry.iloc[0]

    print(f"[build_city_graph] Download grafo pedonale da OSM...")
    G = ox.graph_from_polygon(polygon, network_type="walk", simplify=True)
    print(f"[build_city_graph] Grafo costruito: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
    return G, polygon


def fetch_street_lamps(polygon, cache_path: Path, use_cache: bool = True) -> list[dict]:
    """Scarica lampioni OSM via Overpass (con cache)."""
    import requests

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if use_cache and cache_path.exists():
        lamps = json.loads(cache_path.read_text(encoding="utf-8"))
        print(f"[fetch_street_lamps] Caricati {len(lamps)} lampioni da cache ({cache_path})")
        return lamps

    print("[fetch_street_lamps] Download lampioni da Overpass API...")
    minx, miny, maxx, maxy = polygon.bounds
    south, west, north, east = miny, minx, maxy, maxx

    query = f"""
    [out:json][timeout:120];
    node["highway"="street_lamp"]({south},{west},{north},{east});
    out;
    """.strip()

    resp = requests.get(OVERPASS_URL, params={"data": query}, timeout=180)
    resp.raise_for_status()
    data = resp.json()
    lamps = data.get("elements", [])

    cache_path.write_text(json.dumps(lamps), encoding="utf-8")
    print(f"[fetch_street_lamps] Trovati {len(lamps)} lampioni, salvati in {cache_path}")
    return lamps


def lamps_to_geodataframe(lamps: list[dict]) -> gpd.GeoDataFrame:
    """Converte lampioni Overpass → GeoDataFrame EPSG:4326."""
    if not lamps:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    pts = [
        Point(l["lon"], l["lat"])
        for l in lamps
        if isinstance(l, dict) and "lon" in l and "lat" in l
    ]
    return gpd.GeoDataFrame({"geometry": pts}, geometry="geometry", crs="EPSG:4326")


def _normalize_lit_value(val) -> bool | None:
    """Normalizza il tag OSM 'lit' in True/False/None."""
    if pd.isna(val):
        return None
    s = str(val).strip().lower()
    if s in ("yes", "true", "1"):
        return True
    if s in ("no", "false", "0"):
        return False
    return None


def enrich_graph_with_lighting(
    G: nx.MultiDiGraph,
    polygon,
    lamp_cache_path: Path,
    lamp_buffer_meters: int = LAMP_BUFFER_METERS,
    use_cache: bool = True,
) -> nx.MultiDiGraph:
    """Arricchisce gli archi con lamp_count, lamp_density, lit_tag, lighting_score."""

    print("[enrich_lighting] Recupero lampioni...")
    lamps = fetch_street_lamps(polygon, cache_path=lamp_cache_path, use_cache=use_cache)
    lamp_gdf = lamps_to_geodataframe(lamps)
    print(f"[enrich_lighting] {len(lamp_gdf)} lampioni come punti geometrici")

    print("[enrich_lighting] Calcolo lamp_count e lamp_density per arco...")
    G_proj = ox.project_graph(G)
    _, edges_proj = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)

    if lamp_gdf.empty:
        nx.set_edge_attributes(G, {k: 0 for k in edges_proj.index}, "lamp_count")
        nx.set_edge_attributes(G, {k: 0.0 for k in edges_proj.index}, "lamp_density")
    else:
        lamp_proj = lamp_gdf.to_crs(edges_proj.crs)

        edges_buf = edges_proj[["geometry"]].copy()
        edges_buf["geometry"] = edges_buf.geometry.buffer(lamp_buffer_meters)
        edges_buf = edges_buf.reset_index(drop=True)

        joined = gpd.sjoin(lamp_proj[["geometry"]], edges_buf, how="left", predicate="within")
        joined = joined.dropna(subset=["index_right"])
        joined["index_right"] = joined["index_right"].astype(int)
        counts = joined.groupby("index_right").size().rename("lamp_count")
        counts.index = edges_proj.index[counts.index]

        edges_proj = edges_proj.copy()
        edges_proj["lamp_count"] = counts
        edges_proj["lamp_count"] = edges_proj["lamp_count"].fillna(0).astype(int)

        length_m = edges_proj.get("length")
        if length_m is None:
            length_m = edges_proj.geometry.length
        length_km = (length_m.astype(float) / 1000.0).replace(0, np.nan)
        edges_proj["lamp_density"] = (edges_proj["lamp_count"] / length_km).fillna(0.0)

        nx.set_edge_attributes(G, edges_proj["lamp_count"].to_dict(), "lamp_count")
        nx.set_edge_attributes(G, edges_proj["lamp_density"].to_dict(), "lamp_density")

    # Tag 'lit'
    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    if "lit" in edges.columns:
        lit_bool = edges["lit"].apply(_normalize_lit_value)
        lit_tag = lit_bool.map(lambda b: 1.0 if b is True else (0.0 if b is False else np.nan))
    else:
        lit_tag = pd.Series(np.nan, index=edges.index)

    lamp_density = edges.get("lamp_density")
    if lamp_density is None:
        lamp_density = pd.Series(0.0, index=edges.index)

    if len(lamp_density) == 0:
        density_score = pd.Series(0.0, index=edges.index)
    else:
        p95 = float(np.nanpercentile(lamp_density.astype(float), 95))
        density_score = (lamp_density.astype(float) / p95).replace([np.inf, -np.inf], np.nan)
        density_score = density_score.fillna(0.0).clip(0, 1) if p95 > 0 else 0.0
        if not isinstance(density_score, pd.Series):
            density_score = pd.Series(density_score, index=edges.index)

    lighting_score = lit_tag.copy().fillna(density_score).clip(0, 1)

    nx.set_edge_attributes(G, lit_tag.to_dict(), "lit_tag")
    nx.set_edge_attributes(G, lighting_score.to_dict(), "lighting_score")

    n_lit_known = int(lit_tag.notna().sum())
    n_lit_yes = int((lit_tag == 1.0).sum())
    n_lamps = int(edges.get("lamp_count", pd.Series(0)).sum()) if "lamp_count" in edges.columns else 0
    print(f"[enrich_lighting] Tag 'lit' noto su {n_lit_known}/{len(lit_tag)} archi ({n_lit_yes} illuminati)")
    print(f"[enrich_lighting] lighting_score medio: {lighting_score.mean():.3f}")
    return G


def graph_to_edges_df(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Converte il grafo in DataFrame con le colonne attese dal modello."""
    edges = ox.graph_to_gdfs(G, nodes=False).reset_index()

    # Assicura che tutte le colonne richieste esistano
    expected_cols = [
        "u", "v", "key", "osmid", "name", "highway", "length", "oneway",
        "lanes", "maxspeed", "surface", "lit",
        "lamp_count", "lamp_density", "lit_tag", "lighting_score",
    ]
    for col in expected_cols:
        if col not in edges.columns:
            edges[col] = np.nan

    # Serializza strutture complesse (list/dict) come JSON
    def _to_csv_cell(x):
        if isinstance(x, (dict, list, tuple, set)):
            return json.dumps(list(x) if isinstance(x, set) else x, ensure_ascii=False)
        return x

    for col in expected_cols:
        edges[col] = edges[col].map(_to_csv_cell)

    return edges


def apply_risk_model(model_bundle: dict, df: pd.DataFrame) -> pd.Series:
    """Applica il modello hurdle e ritorna il risk_score per ogni arco."""
    clf = model_bundle["classifier"]
    reg = model_bundle["regressor"]
    expected_features = model_bundle["feature_names"]

    X = prepare_features(df)

    # Allinea le colonne: aggiungi mancanti, rimuovi extra
    for col in expected_features:
        if col not in X.columns:
            X[col] = 0
    X = X[expected_features]

    p_crime = clf.predict_proba(X)[:, 1]
    severity_log = reg.predict(X)
    severity = np.expm1(severity_log).clip(0)

    risk = p_crime * severity
    return pd.Series(risk, index=df.index, name="risk_score")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print(f"[predict_risk] Pipeline per: {PLACE_NAME}")
    print("=" * 60)

    # --- Carica modello ---
    print(f"\n--- STEP 1/5: Caricamento modello ---")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modello non trovato: {MODEL_PATH}\n"
            "Esegui prima: python notebooks/train_risk_model.py"
        )
    model_bundle = joblib.load(MODEL_PATH)
    stats = model_bundle["training_stats"]
    print(f"  Modello caricato da {MODEL_PATH}")
    print(f"  Training: {stats['n_edges']} archi, AUC={stats['auc']:.3f}, R²={stats['reg_r2']:.3f}")

    # --- Costruzione grafo ---
    print(f"\n--- STEP 2/5: Costruzione grafo OSM ---")
    G, polygon = build_city_graph(PLACE_NAME)

    # --- Illuminazione ---
    print(f"\n--- STEP 3/5: Arricchimento con illuminazione ---")
    lamp_cache = CACHE_DIR / "bari_street_lamps.json"
    G = enrich_graph_with_lighting(G, polygon, lamp_cache_path=lamp_cache)

    # --- Predizione ---
    print(f"\n--- STEP 4/5: Calcolo risk_score per ogni arco ---")
    edges_df = graph_to_edges_df(G)
    print(f"  Archi totali: {len(edges_df)}")

    risk_scores = apply_risk_model(model_bundle, edges_df)
    edges_df["risk_score"] = risk_scores.values

    # Normalizza risk_score in percentili [0, 100] per interpretabilità
    edges_df["risk_percentile"] = edges_df["risk_score"].rank(pct=True) * 100

    print(f"\n  Risk score – statistiche:")
    print(f"    min:    {risk_scores.min():.4f}")
    print(f"    median: {risk_scores.median():.4f}")
    print(f"    mean:   {risk_scores.mean():.4f}")
    print(f"    P90:    {np.percentile(risk_scores, 90):.4f}")
    print(f"    P95:    {np.percentile(risk_scores, 95):.4f}")
    print(f"    max:    {risk_scores.max():.4f}")

    # Distribuzione per fasce
    bins = [0, 5, 15, 30, 60, float("inf")]
    labels_risk = ["Molto basso", "Basso", "Medio", "Alto", "Molto alto"]
    edges_df["risk_category"] = pd.cut(
        edges_df["risk_score"], bins=bins, labels=labels_risk, right=False
    )
    print(f"\n  Distribuzione risk_category:")
    print(edges_df["risk_category"].value_counts().sort_index().to_string())

    # Top archi pericolosi
    print(f"\n  Top 15 archi più pericolosi:")
    top_cols = ["name", "highway", "length", "lamp_count", "lamp_density",
                "lighting_score", "risk_score", "risk_percentile"]
    top = edges_df.nlargest(15, "risk_score")[top_cols]
    print(top.to_string(index=False))

    # --- Export ---
    print(f"\n--- STEP 5/5: Export CSV + mappa HTML ---")
    out_csv = Path("data/bari_edges_risk.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Aggiungi geometria WKT e centroide
    edges_gdf = ox.graph_to_gdfs(G, nodes=False).reset_index()
    edges_df["geometry_wkt"] = edges_gdf.geometry.to_wkt()
    # Proietta in metrico per centroide accurato, poi torna a 4326
    edges_projected = edges_gdf.geometry.to_crs(epsg=3857)
    cent = edges_projected.centroid.to_crs(epsg=4326)
    edges_df["centroid_lon"] = cent.x.values
    edges_df["centroid_lat"] = cent.y.values

    export_cols = [
        "u", "v", "key", "osmid", "name", "highway", "length", "oneway",
        "lanes", "maxspeed", "surface", "lit",
        "lamp_count", "lamp_density", "lit_tag", "lighting_score",
        "risk_score", "risk_percentile", "risk_category",
        "geometry_wkt", "centroid_lon", "centroid_lat",
    ]
    for col in export_cols:
        if col not in edges_df.columns:
            edges_df[col] = np.nan

    edges_df[export_cols].to_csv(out_csv, index=False)
    print(f"  CSV salvato: {out_csv} ({len(edges_df)} righe)")

    # Mappa HTML
    _generate_risk_map(G, edges_df, out_html=Path("bari_risk_map.html"))

    print("\n" + "=" * 60)
    print("[predict_risk] Pipeline completata!")
    print(f"  -> CSV:  {out_csv}")
    print(f"  -> HTML: bari_risk_map.html")
    print("=" * 60)


def _generate_risk_map(G, edges_df, out_html: Path):
    """Genera mappa Folium con archi colorati per livello di rischio."""
    import folium
    from folium.plugins import HeatMap

    print("[map] Generazione mappa Folium...")

    _, edges_gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    risk_vals = edges_df["risk_score"].values

    # Soglie per colori
    p80 = np.percentile(risk_vals, 80)
    p95 = np.percentile(risk_vals, 95)

    def _color(v):
        if v <= 0:
            return "#2b83ba"  # blu – nessun rischio
        if v <= p80:
            return "#abdda4"  # verde chiaro
        if v <= p95:
            return "#f46d43"  # arancione
        return "#d7191c"  # rosso

    colors = [_color(v) for v in risk_vals]

    m = ox.plot_graph_folium(
        G,
        edge_color=colors,
        edge_width=2,
        zoom=14,
    )

    m.save(str(out_html))
    print(f"[map] Mappa salvata in {out_html}")


if __name__ == "__main__":
    main()
