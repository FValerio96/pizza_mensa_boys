"""
chicago_crime_graph.py

Modulo per:
- scaricare un sottoinsieme del dataset 'Crimes - 2001 to Present' (Chicago)
- costruire un grafo OSM per il quartiere Loop
- arricchire gli archi del grafo con il numero di crimini vicini (num_crimes)

Obiettivo pratico:
- creare un dataset "edge-level" (un record per arco stradale) che contenga
    sia attributi OSM della strada (es. highway, name, surface, lit, length, ...)
    sia segnali esterni come crimini agganciati all'arco e feature di illuminazione.

Il CSV risultante può poi essere usato in un altro modulo per analisi statistiche,
ad esempio per misurare quanto il tag OSM `lit` sia correlato a `crimes_per_km`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import json
import math
import os

import requests
import pandas as pd
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
from shapely.geometry import Point

# Endpoint ufficiale Socrata per 'Crimes - 2001 to Present'
# https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2
CRIME_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.csv"

# Overpass endpoint (per lampioni, ecc.)
OVERPASS_URL = "https://overpass.kumi.systems/api/interpreter"

# Raggio (metri) per associare lampioni agli archi
LAMP_BUFFER_METERS = 30


def download_chicago_crimes(
    output_csv: Path,
    year_min: int = 2020,
    limit: int = 50000,
    app_token: Optional[str] = None,
) -> Path:
    """
    Scarica un sottoinsieme del dataset dei crimini di Chicago come CSV.

    Parametri
    ---------
    output_csv : Path
        Percorso del file CSV da salvare.
    year_min : int
        Anno minimo dei crimini da scaricare (filtraggio lato API).
    limit : int
        Numero massimo di righe da scaricare.
    app_token : Optional[str]
        Eventuale token Socrata (non obbligatorio ma consigliato).

    Ritorna
    -------
    Path
        Percorso al file CSV salvato.
    """
    params = {
        "$limit": limit,
        "$where": f"year >= {year_min} AND latitude IS NOT NULL AND longitude IS NOT NULL",
    }
    headers = {}
    if app_token:
        headers["X-App-Token"] = app_token

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"[download_chicago_crimes] Scaricamento da Socrata (year >= {year_min}, limit={limit})...")
    resp = requests.get(CRIME_URL, params=params, headers=headers, timeout=60)
    resp.raise_for_status()

    output_csv.write_bytes(resp.content)
    size_mb = len(resp.content) / (1024 * 1024)
    print(f"[download_chicago_crimes] Salvato {output_csv} ({size_mb:.1f} MB)")
    return output_csv


def load_crimes_as_gdf(csv_path: Path) -> gpd.GeoDataFrame:
    """
    Carica il CSV dei crimini in un GeoDataFrame (CRS EPSG:4326).

    Si aspetta colonne 'latitude' e 'longitude' nel dataset.
    """
    print(f"[load_crimes_as_gdf] Caricamento CSV {csv_path}...")
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"[load_crimes_as_gdf] {len(df)} crimini con coordinate valide")

    geometry = [
        Point(lon, lat) for lon, lat in zip(df["longitude"], df["latitude"])
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf


def build_loop_graph() -> Tuple[nx.MultiDiGraph, gpd.GeoSeries]:
    """
    Costruisce il grafo pedonale OSM per il quartiere Loop (Chicago).

    Ritorna
    -------
    G : nx.MultiDiGraph
        Grafo pedonale estratto da OSM.
    polygon : gpd.GeoSeries
        Poligono del Loop (geometria area).
    """
    place = "Loop, Chicago, Illinois, USA"
    print(f"[build_loop_graph] Geocoding '{place}'...")
    gdf_place = ox.geocode_to_gdf(place)
    polygon = gdf_place.geometry.iloc[0]

    print("[build_loop_graph] Download grafo pedonale da OSM...")
    G = ox.graph_from_polygon(
        polygon,
        network_type="walk",
        simplify=True,
    )
    print(f"[build_loop_graph] Grafo costruito: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")
    return G, polygon


def fetch_street_lamps_for_polygon(
    polygon,
    cache_path: Path = Path("cache/chicago_loop_street_lamps.json"),
    use_cache: bool = True,
) -> list[dict]:
    """Scarica lampioni OSM (node highway=street_lamp) via Overpass.

    Nota: query basata sul bounding box del poligono (semplice e robusta).
    """
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
    """Converte lampioni (lista Overpass) in GeoDataFrame EPSG:4326."""
    if not lamps:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")

    pts = [
        Point(l["lon"], l["lat"])
        for l in lamps
        if isinstance(l, dict) and "lon" in l and "lat" in l
    ]
    return gpd.GeoDataFrame({"geometry": pts}, geometry="geometry", crs="EPSG:4326")


def clip_crimes_to_polygon(
    crimes_gdf: gpd.GeoDataFrame, polygon
) -> gpd.GeoDataFrame:
    """
    Filtra i crimini mantenendo solo quelli che cadono dentro il poligono dato.
    """
    # Assumo polygon in EPSG:4326
    crimes_clipped = crimes_gdf[crimes_gdf.within(polygon)]
    print(f"[clip_crimes_to_polygon] {len(crimes_clipped)}/{len(crimes_gdf)} crimini dentro il poligono")
    return crimes_clipped


def attach_crimes_to_graph(
    G: nx.MultiDiGraph,
    crimes_gdf: gpd.GeoDataFrame,
) -> Tuple[nx.MultiDiGraph, gpd.GeoDataFrame]:
    """
    Associa ogni crimine all'arco più vicino e aggiunge 'num_crimes' agli archi.

    Passi:
    - converte grafo e crimini in CRS metrico coerente (UTM scelto da OSMnx)
    - usa sjoin_nearest per trovare l'arco più vicino per ogni crimine
    - conta il numero di crimini per arco
    - aggiunge 'num_crimes' come attributo sia in edges_gdf che in G

    Ritorna
    -------
    G_enriched : nx.MultiDiGraph
        Grafo con attributo 'num_crimes' sugli archi.
    edges_enriched : gpd.GeoDataFrame
        GeoDataFrame degli archi con colonna 'num_crimes'.
    """
    print(f"[attach_crimes_to_graph] Proiezione metrica e spatial join di {len(crimes_gdf)} crimini...")
    # Proiezione metrica coerente (UTM)
    G_proj = ox.project_graph(G)
    _, edges_proj = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)
    crimes_proj = crimes_gdf.to_crs(edges_proj.crs).copy()

    # Spatial join: ogni crimine al segmento più vicino
    # edges_proj ha MultiIndex (u, v, key): resettiamo per avere un indice intero
    edges_flat = edges_proj[["geometry"]].reset_index()
    sjoin = gpd.sjoin_nearest(
        crimes_proj[["geometry"]],
        edges_flat[["geometry"]],
        how="left",
        distance_col="dist_crime_edge",
    )

    # Conta crimini per arco (index_right ora punta all'indice intero di edges_flat)
    crime_counts = (
        sjoin.groupby("index_right")
        .size()
        .rename("num_crimes")
    )

    # Rimappa dall'indice intero al MultiIndex originale
    idx_map = edges_flat.index  # RangeIndex → posizionale
    crime_counts.index = edges_proj.index[crime_counts.index]

    # Unisci ai bordi proiettati
    edges_proj = edges_proj.join(crime_counts.to_frame(), how="left")
    edges_proj["num_crimes"] = edges_proj["num_crimes"].fillna(0).astype(int)

    # Calcola crimes_per_km (normalizza per lunghezza arco)
    length_m = edges_proj.get("length")
    if length_m is None:
        length_m = edges_proj.geometry.length
    length_km = (length_m.astype(float) / 1000.0).replace(0, np.nan)
    edges_proj["crimes_per_km"] = (edges_proj["num_crimes"].astype(float) / length_km).fillna(0.0)

    # Aggiorna gli attributi del grafo originale (stesse edge keys u,v,k)
    nx.set_edge_attributes(G, edges_proj["num_crimes"].to_dict(), "num_crimes")
    nx.set_edge_attributes(G, edges_proj["crimes_per_km"].to_dict(), "crimes_per_km")

    total_crimes = int(edges_proj["num_crimes"].sum())
    edges_with_crimes = int((edges_proj["num_crimes"] > 0).sum())
    print(f"[attach_crimes_to_graph] {total_crimes} crimini assegnati a {edges_with_crimes}/{len(edges_proj)} archi")

    # Ritorna anche edges in EPSG:4326 con colonna num_crimes
    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
    if "num_crimes" not in edges.columns:
        edges["num_crimes"] = 0
    return G, edges


def _normalize_lit_value(v) -> Optional[bool]:
    """Normalizza il tag OSM 'lit' in booleano.

    Ritorna:
    - True se lit=yes/true/1/24_7
    - False se lit=no/false/0
    - None se non determinabile
    """
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (list, tuple, set)):
        for item in v:
            norm = _normalize_lit_value(item)
            if norm is not None:
                return norm
        return None

    s = str(v).strip().lower().replace(" ", "")
    if s in {"yes", "true", "1", "24/7", "24_7"}:
        return True
    if s in {"no", "false", "0"}:
        return False
    return None


def enrich_graph_with_lighting(
    G: nx.MultiDiGraph,
    polygon,
    lamp_buffer_meters: int = LAMP_BUFFER_METERS,
    use_cache: bool = True,
) -> nx.MultiDiGraph:
    """Arricchisce gli archi con segnali di illuminazione ("luci").

    Aggiunge agli archi:
    - lamp_count: numero di lampioni vicini (entro buffer)
    - lamp_density: lampioni per km
    - lit_tag: 1/0 se il tag OSM 'lit' è disponibile, altrimenti NaN
    - lighting_score: [0,1] combinando lit_tag e densità lampioni
    """
    print("[enrich_graph_with_lighting] Recupero lampioni...")
    lamps = fetch_street_lamps_for_polygon(polygon=polygon, use_cache=use_cache)
    lamp_gdf = lamps_to_geodataframe(lamps)
    print(f"[enrich_graph_with_lighting] {len(lamp_gdf)} lampioni come punti geometrici")

    print("[enrich_graph_with_lighting] Calcolo lamp_count e lamp_density per arco...")
    # Proiezione metrica coerente
    G_proj = ox.project_graph(G)
    _, edges_proj = ox.graph_to_gdfs(G_proj, nodes=True, edges=True)

    if lamp_gdf.empty:
        nx.set_edge_attributes(G, {k: 0 for k in edges_proj.index}, "lamp_count")
        nx.set_edge_attributes(G, {k: 0.0 for k in edges_proj.index}, "lamp_density")
    else:
        lamp_proj = lamp_gdf.to_crs(edges_proj.crs)

        edges_buf = edges_proj[["geometry"]].copy()
        edges_buf["geometry"] = edges_buf.geometry.buffer(lamp_buffer_meters)
        # MultiIndex (u,v,key) → indice intero per sjoin
        edges_buf = edges_buf.reset_index(drop=True)

        joined = gpd.sjoin(lamp_proj[["geometry"]], edges_buf, how="left", predicate="within")
        joined = joined.dropna(subset=["index_right"])
        joined["index_right"] = joined["index_right"].astype(int)
        counts = joined.groupby("index_right").size().rename("lamp_count")
        # Rimappa indice intero → MultiIndex originale
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

    # Normalizza densità usando p95 per robustezza
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
    print(f"[enrich_graph_with_lighting] Tag 'lit' noto su {n_lit_known}/{len(lit_tag)} archi ({n_lit_yes} illuminati)")
    print(f"[enrich_graph_with_lighting] lighting_score medio: {lighting_score.mean():.3f}")
    return G


def export_edges_dataset_csv(
    G: nx.MultiDiGraph,
    output_csv: Path,
    include_geometry_wkt: bool = True,
) -> Path:
    """Esporta un dataset edge-level in CSV.

    Il CSV contiene una riga per (u, v, key) con:
    - attributi OSM principali dell'arco (se presenti)
    - feature calcolate: num_crimes, crimes_per_km, lamp_count, lamp_density,
      lit_tag, lighting_score
    - opzionalmente: geometry in WKT e centroid lat/lon
    """
    print(f"[export_edges_dataset_csv] Preparazione dataset per {output_csv}...")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    edges = ox.graph_to_gdfs(G, nodes=False).copy()
    edges = edges.reset_index()  # porta u,v,key a colonne

    # Colonne "attese" ma opzionali (dipendono da OSM / osmnx)
    preferred_cols = [
        "u",
        "v",
        "key",
        "osmid",
        "name",
        "highway",
        "length",
        "oneway",
        "lanes",
        "maxspeed",
        "surface",
        "lit",
        # feature calcolate
        "num_crimes",
        "crimes_per_km",
        "lamp_count",
        "lamp_density",
        "lit_tag",
        "lighting_score",
    ]

    for col in preferred_cols:
        if col not in edges.columns:
            edges[col] = np.nan

    # CSV-friendly: serializza strutture complesse (list/dict) come JSON
    def _to_csv_cell(x):
        if isinstance(x, (dict, list, tuple, set)):
            return json.dumps(list(x) if isinstance(x, set) else x, ensure_ascii=False)
        return x

    for col in preferred_cols:
        edges[col] = edges[col].map(_to_csv_cell)

    if include_geometry_wkt and "geometry" in edges.columns:
        edges["geometry_wkt"] = edges.geometry.to_wkt()
        cent = edges.geometry.centroid
        edges["centroid_lon"] = cent.x
        edges["centroid_lat"] = cent.y

    out = edges[preferred_cols + (["geometry_wkt", "centroid_lon", "centroid_lat"] if include_geometry_wkt else [])]
    out.to_csv(output_csv, index=False)
    print(f"[export_edges_dataset_csv] Salvate {len(out)} righe x {len(out.columns)} colonne in {output_csv}")
    return output_csv


def make_folium_map_with_crimes(
    G: nx.MultiDiGraph,
    crimes_gdf: Optional[gpd.GeoDataFrame] = None,
    edge_metric: str = "risk",
) -> "folium.Map":
    """
    Crea una mappa Folium:
    - archi colorati in base a una metrica (default: 'risk')
    - crimini come heatmap (se crimes_gdf è fornito)
    """
    import folium
    from folium.plugins import HeatMap

    # Estrai edges in GeoDataFrame
    _, edges = ox.graph_to_gdfs(G, nodes=True, edges=True)

    metric = edge_metric if edge_metric in edges.columns else "num_crimes"
    values = edges[metric].astype(float).fillna(0.0)
    if len(values) > 0:
        q1, q2 = np.quantile(values, [0.8, 0.95])
    else:
        q1, q2 = 0.0, 0.0

    def edge_color(v: float) -> str:
        if v <= 0:
            return "#aaaaaa"  # grigio
        if v <= q1:
            return "#fee08b"  # giallo chiaro
        if v <= q2:
            return "#f46d43"  # arancio
        return "#a50026"  # rosso scuro

    edges["color"] = values.apply(edge_color)

    m = ox.plot_graph_folium(
        G,
        edge_color=edges["color"].tolist(),
        edge_width=2,
        zoom=14,
    )

    if crimes_gdf is not None and not crimes_gdf.empty:
        crimes_4326 = crimes_gdf.to_crs("EPSG:4326")
        heat_data = list(zip(crimes_4326.geometry.y, crimes_4326.geometry.x))
        HeatMap(heat_data, name="Crimes heatmap", radius=8, blur=10).add_to(m)
        folium.LayerControl().add_to(m)
    return m


def main():
    """
    Esempio di utilizzo:
    - scarica crimini recenti
    - costruisce grafo per il Loop
    - filtra crimini nel Loop
    - arricchisce il grafo con num_crimes
    - arricchisce il grafo con luci (lampioni, tag lit)
    - esporta un CSV edge-level (strada + crimini + luci)
    - salva opzionalmente una mappa HTML
    """
    print("=" * 60)
    print("[main] Avvio pipeline chicago_crime_graph")
    print("=" * 60)

    print("\n--- STEP 1/6: Download crimini da Socrata ---")
    out_csv = Path("data/chicago_crimes_sample.csv")
    app_token = os.environ.get("SOCRATA_APP_TOKEN")
    download_chicago_crimes(out_csv, year_min=2022, limit=50000, app_token=app_token)

    print("\n--- STEP 2/6: Caricamento crimini in GeoDataFrame ---")
    crimes_gdf = load_crimes_as_gdf(out_csv)

    print("\n--- STEP 3/6: Costruzione grafo OSM (Loop, Chicago) ---")
    G, polygon = build_loop_graph()

    print("\n--- STEP 4/6: Aggancio crimini agli archi del grafo ---")
    crimes_loop = clip_crimes_to_polygon(crimes_gdf, polygon)
    G_enriched, _ = attach_crimes_to_graph(G, crimes_loop)

    print("\n--- STEP 5/6: Arricchimento con dati illuminazione (lampioni + tag lit) ---")
    G_enriched = enrich_graph_with_lighting(
        G_enriched,
        polygon=polygon,
        lamp_buffer_meters=LAMP_BUFFER_METERS,
        use_cache=True,
    )

    print("\n--- STEP 6/6: Export dataset CSV + mappa HTML ---")
    edges_csv = export_edges_dataset_csv(G_enriched, Path("data/chicago_loop_edges_dataset.csv"))

    print("[main] Generazione mappa Folium (sanity-check)...")
    m = make_folium_map_with_crimes(G_enriched, crimes_gdf=crimes_loop, edge_metric="num_crimes")
    m.save("loop_crime_graph.html")

    print("\n" + "=" * 60)
    print("[main] Pipeline completata!")
    print(f"  -> Dataset archi: {edges_csv}")
    print("  -> Mappa HTML:    loop_crime_graph.html")
    print("=" * 60)


if __name__ == "__main__":
    main()