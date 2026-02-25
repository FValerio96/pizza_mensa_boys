"""
SafeWalk Pipeline — Multi-Criteria Routing per Bari
====================================================

Questo script implementa un sistema di routing multi-criterio che suggerisce
il percorso migliore in base a:
  1. Sicurezza (luminosità stradale, orario, vivacità della zona)
  2. Carbon footprint (a piedi vs bus)
  3. Tempo a disposizione

Dati utilizzati:
  - Grafo pedonale di Bari (osmnx / OpenStreetMap)
  - Lampioni stradali (Overpass API / OpenStreetMap)
  - Fermate bus AMTAB (fermate.csv)
  - Orari transiti bus (orari_fermate.csv)
  - Consumi carburante AMTAB (consumi_amtab.csv)
  - Stazioni bike sharing (postazionibikesharing.csv)

Per eseguire: python safewalk_pipeline.py
Oppure copiare le sezioni in celle di un notebook Jupyter.
"""

import json
import os
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import Point

warnings.filterwarnings("ignore")

# ===========================================================================
# CONFIGURAZIONE
# ===========================================================================
DATA_DIR = Path(__file__).parent.parent / "ai_challenge" / "data"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Bounding box per la zona di interesse (centro di Bari)
BBOX_SOUTH = 41.10
BBOX_NORTH = 41.15
BBOX_WEST = 16.82
BBOX_EAST = 16.90

# Velocità medie (km/h)
WALKING_SPEED_KMH = 5.0
BUS_SPEED_KMH = 18.0
BIKE_SPEED_KMH = 15.0

# Emissioni CO₂
# Bus AMTAB: consumo medio ~2.2 L/km diesel → ~5.7 kg CO₂/km (per veicolo)
# Con ~30 passeggeri medi → ~0.19 kg CO₂/km per passeggero
CO2_PIEDI_KG_PER_KM = 0.0
CO2_BICI_KG_PER_KM = 0.0
CO2_BUS_KG_PER_KM_PASSEGGERO = 0.19
CO2_AUTO_KG_PER_KM = 0.12  # auto media per confronto

# Raggio per associare lampioni agli archi (in metri)
LAMP_BUFFER_METERS = 30


# ===========================================================================
# FASE 1: Costruzione del Grafo Pedonale
# ===========================================================================
def build_walking_graph(use_cache=True):
    """
    Scarica il grafo pedonale di Bari da OpenStreetMap usando osmnx.
    Se il cache esiste, lo carica da file.
    """
    cache_path = OUTPUT_DIR / "bari_walk_graph.graphml"

    if use_cache and cache_path.exists():
        print("📂 Caricamento grafo da cache...")
        G = ox.load_graphml(cache_path)
    else:
        print("🌐 Download grafo pedonale di Bari da OpenStreetMap...")
        G = ox.graph_from_bbox(
            bbox=(BBOX_NORTH, BBOX_SOUTH, BBOX_EAST, BBOX_WEST),
            network_type="walk",
        )
        ox.save_graphml(G, cache_path)
        print(f"   Salvato in {cache_path}")

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    print(f"   ✅ Grafo caricato: {n_nodes} nodi, {n_edges} archi")
    return G


# ===========================================================================
# FASE 2: Scaricare i Lampioni da Overpass API
# ===========================================================================
def fetch_street_lamps(use_cache=True):
    """
    Scarica i lampioni stradali dalla Overpass API (OpenStreetMap).
    """
    cache_path = OUTPUT_DIR / "lamps.json"

    if use_cache and cache_path.exists():
        print("📂 Caricamento lampioni da cache...")
        with open(cache_path, "r") as f:
            lamps = json.load(f)
    else:
        print("🌐 Download lampioni da Overpass API...")
        query = f"""
        [out:json][timeout:60];
        node["highway"="street_lamp"]({BBOX_SOUTH},{BBOX_WEST},{BBOX_NORTH},{BBOX_EAST});
        out;
        """
        url = "https://overpass.kumi.systems/api/interpreter"
        response = requests.get(url, params={"data": query}, timeout=120)
        data = response.json()
        lamps = data.get("elements", [])

        with open(cache_path, "w") as f:
            json.dump(lamps, f)
        print(f"   Salvati in {cache_path}")

    print(f"   ✅ Lampioni trovati: {len(lamps)}")
    return lamps


def lamps_to_geodataframe(lamps):
    """Converti la lista di lampioni in un GeoDataFrame proiettato in metri."""
    if not lamps:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:32633")

    gdf = gpd.GeoDataFrame(
        lamps,
        geometry=[Point(l["lon"], l["lat"]) for l in lamps],
        crs="EPSG:4326",
    )
    return gdf.to_crs("EPSG:32633")


# ===========================================================================
# FASE 3: Caricare i Dati CSV
# ===========================================================================
def load_csv_data():
    """Carica tutti i dataset CSV."""
    print("📂 Caricamento dati CSV...")

    # Fermate bus
    fermate = pd.read_csv(DATA_DIR / "fermate.csv")
    print(f"   Fermate bus: {len(fermate)} righe")

    # Bike sharing
    bike = pd.read_csv(DATA_DIR / "postazionibikesharing.csv", encoding="latin-1")
    print(f"   Stazioni bike sharing: {len(bike)} righe")

    # Orari transiti
    orari = pd.read_csv(DATA_DIR / "orari_fermate.csv")
    print(f"   Orari transiti: {len(orari)} righe")

    # Consumi AMTAB
    consumi = pd.read_csv(DATA_DIR / "consumi_amtab.csv")
    print(f"   Consumi AMTAB: {len(consumi)} righe")

    return fermate, bike, orari, consumi


def compute_transit_frequency(orari):
    """
    Calcola la frequenza di transiti per quartiere e ora.
    Ritorna un DataFrame con colonne: quartiere, ora, transiti_totali.
    """
    col_quartiere = "hits_hits__source_quartiere"
    col_ora = "hits_hits__source_ora"
    col_rilevazioni = "hits_hits__source_totale_rilevazioni"

    # Filtrare righe valide
    df = orari[[col_quartiere, col_ora, col_rilevazioni]].dropna()
    df[col_ora] = df[col_ora].astype(int)
    df[col_rilevazioni] = pd.to_numeric(df[col_rilevazioni], errors="coerce").fillna(0)

    # Aggregare
    freq = (
        df.groupby([col_quartiere, col_ora])[col_rilevazioni]
        .sum()
        .reset_index()
    )
    freq.columns = ["quartiere", "ora", "transiti_totali"]

    print(f"   ✅ Frequenza transiti calcolata: {len(freq)} combinazioni quartiere/ora")
    return freq


def compute_avg_bus_co2(consumi):
    """
    Calcola il consumo medio di CO₂ per km dei bus AMTAB.
    Basato su: consumo carburante (L/km) → CO₂ (kg/km).
    """
    col_consumo = "hits_hits__source_ConsumoMedioBuonoCorrente"

    # Filtrare valori validi e positivi
    vals = pd.to_numeric(consumi[col_consumo], errors="coerce").dropna()
    vals = vals[vals > 0]

    if len(vals) == 0:
        print("   ⚠️ Nessun dato di consumo valido, uso default")
        return 5.7  # kg CO₂/km per veicolo (default)

    avg_consumption_l_per_km = vals.mean()
    # Conversione: 1L diesel ≈ 2.6 kg CO₂
    avg_co2_per_km = avg_consumption_l_per_km * 2.6

    print(f"   ✅ Consumo medio bus: {avg_consumption_l_per_km:.2f} L/km")
    print(f"      → CO₂ per veicolo: {avg_co2_per_km:.1f} kg/km")
    print(f"      → CO₂ per passeggero (~30 pax): {avg_co2_per_km / 30:.3f} kg/km")
    return avg_co2_per_km


# ===========================================================================
# FASE 4: Arricchire il Grafo con Dati di Sicurezza
# ===========================================================================
def enrich_graph_with_lamps(G, lamp_gdf):
    """
    Per ogni arco del grafo, calcola la densità di lampioni
    (lampioni per km) entro un raggio di LAMP_BUFFER_METERS.
    """
    print("🔧 Arricchimento grafo con dati lampioni...")

    if lamp_gdf.empty:
        print("   ⚠️ Nessun lampione disponibile, imposto densità = 0")
        for u, v, k, data in G.edges(data=True, keys=True):
            data["lamp_density"] = 0.0
            data["lamp_count"] = 0
        return G

    # Convertire archi in GeoDataFrame proiettato in metri
    edges_gdf = ox.graph_to_gdfs(G, nodes=False)
    edges_gdf = edges_gdf.to_crs("EPSG:32633")

    # Creare buffer attorno a ogni arco
    edges_gdf["buffer_geom"] = edges_gdf.geometry.buffer(LAMP_BUFFER_METERS)
    buffer_gdf = edges_gdf.set_geometry("buffer_geom")

    # Spatial join: contare lampioni per arco
    joined = gpd.sjoin(
        buffer_gdf[["buffer_geom", "length"]].set_geometry("buffer_geom"),
        lamp_gdf[["geometry"]],
        how="left",
        predicate="contains",
    )

    # Contare lampioni per arco (basato sull'indice)
    lamp_counts = joined.groupby(joined.index).agg(
        lamp_count=("index_right", "count")
    )

    # Dove non ci sono match (NaN in index_right), il count è comunque >= 1
    # per via del left join — correggiamo i NaN
    # Se index_right era NaN per un arco → 0 lampioni
    for idx in lamp_counts.index:
        row = joined.loc[idx] if idx in joined.index else None
        if row is not None:
            if isinstance(row, pd.DataFrame):
                # Più righe → almeno un lampione
                if row["index_right"].isna().all():
                    lamp_counts.loc[idx, "lamp_count"] = 0
            else:
                # Una sola riga
                if pd.isna(row["index_right"]):
                    lamp_counts.loc[idx, "lamp_count"] = 0

    # Scrivere nel grafo
    edge_keys = list(G.edges(keys=True))
    edges_index = edges_gdf.index.tolist()

    for i, (u, v, k) in enumerate(edge_keys):
        idx = (u, v, k) if (u, v, k) in edges_gdf.index else None
        if idx is None:
            # Provare senza la key
            candidates = [(uu, vv, kk) for uu, vv, kk in edges_index if uu == u and vv == v]
            if candidates:
                idx = candidates[0]

        if idx is not None and idx in lamp_counts.index:
            count = int(lamp_counts.loc[idx, "lamp_count"])
        else:
            count = 0

        length_km = G[u][v][k].get("length", 1) / 1000
        density = count / max(length_km, 0.001)

        G[u][v][k]["lamp_count"] = count
        G[u][v][k]["lamp_density"] = density

    # Calcolare il massimo per normalizzazione
    densities = [G[u][v][k].get("lamp_density", 0) for u, v, k in G.edges(keys=True)]
    max_density = max(densities) if densities else 1

    for u, v, k in G.edges(keys=True):
        density = G[u][v][k].get("lamp_density", 0)
        # Normalizzare: 0 = massima insicurezza, 1 = massima sicurezza
        G[u][v][k]["safety_normalized"] = density / max(max_density, 0.001)

    n_with_lamps = sum(1 for u, v, k in G.edges(keys=True) if G[u][v][k]["lamp_count"] > 0)
    print(f"   ✅ Archi con almeno 1 lampione: {n_with_lamps}/{G.number_of_edges()}")
    return G


def enrich_graph_with_stops(G, fermate, bike):
    """
    Associa fermate bus e stazioni bike sharing ai nodi più vicini del grafo.
    """
    print("🔧 Associazione fermate bus e bike sharing...")

    # Fermate bus
    bus_count = 0
    for _, row in fermate.iterrows():
        try:
            lat, lon = float(row["latitudine"]), float(row["longitudine"])
            nearest = ox.nearest_nodes(G, lon, lat)
            if "fermate_bus" not in G.nodes[nearest]:
                G.nodes[nearest]["fermate_bus"] = []
            G.nodes[nearest]["fermate_bus"].append(row["idFermata"])
            G.nodes[nearest]["ha_fermata_bus"] = True
            bus_count += 1
        except Exception:
            pass
    print(f"   Fermate bus associate: {bus_count}")

    # Bike sharing
    bike_count = 0
    for _, row in bike.iterrows():
        try:
            lat, lon = float(row["Lat"]), float(row["Long"])
            nearest = ox.nearest_nodes(G, lon, lat)
            G.nodes[nearest]["ha_bike_sharing"] = True
            G.nodes[nearest]["bici_disponibili"] = int(row["Numero Bici"])
            bike_count += 1
        except Exception:
            pass
    print(f"   Stazioni bike sharing associate: {bike_count}")

    return G


# ===========================================================================
# FASE 5: Calcolo Pesi e Routing Multi-Criterio
# ===========================================================================
def compute_edge_weight(data, ora, alpha=0.33, beta=0.33, gamma=0.34):
    """
    Calcola il peso combinato di un arco.

    Parametri:
        data:  attributi dell'arco
        ora:   ora del giorno (0-23)
        alpha: peso sicurezza (0-1)
        beta:  peso carbon footprint (0-1)
        gamma: peso tempo (0-1)

    Ritorna un peso normalizzato (float ≥ 0).
    """
    length_km = data.get("length", 0) / 1000

    # --- TEMPO (minuti) ---
    tempo_min = (length_km / WALKING_SPEED_KMH) * 60

    # --- CARBON FOOTPRINT ---
    # A piedi = 0, ma se confrontiamo con l'auto, il "risparmio" è un beneficio
    co2 = CO2_PIEDI_KG_PER_KM * length_km

    # --- SICUREZZA ---
    safety_norm = data.get("safety_normalized", 0.0)

    # Di giorno (7-20), la sicurezza è quasi irrilevante
    # Di notte, l'illuminazione conta molto
    if 7 <= ora <= 20:
        insicurezza = 0.1 * (1 - safety_norm)
    else:
        insicurezza = 1.0 * (1 - safety_norm)

    # Normalizzare tempo e CO₂ su scala simile a insicurezza (0-1)
    # Usiamo una lunghezza di riferimento (1 km) per normalizzare
    tempo_normalizzato = min(tempo_min / 12.0, 1.0)  # 12 min = ~ 1 km a piedi
    co2_normalizzato = co2  # Già ~0 per camminata

    peso = (
        alpha * insicurezza
        + beta * co2_normalizzato
        + gamma * tempo_normalizzato
    )

    # Garantire che il peso sia ≥ un minimo (per evitare peso 0)
    return max(peso, 0.001)


def find_route(G, origin_latlon, dest_latlon, ora=12, alpha=0.33, beta=0.33, gamma=0.34):
    """
    Trova il percorso ottimale tra due punti con pesi multi-criterio.

    Parametri:
        G:             grafo arricchito
        origin_latlon: (latitudine, longitudine) partenza
        dest_latlon:   (latitudine, longitudine) arrivo
        ora:           ora del giorno (0-23)
        alpha:         peso sicurezza
        beta:          peso carbon footprint
        gamma:         peso tempo

    Ritorna: (lista_nodi, metriche)
    """
    # Trovare nodi più vicini
    orig_node = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest_node = ox.nearest_nodes(G, dest_latlon[1], dest_latlon[0])

    print(f"🗺️  Routing da nodo {orig_node} a nodo {dest_node}")
    print(f"   Ora: {ora}:00 | Pesi: sicurezza={alpha}, eco={beta}, tempo={gamma}")

    # Funzione peso custom
    def weight_fn(u, v, data):
        return compute_edge_weight(data, ora, alpha, beta, gamma)

    # Dijkstra con peso custom
    try:
        route = nx.shortest_path(G, orig_node, dest_node, weight=weight_fn)
    except nx.NetworkXNoPath:
        print("   ❌ Nessun percorso trovato!")
        return None, None

    # Calcolare metriche del percorso
    total_length_m = 0
    total_lamps = 0
    total_safety = 0
    n_edges = 0

    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        # Prendere il primo arco disponibile tra u e v
        edge_data = G[u][v][0] if 0 in G[u][v] else G[u][v][list(G[u][v].keys())[0]]
        total_length_m += edge_data.get("length", 0)
        total_lamps += edge_data.get("lamp_count", 0)
        total_safety += edge_data.get("safety_normalized", 0)
        n_edges += 1

    total_length_km = total_length_m / 1000
    tempo_piedi_min = (total_length_km / WALKING_SPEED_KMH) * 60
    tempo_bici_min = (total_length_km / BIKE_SPEED_KMH) * 60
    co2_risparmiata = CO2_AUTO_KG_PER_KM * total_length_km
    avg_safety = total_safety / max(n_edges, 1)

    metriche = {
        "distanza_km": round(total_length_km, 2),
        "tempo_piedi_min": round(tempo_piedi_min, 1),
        "tempo_bici_min": round(tempo_bici_min, 1),
        "lampioni_sul_percorso": total_lamps,
        "safety_score_medio": round(avg_safety * 100, 1),  # percentuale
        "co2_risparmiata_vs_auto_kg": round(co2_risparmiata, 3),
    }

    print(f"   ✅ Percorso trovato!")
    print(f"      Distanza: {metriche['distanza_km']} km")
    print(f"      Tempo a piedi: {metriche['tempo_piedi_min']} min")
    print(f"      Tempo in bici: {metriche['tempo_bici_min']} min")
    print(f"      Lampioni: {metriche['lampioni_sul_percorso']}")
    print(f"      Safety score: {metriche['safety_score_medio']}%")
    print(f"      CO₂ risparmiata vs auto: {metriche['co2_risparmiata_vs_auto_kg']} kg")

    return route, metriche


# ===========================================================================
# FASE 6: Visualizzazione con Folium
# ===========================================================================
def visualize_route(G, route, lamps, metriche, filename="safewalk_map.html"):
    """
    Crea una mappa HTML interattiva con il percorso e i lampioni.
    """
    try:
        import folium
    except ImportError:
        print("   ⚠️ folium non installato. Installa con: pip install folium")
        return

    print(f"🗺️  Generazione mappa interattiva...")

    # Centro della mappa
    center_lat = np.mean([G.nodes[n]["y"] for n in route])
    center_lon = np.mean([G.nodes[n]["x"] for n in route])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    # Disegnare il percorso
    route_coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]
    folium.PolyLine(
        route_coords,
        color="blue",
        weight=5,
        opacity=0.8,
        popup=f"Distanza: {metriche['distanza_km']} km<br>"
              f"Tempo: {metriche['tempo_piedi_min']} min<br>"
              f"Safety: {metriche['safety_score_medio']}%",
    ).add_to(m)

    # Marker partenza e arrivo
    folium.Marker(
        route_coords[0],
        popup="🟢 Partenza",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        route_coords[-1],
        popup="🔴 Arrivo",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    # Lampioni come cerchi gialli
    lamp_group = folium.FeatureGroup(name="Lampioni")
    for lamp in lamps:
        folium.CircleMarker(
            [lamp["lat"], lamp["lon"]],
            radius=3,
            color="orange",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.7,
            popup="💡 Lampione",
        ).add_to(lamp_group)
    lamp_group.add_to(m)

    # Layer control
    folium.LayerControl().add_to(m)

    # Salvare
    output_path = OUTPUT_DIR / filename
    m.save(str(output_path))
    print(f"   ✅ Mappa salvata in: {output_path}")
    return m


def visualize_comparison(G, routes_dict, lamps, filename="safewalk_comparison.html"):
    """
    Confronta più percorsi su una singola mappa.
    routes_dict: { "nome": (route, metriche, colore) }
    """
    try:
        import folium
    except ImportError:
        print("   ⚠️ folium non installato.")
        return

    print(f"🗺️  Generazione mappa comparativa...")

    # Centro basato sul primo percorso
    first_route = list(routes_dict.values())[0][0]
    center_lat = np.mean([G.nodes[n]["y"] for n in first_route])
    center_lon = np.mean([G.nodes[n]["x"] for n in first_route])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

    colors = ["blue", "red", "green", "purple", "orange"]

    for i, (nome, (route, metriche, colore)) in enumerate(routes_dict.items()):
        if route is None:
            continue
        color = colore if colore else colors[i % len(colors)]
        coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in route]

        folium.PolyLine(
            coords,
            color=color,
            weight=4,
            opacity=0.7,
            popup=(
                f"<b>{nome}</b><br>"
                f"Distanza: {metriche['distanza_km']} km<br>"
                f"Tempo: {metriche['tempo_piedi_min']} min<br>"
                f"Safety: {metriche['safety_score_medio']}%<br>"
                f"CO₂ risparmiata: {metriche['co2_risparmiata_vs_auto_kg']} kg"
            ),
        ).add_to(m)

    # Lampioni
    lamp_group = folium.FeatureGroup(name="Lampioni")
    for lamp in lamps:
        folium.CircleMarker(
            [lamp["lat"], lamp["lon"]],
            radius=2,
            color="orange",
            fill=True,
            fill_color="yellow",
            fill_opacity=0.6,
        ).add_to(lamp_group)
    lamp_group.add_to(m)

    folium.LayerControl().add_to(m)

    output_path = OUTPUT_DIR / filename
    m.save(str(output_path))
    print(f"   ✅ Mappa salvata in: {output_path}")
    return m


# ===========================================================================
# FASE 7 (BONUS): Modello ML per Safety Score
# ===========================================================================
def train_safety_model(G):
    """
    Addestra un modello di regressione per predire il safety score
    di un arco in base alle sue features.

    NOTA: il target è sintetico (basato su regole), poiché non abbiamo
    dati reali di criminalità/incidenti.
    """
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
    except ImportError:
        print("   ⚠️ scikit-learn non installato.")
        return None

    print("🤖 Addestramento modello ML per safety score...")

    # Preparare dataset
    rows = []
    for u, v, k, data in G.edges(data=True, keys=True):
        length = data.get("length", 0)
        lamp_density = data.get("lamp_density", 0)
        lamp_count = data.get("lamp_count", 0)
        highway_type = data.get("highway", "unknown")

        # Highway type encoding (semplice)
        highway_score = 0.5
        if isinstance(highway_type, str):
            if highway_type in ("primary", "secondary", "trunk"):
                highway_score = 0.8
            elif highway_type in ("tertiary", "residential"):
                highway_score = 0.6
            elif highway_type in ("footway", "path", "service"):
                highway_score = 0.3

        # Simulare diverse ore del giorno
        for ora in range(24):
            # Target sintetico
            safety = 50.0  # base
            safety += min(lamp_density * 2, 30)  # max +30 per illuminazione
            if highway_type in ("primary", "secondary"):
                safety += 10
            if not (7 <= ora <= 20):
                safety -= 25  # penalità notturna
            else:
                safety += 10  # bonus diurno
            safety = max(0, min(100, safety))

            rows.append({
                "length": length,
                "lamp_density": lamp_density,
                "lamp_count": lamp_count,
                "highway_score": highway_score,
                "ora": ora,
                "is_notte": int(not (7 <= ora <= 20)),
                "safety_target": safety,
            })

    df = pd.DataFrame(rows)

    features = ["length", "lamp_density", "lamp_count", "highway_score", "ora", "is_notte"]
    X = df[features]
    y = df["safety_target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"   ✅ Modello addestrato!")
    print(f"      MAE: {mae:.2f}")
    print(f"      R²:  {r2:.4f}")
    print(f"      Feature importances:")
    for feat, imp in sorted(zip(features, model.feature_importances_), key=lambda x: -x[1]):
        print(f"         {feat}: {imp:.3f}")

    return model


# ===========================================================================
# MAIN: Esecuzione Pipeline
# ===========================================================================
def main():
    print("=" * 60)
    print("  SafeWalk Pipeline — Multi-Criteria Routing per Bari")
    print("=" * 60)
    print()

    # --- FASE 1: Grafo ---
    G = build_walking_graph(use_cache=True)
    print()

    # --- FASE 2: Lampioni ---
    lamps = fetch_street_lamps(use_cache=True)
    lamp_gdf = lamps_to_geodataframe(lamps)
    print()

    # --- FASE 3: Dati CSV ---
    fermate, bike, orari, consumi = load_csv_data()
    transit_freq = compute_transit_frequency(orari)
    avg_co2 = compute_avg_bus_co2(consumi)
    print()

    # --- FASE 4: Arricchire grafo ---
    G = enrich_graph_with_lamps(G, lamp_gdf)
    G = enrich_graph_with_stops(G, fermate, bike)
    print()

    # --- FASE 5: Routing ---
    # ESEMPIO: dal Politecnico di Bari alla Stazione Centrale
    origin = (41.1087, 16.8785)   # Politecnico di Bari (approssimativo)
    dest = (41.1173, 16.8693)     # Stazione Centrale Bari (approssimativo)

    print("=" * 60)
    print("  SCENARIO: Politecnico → Stazione Centrale (ore 22:00)")
    print("=" * 60)

    # Percorso bilanciato
    route_balanced, metriche_balanced = find_route(
        G, origin, dest, ora=22, alpha=0.33, beta=0.33, gamma=0.34
    )
    print()

    # Percorso più sicuro
    route_safe, metriche_safe = find_route(
        G, origin, dest, ora=22, alpha=0.8, beta=0.0, gamma=0.2
    )
    print()

    # Percorso più veloce
    route_fast, metriche_fast = find_route(
        G, origin, dest, ora=22, alpha=0.0, beta=0.0, gamma=1.0
    )
    print()

    # --- FASE 6: Visualizzazione ---
    if route_balanced:
        visualize_route(G, route_balanced, lamps, metriche_balanced, "safewalk_balanced.html")
    print()

    # Mappa comparativa
    routes_dict = {}
    if route_balanced:
        routes_dict["⚖️ Bilanciato"] = (route_balanced, metriche_balanced, "blue")
    if route_safe:
        routes_dict["🛡️ Più sicuro"] = (route_safe, metriche_safe, "green")
    if route_fast:
        routes_dict["⚡ Più veloce"] = (route_fast, metriche_fast, "red")

    if routes_dict:
        visualize_comparison(G, routes_dict, lamps, "safewalk_comparison.html")
    print()

    # --- FASE 7: Modello ML (bonus) ---
    model = train_safety_model(G)
    print()

    # Tabella comparativa finale
    print("=" * 60)
    print("  RIEPILOGO PERCORSI")
    print("=" * 60)
    print(f"{'Criterio':<20} {'Bilanciato':>12} {'Più sicuro':>12} {'Più veloce':>12}")
    print("-" * 60)
    for key in ["distanza_km", "tempo_piedi_min", "safety_score_medio", "co2_risparmiata_vs_auto_kg"]:
        label = {
            "distanza_km": "Distanza (km)",
            "tempo_piedi_min": "Tempo (min)",
            "safety_score_medio": "Safety (%)",
            "co2_risparmiata_vs_auto_kg": "CO₂ risp. (kg)",
        }[key]
        vals = []
        for m in [metriche_balanced, metriche_safe, metriche_fast]:
            vals.append(str(m[key]) if m else "N/A")
        print(f"{label:<20} {vals[0]:>12} {vals[1]:>12} {vals[2]:>12}")
    print()
    print("✅ Pipeline completata! Controlla la cartella 'notebooks/output/' per le mappe.")


if __name__ == "__main__":
    main()
