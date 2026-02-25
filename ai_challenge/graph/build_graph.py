"""
build_graph.py
==============
Costruisce un grafo multi-peso per il routing urbano di Bari integrando:
  - Rete stradale/pedonale OSM (via osmnx)
  - Lampioni (Overpass API - highway=street_lamp)
  - Fermate bus AMTAB (fermate.csv)
  - Stazioni bike sharing (postazionibikesharing.csv)
  - Consumi carburante (consumi_amtab.csv) → peso ecologico
  - Orari/frequenza fermate (orari_fermate.csv) → frequenza servizio

Pesi per arco:
  - w_tempo     : tempo di percorrenza normalizzato [0,1]
  - w_ecologia  : emissioni CO2 normalizzate [0,1]
  - w_sicurezza : rischio (1 - safety_score) normalizzato [0,1]

Output: graph.graphml (grafo completo serializzato)

Utilizzo:
    python build_graph.py

Dipendenze:
    pip install osmnx==1.9.1 networkx geopandas pandas requests shapely scipy
"""

# ─────────────────────────────────────────────
import os
import math
import pickle
import requests
import numpy as np
import pandas as pd
import osmnx as ox
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point
from scipy.spatial import cKDTree

import folium
from haversine import haversine, Unit
# ─────────────────────────────────────────────

BARI_PLACE = "Bari, Italy"

# Bounding box per query Overpass (south, west, north, east)
BARI_BBOX = (41.05, 16.78, 41.20, 16.98)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

SPEED_WALK_KMH = 5.0    # velocità pedonale
SPEED_BIKE_KMH = 15.0   # velocità ciclabile
SPEED_BUS_KMH  = 20.0   # velocità media bus in città
TRANSFER_TIME_MIN = 2.0  # minuti di attesa trasbordo

CO2_DIESEL_KG_PER_LITER = 2.64  # kg CO2 per litro gasolio
TRANSFER_DIST_THRESHOLD_M = 150   # distanza max (m) fermata → nodo rete OSM
MAX_BUS_STOP_LINK_M      = 500   # distanza max (m) tra due fermate bus consecutive

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "graph.pkl")

# ─────────────────────────────────────────────
# 1. Rete OSM
# ─────────────────────────────────────────────

# ─── Tipi di highway transitabili a piedi/bici ma NON in auto/bus ────────────
_WALK_ONLY_HIGHWAYS = {
    "footway", "path", "steps", "pedestrian", "track", "cycleway",
    "living_street",
}


def load_osm_graph():
    """
    Scarica la rete pedonale e stradale di Bari da OSM e le combina
    in un unico MultiDiGraph multimodale. Cache local per velocizzare i restart.
    """
    cache_file = "graph_osm_cache.graphml"
    
    if os.path.exists(cache_file):
        print(f"[1/7] Caricamento grafo OSM dalla cache ({cache_file})...")
        G_walk = ox.load_graphml(cache_file)
        
        # Ripristino allowed_modes da stringa a set
        for u, v, k, data in G_walk.edges(keys=True, data=True):
            if 'allowed_modes' in data and isinstance(data['allowed_modes'], str):
                try:
                    data['allowed_modes'] = set(eval(data['allowed_modes']))
                except:
                    pass
        print(f"      Grafo combinato \u2192 Nodi: {G_walk.number_of_nodes()}, Archi: {G_walk.number_of_edges()}")
        return G_walk

    print("[1/7] Scaricamento grafo OSM walk + drive ... (potrebbe richiedere minuti)")

    G_walk = ox.graph_from_place(BARI_PLACE, network_type="walk")
    G_drive = ox.graph_from_place(BARI_PLACE, network_type="drive")

    G_walk  = ox.add_edge_speeds(G_walk)
    G_walk  = ox.add_edge_travel_times(G_walk)
    G_drive = ox.add_edge_speeds(G_drive)
    G_drive = ox.add_edge_travel_times(G_drive)

    print(f"      Walk  \u2192 Nodi: {G_walk.number_of_nodes()},  Archi: {G_walk.number_of_edges()}")
    print(f"      Drive \u2192 Nodi: {G_drive.number_of_nodes()}, Archi: {G_drive.number_of_edges()}")

    # --- Annota archi walk --------------------------------------------------
    for u, v, k, data in G_walk.edges(keys=True, data=True):
        hw = data.get("highway", "unclassified")
        if isinstance(hw, list):
            hw = hw[0]
        if hw in _WALK_ONLY_HIGHWAYS:
            data["allowed_modes"] = {"walk", "bike"}
        else:
            data["allowed_modes"] = {"walk", "bike", "drive", "bus"}

    # --- Aggiungi al grafo walk gli archi drive mancanti --------------------
    drive_added = 0
    for u, v, k, data in G_drive.edges(keys=True, data=True):
        # Aggiunge i nodi drive non ancora presenti
        for node in (u, v):
            if node not in G_walk.nodes:
                nd = G_drive.nodes[node]
                G_walk.add_node(node, **nd)
        if not G_walk.has_edge(u, v):
            new_data = dict(data)
            new_data["allowed_modes"] = {"drive", "bus"}
            G_walk.add_edge(u, v, **new_data)
            drive_added += 1
        else:
            # Arco gi\u00e0 presente: espande i modi ammessi
            for edata in G_walk[u][v].values():
                edata["allowed_modes"] = frozenset(edata.get("allowed_modes", set()) | {"drive", "bus"})

    print(f"      Archi drive aggiuntivi aggiunti: {drive_added}")
    print(f"      Grafo combinato \u2192 Nodi: {G_walk.number_of_nodes()}, "
          f"Archi: {G_walk.number_of_edges()}")
          
    print(f"      Salvataggio cache in {cache_file}...")
    # Converte i set in stringhe per il salvataggio
    H = G_walk.copy()
    for u, v, key, data in H.edges(keys=True, data=True):
        if 'allowed_modes' in data:
            data['allowed_modes'] = repr(list(data['allowed_modes']))
    ox.save_graphml(H, cache_file)
    
    return G_walk


# ─────────────────────────────────────────────
# 2. Lampioni da campo `lit` OSM
# ─────────────────────────────────────────────

# Valori OSM del tag `lit` che indicano illuminazione reale
LIT_YES_VALUES = {"yes", "24/7", "sunset-sunrise", "automatic", "interval"}


def fetch_lampioni_from_graph(G) -> list:
    """
    Estrae i punti illuminati direttamente dal grafo OSM già caricato:
    per ogni arco con lit in LIT_YES_VALUES, aggiunge i nodi estremi come
    punti di lampione, con source_tag = valore del campo `lit`.
    Nessuna chiamata di rete. Restituisce una lista deduplicata.
    """
    print("[2a/7] Estrazione lampioni dagli archi OSM (campo lit) ...")
    nodes_data = dict(G.nodes(data=True))
    seen: dict = {}  # (lat, lon) -> record

    for u, v, data in G.edges(data=True):
        lit = data.get("lit", None)
        if isinstance(lit, list):
            lit = lit[0]
        if not lit or lit.lower() not in LIT_YES_VALUES:
            continue
        # Aggiungi entrambi gli estremi dell'arco come punti illuminati
        for nid in (u, v):
            nd = nodes_data.get(nid, {})
            lat, lon = nd.get("y"), nd.get("x")
            if lat is None or lon is None:
                continue
            key = (round(lat, 6), round(lon, 6))
            if key not in seen:
                seen[key] = {"lat": lat, "lon": lon, "source_tag": f"lit={lit}", "lit": lit}

    print(f"      Lampioni da campo lit OSM: {len(seen)}")
    return list(seen.values())


def fetch_lampioni_from_overpass() -> list:
    """
    Integrazione opzionale: scarica da Overpass le WAY con lit=yes/24/7/sunset-sunrise
    e ne estrae i nodi come punti illuminati aggiuntivi.
    Fallback silenzioso in caso di errore.
    """
    print("[2b/7] Integrazione lampioni da Overpass API (way con lit) ...")
    south, west, north, east = BARI_BBOX
    lit_filter = "|".join(sorted(LIT_YES_VALUES))
    query = f"""
[out:json][timeout:60];
(
  way["lit"="yes"]({south},{west},{north},{east});
  way["lit"="24/7"]({south},{west},{north},{east});
  way["lit"="sunset-sunrise"]({south},{west},{north},{east});
);
out geom;
"""
    url = "https://overpass-api.de/api/interpreter"
    try:
        resp = requests.get(url, params={"data": query}, timeout=90)
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
        seen: dict = {}
        for el in elements:
            lit_val = el.get("tags", {}).get("lit", "yes")
            # Le way con `out geom` hanno la lista di nodi con lat/lon
            for node in el.get("geometry", []):
                lat, lon = node.get("lat"), node.get("lon")
                if lat is None or lon is None:
                    continue
                key = (round(lat, 6), round(lon, 6))
                if key not in seen:
                    seen[key] = {"lat": lat, "lon": lon,
                                 "source_tag": f"way_lit={lit_val}",
                                 "lit": lit_val}
        print(f"      Nodi illuminati da Overpass (way lit): {len(seen)}")
        return list(seen.values())
    except Exception as e:
        print(f"      ATTENZIONE: Overpass non disponibile ({e}). Skip integrazione.")
        return []


def merge_lampioni(from_graph: list, from_overpass: list) -> list:
    """
    Unisce i due set di lampioni deduplicando per coordinate arrotondate.
    I punti dal grafo OSM hanno priorità (source_tag più preciso).
    """
    merged: dict = {}
    for lamp in from_graph:
        key = (round(lamp["lat"], 6), round(lamp["lon"], 6))
        merged[key] = lamp
    for lamp in from_overpass:
        key = (round(lamp["lat"], 6), round(lamp["lon"], 6))
        if key not in merged:
            merged[key] = lamp
    total = len(merged)
    print(f"      Lampioni totali (merged): {total} "
          f"(grafo: {len(from_graph)}, Overpass: {len(from_overpass)})")
    return list(merged.values())


def build_lamp_kdtree(lamps: list):
    """Costruisce un KD-tree (lat/lon) per ricerca rapida dei lampioni vicini."""
    if not lamps:
        return None, None
    coords = np.array([[l["lat"], l["lon"]] for l in lamps])
    tree = cKDTree(coords)
    return tree, coords


# ─────────────────────────────────────────────
# 3. Fermate bus AMTAB
# ─────────────────────────────────────────────

def load_fermate():
    """Carica fermate.csv."""
    print("[3/7] Caricamento fermate bus ...")
    path = os.path.join(DATA_DIR, "fermate.csv")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "idFermata": "id",
        "descrizioneFermata": "desc",
        "latitudine": "lat",
        "longitudine": "lon",
    })
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    print(f"      Fermate caricate: {len(df)}")
    return df


# ─────────────────────────────────────────────
# 4. Bike sharing
# ─────────────────────────────────────────────

def load_bikesharing():
    """Carica postazionibikesharing.csv."""
    print("[4/7] Caricamento stazioni bike sharing ...")
    path = os.path.join(DATA_DIR, "postazionibikesharing.csv")
    df = pd.read_csv(path, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Denominazione": "name",
        "Lat": "lat",
        "Long": "lon",
        "Numero Bici": "num_bici",
    })
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["num_bici"] = pd.to_numeric(df["num_bici"], errors="coerce").fillna(0)
    df = df.dropna(subset=["lat", "lon"])
    # Considera solo stazioni con bici disponibili
    df = df[df["num_bici"] > 0]
    print(f"      Stazioni bike sharing con bici disponibili: {len(df)}")
    return df


# ─────────────────────────────────────────────
# 5. Consumi → peso ecologico
# ─────────────────────────────────────────────

def compute_emission_per_km():
    """
    Calcola emissioni medie di CO2 (kg/km) per modello di veicolo.
    Restituisce la media globale da usare come emissione degli archi bus.
    """
    print("[5/7] Calcolo emissioni bus ...")
    path = os.path.join(DATA_DIR, "consumi_amtab.csv")
    df = pd.read_csv(path)
    col_consumo = "hits_hits__source_ConsumoMedioBuonoCorrente"
    col_modello = "hits_hits__source_modello"

    df[col_consumo] = pd.to_numeric(df[col_consumo], errors="coerce")
    df = df.dropna(subset=[col_consumo])
    df = df[df[col_consumo] > 0]

    # Consumo medio per modello (Km/L nel dataset AMTAB)
    km_al_litro_per_modello = df.groupby(col_modello)[col_consumo].mean()
    # Converte in L/km -> 1 / (Km/L)
    litri_al_km = 1.0 / km_al_litro_per_modello
    # Emissioni: gasolio -> 2.64 kg CO2/L
    emissioni = litri_al_km * CO2_DIESEL_KG_PER_LITER
    media_globale = emissioni.mean()
    print(f"      Emissione media bus: {media_globale:.4f} kg CO2/km")
    return float(media_globale)


# ─────────────────────────────────────────────
# 6. Orari fermate → frequenza servizio
# ─────────────────────────────────────────────

def compute_freq_per_quartiere():
    """
    Calcola il numero medio di corse uniche per quartiere per ogni ora.
    Restituisce un dict: {(id_quartiere, ora): freq_normalizzata [0,1]}
    """
    print("[6/7] Calcolo frequenza bus per quartiere/ora ...")
    path = os.path.join(DATA_DIR, "orari_fermate.csv")
    df = pd.read_csv(path)
    col_quartiere = "hits_hits__source_id_quartiere"
    col_ora = "hits_hits__source_ora"
    col_corsa = "hits_hits__source_id_corsa"

    df[col_quartiere] = pd.to_numeric(df[col_quartiere], errors="coerce")
    df[col_ora] = pd.to_numeric(df[col_ora], errors="coerce")
    df = df.dropna(subset=[col_quartiere, col_ora, col_corsa])

    freq = df.groupby([col_quartiere, col_ora])[col_corsa].nunique()
    max_freq = freq.max() if len(freq) > 0 else 1
    freq_norm = (freq / max_freq).to_dict()
    print(f"      Coppie (quartiere, ora) trovate: {len(freq_norm)}")
    return freq_norm


# ─────────────────────────────────────────────
# 7. Attribuzione pesi agli archi OSM
# ─────────────────────────────────────────────

HIGHWAY_SAFETY = {
    "pedestrian": 1.0,
    "cycleway":   0.9,
    "footway":    0.9,
    "path":       0.85,
    "living_street": 0.8,
    "residential":0.7,
    "unclassified":0.65,
    "tertiary":   0.55,
    "secondary":  0.5,
    "primary":    0.3,
    "trunk":      0.1,
    "motorway":   0.0,
}

LIT_SCORE = {
    "yes":   1.0,
    "24/7":  1.0,
    "sunset-sunrise": 1.0,
    "no":    0.0,
    "limited": 0.4,
}


def haversine_m(lat1, lon1, lat2, lon2):
    """Distanza in metri tra due coordinate geografiche."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def lamp_density_score(lamp_tree, lat, lon, radius_deg=0.002):
    """
    Restituisce un punteggio [0,1] basato sulla densità di lampioni
    entro ~200m dal punto (lat, lon).
    """
    if lamp_tree is None:
        return 0.0
    count = len(lamp_tree.query_ball_point([lat, lon], radius_deg))
    # Normalizzazione: ≥10 lampioni → punteggio 1.0
    return min(count / 10.0, 1.0)


def assign_osm_edge_weights(G, lamp_tree, emission_kg_km):
    """
    Aggiunge attributi di peso a ogni arco del grafo OSM multimodale:
      - w_tempo_raw    (minuti, in base alla velocità dell'arco)
      - w_eco_raw      (kg CO2: 0 per walk/bike, emission_kg_km per drive/bus)
      - w_sicurezza    (rischio [0,1], non normalizzato)
      - modalita       ('walk' | 'drive' | 'mixed')
    L'attributo `allowed_modes` (già impostato da load_osm_graph) viene
    preservato invariato.
    """
    print("[7/7] Calcolo pesi archi OSM ...")
    nodes_data = dict(G.nodes(data=True))

    for u, v, data in G.edges(data=True):
        # Salta archi non-OSM già processati (bus, transfer, lampione)
        if data.get("modalita") in ("bus", "transfer", "lampione"):
            continue

        length_m = data.get("length", 0)  # metri
        length_km = length_m / 1000.0

        # ── MODALITÀ PRINCIPALE dell'arco ──────────────────
        modes = data.get("allowed_modes", {"walk", "bike"})
        if "drive" in modes or "bus" in modes:
            if "walk" in modes or "bike" in modes:
                main_mode = "mixed"
            else:
                main_mode = "drive"
        else:
            main_mode = "walk"

        # ── TEMPO ──────────────────────────────────────────
        speed = data.get("speed_kph", SPEED_WALK_KMH)
        if isinstance(speed, list):
            speed = float(speed[0])
        if speed <= 0:
            speed = SPEED_WALK_KMH
        w_tempo = (length_km / speed) * 60  # minuti

        # ── ECOLOGIA ───────────────────────────────────────
        # Le emissioni si contano solo se l'arco è percorso con veicolo motorizzato
        if main_mode in ("drive", "mixed"):
            w_eco = length_km * emission_kg_km
        else:
            w_eco = 0.0

        # ── SICUREZZA ──────────────────────────────────────
        highway = data.get("highway", "unclassified")
        if isinstance(highway, list):
            highway = highway[0]
        hw_score = HIGHWAY_SAFETY.get(highway, 0.5)

        lit = data.get("lit", None)
        if isinstance(lit, list):
            lit = lit[0]
        lit_score_val = LIT_SCORE.get(lit, 0.3) if lit else 0.3

        # Densità lampioni vicini al punto medio dell'arco
        u_data = nodes_data.get(u, {})
        v_data = nodes_data.get(v, {})
        mid_lat = (u_data.get("y", 0) + v_data.get("y", 0)) / 2
        mid_lon = (u_data.get("x", 0) + v_data.get("x", 0)) / 2
        lamp_score = lamp_density_score(lamp_tree, mid_lat, mid_lon)

        safety_score = 0.4 * lit_score_val + 0.3 * hw_score + 0.3 * lamp_score
        w_sicurezza = 1.0 - safety_score  # rischio (da minimizzare)

        data["w_tempo_raw"]    = w_tempo
        data["w_eco_raw"]      = w_eco
        data["w_sicurezza_raw"] = w_sicurezza
        data["modalita"]       = main_mode

    print(f"      Pesi calcolati per {G.number_of_edges()} archi OSM.")


# ─────────────────────────────────────────────
# 8. Aggiunta nodi bus, bike sharing e lampioni
#    (Ottimizzato con KDTree batch)
# ─────────────────────────────────────────────

def _build_osm_kdtree(G):
    """
    Pre-calcola un KDTree su tutti i nodi OSM (interi) del grafo.
    Restituisce (tree, osm_node_list) dove osm_node_list[i] è il node_id.
    """
    osm_nodes = []
    coords = []
    for n, d in G.nodes(data=True):
        if d.get("node_type") is None:  # nodo OSM puro (senza tipo speciale)
            osm_nodes.append(n)
            coords.append([d.get("y", 0), d.get("x", 0)])
    tree = cKDTree(np.array(coords))
    return tree, osm_nodes


def add_bus_stops(G, fermate_df, osm_tree, osm_nodes):
    """
    Aggiunge le fermate bus come nodi e le collega al nodo OSM più vicino
    con archi di transfer bidirezionali (modalita='transfer').
    Usa KDTree batch per velocità.
    """
    print("    Aggiunta nodi fermate bus ...")

    # Aggiunge tutti i nodi bus
    bus_ids = []
    bus_coords = []
    for _, row in fermate_df.iterrows():
        node_id = f"bus_{row['id']}"
        G.add_node(node_id,
                   x=row["lon"], y=row["lat"],
                   node_type="bus_stop",
                   fermata_id=row["id"],
                   desc=row.get("desc", ""))
        bus_ids.append(node_id)
        bus_coords.append([row["lat"], row["lon"]])

    # Batch nearest-neighbor
    bus_coords_arr = np.array(bus_coords)
    _, indices = osm_tree.query(bus_coords_arr)

    count = 0
    for i, node_id in enumerate(bus_ids):
        nearest = osm_nodes[indices[i]]
        n_data = G.nodes[nearest]
        dist_m = haversine_m(bus_coords[i][0], bus_coords[i][1],
                             n_data.get("y", 0), n_data.get("x", 0))
        if dist_m <= TRANSFER_DIST_THRESHOLD_M:
            for src, dst in [(nearest, node_id), (node_id, nearest)]:
                G.add_edge(src, dst,
                           length=dist_m,
                           w_tempo_raw=TRANSFER_TIME_MIN,
                           w_eco_raw=0.0,
                           w_sicurezza_raw=0.0,
                           modalita="transfer",
                           allowed_modes={"walk", "bike", "drive", "bus"})
            count += 1
    print(f"      Fermate bus aggiunte con transfer: {count}/{len(fermate_df)}")


def add_bus_routes(G, fermate_df, emission_kg_km: float):
    """
    Collega le fermate bus consecutive (entro MAX_BUS_STOP_LINK_M) con archi
    diretti di modalita='bus'. Simula le tratte effettive percorse dal bus.
    """
    print("    Aggiunta archi bus-bus tra fermate vicine ...")

    bus_nodes = [
        (f"bus_{row['id']}", row["lat"], row["lon"])
        for _, row in fermate_df.iterrows()
        if f"bus_{row['id']}" in G.nodes
    ]

    if not bus_nodes:
        print("      Nessun nodo fermata trovato nel grafo, skip.")
        return

    coords = np.array([[lat, lon] for _, lat, lon in bus_nodes])
    tree = cKDTree(coords)
    radius_deg = MAX_BUS_STOP_LINK_M / 111_000.0

    count = 0
    for i, (nid_a, lat_a, lon_a) in enumerate(bus_nodes):
        indices = tree.query_ball_point([lat_a, lon_a], radius_deg)
        for j in indices:
            if j == i:
                continue
            nid_b, lat_b, lon_b = bus_nodes[j]
            dist_m = haversine_m(lat_a, lon_a, lat_b, lon_b)
            if dist_m > MAX_BUS_STOP_LINK_M:
                continue
            w_tempo = (dist_m / 1000.0 / SPEED_BUS_KMH) * 60
            w_eco = (dist_m / 1000.0) * emission_kg_km / 50.0
            for src, dst in [(nid_a, nid_b), (nid_b, nid_a)]:
                if not G.has_edge(src, dst):
                    G.add_edge(src, dst,
                               length=dist_m,
                               w_tempo_raw=w_tempo,
                               w_eco_raw=w_eco,
                               w_sicurezza_raw=0.2,
                               modalita="bus",
                               allowed_modes={"bus"})
                    count += 1

    print(f"      Archi bus-bus aggiunti: {count}")


def add_bike_stations(G, bike_df, osm_tree, osm_nodes):
    """
    Aggiunge le stazioni di bike sharing come nodi e le collega alla rete OSM.
    Usa KDTree batch per velocità.
    """
    print("    Aggiunta nodi bike sharing ...")

    bike_ids = []
    bike_coords = []
    for _, row in bike_df.iterrows():
        node_id = f"bike_{row.name}"
        G.add_node(node_id,
                   x=row["lon"], y=row["lat"],
                   node_type="bike_station",
                   name=row.get("name", ""),
                   num_bici=row["num_bici"])
        bike_ids.append(node_id)
        bike_coords.append([row["lat"], row["lon"]])

    bike_coords_arr = np.array(bike_coords)
    _, indices = osm_tree.query(bike_coords_arr)

    count = 0
    for i, node_id in enumerate(bike_ids):
        nearest = osm_nodes[indices[i]]
        n_data = G.nodes[nearest]
        dist_m = haversine_m(bike_coords[i][0], bike_coords[i][1],
                             n_data.get("y", 0), n_data.get("x", 0))
        if dist_m <= TRANSFER_DIST_THRESHOLD_M:
            for src, dst in [(nearest, node_id), (node_id, nearest)]:
                G.add_edge(src, dst,
                           length=dist_m,
                           w_tempo_raw=TRANSFER_TIME_MIN,
                           w_eco_raw=0.0,
                           w_sicurezza_raw=0.0,
                           modalita="transfer",
                           allowed_modes={"walk", "bike"})
            count += 1
    print(f"      Stazioni bike sharing aggiunte con transfer: {count}/{len(bike_df)}")


def add_lampioni(G, lamps: list, osm_tree, osm_nodes) -> None:
    """
    Aggiunge i punti illuminati come nodi (node_type='lampione') nel grafo
    e li collega al nodo OSM più vicino con un arco di modalita='lampione'.
    Usa KDTree batch per velocità.
    """
    print("    Aggiunta nodi lampioni ...")
    if not lamps:
        print("      Nessun lampione trovato.")
        return

    # Aggiunge tutti i nodi lampione
    lamp_ids = []
    lamp_coords = []
    for i, lamp in enumerate(lamps):
        node_id = f"lamp_{i}"
        G.add_node(node_id,
                   x=lamp["lon"], y=lamp["lat"],
                   node_type="lampione",
                   source_tag=lamp.get("source_tag", "lit=yes"),
                   lit=lamp.get("lit", "yes"))
        lamp_ids.append(node_id)
        lamp_coords.append([lamp["lat"], lamp["lon"]])

    # Batch nearest-neighbor
    lamp_coords_arr = np.array(lamp_coords)
    _, indices = osm_tree.query(lamp_coords_arr)

    count = 0
    for i, node_id in enumerate(lamp_ids):
        nearest = osm_nodes[indices[i]]
        n_data = G.nodes[nearest]
        dist_m = haversine_m(lamp_coords[i][0], lamp_coords[i][1],
                             n_data.get("y", 0), n_data.get("x", 0))
        if dist_m <= TRANSFER_DIST_THRESHOLD_M:
            G.add_edge(nearest, node_id,
                       length=dist_m,
                       w_tempo_raw=0.0,
                       w_eco_raw=0.0,
                       w_sicurezza_raw=0.0,
                       modalita="lampione")
            count += 1
    print(f"      Lampioni aggiunti al grafo: {len(lamps)} (con arco: {count})")


# ─────────────────────────────────────────────
# 9. Normalizzazione pesi globale
# ─────────────────────────────────────────────

def normalize_weights(G):
    """
    Normalizza w_tempo_raw, w_eco_raw, w_sicurezza_raw in [0,1]
    e salva come w_tempo, w_ecologia, w_sicurezza.
    """
    print("    Normalizzazione pesi ...")
    tempi = [d.get("w_tempo_raw", 0) for _, _, d in G.edges(data=True)]
    ecos  = [d.get("w_eco_raw", 0)   for _, _, d in G.edges(data=True)]
    sics  = [d.get("w_sicurezza_raw", 0) for _, _, d in G.edges(data=True)]

    max_tempo = max(tempi) if tempi else 1
    max_eco   = max(ecos)  if max(ecos) > 0 else 1
    max_sic   = max(sics)  if max(sics) > 0 else 1

    for u, v, data in G.edges(data=True):
        data["w_tempo"]    = data.get("w_tempo_raw", 0)    / max_tempo
        data["w_ecologia"] = data.get("w_eco_raw", 0)      / max_eco
        data["w_sicurezza"]= data.get("w_sicurezza_raw", 0)/ max_sic

    print("      Normalizzazione completata.")


# ─────────────────────────────────────────────
# 10. Salvataggio
# ─────────────────────────────────────────────

def save_graph(G):
    """Salva il grafo in formato pickle (preserva tutti gli attributi Python)."""
    print(f"    Salvataggio grafo in {OUTPUT_FILE} ...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("      Grafo salvato.")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  BUILD GRAPH — routing urbano Bari")
    print("=" * 60)

    # 1. Grafo OSM
    G = load_osm_graph()

    # 2. Lampioni da campo lit OSM (senza rete)
    lamps_from_graph = fetch_lampioni_from_graph(G)

    # 2b. Integrazione opzionale da Overpass (way con lit=yes)
    lamps_from_overpass = fetch_lampioni_from_overpass()

    # Unione dei due set (deduplicata)
    lamps = merge_lampioni(lamps_from_graph, lamps_from_overpass)
    lamp_tree, _ = build_lamp_kdtree(lamps)

    # 3. Fermate bus
    fermate_df = load_fermate()

    # 4. Bike sharing
    bike_df = load_bikesharing()

    # 5. Emissioni bus
    emission_kg_km = compute_emission_per_km()

    # 6. Frequenza (opzionale — disponibile per uso futuro nei pesi bus)
    freq_map = compute_freq_per_quartiere()

    # 7. Pesi archi OSM
    assign_osm_edge_weights(G, lamp_tree, emission_kg_km)

    # 7b. KDTree batch per nodi OSM (velocizza l'aggiunta dei nodi)
    print("    Costruzione KDTree batch nodi OSM ...")
    osm_tree, osm_nodes = _build_osm_kdtree(G)

    # 8. Aggiungi nodi aggiuntivi (con KDTree batch)
    add_bus_stops(G, fermate_df, osm_tree, osm_nodes)
    add_bus_routes(G, fermate_df, emission_kg_km)
    add_bike_stations(G, bike_df, osm_tree, osm_nodes)
    add_lampioni(G, lamps, osm_tree, osm_nodes)

    # 9. Normalizza
    normalize_weights(G)

    # 10. Salva metadati globali per consistenza nel router
    G.graph["emissioni_auto_kg_km"] = 0.15
    G.graph["emissioni_bus_kg_km_per_persona"] = float(emission_kg_km / 50.0)

    # 11. Salva
    save_graph(G)

    print("\n✅ Grafo costruito con successo!")
    print(f"   Nodi totali : {G.number_of_nodes()}")
    print(f"   Archi totali: {G.number_of_edges()}")
    # Riepilogo per tipo di nodo
    type_counts = {}
    for _, d in G.nodes(data=True):
        t = d.get("node_type", "osm")
        type_counts[t] = type_counts.get(t, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"   - {t}: {c} nodi")
    print(f"   File output : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
