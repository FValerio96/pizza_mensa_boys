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

# Velocità medie (km/h)
WALKING_SPEED_KMH = 5.0
BUS_SPEED_KMH = 30.0
BIKE_SPEED_KMH = 17.0

# Emissioni CO₂
# Bus AMTAB: consumo medio ~2.2 L/km diesel → ~5.7 kg CO₂/km (per veicolo)
# Con ~30 passeggeri medi → ~0.19 kg CO₂/km per passeggero
CO2_PIEDI_KG_PER_KM = 0.0
CO2_BICI_KG_PER_KM = 0.0
CO2_BUS_KG_PER_KM_PASSEGGERO = 0.19
CO2_AUTO_KG_PER_KM = 0.5  # auto media per confronto

# Raggio per associare lampioni agli archi (in metri)
LAMP_BUFFER_METERS = 30


# ===========================================================================
# FASE 1: Costruzione del Grafo Pedonale
# ===========================================================================
def build_walking_graph(use_cache=True):
    """
    Scarica il grafo pedonale di Bari da OpenStreetMap usando osmnx.
    Se il cache esiste, lo carica da file.
    Aggiunge il tag 'lit' (illuminazione) ai tag WAY scaricati.
    """
    cache_path = OUTPUT_DIR / "bari_walk_graph.graphml"

    # Aggiungere 'lit' ai tag degli archi scaricati da OSM
    default_tags = ox.settings.useful_tags_way
    if "lit" not in default_tags:
        ox.settings.useful_tags_way = list(default_tags) + ["lit"]

    if use_cache and cache_path.exists():
        print("📂 Caricamento grafo da cache...")
        G = ox.load_graphml(cache_path)
    else:
        print("🌐 Download grafo pedonale di Bari da OpenStreetMap...")
        G = ox.graph_from_place("Bari, Italy", network_type="all")
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
        query = """
        [out:json][timeout:60];
        area["name"="Bari"]["admin_level"="8"]->.bari;
        node["highway"="street_lamp"](area.bari);
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


def fetch_lit_ways(use_cache=True):
    """
    Scarica da Overpass gli ID delle way taggate lit=yes/24/7/automatic a Bari.
    Ritorna un set di osmid (int).
    """
    cache_path = OUTPUT_DIR / "lit_ways.json"

    if use_cache and cache_path.exists():
        print("📂 Caricamento strade illuminate da cache...")
        with open(cache_path, "r") as f:
            lit_ids = json.load(f)
    else:
        print("🌐 Download strade illuminate (lit=yes) da Overpass API...")
        query = """
        [out:json][timeout:60];
        area["name"="Bari"]["admin_level"="8"]->.bari;
        way["lit"~"yes|24/7|automatic"](area.bari);
        out ids;
        """
        url = "https://overpass.kumi.systems/api/interpreter"
        response = requests.get(url, params={"data": query}, timeout=120)
        data = response.json()
        lit_ids = [e["id"] for e in data.get("elements", [])]

        with open(cache_path, "w") as f:
            json.dump(lit_ids, f)
        print(f"   Salvati in {cache_path}")

    lit_set = set(lit_ids)
    print(f"   ✅ Strade illuminate trovate: {len(lit_set)}")
    return lit_set


def fetch_bus_routes(use_cache=True):
    """
    Scarica le linee bus (relazioni route=bus) da Overpass per Bari.
    Costruisce un mapping dalle fermate OSM alle linee bus che le servono.
    Ritorna (osm_bus_stops, line_info) dove:
      - osm_bus_stops: lista di {lat, lon, lines: [ref, ...]}
      - line_info: dict {ref: {"name": str}}
    """
    cache_path = OUTPUT_DIR / "bus_routes.json"

    if use_cache and cache_path.exists():
        print("📂 Caricamento linee bus da cache...")
        with open(cache_path, "r") as f:
            cached = json.load(f)
        print(f"   ✅ Linee bus: {len(cached['line_info'])}, "
              f"fermate OSM con info linee: {len(cached['osm_bus_stops'])}")
        return cached["osm_bus_stops"], cached["line_info"]

    print("🌐 Download linee bus da Overpass API...")
    urls = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]

    def _overpass_query(query, retries=3, wait=10):
        import time as _time
        for attempt in range(retries):
            for url in urls:
                try:
                    resp = requests.post(url, data={"data": query}, timeout=180)
                    if resp.status_code == 200 and resp.text.strip().startswith("{"):
                        return resp.json()
                    if resp.status_code == 429:
                        print(f"   ⏳ Rate limit, attendo {wait}s...")
                        _time.sleep(wait)
                except Exception:
                    continue
            if attempt < retries - 1:
                print(f"   ⏳ Retry {attempt+2}/{retries} tra {wait}s...")
                _time.sleep(wait)
        return None

    # Step 1: Scarica le relazioni bus con i loro membri
    query1 = """
    [out:json][timeout:120];
    area["name"="Bari"]["admin_level"="8"]->.bari;
    rel["type"="route"]["route"="bus"](area.bari);
    out body;
    """
    data1 = _overpass_query(query1)
    if not data1:
        print("   ⚠️ Overpass non raggiungibile, nessuna info linee bus.")
        return [], {}
    routes = data1.get("elements", [])

    # Step 2: Scarica le coordinate dei nodi fermata (stop + platform)
    query2 = """
    [out:json][timeout:60];
    area["name"="Bari"]["admin_level"="8"]->.bari;
    rel["type"="route"]["route"="bus"](area.bari)->.routes;
    (node(r.routes:"stop"); node(r.routes:"platform"););
    out;
    """
    data2 = _overpass_query(query2)
    stop_nodes = data2.get("elements", []) if data2 else []
    node_coords = {n["id"]: (n["lat"], n["lon"]) for n in stop_nodes}

    # Parse: per ogni linea, estrai ref e fermate
    line_info = {}
    node_to_lines = {}  # osm_node_id -> set of line refs

    for rel in routes:
        tags = rel.get("tags", {})
        ref = tags.get("ref", "").strip()
        if not ref:
            ref = tags.get("name", f"#{rel['id']}")
        name = tags.get("name", ref)

        if ref not in line_info:
            line_info[ref] = {"name": name}

        for member in rel.get("members", []):
            role = member.get("role", "")
            if member["type"] == "node" and (
                "stop" in role or "platform" in role
            ):
                nid = member["ref"]
                if nid not in node_to_lines:
                    node_to_lines[nid] = set()
                node_to_lines[nid].add(ref)

    # Costruire lista osm_bus_stops
    osm_bus_stops = []
    for nid, lines in node_to_lines.items():
        if nid in node_coords:
            lat, lon = node_coords[nid]
            osm_bus_stops.append({
                "lat": lat,
                "lon": lon,
                "lines": sorted(lines),
            })

    cached = {
        "osm_bus_stops": osm_bus_stops,
        "line_info": line_info,
    }
    with open(cache_path, "w") as f:
        json.dump(cached, f)
    print(f"   Salvate in {cache_path}")

    print(f"   ✅ Linee bus: {len(line_info)}, "
          f"fermate OSM con info linee: {len(osm_bus_stops)}")
    return osm_bus_stops, line_info


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


def load_edge_risk():
    """Carica il file bari_edges_risk.csv e restituisce un dizionario
    {(u, v, key): {"risk_score": float, "risk_category": str}}."""
    csv_path = DATA_DIR / "bari_edges_risk.csv"
    print(f"📂 Caricamento rischio archi da {csv_path}...")
    df = pd.read_csv(csv_path, usecols=["u", "v", "key", "risk_score", "risk_category"])
    # Vectorized: molto più veloce di iterrows()
    keys = list(zip(df["u"].astype(int), df["v"].astype(int), df["key"].astype(int)))
    vals = [{"risk_score": float(r), "risk_category": str(c)}
            for r, c in zip(df["risk_score"], df["risk_category"])]
    risk_map = dict(zip(keys, vals))
    print(f"   ✅ Rischio caricato per {len(risk_map)} archi")
    return risk_map


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
def enrich_graph_with_lamps(G, lamp_gdf, lit_way_ids=None):
    """
    Per ogni arco del grafo, calcola la densità di lampioni
    (lampioni per km) entro un raggio di LAMP_BUFFER_METERS.
    Marca anche is_lit=1 se l'osmid è tra le way illuminate.
    """
    print("🔧 Arricchimento grafo con dati lampioni...")
    if lit_way_ids is None:
        lit_way_ids = set()

    if lamp_gdf.empty:
        print("   ⚠️ Nessun lampione disponibile, imposto densità = 0")
        for u, v, k, data in G.edges(data=True, keys=True):
            data["lamp_density"] = 0.0
            data["lamp_count"] = 0
            data["is_lit"] = 0
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
    # count() esclude automaticamente i NaN, quindi archi senza lampioni → 0
    lamp_counts = joined.groupby(joined.index)["index_right"].count().rename("lamp_count")

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
            count = int(lamp_counts[idx])
        else:
            count = 0

        # FALLBACK: Se non ci sono lampioni dal spatial join,
        # controlla il tag "lit" già presente negli archi OSM.
        # osmnx può salvare il valore come stringa, lista o stringa-lista da graphml
        if count == 0:
            lit_tag = G[u][v][k].get("lit", "no")
            # Normalizzare: lista → primo elemento, stringified list → parse
            if isinstance(lit_tag, list):
                lit_tag = lit_tag[0] if lit_tag else "no"
            lit_tag = str(lit_tag).strip("[']\" ")
            if lit_tag in ("yes", "24/7", "automatic"):
                length_km_fallback = G[u][v][k].get("length", 1) / 1000
                # Stima approssimativa: ~1 lampione ogni 30m
                count = max(1, int((length_km_fallback * 1000) / 30))

        length_km = G[u][v][k].get("length", 1) / 1000
        density = count / max(length_km, 0.001)

        G[u][v][k]["lamp_count"] = count
        G[u][v][k]["lamp_density"] = density

        # Marcare is_lit da Overpass lit ways (match per osmid)
        osmid = G[u][v][k].get("osmid", None)
        is_lit = False
        if osmid is not None:
            if isinstance(osmid, (list, tuple)):
                is_lit = any(int(oid) in lit_way_ids for oid in osmid)
            else:
                is_lit = int(osmid) in lit_way_ids
        # Se count > 0 da lampioni, consideriamo anche illuminato
        if count > 0:
            is_lit = True
        G[u][v][k]["is_lit"] = int(is_lit)

    # --- Calcolo safety_normalized rule-based (0-1) ---
    # Ogni arco riceve un punteggio basato su:
    #   - Illuminazione (is_lit o lamp_count > 0): +0.40
    #   - Densità lampioni (normalizzata): fino a +0.25
    #   - Tipo di strada: fino a +0.20
    #   - Prossimità trasporto pubblico: fino a +0.15
    densities = [G[u][v][k].get("lamp_density", 0) for u, v, k in G.edges(keys=True)]
    max_density = max(densities) if densities else 1

    for u, v, k in G.edges(keys=True):
        data = G[u][v][k]
        score = 0.0

        # Illuminazione: componente dominante
        is_lit = int(float(data.get("is_lit", 0)))
        lamp_count = int(float(data.get("lamp_count", 0)))
        if is_lit or lamp_count > 0:
            score += 0.40

        # Densità lampioni (normalizzata)
        density = float(data.get("lamp_density", 0))
        score += 0.25 * (density / max(max_density, 0.001))

        # Tipo di strada
        highway_type = data.get("highway", "unknown")
        if isinstance(highway_type, list):
            highway_type = highway_type[0]
        hw = str(highway_type)
        if hw in ("primary", "secondary", "trunk"):
            score += 0.20
        elif hw in ("tertiary", "residential", "living_street"):
            score += 0.15
        elif hw in ("pedestrian", "cycleway"):
            score += 0.10
        elif hw in ("footway", "path", "service", "track"):
            score += 0.05

        # Prossimità trasporto pubblico
        has_bus = (
            G.nodes[u].get("ha_fermata_bus", False)
            or G.nodes[v].get("ha_fermata_bus", False)
        )
        has_bike = (
            G.nodes[u].get("ha_bike_sharing", False)
            or G.nodes[v].get("ha_bike_sharing", False)
        )
        if has_bus:
            score += 0.10
        if has_bike:
            score += 0.05

        data["safety_normalized"] = max(0.0, min(1.0, score))
        data["safety_base"] = data["safety_normalized"]

    n_with_lamps = sum(1 for u, v, k in G.edges(keys=True) if G[u][v][k]["lamp_count"] > 0)
    n_is_lit = sum(1 for u, v, k in G.edges(keys=True) if G[u][v][k]["is_lit"])
    scores = [G[u][v][k]["safety_normalized"] for u, v, k in G.edges(keys=True)]
    n_safe = sum(1 for s in scores if s >= 0.65)
    n_medium = sum(1 for s in scores if 0.25 < s < 0.65)
    n_danger = sum(1 for s in scores if s <= 0.25)
    print(f"   ✅ Archi con almeno 1 lampione: {n_with_lamps}/{G.number_of_edges()}")
    print(f"   ✅ Archi illuminati (is_lit): {n_is_lit}/{G.number_of_edges()}")
    print(f"   ✅ Distribuzione sicurezza:")
    print(f"      🟢 Sicuro (≥0.65):  {n_safe}")
    print(f"      🟠 Medio (0.25-0.65): {n_medium}")
    print(f"      🔴 Pericolo (≤0.25): {n_danger}")
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
            G.nodes[nearest]["nome_stazione_bike"] = row["Denominazione"]
            bike_count += 1
        except Exception:
            pass
    print(f"   Stazioni bike sharing associate: {bike_count}")
    return G


def enrich_graph_with_risk(G, risk_map):
    """Arricchisce il grafo con i dati di rischio dal CSV.
    Per ogni arco, assegna safety_normalized (invertito: più è alto, più è sicuro)
    e risk_category per la visualizzazione."""
    print("🔧 Arricchimento grafo con dati di rischio dal CSV...")

    # Trovo il max risk_score per normalizzare
    max_risk = max((v["risk_score"] for v in risk_map.values()), default=1.0)
    if max_risk <= 0:
        max_risk = 1.0

    matched = 0
    unmatched = 0
    for u, v, k in G.edges(keys=True):
        key = (u, v, k)
        if key in risk_map:
            rs = risk_map[key]["risk_score"]
            cat = risk_map[key]["risk_category"]
            # safety_normalized: 1 = sicuro, 0 = pericoloso
            safety = max(0.0, min(1.0, 1.0 - (rs / max_risk)))
            G[u][v][k]["safety_normalized"] = safety
            G[u][v][k]["safety_base"] = safety
            G[u][v][k]["risk_score"] = rs
            G[u][v][k]["risk_category"] = cat
            matched += 1
        else:
            # Arco non presente nel CSV: assegna sicurezza media
            G[u][v][k]["safety_normalized"] = 0.5
            G[u][v][k]["safety_base"] = 0.5
            G[u][v][k]["risk_score"] = 0.0
            G[u][v][k]["risk_category"] = "Sconosciuto"
            unmatched += 1

    # Statistiche
    scores = [G[u][v][k]["safety_normalized"] for u, v, k in G.edges(keys=True)]
    n_safe = sum(1 for s in scores if s >= 0.6)
    n_medium = sum(1 for s in scores if 0.15 < s < 0.6)
    n_danger = sum(1 for s in scores if s <= 0.15)
    print(f"   ✅ Archi con rischio dal CSV: {matched}/{G.number_of_edges()}")
    print(f"   ⚠️ Archi senza dati (default 0.5): {unmatched}")
    print(f"   ✅ Distribuzione sicurezza:")
    print(f"      🟢 Sicuro (≥0.60):  {n_safe}")
    print(f"      🟠 Medio (0.15-0.60): {n_medium}")
    print(f"      🔴 Pericolo (≤0.15): {n_danger}")
    return G


def precompute_stop_nodes(G, fermate_df, bike_df):
    """Pre-calcola ox.nearest_nodes per tutte le fermate bus e stazioni bike.
    Ritorna (bus_nodes_list, bike_stations_list) pronti per il routing."""
    print("🔧 Pre-calcolo nodi fermate e stazioni bike...")
    bus_nodes = []
    for _, row in fermate_df.iterrows():
        try:
            lat, lon = float(row["latitudine"]), float(row["longitudine"])
            node = ox.nearest_nodes(G, lon, lat)
            bus_nodes.append({
                "node": node,
                "id": row["idFermata"],
                "nome": row["descrizioneFermata"],
                "lat": lat,
                "lon": lon,
            })
        except Exception:
            pass
    print(f"   Fermate bus pre-calcolate: {len(bus_nodes)}")

    bike_stations = []
    for _, row in bike_df.iterrows():
        try:
            lat, lon = float(row["Lat"]), float(row["Long"])
            node = ox.nearest_nodes(G, lon, lat)
            bike_stations.append({
                "node": node,
                "nome": row["Denominazione"],
                "lat": lat,
                "lon": lon,
                "bici": int(row["Numero Bici"]),
            })
        except Exception:
            pass
    print(f"   Stazioni bike pre-calcolate: {len(bike_stations)}")
    return bus_nodes, bike_stations

    return G




# ===========================================================================
# FASE 4b: Bonus Notturno di Sicurezza
# ===========================================================================
def apply_night_bonus(G, ora):
    """
    Tra le 19:00 e le 05:00, aumenta la sicurezza delle strade illuminate
    e riduce quella delle strade al buio.
    """
    is_night = (ora >= 19 or ora <= 5)
    for u, v, k in G.edges(keys=True):
        data = G[u][v][k]
        base = float(data.get("safety_base", data.get("safety_normalized", 0)))
        if is_night:
            is_lit = int(float(data.get("is_lit", 0)))
            lamp_count = int(float(data.get("lamp_count", 0)))
            if is_lit or lamp_count > 0:
                data["safety_normalized"] = min(1.0, base + 0.15)
            else:
                data["safety_normalized"] = max(0.0, base - 0.10)
        else:
            data["safety_normalized"] = base


# ===========================================================================
# FASE 5: Routing Multi-Criterio (Bici + Bus)
# ===========================================================================
def _edge_weight_bike(data, ora):
    """Peso per routing bici: privilegia sicurezza di notte, velocità di giorno."""
    length_km = data.get("length", 0) / 1000
    tempo_min = (length_km / BIKE_SPEED_KMH) * 60
    safety = float(data.get("safety_normalized", 0))

    # Bonus/malus per tipo strada
    hw = str(data.get("highway", ""))
    if isinstance(data.get("highway"), list):
        hw = str(data["highway"][0])
    hw_bonus = 0.0
    if hw in ("cycleway",):
        hw_bonus = -0.3   # preferisci ciclabili
    elif hw in ("primary", "trunk", "motorway"):
        hw_bonus = 0.5    # evita strade ad alto traffico
    elif hw in ("residential", "living_street", "pedestrian"):
        hw_bonus = -0.1

    if 7 <= ora <= 20:
        insicurezza = 0.2 * (1 - safety)
    else:
        insicurezza = 0.8 * (1 - safety)

    tempo_norm = min(tempo_min / 12.0, 1.0)
    peso = 0.4 * insicurezza + 0.6 * tempo_norm + hw_bonus * length_km
    return max(peso, 0.001)


def _path_length(G, path):
    """Calcola la lunghezza totale in metri di un percorso (lista di nodi)."""
    total = 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        ed = G[u][v][0] if 0 in G[u][v] else G[u][v][list(G[u][v].keys())[0]]
        total += ed.get("length", 0)
    return total


def _geo_dist(lat1, lon1, lat2, lon2):
    """Distanza euclidea approssimata tra due coordinate."""
    return ((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2) ** 0.5


def find_bike_route(G, origin_latlon, dest_latlon, bike_df, ora=12,
                    precomputed_bike=None):
    """
    Percorso bici con bike sharing: cammina alla stazione più vicina,
    pedala fino alla stazione più vicina alla destinazione,
    poi cammina fino a destinazione.
    Ritorna (segments, metriche).
    Se precomputed_bike è fornito, salta il calcolo ox.nearest_nodes.
    """
    orig = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest = ox.nearest_nodes(G, dest_latlon[1], dest_latlon[0])

    if precomputed_bike is not None:
        bike_stations = precomputed_bike
    else:
        bike_stations = []
        for _, row in bike_df.iterrows():
            try:
                lat, lon = float(row["Lat"]), float(row["Long"])
                node = ox.nearest_nodes(G, lon, lat)
                bike_stations.append({
                    "node": node,
                    "nome": row["Denominazione"],
                    "lat": lat,
                    "lon": lon,
                    "bici": int(row["Numero Bici"]),
                })
            except Exception:
                pass

    if len(bike_stations) < 2:
        print("   ❌ Stazioni bike sharing insufficienti!")
        return None, None

    orig_lat, orig_lon = origin_latlon
    dest_lat, dest_lon = dest_latlon

    for s in bike_stations:
        s["dist_orig"] = _geo_dist(s["lat"], s["lon"], orig_lat, orig_lon)
        s["dist_dest"] = _geo_dist(s["lat"], s["lon"], dest_lat, dest_lon)

    # Stazioni vicine all'origine CON bici disponibili
    near_orig = [s for s in sorted(bike_stations, key=lambda s: s["dist_orig"])[:10]
                 if s["bici"] > 0]
    near_dest = sorted(bike_stations, key=lambda s: s["dist_dest"])[:10]

    if not near_orig:
        print("   ❌ Nessuna stazione con bici disponibili vicina alla partenza!")
        return None, None

    # Trovare la stazione più vicina all'origine (a piedi)
    best_start = None
    best_start_dist = float("inf")
    for s in near_orig:
        try:
            d = nx.shortest_path_length(G, orig, s["node"], weight="length")
            if d < best_start_dist:
                best_start_dist = d
                best_start = s
        except nx.NetworkXNoPath:
            pass

    # Trovare la stazione più vicina alla destinazione (a piedi)
    best_end = None
    best_end_dist = float("inf")
    for s in near_dest:
        if best_start and s["node"] == best_start["node"]:
            continue
        try:
            d = nx.shortest_path_length(G, s["node"], dest, weight="length")
            if d < best_end_dist:
                best_end_dist = d
                best_end = s
        except nx.NetworkXNoPath:
            pass

    if not best_start or not best_end:
        print("   ❌ Impossibile trovare stazioni bike sharing raggiungibili!")
        return None, None

    # Segmento 1: Cammina → stazione partenza
    try:
        walk1 = nx.shortest_path(G, orig, best_start["node"], weight="length")
    except nx.NetworkXNoPath:
        walk1 = [orig]

    # Segmento 2: Bici tra le due stazioni
    def wfn(u, v, data):
        return _edge_weight_bike(data, ora)
    try:
        bike_path = nx.shortest_path(G, best_start["node"],
                                     best_end["node"], weight=wfn)
    except nx.NetworkXNoPath:
        print("   ❌ Nessun percorso bici trovato!")
        return None, None

    # Segmento 3: Cammina stazione arrivo → destinazione
    try:
        walk2 = nx.shortest_path(G, best_end["node"], dest, weight="length")
    except nx.NetworkXNoPath:
        walk2 = [best_end["node"]]

    walk1_m = _path_length(G, walk1)
    bike_m = _path_length(G, bike_path)
    walk2_m = _path_length(G, walk2)
    total_m = walk1_m + bike_m + walk2_m

    walk_time = ((walk1_m + walk2_m) / 1000 / WALKING_SPEED_KMH) * 60
    bike_time = (bike_m / 1000 / BIKE_SPEED_KMH) * 60

    segments = [
        {"type": "walk", "route": walk1,
         "info": f"🚶 Cammina fino a: stazione {best_start['nome']} "
                 f"({best_start['bici']} bici disponibili)"},
        {"type": "bike", "route": bike_path,
         "info": f"🚲 Bici: {best_start['nome']} → {best_end['nome']}"},
        {"type": "walk", "route": walk2,
         "info": f"🚶 Cammina fino a destinazione"},
    ]

    # Safety score medio
    all_nodes = walk1 + bike_path[1:] + walk2[1:]
    total_safety, n_e = 0.0, 0
    for i in range(len(all_nodes) - 1):
        u, v = all_nodes[i], all_nodes[i + 1]
        ed = G[u][v][0] if 0 in G[u][v] else G[u][v][list(G[u][v].keys())[0]]
        total_safety += float(ed.get("safety_normalized", 0))
        n_e += 1

    metriche = {
        "distanza_km": round(total_m / 1000, 2),
        "tempo_min": round(walk_time + bike_time, 1),
        "safety_score": round((total_safety / max(n_e, 1)) * 100, 1),
        "co2_kg": 0.0,
        "mezzo": "🚲 Bici",
        "stazione_partenza": best_start["nome"],
        "stazione_arrivo": best_end["nome"],
        "bici_disponibili": best_start["bici"],
        "walk1_min": round((walk1_m / 1000 / WALKING_SPEED_KMH) * 60, 1),
        "bike_min": round(bike_time, 1),
        "walk2_min": round((walk2_m / 1000 / WALKING_SPEED_KMH) * 60, 1),
    }
    return segments, metriche


def _find_lines_for_stop(lat, lon, osm_bus_stops, threshold=0.003):
    """Trova le linee bus OSM vicine a una coordinata (entro threshold gradi).
    Raccoglie le linee da TUTTE le fermate OSM nel raggio."""
    all_lines = set()
    for s in osm_bus_stops:
        d = _geo_dist(s["lat"], s["lon"], lat, lon)
        if d < threshold:
            all_lines.update(s["lines"])
    return sorted(all_lines)


def find_bus_route(G, origin_latlon, dest_latlon, fermate_df,
                   osm_bus_stops=None, ora=12, precomputed_bus=None):
    """
    Trova il percorso bus: cammina alla fermata più vicina all'origine,
    prendi il bus fino alla fermata più vicina alla destinazione,
    poi cammina fino a destinazione.
    Usa osm_bus_stops per indicare quale linea prendere.
    Se precomputed_bus è fornito, salta il calcolo ox.nearest_nodes.
    Ritorna (segments, metriche).
    """
    orig = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest = ox.nearest_nodes(G, dest_latlon[1], dest_latlon[0])

    if precomputed_bus is not None:
        bus_nodes = precomputed_bus
    else:
        bus_nodes = []
        for _, row in fermate_df.iterrows():
            try:
                lat, lon = float(row["latitudine"]), float(row["longitudine"])
                node = ox.nearest_nodes(G, lon, lat)
                bus_nodes.append({
                    "node": node,
                    "id": row["idFermata"],
                    "nome": row["descrizioneFermata"],
                    "lat": lat,
                    "lon": lon,
                })
            except Exception:
                pass

    if len(bus_nodes) < 2:
        print("   ❌ Fermate bus insufficienti!")
        return None, None

    orig_lat, orig_lon = origin_latlon
    dest_lat, dest_lon = dest_latlon

    for stop in bus_nodes:
        stop["dist_orig"] = _geo_dist(stop["lat"], stop["lon"], orig_lat, orig_lon)
        stop["dist_dest"] = _geo_dist(stop["lat"], stop["lon"], dest_lat, dest_lon)

    near_orig = sorted(bus_nodes, key=lambda s: s["dist_orig"])[:15]
    near_dest = sorted(bus_nodes, key=lambda s: s["dist_dest"])[:15]

    # Fase A: Trovare la fermata più vicina all'origine (camminando)
    best_start_stop = None
    best_start_dist = float("inf")
    for stop in near_orig:
        try:
            d = nx.shortest_path_length(G, orig, stop["node"], weight="length")
            if d < best_start_dist:
                best_start_dist = d
                best_start_stop = stop
        except nx.NetworkXNoPath:
            pass

    # Fase B: Trovare la fermata più vicina alla destinazione (camminando)
    best_end_stop = None
    best_end_dist = float("inf")
    for stop in near_dest:
        if best_start_stop and stop["node"] == best_start_stop["node"]:
            continue
        try:
            d = nx.shortest_path_length(G, stop["node"], dest, weight="length")
            if d < best_end_dist:
                best_end_dist = d
                best_end_stop = stop
        except nx.NetworkXNoPath:
            pass

    if not best_start_stop or not best_end_stop:
        print("   ❌ Impossibile trovare fermate raggiungibili!")
        return None, None

    # --- Identificare le linee bus ---
    bus_line_str = ""
    if osm_bus_stops:
        lines_start = _find_lines_for_stop(
            best_start_stop["lat"], best_start_stop["lon"], osm_bus_stops)
        lines_end = _find_lines_for_stop(
            best_end_stop["lat"], best_end_stop["lon"], osm_bus_stops)
        common = sorted(set(lines_start) & set(lines_end))
        if common:
            bus_line_str = f"Linea {', '.join(common)}"
        elif lines_start:
            bus_line_str = f"Linee disponibili: {', '.join(sorted(lines_start[:5]))}"

    # Segmento 1: Cammina → fermata partenza
    try:
        walk1 = nx.shortest_path(G, orig, best_start_stop["node"], weight="length")
    except nx.NetworkXNoPath:
        walk1 = [orig]

    # Segmento 2: Bus tra le due fermate
    try:
        bus_path = nx.shortest_path(G, best_start_stop["node"],
                                    best_end_stop["node"], weight="length")
    except nx.NetworkXNoPath:
        print("   ❌ Nessun percorso bus trovato!")
        return None, None

    # Segmento 3: Cammina fermata arrivo → destinazione
    try:
        walk2 = nx.shortest_path(G, best_end_stop["node"], dest, weight="length")
    except nx.NetworkXNoPath:
        walk2 = [best_end_stop["node"]]

    walk1_m = _path_length(G, walk1)
    bus_m = _path_length(G, bus_path)
    walk2_m = _path_length(G, walk2)
    total_m = walk1_m + bus_m + walk2_m

    walk_time = ((walk1_m + walk2_m) / 1000 / WALKING_SPEED_KMH) * 60
    bus_time = (bus_m / 1000 / BUS_SPEED_KMH) * 60
    total_time = walk_time + bus_time

    co2 = CO2_BUS_KG_PER_KM_PASSEGGERO * (bus_m / 1000)

    bus_label = f"🚌 {bus_line_str}: " if bus_line_str else "🚌 Bus: "
    segments = [
        {"type": "walk", "route": walk1,
         "info": f"🚶 Cammina fino a: {best_start_stop['nome']}"},
        {"type": "bus", "route": bus_path,
         "info": f"{bus_label}{best_start_stop['nome']} → {best_end_stop['nome']}"},
        {"type": "walk", "route": walk2,
         "info": f"🚶 Cammina fino a destinazione"},
    ]

    metriche = {
        "distanza_km": round(total_m / 1000, 2),
        "tempo_min": round(total_time, 1),
        "safety_score": 0.0,
        "co2_kg": round(co2, 3),
        "mezzo": "🚌 Bus",
        "fermata_partenza": best_start_stop["nome"],
        "fermata_arrivo": best_end_stop["nome"],
        "linea_bus": bus_line_str,
        "walk1_min": round((walk1_m / 1000 / WALKING_SPEED_KMH) * 60, 1),
        "bus_min": round(bus_time, 1),
        "walk2_min": round((walk2_m / 1000 / WALKING_SPEED_KMH) * 60, 1),
    }

    # Safety score medio su tutto il percorso
    all_nodes = walk1 + bus_path[1:] + walk2[1:]
    total_safety, n_e = 0.0, 0
    for i in range(len(all_nodes) - 1):
        u, v = all_nodes[i], all_nodes[i + 1]
        ed = G[u][v][0] if 0 in G[u][v] else G[u][v][list(G[u][v].keys())[0]]
        total_safety += float(ed.get("safety_normalized", 0))
        n_e += 1
    metriche["safety_score"] = round((total_safety / max(n_e, 1)) * 100, 1)

    return segments, metriche


def find_safe_route(G, origin_latlon, dest_latlon, ora=12):
    """
    Calcola il percorso più sicuro a piedi da partenza a destinazione.
    Il peso di ogni arco è inversamente proporzionale alla sicurezza.
    Ritorna (segments, metriche).
    """
    orig = ox.nearest_nodes(G, origin_latlon[1], origin_latlon[0])
    dest = ox.nearest_nodes(G, dest_latlon[1], dest_latlon[0])

    def safety_weight(u, v, data):
        """Peso che penalizza strade pericolose."""
        length_km = data.get("length", 1) / 1000
        safety = float(data.get("safety_normalized", 0.5))
        # Inversione: più è pericolosa, più pesa
        risk_factor = 1.0 - safety  # 0 = sicura, 1 = pericolosa
        return (0.8 * risk_factor + 0.2) * length_km

    try:
        path = nx.shortest_path(G, orig, dest, weight=safety_weight)
    except nx.NetworkXNoPath:
        print("   ❌ Nessun percorso sicuro trovato!")
        return None, None

    total_m = _path_length(G, path)
    walk_time = (total_m / 1000 / WALKING_SPEED_KMH) * 60

    # Safety score medio
    total_safety, n_e = 0.0, 0
    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        ed = G[u][v][0] if 0 in G[u][v] else G[u][v][list(G[u][v].keys())[0]]
        total_safety += float(ed.get("safety_normalized", 0))
        n_e += 1

    segments = [
        {"type": "walk", "route": path,
         "info": "🛡️ Percorso più sicuro a piedi"},
    ]

    metriche = {
        "distanza_km": round(total_m / 1000, 2),
        "tempo_min": round(walk_time, 1),
        "safety_score": round((total_safety / max(n_e, 1)) * 100, 1),
        "co2_kg": 0.0,
        "mezzo": "🛡️ Sicuro",
    }
    return segments, metriche


# ===========================================================================
# FASE 6: Visualizzazione con Folium
# ===========================================================================
def _safety_color(safety_norm):
    """>=0.75 verde (sicuro), <=0.25 rosso (pericolo), resto arancione."""
    if safety_norm >= 0.65:
        return "green"
    elif safety_norm > 0.25:
        return "orange"
    else:
        return "red"


def _path_coords(G, route):
    """Estrae le coordinate (lat, lon) da una lista di nodi."""
    coords = []
    for i in range(len(route) - 1):
        u, v = route[i], route[i + 1]
        ed = G[u][v][0] if 0 in G[u][v] else G[u][v][list(G[u][v].keys())[0]]
        if "geometry" in ed:
            seg = [(lat, lon) for lon, lat in ed["geometry"].coords]
        else:
            seg = [
                (float(G.nodes[u]["y"]), float(G.nodes[u]["x"])),
                (float(G.nodes[v]["y"]), float(G.nodes[v]["x"])),
            ]
        if coords and coords[-1] == seg[0]:
            coords.extend(seg[1:])
        else:
            coords.extend(seg)
    if not coords and route:
        coords = [(float(G.nodes[route[0]]["y"]), float(G.nodes[route[0]]["x"]))]
    return coords


def visualize_map(G, bike_segments, bike_metriche, bus_segments, bus_metriche,
                  fermate_df=None, bike_df=None, filename="safewalk_map.html"):
    """
    Genera un'unica mappa HTML con:
      - Tutti gli archi colorati per sicurezza (verde/arancione/rosso)
      - Percorso bici in viola (camminate tratteggiate, bici pieno)
      - Percorso bus in blu (camminate tratteggiate, bus pieno)
      - Pallini blu = fermate bus, pallini gialli = stazioni bike sharing
      - Marker partenza, arrivo, fermate bus usate, stazioni bike usate
    """
    try:
        import folium
    except ImportError:
        print("   ⚠️ folium non installato.")
        return

    print("🗺️  Generazione mappa interattiva...")

    # Centro mappa
    center_lat, center_lon = 41.117, 16.87
    if bike_segments:
        all_route = bike_segments[0]["route"]
        for seg in bike_segments[1:]:
            all_route = all_route + seg["route"][1:]
        if all_route:
            center_lat = np.mean([float(G.nodes[n]["y"]) for n in all_route])
            center_lon = np.mean([float(G.nodes[n]["x"]) for n in all_route])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=14,
                   tiles="cartodbpositron")

    # --- Layer: sicurezza strade ---
    safety_grp = folium.FeatureGroup(name="Sicurezza strade", show=True)
    for u, v, k, data in G.edges(data=True, keys=True):
        safety = float(data.get("safety_normalized", 0))
        color = _safety_color(safety)
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            coords = [
                (float(G.nodes[u]["y"]), float(G.nodes[u]["x"])),
                (float(G.nodes[v]["y"]), float(G.nodes[v]["x"])),
            ]
        folium.PolyLine(coords, color=color, weight=2, opacity=0.5).add_to(safety_grp)
    safety_grp.add_to(m)

    # --- Layer: Fermate bus (pallini blu) ---
    if fermate_df is not None:
        bus_stop_grp = folium.FeatureGroup(name="🚏 Fermate Bus", show=True)
        for _, row in fermate_df.iterrows():
            try:
                lat = float(row["latitudine"])
                lon = float(row["longitudine"])
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=4,
                    color="blue",
                    fill=True,
                    fill_color="blue",
                    fill_opacity=0.7,
                    popup=f"🚏 {row['descrizioneFermata']}",
                ).add_to(bus_stop_grp)
            except Exception:
                pass
        bus_stop_grp.add_to(m)

    # --- Layer: Stazioni bike sharing (pallini gialli) ---
    if bike_df is not None:
        bike_st_grp = folium.FeatureGroup(name="🚲 Stazioni Bike Sharing", show=True)
        for _, row in bike_df.iterrows():
            try:
                lat = float(row["Lat"])
                lon = float(row["Long"])
                folium.CircleMarker(
                    location=[lat, lon],
                    radius=6,
                    color="goldenrod",
                    fill=True,
                    fill_color="yellow",
                    fill_opacity=0.9,
                    popup=(f"🚲 <b>{row['Denominazione']}</b><br>"
                           f"Bici: {row['Numero Bici']}"),
                ).add_to(bike_st_grp)
            except Exception:
                pass
        bike_st_grp.add_to(m)

    # --- Layer: Percorso bici (viola) ---
    if bike_segments and bike_metriche:
        bike_grp = folium.FeatureGroup(name="🚲 Percorso Bici", show=True)
        seg_colors_bike = {"walk": "darkmagenta", "bike": "purple"}
        seg_dash_bike = {"walk": "5 10", "bike": None}
        seg_weight_bike = {"walk": 4, "bike": 6}

        for seg in bike_segments:
            sc = _path_coords(G, seg["route"])
            if len(sc) < 2:
                continue
            line_opts = {
                "color": seg_colors_bike[seg["type"]],
                "weight": seg_weight_bike[seg["type"]],
                "opacity": 0.85,
                "popup": seg["info"],
            }
            if seg_dash_bike[seg["type"]]:
                line_opts["dash_array"] = seg_dash_bike[seg["type"]]
            folium.PolyLine(sc, **line_opts).add_to(bike_grp)

        # Marker stazioni bike usate
        if len(bike_segments) >= 2:
            # Stazione partenza bici (fine segmento walk1)
            if len(bike_segments[0]["route"]) > 1:
                sc0 = _path_coords(G, bike_segments[0]["route"])
                folium.Marker(
                    sc0[-1],
                    popup=f"🚲 Stazione: {bike_metriche.get('stazione_partenza', '')}",
                    icon=folium.Icon(color="purple", icon="bicycle", prefix="fa"),
                ).add_to(bike_grp)
            # Stazione arrivo bici (fine segmento bike)
            if len(bike_segments) > 1 and len(bike_segments[1]["route"]) > 1:
                sc1 = _path_coords(G, bike_segments[1]["route"])
                folium.Marker(
                    sc1[-1],
                    popup=f"🚲 Stazione: {bike_metriche.get('stazione_arrivo', '')}",
                    icon=folium.Icon(color="purple", icon="bicycle", prefix="fa"),
                ).add_to(bike_grp)
        bike_grp.add_to(m)

    # --- Layer: Percorso bus (blu) ---
    if bus_segments and bus_metriche:
        bus_grp = folium.FeatureGroup(name="🚌 Percorso Bus", show=True)
        seg_colors = {"walk": "darkblue", "bus": "blue"}
        seg_dash = {"walk": "5 10", "bus": None}
        seg_weight = {"walk": 4, "bus": 6}

        for seg in bus_segments:
            sc = _path_coords(G, seg["route"])
            if len(sc) < 2:
                continue
            line_opts = {
                "color": seg_colors[seg["type"]],
                "weight": seg_weight[seg["type"]],
                "opacity": 0.85,
                "popup": seg["info"],
            }
            if seg_dash[seg["type"]]:
                line_opts["dash_array"] = seg_dash[seg["type"]]
            folium.PolyLine(sc, **line_opts).add_to(bus_grp)

        # Marker fermate bus usate
        if len(bus_segments[0]["route"]) > 1:
            sc0 = _path_coords(G, bus_segments[0]["route"])
            folium.Marker(
                sc0[-1],
                popup=(f"🚏 Fermata: {bus_metriche.get('fermata_partenza', '')}"
                       f"<br>{bus_metriche.get('linea_bus', '')}"),
                icon=folium.Icon(color="blue", icon="bus", prefix="fa"),
            ).add_to(bus_grp)
        if len(bus_segments) > 1 and len(bus_segments[1]["route"]) > 1:
            sc1 = _path_coords(G, bus_segments[1]["route"])
            folium.Marker(
                sc1[-1],
                popup=(f"🚏 Fermata: {bus_metriche.get('fermata_arrivo', '')}"
                       f"<br>{bus_metriche.get('linea_bus', '')}"),
                icon=folium.Icon(color="blue", icon="bus", prefix="fa"),
            ).add_to(bus_grp)
        bus_grp.add_to(m)

    # --- Marker partenza e arrivo ---
    first_route = None
    if bike_segments:
        first_route = bike_segments[0]["route"]
        last_route = bike_segments[-1]["route"]
    elif bus_segments:
        first_route = bus_segments[0]["route"]
        last_route = bus_segments[-1]["route"]

    if first_route:
        folium.Marker(
            (float(G.nodes[first_route[0]]["y"]),
             float(G.nodes[first_route[0]]["x"])),
            popup="🟢 Partenza",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)
        folium.Marker(
            (float(G.nodes[last_route[-1]]["y"]),
             float(G.nodes[last_route[-1]]["x"])),
            popup="🔴 Arrivo",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

    # --- Legenda ---
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                border:2px solid grey; font-size:13px; line-height:1.8;">
        <b>Legenda</b><br>
        <span style="color:green;">&#9644;</span> Sicuro (&ge;75%)<br>
        <span style="color:orange;">&#9644;</span> Media (25-75%)<br>
        <span style="color:red;">&#9644;</span> Pericolo (&le;25%)<br>
        <span style="color:purple;">&#9644;&#9644;</span> 🚲 Percorso Bici<br>
        <span style="color:blue;">&#9644;&#9644;</span> 🚌 Percorso Bus<br>
        <span style="color:darkblue;">- - -</span> 🚶 Camminata<br>
        <span style="color:blue;">&#9679;</span> Fermata bus &nbsp;
        <span style="color:goldenrod;">&#9679;</span> Stazione bike
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    # --- Info box con riepilogo ---
    info_parts = []
    if bike_metriche:
        staz = (f"<br>📍 {bike_metriche.get('stazione_partenza', '')} → "
                f"{bike_metriche.get('stazione_arrivo', '')}" if "stazione_partenza" in bike_metriche else "")
        info_parts.append(
            f"<b>🚲 Bici</b>: {bike_metriche['distanza_km']} km, "
            f"{bike_metriche['tempo_min']} min "
            f"(🚶 {bike_metriche.get('walk1_min', 0)}+"
            f"{bike_metriche.get('walk2_min', 0)} min), "
            f"Safety {bike_metriche['safety_score']}%{staz}"
        )
    if bus_metriche:
        linea = bus_metriche.get("linea_bus", "")
        linea_str = f"<br>🚌 {linea}" if linea else ""
        info_parts.append(
            f"<b>🚌 Bus</b>: {bus_metriche['distanza_km']} km, "
            f"{bus_metriche['tempo_min']} min "
            f"(🚶 {bus_metriche['walk1_min']}+{bus_metriche['walk2_min']} min), "
            f"Safety {bus_metriche['safety_score']}%, "
            f"CO₂ {bus_metriche['co2_kg']} kg{linea_str}"
        )
    if info_parts:
        info_html = f"""
        <div style="position:fixed; top:15px; left:60px; z-index:1000;
                    background:white; padding:12px 16px; border-radius:8px;
                    border:2px solid grey; font-size:13px; line-height:1.8;
                    max-width:520px;">
            <b>Riepilogo Percorsi</b><br>{'<br>'.join(info_parts)}
        </div>
        """
        m.get_root().html.add_child(folium.Element(info_html))

    folium.LayerControl().add_to(m)

    output_path = OUTPUT_DIR / filename
    m.save(str(output_path))
    print(f"   ✅ Mappa salvata in: {output_path}")
    return m


def precompute_safety_geojson(G):
    """Pre-calcola i dati GeoJSON per il layer sicurezza strade.
    Deve essere chiamata UNA VOLTA all'avvio, non per ogni richiesta.
    Ritorna una lista di dict pronti per folium."""
    import time as _time
    t0 = _time.time()
    print("🟢 Pre-calcolo layer sicurezza strade...")
    features = []
    for u, v, k, data in G.edges(data=True, keys=True):
        safety = float(data.get("safety_normalized", 0.5))
        color = _safety_color(safety)
        if "geometry" in data:
            coords = [(lat, lon) for lon, lat in data["geometry"].coords]
        else:
            try:
                coords = [
                    (float(G.nodes[u]["y"]), float(G.nodes[u]["x"])),
                    (float(G.nodes[v]["y"]), float(G.nodes[v]["x"])),
                ]
            except (KeyError, ValueError):
                continue
        cat = data.get("risk_category", "Sconosciuto")
        rs = data.get("risk_score", 0)
        features.append({
            "coords": coords,
            "color": color,
            "tooltip": f"Sicurezza: {safety:.0%} | Rischio: {rs:.1f} | {cat}",
        })
    print(f"   ✅ Pre-calcolati {len(features)} archi in {_time.time()-t0:.1f}s")
    return features


def visualize_map_light(G, segments_list=None, bike_segments=None,
                        bike_metriche=None, bus_segments=None,
                        bus_metriche=None, safe_segments=None,
                        safe_metriche=None, custom_segments=None,
                        custom_metriche=None,
                        safety_geojson=None,
                        filename="route_map.html"):
    """Versione della mappa con layer sicurezza strade e percorsi calcolati.
    Se safety_geojson è fornito, usa dati precalcolati (veloce).
    Altrimenti itera il grafo (lento)."""
    import folium

    center_lat, center_lon = 41.117, 16.87
    all_nodes = []
    for segs in [bike_segments, bus_segments, safe_segments, custom_segments]:
        if segs:
            for seg in segs:
                all_nodes.extend(seg["route"])
    if all_nodes:
        center_lat = np.mean([float(G.nodes[n]["y"]) for n in all_nodes])
        center_lon = np.mean([float(G.nodes[n]["x"]) for n in all_nodes])

    m = folium.Map(location=[center_lat, center_lon], zoom_start=15,
                   tiles="cartodbpositron")

    # --- Layer: sicurezza strade (colorato) ---
    safety_grp = folium.FeatureGroup(name="🛡️ Sicurezza strade", show=True)
    if safety_geojson:
        # Usa dati pre-calcolati (veloce!)
        for feat in safety_geojson:
            folium.PolyLine(
                feat["coords"], color=feat["color"],
                weight=2, opacity=0.5, tooltip=feat["tooltip"]
            ).add_to(safety_grp)
    else:
        # Fallback: itera il grafo (lento)
        for u, v, k, data in G.edges(data=True, keys=True):
            safety = float(data.get("safety_normalized", 0.5))
            color = _safety_color(safety)
            if "geometry" in data:
                coords = [(lat, lon) for lon, lat in data["geometry"].coords]
            else:
                try:
                    coords = [
                        (float(G.nodes[u]["y"]), float(G.nodes[u]["x"])),
                        (float(G.nodes[v]["y"]), float(G.nodes[v]["x"])),
                    ]
                except (KeyError, ValueError):
                    continue
            cat = data.get("risk_category", "Sconosciuto")
            rs = data.get("risk_score", 0)
            tooltip = f"Sicurezza: {safety:.0%} | Rischio: {rs:.1f} | {cat}"
            folium.PolyLine(coords, color=color, weight=2, opacity=0.5,
                            tooltip=tooltip).add_to(safety_grp)
    safety_grp.add_to(m)

    # --- Percorso sicuro ---
    if safe_segments and safe_metriche:
        safe_grp = folium.FeatureGroup(name="🛡️ Percorso Sicuro", show=True)
        for seg in safe_segments:
            sc = _path_coords(G, seg["route"])
            if len(sc) < 2:
                continue
            folium.PolyLine(sc, color="#2E7D32", weight=6, opacity=0.9,
                            popup=seg["info"]).add_to(safe_grp)
        safe_grp.add_to(m)

    # --- Percorso personalizzato ---
    if custom_segments and custom_metriche:
        custom_grp = folium.FeatureGroup(name="⚙️ Percorso Personalizzato", show=True)
        seg_styles = {
            "walk": {"color": "#FF6F00", "weight": 4, "dash_array": "5 10"},
            "bike": {"color": "#E65100", "weight": 6},
            "bus": {"color": "#E65100", "weight": 6},
            "drive": {"color": "#E65100", "weight": 6},
        }
        for seg in custom_segments:
            sc = _path_coords(G, seg["route"])
            if len(sc) < 2:
                continue
            st = seg_styles.get(seg.get("type", "walk"), seg_styles["walk"])
            opts = {"opacity": 0.9, "popup": seg.get("info", ""), **st}
            folium.PolyLine(sc, **opts).add_to(custom_grp)
        custom_grp.add_to(m)

    # --- Percorso bici ---
    if bike_segments and bike_metriche:
        bike_grp = folium.FeatureGroup(name="🚲 Percorso Bici", show=True)
        styles = {
            "walk": {"color": "darkmagenta", "weight": 4, "dash_array": "5 10"},
            "bike": {"color": "purple", "weight": 6},
        }
        for seg in bike_segments:
            sc = _path_coords(G, seg["route"])
            if len(sc) < 2:
                continue
            opts = {"opacity": 0.85, "popup": seg["info"],
                    **styles[seg["type"]]}
            folium.PolyLine(sc, **opts).add_to(bike_grp)

        # Marker stazioni
        if len(bike_segments) >= 2 and len(bike_segments[0]["route"]) > 1:
            sc0 = _path_coords(G, bike_segments[0]["route"])
            folium.Marker(
                sc0[-1],
                popup=f"🚲 {bike_metriche.get('stazione_partenza', '')}",
                icon=folium.Icon(color="purple", icon="bicycle", prefix="fa"),
            ).add_to(bike_grp)
        if len(bike_segments) > 1 and len(bike_segments[1]["route"]) > 1:
            sc1 = _path_coords(G, bike_segments[1]["route"])
            folium.Marker(
                sc1[-1],
                popup=f"🚲 {bike_metriche.get('stazione_arrivo', '')}",
                icon=folium.Icon(color="purple", icon="bicycle", prefix="fa"),
            ).add_to(bike_grp)
        bike_grp.add_to(m)

    # --- Percorso bus ---
    if bus_segments and bus_metriche:
        bus_grp = folium.FeatureGroup(name="🚌 Percorso Bus", show=True)
        styles = {
            "walk": {"color": "darkblue", "weight": 4, "dash_array": "5 10"},
            "bus": {"color": "blue", "weight": 6},
        }
        for seg in bus_segments:
            sc = _path_coords(G, seg["route"])
            if len(sc) < 2:
                continue
            opts = {"opacity": 0.85, "popup": seg["info"],
                    **styles[seg["type"]]}
            folium.PolyLine(sc, **opts).add_to(bus_grp)

        if len(bus_segments[0]["route"]) > 1:
            sc0 = _path_coords(G, bus_segments[0]["route"])
            folium.Marker(
                sc0[-1],
                popup=(f"🚏 {bus_metriche.get('fermata_partenza', '')}"
                       f"<br>{bus_metriche.get('linea_bus', '')}"),
                icon=folium.Icon(color="blue", icon="bus", prefix="fa"),
            ).add_to(bus_grp)
        if len(bus_segments) > 1 and len(bus_segments[1]["route"]) > 1:
            sc1 = _path_coords(G, bus_segments[1]["route"])
            folium.Marker(
                sc1[-1],
                popup=(f"🚏 {bus_metriche.get('fermata_arrivo', '')}"
                       f"<br>{bus_metriche.get('linea_bus', '')}"),
                icon=folium.Icon(color="blue", icon="bus", prefix="fa"),
            ).add_to(bus_grp)
        bus_grp.add_to(m)

    # --- Marker partenza e arrivo ---
    first_route = None
    last_route = None
    for segs in [safe_segments, custom_segments, bike_segments, bus_segments]:
        if segs:
            if first_route is None:
                first_route = segs[0]["route"]
            last_route = segs[-1]["route"]
    if first_route:
        folium.Marker(
            (float(G.nodes[first_route[0]]["y"]),
             float(G.nodes[first_route[0]]["x"])),
            popup="🟢 Partenza",
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(m)
        folium.Marker(
            (float(G.nodes[last_route[-1]]["y"]),
             float(G.nodes[last_route[-1]]["x"])),
            popup="🔴 Arrivo",
            icon=folium.Icon(color="red", icon="stop"),
        ).add_to(m)

    # --- Legenda ---
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
                background:white; padding:12px 16px; border-radius:8px;
                border:2px solid grey; font-size:13px; line-height:1.8;">
        <b>Pericolosità strade</b><br>
        <span style="color:green;">&#9644;</span> Sicura<br>
        <span style="color:orange;">&#9644;</span> Media<br>
        <span style="color:red;">&#9644;</span> Pericolosa
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)
    output_path = OUTPUT_DIR / filename
    m.save(str(output_path))
    return m
def geocode_place(query):
    """Geocodifica un luogo usando osmnx (Nominatim). Ritorna (lat, lon)."""
    try:
        result = ox.geocode(query + ", Bari, Italy")
        return result  # (lat, lon)
    except Exception:
        return None


def main():
    print("=" * 60)
    print("  SafeWalk — Routing Sicuro e Sostenibile per Bari")
    print("=" * 60)
    print()

    # --- FASE 1: Grafo ---
    G = build_walking_graph(use_cache=True)
    print()

    # --- FASE 2: Lampioni, strade illuminate, linee bus ---
    lamps = fetch_street_lamps(use_cache=True)
    lamp_gdf = lamps_to_geodataframe(lamps)
    lit_way_ids = fetch_lit_ways(use_cache=True)
    osm_bus_stops, line_info = fetch_bus_routes(use_cache=True)
    print()

    # --- FASE 3: Dati CSV ---
    fermate, bike, orari, consumi = load_csv_data()
    transit_freq = compute_transit_frequency(orari)
    avg_co2 = compute_avg_bus_co2(consumi)
    print()

    # --- FASE 4: Arricchire grafo ---
    G = enrich_graph_with_stops(G, fermate, bike)
    G = enrich_graph_with_lamps(G, lamp_gdf, lit_way_ids)
    print()

    # --- INPUT UTENTE ---
    print("=" * 60)
    print("  Inserisci i dati del percorso")
    print("=" * 60)

    origin_str = input("📍 Luogo di partenza: ").strip()
    dest_str = input("📍 Luogo di destinazione: ").strip()
    ora_str = input("🕐 Orario (0-23): ").strip()

    # Default di fallback
    if not origin_str:
        origin_str = "Politecnico di Bari"
    if not dest_str:
        dest_str = "Stazione Centrale Bari"
    ora = int(ora_str) if ora_str.isdigit() and 0 <= int(ora_str) <= 23 else 12

    print(f"\n   Partenza:     {origin_str}")
    print(f"   Destinazione: {dest_str}")
    print(f"   Orario:       {ora}:00")
    print()

    # Geocodifica
    origin = geocode_place(origin_str)
    dest = geocode_place(dest_str)

    if not origin:
        print(f"   ❌ Impossibile geolocalizzare: {origin_str}")
        return
    if not dest:
        print(f"   ❌ Impossibile geolocalizzare: {dest_str}")
        return

    print(f"   📌 Partenza:     ({origin[0]:.5f}, {origin[1]:.5f})")
    print(f"   📌 Destinazione: ({dest[0]:.5f}, {dest[1]:.5f})")
    print()

    # --- FASE 5: Routing ---
    print("=" * 60)
    print("  🚲 PERCORSO BICI (sostenibile)")
    print("=" * 60)
    bike_segments, bike_metriche = find_bike_route(G, origin, dest, bike, ora)
    if bike_metriche:
        print(f"   🚶 Cammina fino a: stazione {bike_metriche['stazione_partenza']} "
              f"({bike_metriche['walk1_min']} min)")
        print(f"   🚲 Pedala fino a:  stazione {bike_metriche['stazione_arrivo']} "
              f"({bike_metriche['bike_min']} min)")
        print(f"   🚶 Cammina fino a destinazione ({bike_metriche['walk2_min']} min)")
        print(f"   Distanza: {bike_metriche['distanza_km']} km")
        print(f"   Tempo:    {bike_metriche['tempo_min']} min")
        print(f"   Safety:   {bike_metriche['safety_score']}%")
        print(f"   CO₂:      {bike_metriche['co2_kg']} kg")
    print()

    print("=" * 60)
    print("  🚌 PERCORSO BUS")
    print("=" * 60)
    bus_segments, bus_metriche = find_bus_route(
        G, origin, dest, fermate, osm_bus_stops, ora)
    if bus_metriche:
        linea = bus_metriche.get("linea_bus", "")
        linea_str = f" [{linea}]" if linea else ""
        print(f"   🚶 Cammina fino a: {bus_metriche['fermata_partenza']} "
              f"({bus_metriche['walk1_min']} min)")
        print(f"   🚌 Bus{linea_str}: → {bus_metriche['fermata_arrivo']} "
              f"({bus_metriche['bus_min']} min)")
        print(f"   🚶 Cammina fino a destinazione ({bus_metriche['walk2_min']} min)")
        print(f"   Distanza totale: {bus_metriche['distanza_km']} km")
        print(f"   Tempo totale:    {bus_metriche['tempo_min']} min")
        print(f"   Safety:          {bus_metriche['safety_score']}%")
        print(f"   CO₂:             {bus_metriche['co2_kg']} kg")
    print()

    # --- FASE 6: Visualizzazione ---
    visualize_map(G, bike_segments, bike_metriche, bus_segments, bus_metriche,
                  fermate_df=fermate, bike_df=bike)
    print()

    # Riepilogo
    print("=" * 60)
    print("  RIEPILOGO")
    print("=" * 60)
    print(f"{'':>20} {'🚲 Bici':>15} {'🚌 Bus':>15}")
    print("-" * 55)
    for key, label in [("distanza_km", "Distanza (km)"),
                       ("tempo_min", "Tempo (min)"),
                       ("safety_score", "Safety (%)"),
                       ("co2_kg", "CO₂ (kg)")]:
        v1 = str(bike_metriche.get(key, "N/A")) if bike_metriche else "N/A"
        v2 = str(bus_metriche.get(key, "N/A")) if bus_metriche else "N/A"
        print(f"{label:>20} {v1:>15} {v2:>15}")
    print()
    print("✅ Pipeline completata! Apri notebooks/output/safewalk_map.html")


if __name__ == "__main__":
    main()
