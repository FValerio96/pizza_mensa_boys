"""
router.py
=========
Helper per il routing multi-criterio sul grafo costruito da build_graph.py.

Uso tipico:
    from router import load_graph, find_best_path

    G = load_graph()
    path, info = find_best_path(
        G,
        start_lat=41.1177, start_lon=16.8717,
        end_lat=41.1087,   end_lon=16.8719,
        pref_tempo=0.4,
        pref_ecologia=0.3,
        pref_sicurezza=0.3,
    )
    print("Percorso nodi:", path)
    print("Dettagli:", info)
"""

import os
import math
import pickle
import networkx as nx
from functools import partial

GRAPH_FILE = os.path.join(os.path.dirname(__file__), "graph.pkl")


# ─────────────────────────────────────────────
# Caricamento grafo
# ─────────────────────────────────────────────

def load_graph(path: str = GRAPH_FILE) -> nx.MultiDiGraph:
    """Carica il grafo pre-costruito da file pickle."""
    with open(path, "rb") as f:
        G = pickle.load(f)
    return G


# ─────────────────────────────────────────────
# Utilità geografiche
# ─────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    """Distanza in metri tra due coordinate lat/lon."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_osm_node(G: nx.MultiDiGraph, lat: float, lon: float) -> int:
    """
    Restituisce il nodo OSM (intero) più vicino alle coordinate fornite.
    Ignora i nodi non-OSM (bus stop, bike station).
    """
    best_node = None
    best_dist = float("inf")
    for node, data in G.nodes(data=True):
        # Considera solo nodi OSM (identificatore intero)
        if not isinstance(node, int):
            continue
        n_lat = data.get("y", None)
        n_lon = data.get("x", None)
        if n_lat is None or n_lon is None:
            continue
        d = haversine_m(lat, lon, n_lat, n_lon)
        if d < best_dist:
            best_dist = d
            best_node = node
    return best_node


# ─────────────────────────────────────────────
# Calcolo peso combinato
# ─────────────────────────────────────────────

def combined_weight(
    u, v, data: dict,
    alpha: float,
    beta: float,
    gamma: float,
    mode: str = "walk",
    G: nx.MultiDiGraph = None
) -> float:
    """
    Formula di aggregazione che utilizza i cluster AI (se disponibili).
    """
    # Se abbiamo il grafo e l'attributo cluster, usiamo l'intelligenza artificiale
    if G is not None:
        def get_ai_cost(e_data):
            c_id = e_data.get("cluster_id")
            length_m = e_data.get("length", 1.0)
            if c_id is not None and f"centroide_{c_id}_tempo_km" in G.graph:
                # Estraiamo il profilo dal centroide del cluster (3 features)
                t_km = G.graph[f"centroide_{c_id}_tempo_km"]
                s_raw = G.graph[f"centroide_{c_id}_sic"]
                e_km = G.graph.get(f"centroide_{c_id}_eco_km", 0)

                # Ricostruzione costo pesato sulla lunghezza dell'arco
                wt = t_km * (length_m / 1000.0)
                ws = s_raw  # rischio adimensionale per arco
                we = e_km * (length_m / 1000.0)  # ecologia dal cluster AI
                return wt, we, ws
            # Fallback a raw
            return e_data.get("w_tempo_raw", 0), e_data.get("w_eco_raw", 0), e_data.get("w_sicurezza_raw", 0)

        if isinstance(data, dict) and "w_tempo_raw" in data:
            wt, we, ws = get_ai_cost(data)
        elif isinstance(data, dict):
            # Cerca il percorso migliore tra i multiedge usando l'AI
            costs = [get_ai_cost(e) for e in data.values()]
            wt = min(c[0] for c in costs)
            we = min(c[1] for c in costs)
            ws = min(c[2] for c in costs)
        else:
            wt, we, ws = 0, 0, 0
    else:
        # Vecchio metodo deterministico (fallback se grafo non passato)
        if isinstance(data, dict) and "w_tempo_raw" in data:
            wt = data.get("w_tempo_raw", 0)
            we = data.get("w_eco_raw", 0)
            ws = data.get("w_sicurezza_raw", 0)
        elif isinstance(data, dict):
            edges = list(data.values())
            wt = min(e.get("w_tempo_raw", 0) for e in edges)
            we = min(e.get("w_eco_raw", 0) for e in edges)
            ws = min(e.get("w_sicurezza_raw", 0) for e in edges)
        else:
            wt, we, ws = 0, 0, 0

    # Forza eco a 0 per mezzi non motorizzati
    if mode in ("walk", "piedi", "bike", "bici"):
        we = 0.0

    return alpha * wt + beta * we + gamma * ws


# ─────────────────────────────────────────────
# Euristica A* (distanza in linea d'aria)
# ─────────────────────────────────────────────

def _make_heuristic(G: nx.MultiDiGraph, alpha: float, mode: str = "walk"):
    node_coords = {}
    for n, d in G.nodes(data=True):
        lat = d.get("y", None)
        lon = d.get("x", None)
        if lat is not None and lon is not None:
            node_coords[n] = (lat, lon)

    # Velocità di riferimento per l'euristica (stima inferiore del tempo)
    SPEEDS = {
        "walk": 5.0,
        "bike": 15.0,
        "bus": 20.0,
        "drive": 30.0
    }
    speed_kmh = SPEEDS.get(mode, 5.0)
    speed_ms = speed_kmh * 1000 / 3600

    def heuristic(u, v):
        if u not in node_coords or v not in node_coords:
            return 0
        lat1, lon1 = node_coords[u]
        lat2, lon2 = node_coords[v]
        dist_m = haversine_m(lat1, lon1, lat2, lon2)
        # Tempo stimato in minuti
        time_min = (dist_m / speed_ms) / 60
        # Restituisce il contributo al peso alpha * wt (normalizzato approssimativamente)
        return alpha * (time_min / 60)

    return heuristic


# ─────────────────────────────────────────────
# Routing specifico per modalità
# ─────────────────────────────────────────────

def route_for_mode(
    G: nx.MultiDiGraph,
    start_node,
    end_node,
    mode: str,
    alpha: float,
    beta: float,
    gamma: float
) -> list:
    """
    Trova il percorso ottimale vincolato a una specifica modalità di trasporto.
    Filtra gli archi che non permettono il 'mode' richiesto.
    """
    if mode == "piedi": mode = "walk"
    if mode == "bici":  mode = "bike"
    if mode == "auto":  mode = "drive"

    # Filtra il grafo: mantiene solo archi che permettono la modalità
    def filter_edge(u, v, k):
        data = G[u][v][k]
        allowed = data.get("allowed_modes", {"walk"})
        # I transfer sono sempre permessi (se non specificato diversamente)
        if data.get("modalita") == "transfer":
            # Per il bus e l'auto, permettiamo il transfer solo vicino alle fermate/nodi drive
            return True
        return mode in allowed

    # Vista filtrata del grafo
    G_sub = nx.subgraph_view(G, filter_edge=filter_edge)

    weight_fn = lambda u, v, d: combined_weight(u, v, d, alpha, beta, gamma, mode=mode, G=G)
    heuristic = _make_heuristic(G, alpha, mode=mode)

    try:
        path = nx.astar_path(G_sub, start_node, end_node,
                             heuristic=heuristic,
                             weight=weight_fn)
        return path
    except nx.NetworkXNoPath:
        # Tenta Dijkstra come fallback (potrebbe essere lento su grafi grandi)
        try:
            return nx.dijkstra_path(G_sub, start_node, end_node, weight=weight_fn)
        except nx.NetworkXNoPath:
            return []


# ─────────────────────────────────────────────
# Nearest stop/station helpers
# ─────────────────────────────────────────────

def _nearest_special_node(G, lat, lon, node_type: str):
    """
    Trova il nodo speciale (bus_stop o bike_station) più vicino
    alle coordinate fornite. Restituisce (node_id, dist_m) o (None, inf).
    """
    best_node = None
    best_dist = float("inf")
    for n, d in G.nodes(data=True):
        if d.get("node_type") != node_type:
            continue
        dist = haversine_m(lat, lon, d.get("y", 0), d.get("x", 0))
        if dist < best_dist:
            best_dist = dist
            best_node = n
    return best_node, best_dist


def _compute_route_metrics(G, path, mode, alpha, beta, gamma):
    """
    Calcola le metriche per un percorso dato (tempo, eco, sicurezza, distanza).
    """
    tempo_totale = 0.0
    eco_totale = 0.0
    sic_values = []
    dist_m = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        edges = G[u][v]
        best_e = min(edges.values(),
                     key=lambda e: combined_weight(u, v, e, alpha, beta, gamma, mode=mode, G=G))
        tempo_totale += best_e.get("w_tempo_raw", 0)
        if mode in ("walk", "piedi", "bike", "bici"):
            eco_totale += 0.0
        else:
            eco_totale += best_e.get("w_eco_raw", 0)
        dist_m += best_e.get("length", 0)
        sic_values.append(best_e.get("w_sicurezza_raw", 0))

    dist_km = dist_m / 1000.0
    return {
        "tempo_totale_min": round(tempo_totale, 2),
        "eco_totale_kg_co2": round(eco_totale, 4),
        "distanza_km": round(dist_km, 3),
        "sic_media": round(sum(sic_values) / len(sic_values), 3) if sic_values else 0,
        "num_nodi": len(path),
    }


# ─────────────────────────────────────────────
# Routing principale: percorso per OGNI modalità
# ─────────────────────────────────────────────

def find_all_routes(
    G: nx.MultiDiGraph,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    pref_tempo: float = 0.5,
    pref_ecologia: float = 0.2,
    pref_sicurezza: float = 0.3,
) -> dict:
    """
    Calcola un percorso per ciascuna modalità di trasporto (piedi, bici, bus, auto).

    Per bus e bici, il percorso include tratti a piedi dal punto di partenza
    alla fermata/stazione più vicina e dalla fermata/stazione di arrivo alla
    destinazione finale.

    Restituisce un dizionario:
        {
            "piedi": {"path": [...], "info": {...}},
            "bici":  {"path": [...], "info": {...}, "walk_start": [...], "walk_end": [...]},
            "bus":   {"path": [...], "info": {...}, "walk_start": [...], "walk_end": [...]},
            "auto":  {"path": [...], "info": {...}},
            "prefs": {"pref_tempo": ..., "pref_ecologia": ..., "pref_sicurezza": ...},
        }
    """
    total = pref_tempo + pref_ecologia + pref_sicurezza
    alpha = pref_tempo / total
    beta = pref_ecologia / total
    gamma = pref_sicurezza / total

    start_node = nearest_osm_node(G, start_lat, start_lon)
    end_node = nearest_osm_node(G, end_lat, end_lon)

    if start_node is None or end_node is None:
        raise ValueError("Impossibile trovare nodi OSM per le coordinate fornite.")

    results = {
        "prefs": {
            "pref_tempo": round(alpha, 2),
            "pref_ecologia": round(beta, 2),
            "pref_sicurezza": round(gamma, 2),
        }
    }

    # ── 1. PIEDI ──────────────────────────────────────
    path_walk = route_for_mode(G, start_node, end_node, "walk", alpha, beta, gamma)
    if path_walk:
        info_walk = _compute_route_metrics(G, path_walk, "walk", alpha, beta, gamma)
        info_walk["start_node"] = start_node
        info_walk["end_node"] = end_node
        results["piedi"] = {"path": path_walk, "info": info_walk}
    else:
        results["piedi"] = None

    # ── 2. AUTO ──────────────────────────────────────
    path_auto = route_for_mode(G, start_node, end_node, "drive", alpha, beta, gamma)
    if path_auto:
        info_auto = _compute_route_metrics(G, path_auto, "drive", alpha, beta, gamma)
        info_auto["start_node"] = start_node
        info_auto["end_node"] = end_node
        results["auto"] = {"path": path_auto, "info": info_auto}
    else:
        results["auto"] = None

    # ── 3. BUS (con tratti a piedi) ──────────────────
    bus_start, _ = _nearest_special_node(G, start_lat, start_lon, "bus_stop")
    bus_end, _ = _nearest_special_node(G, end_lat, end_lon, "bus_stop")

    if bus_start and bus_end and bus_start != bus_end:
        # Tratto a piedi: start → fermata partenza
        walk_to_bus = route_for_mode(G, start_node, bus_start, "walk", alpha, beta, gamma)
        # Tratto bus: fermata partenza → fermata arrivo (su strade reali)
        # Usiamo "drive" perché il bus percorre strade carrozzabili;
        # i nodi bus_stop sono collegati alla rete OSM via transfer edges.
        path_bus = route_for_mode(G, bus_start, bus_end, "drive", alpha, beta, gamma)
        # Tratto a piedi: fermata arrivo → end
        walk_from_bus = route_for_mode(G, bus_end, end_node, "walk", alpha, beta, gamma)

        if path_bus:
            # Metriche per ogni segmento (bus usa metriche "bus" per CO2 corretta)
            info_walk_to = _compute_route_metrics(G, walk_to_bus, "walk", alpha, beta, gamma) if walk_to_bus else {"tempo_totale_min": 0, "distanza_km": 0, "eco_totale_kg_co2": 0, "sic_media": 0, "num_nodi": 0}
            info_bus = _compute_route_metrics(G, path_bus, "bus", alpha, beta, gamma)
            info_walk_from = _compute_route_metrics(G, walk_from_bus, "walk", alpha, beta, gamma) if walk_from_bus else {"tempo_totale_min": 0, "distanza_km": 0, "eco_totale_kg_co2": 0, "sic_media": 0, "num_nodi": 0}

            # Metriche totali
            info_total = {
                "tempo_totale_min": round(info_walk_to["tempo_totale_min"] + info_bus["tempo_totale_min"] + info_walk_from["tempo_totale_min"], 2),
                "eco_totale_kg_co2": round(info_bus["eco_totale_kg_co2"], 4),
                "distanza_km": round(info_walk_to["distanza_km"] + info_bus["distanza_km"] + info_walk_from["distanza_km"], 3),
                "sic_media": info_bus["sic_media"],
                "num_nodi": info_walk_to["num_nodi"] + info_bus["num_nodi"] + info_walk_from["num_nodi"],
                "start_node": start_node,
                "end_node": end_node,
                "fermata_partenza": bus_start,
                "fermata_arrivo": bus_end,
            }
            results["bus"] = {
                "path": path_bus,
                "info": info_total,
                "walk_start": walk_to_bus or [],
                "walk_end": walk_from_bus or [],
            }
        else:
            results["bus"] = None
    else:
        results["bus"] = None

    # ── 4. BICI (con tratti a piedi) ──────────────────
    bike_start, _ = _nearest_special_node(G, start_lat, start_lon, "bike_station")
    bike_end, _ = _nearest_special_node(G, end_lat, end_lon, "bike_station")

    if bike_start and bike_end and bike_start != bike_end:
        walk_to_bike = route_for_mode(G, start_node, bike_start, "walk", alpha, beta, gamma)
        path_bike = route_for_mode(G, bike_start, bike_end, "bike", alpha, beta, gamma)
        walk_from_bike = route_for_mode(G, bike_end, end_node, "walk", alpha, beta, gamma)

        if path_bike:
            info_walk_to = _compute_route_metrics(G, walk_to_bike, "walk", alpha, beta, gamma) if walk_to_bike else {"tempo_totale_min": 0, "distanza_km": 0, "eco_totale_kg_co2": 0, "sic_media": 0, "num_nodi": 0}
            info_bike = _compute_route_metrics(G, path_bike, "bike", alpha, beta, gamma)
            info_walk_from = _compute_route_metrics(G, walk_from_bike, "walk", alpha, beta, gamma) if walk_from_bike else {"tempo_totale_min": 0, "distanza_km": 0, "eco_totale_kg_co2": 0, "sic_media": 0, "num_nodi": 0}

            info_total = {
                "tempo_totale_min": round(info_walk_to["tempo_totale_min"] + info_bike["tempo_totale_min"] + info_walk_from["tempo_totale_min"], 2),
                "eco_totale_kg_co2": 0.0,
                "distanza_km": round(info_walk_to["distanza_km"] + info_bike["distanza_km"] + info_walk_from["distanza_km"], 3),
                "sic_media": info_bike["sic_media"],
                "num_nodi": info_walk_to["num_nodi"] + info_bike["num_nodi"] + info_walk_from["num_nodi"],
                "start_node": start_node,
                "end_node": end_node,
                "stazione_partenza": bike_start,
                "stazione_arrivo": bike_end,
            }
            results["bici"] = {
                "path": path_bike,
                "info": info_total,
                "walk_start": walk_to_bike or [],
                "walk_end": walk_from_bike or [],
            }
        else:
            results["bici"] = None
    else:
        results["bici"] = None

    return results


# ─────────────────────────────────────────────
# Esempio d'uso
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Caricamento grafo ...")
    G = load_graph()
    print(f"Grafo caricato: {G.number_of_nodes()} nodi, {G.number_of_edges()} archi")

    START = (41.1258, 16.8650)
    END = (41.1202, 16.8727)

    print("\n--- Calcolo percorsi per tutte le modalità ---")
    routes = find_all_routes(G, *START, *END, pref_tempo=0.5, pref_ecologia=0.2, pref_sicurezza=0.3)

    for mode_name in ["piedi", "bici", "bus", "auto"]:
        route = routes.get(mode_name)
        if route is None:
            print(f"\n  {mode_name.upper()}: nessun percorso trovato")
            continue
        info = route["info"]
        print(f"\n  {mode_name.upper()}: {info['num_nodi']} nodi | "
              f"{info['tempo_totale_min']} min | "
              f"{info['distanza_km']} km | "
              f"CO2: {info['eco_totale_kg_co2']} kg")
        if "walk_start" in route and route["walk_start"]:
            print(f"    └ Tratto a piedi iniziale: {len(route['walk_start'])} nodi")
        if "walk_end" in route and route["walk_end"]:
            print(f"    └ Tratto a piedi finale: {len(route['walk_end'])} nodi")

