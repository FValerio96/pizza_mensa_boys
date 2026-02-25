"""
visualize.py
============
Esplorazione visiva del grafo di routing urbano di Bari.

Genera una mappa HTML interattiva (Folium) con:
  - Rete stradale colorata per tipo di peso (tempo / ecologia / sicurezza)
  - Nodi bus (marker rossi)
  - Nodi bike sharing (marker verdi)
  - Lampioni (cerchi gialli)
  - Percorso ottimale evidenziato (opzionale)

Output: visualize_graph.html  (apri nel browser)

Uso:
    python visualize.py                          # mappa base
    python visualize.py --weight sicurezza       # colora archi per sicurezza
    python visualize.py --route 41.1258,16.8650 41.1202,16.8727  # mostra percorso
"""

import argparse
import math
import os
import pickle
import sys
import webbrowser

import folium
from folium.plugins import MiniMap
import networkx as nx

# -------------------------------------------------
# Configurazione
# -------------------------------------------------

GRAPH_FILE = os.path.join(os.path.dirname(__file__), "graph.pkl")
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "visualize_graph.html")

WEIGHT_CHOICES = {
    "tempo":     "w_tempo",
    "ecologia":  "w_ecologia",
    "sicurezza": "w_sicurezza",
}

# -------------------------------------------------
# Utilita colore
# -------------------------------------------------

def value_to_color(value: float, invert: bool = False) -> str:
    """
    Mappa un valore [0,1] in tre fasce di colore nette:
      sicuro  (0.00-0.33) -> verde  #2E7D32
      medio   (0.33-0.66) -> arancio #E65100
      rischioso (0.66-1.0) -> rosso  #B71C1C
    Se invert=True il valore viene prima ribaltato (es. per sicurezza
    dove 0=sicuro e vogliamo verde per il valore basso).
    """
    if invert:
        value = 1 - value
    value = max(0.0, min(1.0, value))
    if value < 0.33:
        return "#2E7D32"
    elif value < 0.66:
        return "#E65100"
    else:
        return "#B71C1C"


def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


# -------------------------------------------------
# Caricamento grafo
# -------------------------------------------------

def load_graph() -> nx.MultiDiGraph:
    if not os.path.exists(GRAPH_FILE):
        print(f"[!] File non trovato: {GRAPH_FILE}")
        print("   Esegui prima: python build_graph.py")
        sys.exit(1)
    with open(GRAPH_FILE, "rb") as f:
        return pickle.load(f)


# -------------------------------------------------
# Centro mappa
# -------------------------------------------------

def compute_center(G: nx.MultiDiGraph) -> tuple[float, float]:
    lats, lons = [], []
    for _, data in G.nodes(data=True):
        if "y" in data and "x" in data:
            lats.append(data["y"])
            lons.append(data["x"])
    if not lats:
        return 41.125, 16.862  # Bari default
    return sum(lats) / len(lats), sum(lons) / len(lons)


# -------------------------------------------------
# Costruzione mappa
# -------------------------------------------------

def build_map(G: nx.MultiDiGraph, weight_key: str = "w_sicurezza",
              route_node_set: set = None) -> folium.Map:
    """
    Costruisce la mappa Folium.
    Se route_node_set e' fornito, mostra SOLO i nodi del percorso
    (bus, bike, lampioni) invece di tutti.
    """
    center = compute_center(G)
    m = folium.Map(
        location=center,
        zoom_start=14,
        tiles="OpenStreetMap",
    )
    MiniMap(toggle_display=True).add_to(m)

    nodes_data = dict(G.nodes(data=True))
    only_route = route_node_set is not None

    # -- Gruppi layer -------------------------------------------------
    layer_edges    = folium.FeatureGroup(name="[STRADE] Rete stradale", show=True)
    layer_transfer = folium.FeatureGroup(name="[TRANSFER] Connessioni",  show=False)
    layer_bus      = folium.FeatureGroup(name="[BUS] Fermate bus",       show=True)
    layer_bike     = folium.FeatureGroup(name="[BIKE] Bike sharing",     show=True)
    layer_lamps    = folium.FeatureGroup(name="[LAMP] Lampioni",         show=True)

    invert = (weight_key == "w_sicurezza")  # 0=sicuro=verde

    # -- Archi OSM ----------------------------------------------------
    print("  Disegno archi ...")
    edge_count = 0
    for u, v, data in G.edges(data=True):
        modalita = data.get("modalita", "walk")
        if modalita in ("transfer", "bus"):
            if modalita == "transfer":
                u_data = nodes_data.get(u, {})
                v_data = nodes_data.get(v, {})
                if all(k in u_data for k in ("y", "x")) and all(k in v_data for k in ("y", "x")):
                    folium.PolyLine(
                        [(u_data["y"], u_data["x"]), (v_data["y"], v_data["x"])],
                        color="#9E9E9E",
                        weight=1,
                        opacity=0.5,
                        dash_array="4 6",
                    ).add_to(layer_transfer)
            continue
        if modalita == "lampione":
            continue

        u_data = nodes_data.get(u, {})
        v_data = nodes_data.get(v, {})
        if not all(k in u_data for k in ("y", "x")) or \
           not all(k in v_data for k in ("y", "x")):
            continue

        lat1, lon1 = u_data["y"], u_data["x"]
        lat2, lon2 = v_data["y"], v_data["x"]

        w = data.get(weight_key, 0.5)
        color = value_to_color(w, invert=invert)

        tooltip = (
            f"<b>Arco OSM</b><br>"
            f"Tempo: {data.get('w_tempo_raw', 0):.2f} min<br>"
            f"CO2: {data.get('w_eco_raw', 0):.4f} kg<br>"
            f"Rischio: {data.get('w_sicurezza_raw', 0):.3f}<br>"
            f"Highway: {data.get('highway', 'N/A')}<br>"
            f"Illuminato: {data.get('lit', 'N/A')}"
        )

        folium.PolyLine(
            [(lat1, lon1), (lat2, lon2)],
            color=color,
            weight=2.5,
            opacity=0.75,
            tooltip=tooltip,
        ).add_to(layer_edges)
        edge_count += 1

    print(f"  Archi disegnati: {edge_count}")

    # -- Nodi bus -----------------------------------------------------
    print("  Disegno fermate bus ...")
    bus_count = 0
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "bus_stop":
            continue
        if only_route and node not in route_node_set:
            continue
        lat, lon = data.get("y"), data.get("x")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            location=(lat, lon),
            radius=6,
            color="#1565C0",
            fill=True,
            fill_color="#1565C0",
            fill_opacity=0.9,
            tooltip=f"[BUS] Fermata: {data.get('desc', data.get('fermata_id', node))}",
        ).add_to(layer_bus)
        bus_count += 1
    print(f"  Fermate bus disegnate: {bus_count}")

    # -- Nodi bike sharing --------------------------------------------
    print("  Disegno stazioni bike sharing ...")
    bike_count = 0
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "bike_station":
            continue
        if only_route and node not in route_node_set:
            continue
        lat, lon = data.get("y"), data.get("x")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            location=(lat, lon),
            radius=8,
            color="#2E7D32",
            fill=True,
            fill_color="#2E7D32",
            fill_opacity=0.9,
            tooltip=f"[BIKE] {data.get('name', node)} - bici: {data.get('num_bici', '?')}",
        ).add_to(layer_bike)
        bike_count += 1
    print(f"  Stazioni bike sharing disegnate: {bike_count}")

    # -- Nodi lampioni ------------------------------------------------
    print("  Disegno lampioni ...")
    lamp_count = 0
    for node, data in G.nodes(data=True):
        if data.get("node_type") != "lampione":
            continue
        if only_route and node not in route_node_set:
            continue
        lat, lon = data.get("y"), data.get("x")
        if lat is None or lon is None:
            continue
        folium.CircleMarker(
            location=(lat, lon),
            radius=3,
            color="#F9A825",
            fill=True,
            fill_color="#F9A825",
            fill_opacity=0.85,
            opacity=0.85,
            tooltip=f"[LAMP] Lampione [{data.get('source_tag', 'street_lamp')}]",
        ).add_to(layer_lamps)
        lamp_count += 1
    print(f"  Lampioni disegnati: {lamp_count}")

    # Aggiungi layer alla mappa
    layer_edges.add_to(m)
    layer_transfer.add_to(m)
    layer_bus.add_to(m)
    layer_bike.add_to(m)
    layer_lamps.add_to(m)

    # -- Legenda ------------------------------------------------------
    legend_html = f"""
    <div style="position:fixed;bottom:40px;left:40px;z-index:9999;
                background:rgba(255,255,255,0.92);border:1px solid #ccc;
                padding:12px 16px;border-radius:8px;
                font-family:sans-serif;color:#111;font-size:13px;line-height:1.9;
                box-shadow:0 2px 8px rgba(0,0,0,0.15)">
      <b>Colore archi: {weight_key.replace('w_','').capitalize()}</b><br>
      <span style="color:#2E7D32;font-size:16px">&#9679;</span> Sicuro (0-33%)<br>
      <span style="color:#E65100;font-size:16px">&#9679;</span> Medio (33-66%)<br>
      <span style="color:#B71C1C;font-size:16px">&#9679;</span> Pericoloso (66-100%)<br>
      <hr style="border-color:#ddd;margin:6px 0">
      <span style="color:#1565C0">&#9679;</span> Fermata Bus&nbsp;&nbsp;
      <span style="color:#2E7D32">&#9679;</span> Bike Sharing&nbsp;&nbsp;
      <span style="color:#F9A825">&#9679;</span> Lampione
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))

    return m


# -------------------------------------------------
# Percorso evidenziato
# -------------------------------------------------

def add_route_to_map(m: folium.Map, G: nx.MultiDiGraph,
                     start_lat: float, start_lon: float,
                     end_lat: float, end_lon: float,
                     prefs: dict) -> set:
    """
    Calcola e disegna TUTTI i percorsi (piedi, bici, bus, auto) sulla mappa.
    Ogni modalita ha un colore e un layer dedicato.
    Per bus e bici, i tratti a piedi sono mostrati come linee tratteggiate.
    Restituisce il set di tutti i node_id di tutti i percorsi.
    """
    try:
        from router import find_all_routes
    except ImportError:
        print("  [!] router.py non trovato - percorso non tracciato.")
        return set()

    print(f"  Calcolo percorsi da ({start_lat},{start_lon}) a ({end_lat},{end_lon}) ...")
    try:
        routes = find_all_routes(
            G,
            start_lat=start_lat, start_lon=start_lon,
            end_lat=end_lat,     end_lon=end_lon,
            **prefs,
        )
    except Exception as e:
        print(f"  [!] Errore nel calcolo percorsi: {e}")
        return set()

    # Configurazione colori e stili per ciascuna modalita
    MODE_STYLE = {
        "piedi": {"color": "#1E88E5", "icon": "[PIEDI]", "label": "A piedi",  "weight": 5},
        "bici":  {"color": "#43A047", "icon": "[BICI]",  "label": "In bici",  "weight": 5},
        "bus":   {"color": "#FB8C00", "icon": "[BUS]",   "label": "In bus",   "weight": 5},
        "auto":  {"color": "#E53935", "icon": "[AUTO]",  "label": "In auto",  "weight": 5},
    }
    WALK_SEG_COLOR = "#90CAF9"  # colore tratti a piedi per bus/bici

    nodes_data = dict(G.nodes(data=True))
    all_path_nodes = set()
    all_coords = []

    def _path_to_coords(path):
        """Converte un percorso (lista di nodi) in lista di (lat, lon)."""
        coords = []
        for n in path:
            nd = nodes_data.get(n, {})
            lat, lon = nd.get("y"), nd.get("x")
            if lat is not None and lon is not None:
                coords.append((lat, lon))
        return coords

    # -- Disegna ogni percorso ----------------------------------------
    for mode_key in ["piedi", "bici", "bus", "auto"]:
        route = routes.get(mode_key)
        if route is None:
            print(f"    {mode_key.upper()}: nessun percorso trovato")
            continue

        style = MODE_STYLE[mode_key]
        info = route["info"]
        path = route["path"]
        all_path_nodes.update(path)

        layer = folium.FeatureGroup(
            name=f"{style['icon']} {style['label']}",
            show=True
        )

        # Percorso principale
        coords = _path_to_coords(path)
        all_coords.extend(coords)
        if coords:
            tooltip = (
                f"{style['icon']} {style['label']}: "
                f"{info['tempo_totale_min']} min | "
                f"{info['distanza_km']} km | "
                f"CO2: {info['eco_totale_kg_co2']} kg"
            )
            # Glow sottostante
            folium.PolyLine(
                coords,
                color=style["color"],
                weight=style["weight"] + 6,
                opacity=0.25,
            ).add_to(layer)
            # Linea principale
            folium.PolyLine(
                coords,
                color=style["color"],
                weight=style["weight"],
                opacity=0.9,
                tooltip=tooltip,
            ).add_to(layer)

        # Tratti a piedi (per bus e bici) - linea tratteggiata
        for walk_key in ["walk_start", "walk_end"]:
            walk_path = route.get(walk_key, [])
            if walk_path:
                all_path_nodes.update(walk_path)
                walk_coords = _path_to_coords(walk_path)
                all_coords.extend(walk_coords)
                if walk_coords:
                    seg_label = "inizio" if walk_key == "walk_start" else "fine"
                    folium.PolyLine(
                        walk_coords,
                        color=WALK_SEG_COLOR,
                        weight=3,
                        opacity=0.8,
                        dash_array="8 6",
                        tooltip=f"[PIEDI] Tratto a piedi ({seg_label})",
                    ).add_to(layer)

        print(f"    {mode_key.upper()}: {info['num_nodi']} nodi | "
              f"{info['tempo_totale_min']} min | {info['distanza_km']} km")

        layer.add_to(m)

    # -- Marker PARTENZA e ARRIVO -------------------------------------
    layer_markers = folium.FeatureGroup(name="[PIN] Partenza / Arrivo", show=True)

    start_icon_html = """
        <div style="
            background:#00C853; border:3px solid #fff; border-radius:50%;
            width:36px; height:36px; display:flex; align-items:center;
            justify-content:center; font-size:20px;
            box-shadow:0 0 10px rgba(0,200,83,0.6);
            font-weight:bold; color:white; line-height:1">S</div>"""
    folium.Marker(
        (start_lat, start_lon),
        popup=folium.Popup(
            f"<b>Partenza</b><br>{start_lat:.6f}, {start_lon:.6f}",
            max_width=200,
        ),
        tooltip="Partenza",
        icon=folium.DivIcon(html=start_icon_html, icon_size=(42, 42), icon_anchor=(21, 21)),
    ).add_to(layer_markers)

    end_icon_html = """
        <div style="
            background:#D32F2F; border:3px solid #fff; border-radius:50%;
            width:36px; height:36px; display:flex; align-items:center;
            justify-content:center; font-size:20px;
            box-shadow:0 0 10px rgba(211,47,47,0.6);
            font-weight:bold; color:white; line-height:1">E</div>"""
    folium.Marker(
        (end_lat, end_lon),
        popup=folium.Popup(
            f"<b>Arrivo</b><br>{end_lat:.6f}, {end_lon:.6f}",
            max_width=200,
        ),
        tooltip="Arrivo",
        icon=folium.DivIcon(html=end_icon_html, icon_size=(42, 42), icon_anchor=(21, 21)),
    ).add_to(layer_markers)

    layer_markers.add_to(m)

    # -- Marker FERMATE BUS e STAZIONI BIKE ---------------------------
    layer_stops = folium.FeatureGroup(name="[STOP] Fermate / Stazioni", show=True)

    def _add_stop_marker(node_id, label, color, icon_letter):
        """Aggiunge un marker diamante per fermata/stazione."""
        nd = nodes_data.get(node_id, {})
        lat, lon = nd.get("y"), nd.get("x")
        if lat is None or lon is None:
            return
        name = nd.get("desc", nd.get("name", str(node_id)))
        icon_html = f"""
            <div style="
                background:{color}; border:2px solid #fff;
                width:28px; height:28px; display:flex; align-items:center;
                justify-content:center; font-size:14px; font-weight:bold;
                box-shadow:0 0 8px rgba(0,0,0,0.4);
                color:white; line-height:1;
                transform:rotate(45deg); border-radius:4px">
                <span style="transform:rotate(-45deg)">{icon_letter}</span>
            </div>"""
        folium.Marker(
            (lat, lon),
            popup=folium.Popup(f"<b>{label}</b><br>{name}<br>{lat:.6f}, {lon:.6f}", max_width=250),
            tooltip=f"{label}: {name}",
            icon=folium.DivIcon(html=icon_html, icon_size=(34, 34), icon_anchor=(17, 17)),
        ).add_to(layer_stops)

    # Fermate bus (se percorso bus esiste)
    bus_route = routes.get("bus")
    if bus_route and bus_route.get("info"):
        bus_info = bus_route["info"]
        if bus_info.get("fermata_partenza"):
            _add_stop_marker(bus_info["fermata_partenza"], "Fermata Bus Partenza", "#FB8C00", "B")
        if bus_info.get("fermata_arrivo"):
            _add_stop_marker(bus_info["fermata_arrivo"], "Fermata Bus Arrivo", "#E65100", "B")

    # Stazioni bike (se percorso bici esiste)
    bike_route = routes.get("bici")
    if bike_route and bike_route.get("info"):
        bike_info = bike_route["info"]
        if bike_info.get("stazione_partenza"):
            _add_stop_marker(bike_info["stazione_partenza"], "Stazione Bike Partenza", "#43A047", "K")
        if bike_info.get("stazione_arrivo"):
            _add_stop_marker(bike_info["stazione_arrivo"], "Stazione Bike Arrivo", "#2E7D32", "K")

    layer_stops.add_to(m)

    # -- Zoom automatico ----------------------------------------------
    if all_coords:
        lats = [c[0] for c in all_coords]
        lons = [c[1] for c in all_coords]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]],
                     padding=(60, 60))

    # -- Pannello confronto percorsi ----------------------------------
    mezzo_rows = ""
    for mode_key in ["piedi", "bici", "bus", "auto"]:
        style = MODE_STYLE[mode_key]
        route = routes.get(mode_key)
        if route is None:
            mezzo_rows += (
                f"<tr style='opacity:0.35'>"
                f"<td style='padding:3px 6px'>"
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"border-radius:50%;background:{style['color']};margin-right:4px'></span>"
                f"{style['label']}</td>"
                f"<td style='padding:3px 6px;text-align:center'>-</td>"
                f"<td style='padding:3px 6px;text-align:center'>-</td>"
                f"<td style='padding:3px 6px;text-align:center'>-</td>"
                f"</tr>"
            )
        else:
            ri = route["info"]
            extra = ""
            if mode_key in ("bus", "bici") and route.get("walk_start"):
                extra = " *"
            mezzo_rows += (
                f"<tr>"
                f"<td style='padding:3px 6px'>"
                f"<span style='display:inline-block;width:10px;height:10px;"
                f"border-radius:50%;background:{style['color']};margin-right:4px'></span>"
                f"{style['label']}{extra}</td>"
                f"<td style='padding:3px 6px;text-align:center'>{ri['tempo_totale_min']} min</td>"
                f"<td style='padding:3px 6px;text-align:center'>{ri['distanza_km']} km</td>"
                f"<td style='padding:3px 6px;text-align:center'>{ri['eco_totale_kg_co2']} kg</td>"
                f"</tr>"
            )

    # Nota a pie di pagina se ci sono tratti a piedi
    has_walk_segments = any(
        routes.get(mk) is not None and (routes[mk].get("walk_start") or routes[mk].get("walk_end"))
        for mk in ["bus", "bici"]
    )
    walk_note = ""
    if has_walk_segments:
        walk_note = (
            "<div style='font-size:10px;color:#90CAF9;margin-top:6px'>"
            "* Include tratti a piedi (linea tratteggiata) "
            "per raggiungere fermate/stazioni"
            "</div>"
        )

    pref_t = prefs.get('pref_tempo', 0.5)
    pref_e = prefs.get('pref_ecologia', 0.2)
    pref_s = prefs.get('pref_sicurezza', 0.3)

    info_html = f"""
    <div style="
        position:fixed; top:20px; right:20px; z-index:9999;
        background:rgba(10,18,40,0.95); padding:14px 18px; border-radius:12px;
        font-family:'Segoe UI',sans-serif; color:white; font-size:13px; line-height:1.6;
        border:1px solid #2a2a3a; box-shadow:0 4px 20px rgba(0,0,0,0.5);
        min-width:290px; max-width:350px;">

      <div style="font-size:15px;font-weight:700;margin-bottom:10px;color:#fff;
                  border-bottom:1px solid #333;padding-bottom:6px">
        Confronto Percorsi
      </div>

      <div style="font-size:10px;color:#888;margin-bottom:8px">
        Preferenze: T={pref_t:.1f} | E={pref_e:.1f} | S={pref_s:.1f}
      </div>

      <table style="width:100%;border-collapse:collapse;font-size:12px;
                    background:rgba(255,255,255,0.05);border-radius:8px;overflow:hidden">
        <thead>
          <tr style="color:#888;font-size:10px;border-bottom:1px solid #333">
            <th style="padding:4px 6px;text-align:left">Mezzo</th>
            <th style="padding:4px 6px;text-align:center">Tempo</th>
            <th style="padding:4px 6px;text-align:center">Distanza</th>
            <th style="padding:4px 6px;text-align:center">CO2</th>
          </tr>
        </thead>
        <tbody>
          {mezzo_rows}
        </tbody>
      </table>

      {walk_note}

      <div style="font-size:10px;color:#666;margin-top:8px;border-top:1px solid #333;padding-top:6px">
        Usa il pannello layer (in alto a destra) per mostrare/nascondere i singoli percorsi.
      </div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(info_html))

    return all_path_nodes


# -------------------------------------------------
# CLI
# -------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualizzazione interattiva del grafo di routing urbano Bari"
    )
    parser.add_argument(
        "--weight", "-w",
        choices=list(WEIGHT_CHOICES.keys()),
        default="sicurezza",
        help="Peso da usare per colorare gli archi (default: sicurezza)",
    )
    parser.add_argument(
        "--route", "-r",
        nargs=2,
        metavar=("START", "END"),
        help='Traccia percorso. Es: --route "41.125,16.865" "41.120,16.872"',
    )
    parser.add_argument(
        "--pref-tempo",      type=float, default=0.4, metavar="ALPHA"
    )
    parser.add_argument(
        "--pref-ecologia",   type=float, default=0.2, metavar="BETA"
    )
    parser.add_argument(
        "--pref-sicurezza",  type=float, default=0.4, metavar="GAMMA"
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Non aprire automaticamente il browser",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    weight_key = WEIGHT_CHOICES[args.weight]

    print("=" * 55)
    print("  VISUALIZZAZIONE GRAFO - Bari routing")
    print("=" * 55)

    print("Caricamento grafo ...")
    G = load_graph()
    print(f"  Nodi: {G.number_of_nodes()} | Archi: {G.number_of_edges()}")

    print(f"Costruzione mappa (colore archi: {args.weight}) ...")

    route_node_set = None
    prefs = dict(
        pref_tempo=args.pref_tempo,
        pref_ecologia=args.pref_ecologia,
        pref_sicurezza=args.pref_sicurezza,
    )

    if args.route:
        try:
            s_lat, s_lon = map(float, args.route[0].split(","))
            e_lat, e_lon = map(float, args.route[1].split(","))
        except ValueError:
            print("[!] Formato coordinate errato. Usa: 'lat,lon'")
            sys.exit(1)

        # Pre-calcola percorsi per sapere quali nodi mostrare
        try:
            from router import find_all_routes
            routes = find_all_routes(
                G,
                start_lat=s_lat, start_lon=s_lon,
                end_lat=e_lat,   end_lon=e_lon,
                **prefs,
            )
            # Usa il percorso piedi (o fallback) per il set di nodi
            for mode_key in ["piedi", "auto", "bus", "bici"]:
                r = routes.get(mode_key)
                if r is not None:
                    route_node_set = set(r["path"])
                    break
            if route_node_set:
                print(f"  Percorso pre-calcolato: {len(route_node_set)} nodi")
        except Exception as e:
            print(f"  [!] Pre-calcolo percorso fallito ({e}), mostro tutti i nodi.")
            route_node_set = None

    # Costruisce la mappa passando route_node_set per filtrare i nodi
    m = build_map(G, weight_key=weight_key, route_node_set=route_node_set)

    if args.route and route_node_set is not None:
        # Disegna tutti i percorsi con layer separati
        add_route_to_map(m, G, s_lat, s_lon, e_lat, e_lon, prefs)

    # LayerControl aggiunto una sola volta alla fine
    folium.LayerControl(collapsed=False).add_to(m)

    m.save(OUTPUT_FILE)
    print(f"\n[OK] Mappa salvata: {OUTPUT_FILE}")

    if not args.no_open:
        print("   Apertura nel browser ...")
        webbrowser.open(f"file://{os.path.abspath(OUTPUT_FILE)}")


if __name__ == "__main__":
    main()
