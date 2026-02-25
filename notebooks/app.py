"""
SafeWalk Web — Interfaccia chatbot per il routing sicuro a Bari
================================================================
Avvia con:  uv run python notebooks/app.py
Poi apri:   http://localhost:5000
"""

import sys
import os
import time

# Assicura che il modulo safewalk_pipeline sia importabile
sys.path.insert(0, os.path.dirname(__file__))
# Assicura che ai_challenge sia importabile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, request, jsonify, send_file
import safewalk_pipeline as sw

app = Flask(__name__)

# ── Stato globale (caricato una sola volta all'avvio) ─────────────────────
pipeline = {}


def init_pipeline():
    """Carica grafo, rischio dal CSV, linee bus, CSV e arricchisci il grafo."""
    print("=" * 60)
    print("  SafeWalk Web — Inizializzazione pipeline")
    print("=" * 60)
    print()

    G = sw.build_walking_graph(use_cache=True)
    osm_bus_stops, line_info = sw.fetch_bus_routes(use_cache=True)
    fermate, bike, orari, consumi = sw.load_csv_data()
    sw.compute_transit_frequency(orari)
    sw.compute_avg_bus_co2(consumi)
    G = sw.enrich_graph_with_stops(G, fermate, bike)

    # Carica rischio archi dal CSV (sostituisce il calcolo dai lampioni)
    risk_map = sw.load_edge_risk()
    G = sw.enrich_graph_with_risk(G, risk_map)

    # Pre-calcola nodi per fermate e stazioni (UNA VOLTA)
    bus_nodes, bike_stations = sw.precompute_stop_nodes(G, fermate, bike)

    # Pre-calcola layer sicurezza strade (UNA VOLTA, evita 56K iterazioni per richiesta)
    safety_geojson = sw.precompute_safety_geojson(G)

    pipeline.update({
        "G": G,
        "fermate": fermate,
        "bike": bike,
        "osm_bus_stops": osm_bus_stops,
        "line_info": line_info,
        "bus_nodes": bus_nodes,
        "bike_stations": bike_stations,
        "safety_geojson": safety_geojson,
    })

    print()
    print("✅  Pipeline pronta! Avvio server web ...")
    print()


# ── Route Flask ───────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/map")
def serve_map():
    """Serve l'ultima mappa generata."""
    path = sw.OUTPUT_DIR / "route_map.html"
    if path.exists():
        return send_file(str(path))
    return "<p>Nessuna mappa disponibile.</p>", 404


@app.route("/api/compute", methods=["POST"])
def api_compute():
    """Calcola il/i percorso/i e genera la mappa."""
    data = request.json
    ora = int(data.get("orario", 12))
    tipo = data.get("tipo", "veloce")  # veloce | sostenibile | sicuro | personalizzata

    G = pipeline["G"]
    fermate = pipeline["fermate"]
    bike_df = pipeline["bike"]
    osm_bus_stops = pipeline["osm_bus_stops"]
    bus_nodes = pipeline["bus_nodes"]
    bike_stations = pipeline["bike_stations"]

    # Applicare bonus notturno
    sw.apply_night_bonus(G, ora)

    # Se arrivano coordinate dirette (click su mappa)
    origin_coords = data.get("origin_coords")  # [lat, lon]
    dest_coords = data.get("dest_coords")      # [lat, lon]

    if origin_coords and dest_coords:
        origin = tuple(origin_coords)
        dest = tuple(dest_coords)
        partenza = f"({origin[0]:.4f}, {origin[1]:.4f})"
        destinazione = f"({dest[0]:.4f}, {dest[1]:.4f})"
    else:
        partenza = data.get("partenza", "").strip()
        destinazione = data.get("destinazione", "").strip()
        origin = sw.geocode_place(partenza)
        if not origin:
            return jsonify({"error": f'Non riesco a trovare "{partenza}".'})
        dest = sw.geocode_place(destinazione)
        if not dest:
            return jsonify({"error": f'Non riesco a trovare "{destinazione}".'})

    is_night = (ora >= 19 or ora <= 5)

    bike_segments, bike_metriche = None, None
    bus_segments, bus_metriche = None, None
    safe_segments, safe_metriche = None, None
    custom_segments, custom_metriche = None, None

    if tipo == "sostenibile":
        bike_segments, bike_metriche = sw.find_bike_route(
            G, origin, dest, bike_df, ora, precomputed_bike=bike_stations)

    elif tipo == "veloce":
        bus_segments, bus_metriche = sw.find_bus_route(
            G, origin, dest, fermate, osm_bus_stops, ora,
            precomputed_bus=bus_nodes)

    elif tipo == "sicuro":
        safe_segments, safe_metriche = sw.find_safe_route(G, origin, dest, ora)

    elif tipo == "personalizzata":
        # Pesi dall'utente (0-100), normalizzati a 0-1
        w_sicurezza = float(data.get("w_sicurezza", 33)) / 100.0
        w_ecologia = float(data.get("w_ecologia", 33)) / 100.0
        w_velocita = float(data.get("w_velocita", 34)) / 100.0

        # Normalizza i pesi perché la somma sia 1
        total_w = w_sicurezza + w_ecologia + w_velocita
        if total_w > 0:
            w_sicurezza /= total_w
            w_ecologia /= total_w
            w_velocita /= total_w

        try:
            from ai_challenge.graph.router import find_all_routes

            routes = find_all_routes(
                G,
                start_lat=origin[0], start_lon=origin[1],
                end_lat=dest[0], end_lon=dest[1],
                pref_tempo=w_velocita,
                pref_ecologia=w_ecologia,
                pref_sicurezza=w_sicurezza,
            )
            # Usa il percorso a piedi come base per il routing personalizzato
            walk_route = routes.get("piedi")
            if walk_route and walk_route.get("path"):
                path = walk_route["path"]
                info = walk_route.get("info", {})
                custom_segments = [
                    {"type": "walk", "route": path,
                     "info": f"⚙️ Percorso personalizzato (Sic:{w_sicurezza:.0%} Eco:{w_ecologia:.0%} Vel:{w_velocita:.0%})"},
                ]
                custom_metriche = {
                    "distanza_km": round(info.get("distanza_km", 0), 2),
                    "tempo_min": round(info.get("tempo_totale_min", 0), 1),
                    "safety_score": round(info.get("sic_media", 0) * 100, 1),
                    "co2_kg": round(info.get("eco_totale_kg_co2", 0), 3),
                    "mezzo": "⚙️ Personalizzato",
                }
        except Exception as e:
            print(f"   ⚠️ Errore routing personalizzato: {e}")
            import traceback
            traceback.print_exc()

    # Genera mappa con layer sicurezza strade (precomputed)
    sw.visualize_map_light(
        G,
        bike_segments=bike_segments,
        bike_metriche=bike_metriche,
        bus_segments=bus_segments,
        bus_metriche=bus_metriche,
        safe_segments=safe_segments,
        safe_metriche=safe_metriche,
        custom_segments=custom_segments,
        custom_metriche=custom_metriche,
        safety_geojson=pipeline.get("safety_geojson"),
        filename="route_map.html",
    )

    result = {
        "origin": list(origin),
        "dest": list(dest),
        "bike": bike_metriche,
        "bus": bus_metriche,
        "safe": safe_metriche,
        "custom": custom_metriche,
        "is_night": is_night,
        "map_url": f"/map?t={int(time.time())}",
        "summary": _build_summary(
            bike_metriche, bus_metriche, safe_metriche, custom_metriche,
            tipo, ora, partenza, destinazione, is_night
        ),
    }
    return jsonify(result)


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Rispondi a domande di follow-up sul percorso calcolato."""
    data = request.json
    message = data.get("message", "")
    result = data.get("result", {})
    response = _handle_discussion(message, result)
    return jsonify({"response": response})


# ── Generazione risposte chatbot ─────────────────────────────────────────
def _build_summary(bike, bus, safe, custom, tipo, ora, partenza, destinazione, is_night):
    """Genera il sommario HTML che il chatbot mostra dopo il calcolo."""
    parts = [
        f"Ecco i risultati per <b>{partenza}</b> → <b>{destinazione}</b> "
        f"alle <b>{ora}:00</b>",
    ]
    if is_night:
        parts.append(
            "🌙 <i>Modalità notturna attiva: le strade illuminate "
            "hanno ricevuto un bonus di sicurezza.</i>"
        )
    parts.append("")

    if safe:
        parts.append("<b>🛡️ Percorso Sicuro:</b>")
        parts.append(
            f"📏 {safe['distanza_km']} km &nbsp;|&nbsp; "
            f"⏱️ {safe['tempo_min']} min &nbsp;|&nbsp; "
            f"🛡️ Sicurezza: <b>{safe['safety_score']}%</b> &nbsp;|&nbsp; "
            f"🌱 0 kg CO₂"
        )
        parts.append("")

    if custom:
        parts.append("<b>⚙️ Percorso Personalizzato:</b>")
        parts.append(
            f"📏 {custom['distanza_km']} km &nbsp;|&nbsp; "
            f"⏱️ {custom['tempo_min']} min &nbsp;|&nbsp; "
            f"🛡️ {custom['safety_score']}% &nbsp;|&nbsp; "
            f"🌱 {custom['co2_kg']} kg CO₂"
        )
        parts.append("")

    if bike:
        parts.append("<b>🚲 Percorso Bici (sostenibile):</b>")
        parts.append(
            f"🚶 Cammina fino a stazione <b>{bike['stazione_partenza']}</b> "
            f"({bike['walk1_min']} min)"
        )
        parts.append(
            f"🚲 Pedala fino a stazione <b>{bike['stazione_arrivo']}</b> "
            f"({bike['bike_min']} min)"
        )
        parts.append(f"🚶 Cammina a destinazione ({bike['walk2_min']} min)")
        parts.append(
            f"📏 {bike['distanza_km']} km &nbsp;|&nbsp; "
            f"⏱️ {bike['tempo_min']} min &nbsp;|&nbsp; "
            f"🛡️ {bike['safety_score']}% &nbsp;|&nbsp; "
            f"🌱 0 kg CO₂"
        )
        parts.append("")

    if bus:
        linea = bus.get("linea_bus", "")
        linea_html = f" — <b>{linea}</b>" if linea else ""
        parts.append("<b>🚌 Percorso Bus (veloce):</b>")
        parts.append(
            f"🚶 Cammina fino a fermata <b>{bus['fermata_partenza']}</b> "
            f"({bus['walk1_min']} min)"
        )
        parts.append(
            f"🚌 Bus fino a <b>{bus['fermata_arrivo']}</b>{linea_html} "
            f"({bus['bus_min']} min)"
        )
        parts.append(f"🚶 Cammina a destinazione ({bus['walk2_min']} min)")
        parts.append(
            f"📏 {bus['distanza_km']} km &nbsp;|&nbsp; "
            f"⏱️ {bus['tempo_min']} min &nbsp;|&nbsp; "
            f"🛡️ {bus['safety_score']}% &nbsp;|&nbsp; "
            f"🌱 {bus['co2_kg']} kg CO₂"
        )
        parts.append("")

    has_any = bike or bus or safe or custom
    if not has_any:
        parts.append(
            "❌ Non è stato possibile calcolare alcun percorso. "
            "Prova con luoghi diversi."
        )
    else:
        parts.append("Chiedimi qualsiasi cosa su questi percorsi! 💬")

    return "<br>".join(parts)


def _handle_discussion(message, result):
    """Gestisce le domande di follow-up dell'utente analizzando keywords."""
    msg = message.lower().strip()
    bike = result.get("bike")
    bus = result.get("bus")
    is_night = result.get("is_night", False)

    if not bike and not bus:
        return (
            "Non ho ancora calcolato un percorso. "
            "Dimmi partenza e destinazione per iniziare!"
        )

    # ── Sicurezza ────────────────────────────────────────────────────
    if any(w in msg for w in [
        "sicur", "pericolos", "buio", "notte", "illumin", "luce", "rischio",
    ]):
        parts = []
        if bike:
            parts.append(f"🚲 Bici: sicurezza <b>{bike['safety_score']}%</b>")
        if bus:
            parts.append(f"🚌 Bus: sicurezza <b>{bus['safety_score']}%</b>")
        if bike and bus:
            if bus["safety_score"] > bike["safety_score"]:
                parts.append(
                    f"Il percorso in <b>bus</b> è più sicuro "
                    f"({bus['safety_score']}% vs {bike['safety_score']}%)."
                )
            else:
                parts.append(
                    f"Il percorso in <b>bici</b> è più sicuro "
                    f"({bike['safety_score']}% vs {bus['safety_score']}%)."
                )
        if is_night:
            parts.append(
                "🌙 Essendo orario notturno (19-05), le strade illuminate "
                "hanno ricevuto un bonus di sicurezza +15%, mentre le strade "
                "al buio hanno subito una penalità di -10%."
            )
        parts.append(
            "<br>Il punteggio si basa su: illuminazione stradale (40%), "
            "densità lampioni (25%), tipo di strada (20%) e "
            "prossimità trasporti (15%)."
        )
        return "<br>".join(parts)

    # ── Tempo ────────────────────────────────────────────────────────
    if any(w in msg for w in [
        "tempo", "veloc", "lento", "quanto ci", "minuti", "durata", "lungo",
    ]):
        parts = []
        if bike:
            parts.append(
                f"🚲 Bici: <b>{bike['tempo_min']} min</b> totali<br>"
                f"&nbsp;&nbsp;🚶 cammino: {bike['walk1_min']}+{bike['walk2_min']} min, "
                f"🚲 pedalo: {bike['bike_min']} min"
            )
        if bus:
            parts.append(
                f"🚌 Bus: <b>{bus['tempo_min']} min</b> totali<br>"
                f"&nbsp;&nbsp;🚶 cammino: {bus['walk1_min']}+{bus['walk2_min']} min, "
                f"🚌 bus: {bus['bus_min']} min"
            )
        if bike and bus:
            faster = "bici" if bike["tempo_min"] < bus["tempo_min"] else "bus"
            diff = abs(round(bike["tempo_min"] - bus["tempo_min"], 1))
            parts.append(
                f"<br>Il <b>{faster}</b> è più veloce di {diff} minuti."
            )
        return "<br>".join(parts)

    # ── Bus / Linee ──────────────────────────────────────────────────
    if any(w in msg for w in [
        "bus", "autobus", "pullman", "linea", "fermata", "quale prend",
    ]):
        if not bus:
            return (
                "Non ho calcolato un percorso bus. "
                "Prova con modalità <b>Veloce</b>."
            )
        linea = bus.get("linea_bus", "")
        parts = [
            "🚌 <b>Dettagli percorso Bus:</b>",
            f"🚶 Cammina fino a: <b>{bus['fermata_partenza']}</b> "
            f"({bus['walk1_min']} min)",
        ]
        if linea:
            parts.append(f"🚌 Prendi il <b>{linea}</b>")
        else:
            parts.append("🚌 Prendi il bus dalla fermata")
        parts.append(
            f"🚏 Scendi a: <b>{bus['fermata_arrivo']}</b> "
            f"({bus['bus_min']} min di viaggio)"
        )
        parts.append(
            f"🚶 Cammina fino a destinazione ({bus['walk2_min']} min)"
        )
        parts.append(
            f"<br>📏 Distanza totale: {bus['distanza_km']} km | "
            f"CO₂: {bus['co2_kg']} kg"
        )
        return "<br>".join(parts)

    # ── Bici / Bike sharing ──────────────────────────────────────────
    if any(w in msg for w in [
        "bici", "bike", "ciclabil", "pedalare", "sharing", "stazione",
    ]):
        if not bike:
            return (
                "Non ho calcolato un percorso bici. "
                "Prova con modalità <b>Sostenibile</b>."
            )
        parts = [
            "🚲 <b>Dettagli percorso Bici:</b>",
            f"🚶 Cammina fino alla stazione: "
            f"<b>{bike['stazione_partenza']}</b> ({bike['walk1_min']} min)",
            f"🚲 Pedala fino alla stazione: "
            f"<b>{bike['stazione_arrivo']}</b> ({bike['bike_min']} min)",
            f"🚶 Cammina a destinazione ({bike['walk2_min']} min)",
            f"<br>🚲 Bici disponibili alla partenza: "
            f"<b>{bike.get('bici_disponibili', 'N/A')}</b>",
            f"📏 Distanza: {bike['distanza_km']} km | Emissioni: 0 kg CO₂ 🌱",
        ]
        return "<br>".join(parts)

    # ── Ecologia / CO₂ ──────────────────────────────────────────────
    if any(w in msg for w in [
        "co2", "inquin", "ambient", "ecolog", "green", "sostenibil",
        "emissioni", "carbon",
    ]):
        parts = []
        dist_ref = 0
        if bike:
            parts.append(
                f"🚲 Bici: <b>0 kg CO₂</b> — zero emissioni! 🌱"
            )
            dist_ref = bike["distanza_km"]
        if bus:
            parts.append(
                f"🚌 Bus: <b>{bus['co2_kg']} kg CO₂</b> per passeggero"
            )
            dist_ref = bus["distanza_km"]
        auto_co2 = round(dist_ref * 0.5, 2) if dist_ref else "?"
        parts.append(
            f"🚗 In auto sarebbero circa <b>{auto_co2} kg CO₂</b>"
        )
        if bike and bus:
            parts.append(
                f"<br>Scegliendo la bici risparmi <b>{bus['co2_kg']} kg CO₂</b> "
                "rispetto al bus e molto di più rispetto all'auto!"
            )
        return "<br>".join(parts)

    # ── Distanza ─────────────────────────────────────────────────────
    if any(w in msg for w in [
        "distanz", "km", "chilometr", "lontano", "quanto dista",
    ]):
        parts = []
        if bike:
            parts.append(f"🚲 Bici: <b>{bike['distanza_km']} km</b>")
        if bus:
            parts.append(f"🚌 Bus: <b>{bus['distanza_km']} km</b>")
        return "<br>".join(parts)

    # ── Consiglio ────────────────────────────────────────────────────
    if any(w in msg for w in [
        "consigl", "miglior", "prefer", "consigliami", "quale scegl",
        "cosa mi", "raccomand", "sugger",
    ]):
        if bike and bus:
            parts = ["Ecco il mio consiglio 🤔:"]
            if bike["tempo_min"] < bus["tempo_min"]:
                parts.append(
                    f"⏱️ Hai fretta → <b>Bici</b> "
                    f"({bike['tempo_min']} vs {bus['tempo_min']} min)"
                )
            else:
                parts.append(
                    f"⏱️ Hai fretta → <b>Bus</b> "
                    f"({bus['tempo_min']} vs {bike['tempo_min']} min)"
                )
            parts.append(
                "🌱 Vuoi essere eco-friendly → <b>Bici</b> (0 emissioni)"
            )
            safer = (
                ("Bus", bus["safety_score"], bike["safety_score"])
                if bus["safety_score"] > bike["safety_score"]
                else ("Bici", bike["safety_score"], bus["safety_score"])
            )
            parts.append(
                f"🛡️ Vuoi sicurezza → <b>{safer[0]}</b> "
                f"({safer[1]}% vs {safer[2]}%)"
            )
            return "<br>".join(parts)
        elif bike:
            return (
                f"🚲 Con la bici: {bike['tempo_min']} min, "
                f"sicurezza {bike['safety_score']}%, 0 emissioni."
            )
        elif bus:
            return (
                f"🚌 Con il bus: {bus['tempo_min']} min, "
                f"sicurezza {bus['safety_score']}%, "
                f"CO₂ {bus['co2_kg']} kg."
            )

    # ── Confronto ────────────────────────────────────────────────────
    if any(w in msg for w in ["confront", "differ", "paragone", "vs"]):
        if bike and bus:
            parts = [
                "<b>Confronto 🚲 Bici vs 🚌 Bus:</b>",
                f"<table style='margin-top:4px;border-collapse:collapse;'>"
                f"<tr><th style='text-align:left;padding:2px 10px;'>Metrica</th>"
                f"<th style='padding:2px 10px;'>🚲 Bici</th>"
                f"<th style='padding:2px 10px;'>🚌 Bus</th></tr>"
                f"<tr><td style='padding:2px 10px;'>Distanza</td>"
                f"<td style='text-align:center;'>{bike['distanza_km']} km</td>"
                f"<td style='text-align:center;'>{bus['distanza_km']} km</td></tr>"
                f"<tr><td style='padding:2px 10px;'>Tempo</td>"
                f"<td style='text-align:center;'>{bike['tempo_min']} min</td>"
                f"<td style='text-align:center;'>{bus['tempo_min']} min</td></tr>"
                f"<tr><td style='padding:2px 10px;'>Sicurezza</td>"
                f"<td style='text-align:center;'>{bike['safety_score']}%</td>"
                f"<td style='text-align:center;'>{bus['safety_score']}%</td></tr>"
                f"<tr><td style='padding:2px 10px;'>CO₂</td>"
                f"<td style='text-align:center;'>0 kg</td>"
                f"<td style='text-align:center;'>{bus['co2_kg']} kg</td></tr>"
                f"</table>",
            ]
            return "".join(parts)
        return "Ho solo un percorso calcolato, non posso confrontare."

    # ── Nuovo percorso ───────────────────────────────────────────────
    if any(w in msg for w in [
        "nuovo", "altro", "ricomincia", "cambia", "diverso", "reset",
    ]):
        return (
            "Per calcolare un nuovo percorso, clicca il pulsante "
            "<b>🔄 Nuovo Percorso</b> in alto nel pannello chat!"
        )

    # ── Default ──────────────────────────────────────────────────────
    return (
        "Puoi chiedermi informazioni su:<br>"
        "• ⏱️ <b>Tempo</b> di percorrenza<br>"
        "• 🚌 <b>Linee bus</b> e fermate<br>"
        "• 🚲 <b>Stazioni bike</b> sharing<br>"
        "• 🛡️ <b>Sicurezza</b> del percorso<br>"
        "• 🌱 <b>Impatto ecologico</b> (CO₂)<br>"
        "• 🔀 <b>Confronto</b> tra bici e bus<br>"
        "• 💡 <b>Consigli</b> su quale percorso scegliere"
    )


# ── Avvio ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_pipeline()
    print("🌐  Server in ascolto su http://localhost:5000")
    print("    Premi Ctrl+C per uscire.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
