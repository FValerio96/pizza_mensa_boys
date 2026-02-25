# AI-Powered Urban Routing -- Bari

Sistema di routing urbano multi-modale con pesatura intelligente degli archi tramite **K-Means Clustering**.

## Pipeline di Esecuzione

```
build_graph.py  --->  cluster_edges.py  --->  router.py  --->  visualize.py
(Grafo fisico)       (AI clustering)        (Pathfinding)     (Mappa + rotte)
```

```bash
cd ai_challenge/graph
pip install -r requirements.txt

# Step 1: Costruzione grafo (scarica OSM, cache automatica)
python build_graph.py

# Step 2: Addestramento AI (clustering K-Means sugli archi)
python cluster_edges.py

# Step 3: Visualizzazione mappa + calcolo percorsi
python visualize.py --route "41.1258,16.8650" "41.1202,16.8727"
```

> **Nota**: Al primo avvio, `build_graph.py` scarica la rete stradale da OSM (~2 min).
> Le esecuzioni successive usano la cache locale (`graph_osm_cache.graphml`).

---

## Architettura del Grafo

### Struttura Dati

Il grafo e' un **`nx.MultiDiGraph`** (orientato, multi-arco) serializzato in formato pickle (`graph.pkl`). Contiene:

- **Nodi OSM**: intersezioni stradali della rete OpenStreetMap, con attributi `x` (longitudine), `y` (latitudine)
- **Nodi bus_stop**: fermate bus AMTAB, collegate ai nodi OSM tramite archi di transfer
- **Nodi bike_station**: stazioni di bike sharing, collegate ai nodi OSM tramite archi di transfer
- **Nodi lampione**: punti luce urbani, usati per il calcolo del punteggio di sicurezza

### Tipi di Archi

| Tipo | `modalita` | `allowed_modes` | Descrizione |
|------|-----------|-----------------|-------------|
| **Stradale** | `walk` | `{walk, bike, drive, bus}` | Archi OSM base, percorribili da tutti |
| **Pedonale** | `walk` | `{walk, bike}` | Sentieri, scale, ZTL (non carrozzabili) |
| **Contromano drive** | `drive` | `{drive, bus}` | Archi aggiunti in senso inverso per strade a senso unico |
| **Transfer** | `transfer` | `{walk, bike, drive, bus}` | Collegano nodi speciali (bus/bike) al nodo OSM piu' vicino |
| **Bus-bus** | `bus` | `{bus}` | Collegano fermate bus vicine (< 500m), archi logici diretti |
| **Lampione** | `lampione` | `{walk}` | Collegano lampioni al nodo OSM piu' vicino (usati solo per densita') |

### Pesi degli Archi

Ogni arco stradale ha tre pesi grezzi (`_raw`) che vengono poi normalizzati in `[0,1]`:

| Peso | Formula | Note |
|------|---------|------|
| `w_tempo_raw` | `(length_km / speed_kph) * 60` | Minuti di percorrenza |
| `w_eco_raw` | `length_km * emission_kg_km` | 0 per walk/bike, emissione reale per drive/bus |
| `w_sicurezza_raw` | `1 - (0.4*lit + 0.3*highway + 0.3*lamp_density)` | Punteggio di rischio composito |

La **normalizzazione** avviene con min-max scaling globale, producendo `w_tempo`, `w_ecologia`, `w_sicurezza`.

### Calcolo Sicurezza (Safety Score)

Il punteggio di sicurezza combina tre fattori:

1. **`lit` score (40%)**: tag `lit=yes/no` da OpenStreetMap. Se mancante, si usa Overpass API per cercare `highway=street_lamp` entro 30m
2. **`highway` score (30%)**: tipo di strada OSM. `residential`/`living_street` = sicure, `trunk`/`primary` = meno sicure
3. **`lamp_density` score (30%)**: rapporto tra lampioni trovati entro un raggio e la lunghezza dell'arco

---

## L'idea del Clustering AI

### Problema

Nel routing tradizionale, il peso di ogni arco e' calcolato come combinazione lineare **deterministica** di tempo, ecologia e sicurezza:

```
w_final = alpha * tempo + beta * ecologia + gamma * sicurezza
```

Questo approccio tratta ogni strada come un'entita' isolata, senza considerare che strade con caratteristiche simili tendono a formare **profili** ricorrenti (es. "arteria veloce ma inquinante", "zona residenziale sicura ma lenta").

### Soluzione: Profilazione AI delle strade

Usiamo il **K-Means Clustering** per identificare automaticamente **5 profili stradali** dalla combinazione di 3 feature estratte da ogni arco:

| Feature | Significato | Unita' |
|---------|-------------|--------|
| `tempo_per_km` | Lentezza della strada | min/km |
| `sicurezza` | Livello di rischio (1 - safety_score) | [0,1] |
| `eco_per_km` | Emissioni CO2 per km percorso | kg/km |

```
  Archi del grafo     StandardScaler        K-Means (k=5)       Centroidi nel Grafo
  (3 feature/arco) -> (normalizzazione) ->  (n_init=10)      -> (metadati globali)
                                                 |
                                                 v
                                          Cluster ID per arco
                                          Nome semantico
```

### Come influenza il routing

Quando il router (`combined_weight`) calcola il peso di un arco:

1. Recupera il **`cluster_id`** dell'arco (assegnato da `cluster_edges.py`)
2. Legge le caratteristiche del **centroide** del cluster dai metadati globali del grafo:
   - `centroide_{id}_tempo_km` -- velocita' media del profilo
   - `centroide_{id}_sic` -- rischio medio del profilo
   - `centroide_{id}_eco_km` -- emissioni medie del profilo
3. Scala queste caratteristiche sulla lunghezza dell'arco per ottenere il costo complessivo
4. Se il cluster non e' disponibile (arco senza dati), usa i pesi grezzi normalizzati come fallback

**Vantaggi**:
- Le strade con **dati rumorosi o incompleti** ereditano le caratteristiche del loro profilo, producendo routing piu' robusto
- L'algoritmo cattura **pattern contestuali** (es. tutte le strade residenziali condividono lo stesso profilo)
- Il numero di parametri effettivi si riduce da N_archi*3 a 5*3 (15 centroidi)

### Esempio di cluster risultanti

| Cluster | Nome semantico | Tempo | Rischio | CO2 |
|---------|----------------|-------|---------|-----|
| 0 | Strada rapida e sicura | 2.1 min/km | 0.15 | 0.003 |
| 1 | Arteria veloce (alta CO2) | 1.8 min/km | 0.45 | 0.021 |
| 2 | Percorso lento (residenziale) | 9.2 min/km | 0.12 | 0.001 |
| 3 | Strada mista | 4.5 min/km | 0.38 | 0.008 |
| 4 | Strada rischiosa | 3.7 min/km | 0.72 | 0.015 |

> I nomi semantici sono assegnati automaticamente in base ai centroidi, e variano ad ogni riesecuzione del clustering.

---

## Routing Multi-Modale

### Modalita' di Trasporto

Il router calcola **4 percorsi indipendenti** in parallelo, uno per ciascuna modalita':

| Modalita' | Archi utilizzati | Velocita' | Emissioni |
|-----------|-----------------|-----------|-----------|
| A piedi | `walk`, `bike` | 5 km/h | 0 |
| In bici | `walk`, `bike` + transfer bike | 15 km/h | 0 |
| In bus | `drive`, `bus` + transfer (strade reali) | 20 km/h | ~0.8 kg/km (media AMTAB) |
| In auto | `drive`, `bus` | 30 km/h | variabile per arco |

### Algoritmo di Pathfinding

Per ogni modalita':

1. **Filtraggio archi**: `route_for_mode()` crea una vista filtrata del grafo mantenendo solo gli archi con `mode in allowed_modes`
2. **A\* Search**: pathfinding con euristica haversine pesata. La funzione di costo e':
   ```
   cost(u,v) = combined_weight(u, v, data, alpha, beta, gamma, mode, G)
   ```
3. **Fallback Dijkstra**: se A\* fallisce (grafo non connesso), tenta Dijkstra come backup
4. **Euristica**: distanza in linea d'aria convertita in tempo stimato a velocita' del modo scelto

### Gestione Percorsi Bus e Bici

Per bus e bici, il percorso e' composto da **3 segmenti**:

```
BUS:  [Partenza] --piedi--> [Fermata A] --strade reali--> [Fermata B] --piedi--> [Arrivo]
BIKE: [Partenza] --piedi--> [Stazione A] --bici--> [Stazione B] --piedi--> [Arrivo]
```

**Dettaglio importante**: il tratto bus tra le fermate segue le **strade reali** (archi con `allowed_modes` contenente `drive`), non gli archi diretti bus-bus. Questo perche' il bus percorre fisicamente la rete viaria. Le fermate bus sono collegate alla rete OSM tramite archi di transfer, quindi il percorso naturalmente segue:

```
Fermata A -> (transfer) -> Nodo OSM -> strade reali ... -> Nodo OSM -> (transfer) -> Fermata B
```

Le fermate piu' vicine a partenza e arrivo vengono trovate con `_nearest_special_node()`, che cerca il nodo con `node_type="bus_stop"` (o `"bike_station"`) a distanza minima haversine.

### Formula di Costo Combinato

La funzione `combined_weight()` calcola il costo di attraversamento di un arco:

```python
def combined_weight(u, v, data, alpha, beta, gamma, mode, G):
    # 1. Tenta costo AI (se cluster disponibile)
    cluster_id = data.get("cluster_id")
    if cluster_id is not None:
        t_km = G.graph[f"centroide_{cluster_id}_tempo_km"]
        s_raw = G.graph[f"centroide_{cluster_id}_sic"]
        e_km = G.graph[f"centroide_{cluster_id}_eco_km"]

        wt = t_km * (length_m / 1000.0)     # tempo scalato sulla lunghezza
        ws = s_raw                            # rischio adimensionale
        we = e_km * (length_m / 1000.0)      # ecologia scalata
        return alpha * wt + beta * we + gamma * ws

    # 2. Fallback: pesi grezzi normalizzati
    wt = data.get("w_tempo", 0.5)
    we = data.get("w_ecologia", 0.3)
    ws = data.get("w_sicurezza", 0.5)
    return alpha * wt + beta * we + gamma * ws
```

---

## Visualizzazione

### Mappa Multi-Percorso

`visualize.py` genera una mappa HTML interattiva (Folium) che mostra **tutti e 4 i percorsi simultaneamente** su layer separati:

| Colore | Modalita' | Layer |
|--------|-----------|-------|
| Blu (#1E88E5) | A piedi | `[PIEDI] A piedi` |
| Verde (#43A047) | In bici | `[BICI] In bici` |
| Arancio (#FB8C00) | In bus | `[BUS] In bus` |
| Rosso (#E53935) | In auto | `[AUTO] In auto` |

Caratteristiche visuali:
- **Linea principale**: colore pieno con glow sottostante per visibilita'
- **Tratti a piedi** (bus/bici): linea tratteggiata azzurra (#90CAF9)
- **Marker Partenza**: cerchio verde con "S"
- **Marker Arrivo**: cerchio rosso con "E"
- **Marker Fermate Bus**: diamante arancio con "B" (partenza e arrivo)
- **Marker Stazioni Bike**: diamante verde con "K" (partenza e arrivo)

### Pannello Confronto

In alto a destra, un pannello overlay mostra la tabella comparativa:

| Mezzo | Tempo | Distanza | CO2 |
|-------|-------|----------|-----|
| A piedi | X min | X km | 0 kg |
| In bici* | X min | X km | 0 kg |
| In bus* | X min | X km | X kg |
| In auto | X min | X km | X kg |

L'asterisco (*) indica che il percorso include tratti a piedi.

### Layer della Mappa Base

Oltre ai percorsi, la mappa base mostra:
- **Rete stradale**: archi colorati per peso selezionato (sicurezza: verde/arancio/rosso)
- **Fermate bus**: cerchi blu
- **Stazioni bike sharing**: cerchi verdi
- **Lampioni**: cerchi gialli piccoli
- **Archi transfer**: linee grigie tratteggiate (nascosti di default)

---

## Sorgenti Dati

| Layer | Fonte | File | Collegamento al grafo |
|-------|-------|----- |-----------------------|
| Rete stradale/pedonale | OSM via `osmnx` | cache automatica | Base del grafo |
| Illuminazione strade | Tag `lit` OSM + Overpass API | query dinamica | Safety score + nodi lampione |
| Fermate bus AMTAB | CSV locale | `data/fermate.csv` | Nodi `bus_<id>` + archi transfer |
| Bike sharing | CSV locale | `data/postazionibikesharing.csv` | Nodi `bike_<idx>` + archi transfer |
| Consumi carburante AMTAB | CSV locale | `data/consumi_amtab.csv` | Emissioni CO2 medie bus (kg/km) |
| Orari fermate AMTAB | CSV locale | `data/orari_fermate.csv` | Frequenza bus per quartiere/ora |

---

## API `router.py`

```python
from router import load_graph, find_all_routes

G = load_graph()

routes = find_all_routes(
    G,
    start_lat=41.1258, start_lon=16.8650,
    end_lat=41.1202,   end_lon=16.8727,
    pref_tempo=0.5,
    pref_ecologia=0.2,
    pref_sicurezza=0.3,
)

# Struttura risultato:
# routes = {
#     "piedi": {
#         "path": [node_id, ...],
#         "info": {
#             "tempo_totale_min": 12.5,
#             "eco_totale_kg_co2": 0.0,
#             "distanza_km": 1.2,
#             "sic_media": 0.35,
#             "num_nodi": 42,
#             "start_node": 123456,
#             "end_node": 789012,
#         }
#     },
#     "bici": {
#         "path": [node_id, ...],           # percorso in bici (tra stazioni)
#         "info": {...,
#             "stazione_partenza": "bike_3",
#             "stazione_arrivo": "bike_7",
#         },
#         "walk_start": [node_id, ...],      # tratto a piedi: start -> stazione
#         "walk_end": [node_id, ...],        # tratto a piedi: stazione -> end
#     },
#     "bus": {
#         "path": [node_id, ...],            # percorso bus (su strade reali)
#         "info": {...,
#             "fermata_partenza": "bus_12",
#             "fermata_arrivo": "bus_45",
#         },
#         "walk_start": [node_id, ...],
#         "walk_end": [node_id, ...],
#     },
#     "auto": {
#         "path": [node_id, ...],
#         "info": {...}
#     },
#     "prefs": {
#         "pref_tempo": 0.5,
#         "pref_ecologia": 0.2,
#         "pref_sicurezza": 0.3
#     },
# }
```

---

## File del Progetto

| File | Descrizione |
|------|-------------|
| `build_graph.py` | Costruzione grafo multimodale: OSM + bus AMTAB + bike sharing + lampioni |
| `cluster_edges.py` | Addestramento K-Means (3 feature, 5 cluster) e annotazione archi |
| `router.py` | Calcolo percorsi A* per tutte le 4 modalita' con costo combinato AI |
| `visualize.py` | Mappa HTML interattiva multi-percorso (Folium) |
| `requirements.txt` | Dipendenze Python |
| `graph.pkl` | Grafo serializzato (generato da build_graph + cluster_edges) |
| `data/` | Dataset CSV AMTAB (fermate, orari, consumi, bike sharing) |

## Velocita' Medie per Modalita'

| Mezzo | Velocita' | Costante |
|-------|-----------|----------|
| A piedi | 5 km/h | `SPEED_WALK_KMH` |
| In bici | 15 km/h | `SPEED_BIKE_KMH` |
| In auto | 30 km/h | `SPEED_DRIVE_KMH` |
| In bus | 20 km/h | `SPEED_BUS_KMH` |

## Parametri Configurabili (CLI)

```bash
python visualize.py [opzioni]

  --weight, -w     Peso per colorare archi: tempo | ecologia | sicurezza (default: sicurezza)
  --route, -r      Coordinate partenza e arrivo: "lat,lon" "lat,lon"
  --pref-tempo     Preferenza tempo [0-1] (default: 0.4)
  --pref-ecologia  Preferenza ecologia [0-1] (default: 0.2)
  --pref-sicurezza Preferenza sicurezza [0-1] (default: 0.4)
  --no-open        Non aprire il browser automaticamente
```
