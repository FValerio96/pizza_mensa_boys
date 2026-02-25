import os
import networkx as nx
import numpy as np
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

GRAPH_PATH = os.path.join(os.path.dirname(__file__), "graph.pkl")
SCALER_PATH = "scaler.pkl"
KMEANS_PATH = "kmeans.pkl"

# Nomi semantici assegnati ai cluster in base ai centroidi
CLUSTER_NAMES = {}


def load_graph():
    if not os.path.exists(GRAPH_PATH):
        raise FileNotFoundError(f"Il file {GRAPH_PATH} non esiste. Esegui prima build_graph.py.")
    print(f"Caricamento del grafo da {GRAPH_PATH}...")
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    return G


def save_graph(G):
    print(f"Salvataggio grafo in {GRAPH_PATH}...")
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Grafo aggiornato salvato in {GRAPH_PATH}")


def _assign_cluster_name(centroid):
    """
    Assegna un nome semantico al cluster basato sulle caratteristiche
    del centroide (tempo/km, sicurezza, ecologia/km).
    """
    tempo_km, sic, eco_km = centroid

    # Classificazione basata su soglie relative
    if sic < 0.3 and tempo_km < 5.0:
        return "Strada rapida e sicura"
    elif sic < 0.3:
        return "Strada sicura"
    elif tempo_km < 3.0 and eco_km > 0.01:
        return "Arteria veloce (alta CO2)"
    elif tempo_km > 8.0:
        return "Percorso lento (residenziale)"
    elif sic > 0.6:
        return "Strada rischiosa"
    else:
        return "Strada mista"


def cluster_edges(G, k=5):
    """
    Estrae 3 feature dagli archi stradali (tempo/km, sicurezza, eco/km),
    applica K-Means per identificare profili stradali distinti,
    e annota ogni arco con il cluster_id corrispondente.

    Le feature sono:
        1. tempo_per_km   — lentezza della strada (min/km)
        2. sic_raw        — rischio sicurezza [0,1]
        3. eco_per_km     — emissioni CO2 per km (kg/km)
    """
    edges_to_cluster = []
    features = []

    # 1. Estrazione delle 3 feature
    print("Estrazione caratteristiche degli archi (3 feature: tempo, sicurezza, ecologia)...")
    for u, v, k_edge, data in G.edges(keys=True, data=True):
        if data.get("modalita") in ["transfer", "lampione", "bus"]:
            continue

        length_km = data.get("length", 1.0) / 1000.0
        if length_km <= 0:
            length_km = 0.001

        # Feature 1: tempo per chilometro (lentezza)
        tempo_per_km = data.get("w_tempo_raw", 0) / length_km
        # Feature 2: rischio sicurezza (adimensionale)
        sic_raw = data.get("w_sicurezza_raw", 0)
        # Feature 3: emissioni CO2 per chilometro
        eco_per_km = data.get("w_eco_raw", 0) / length_km

        # Gestisci NaN e Inf
        for val in [tempo_per_km, sic_raw, eco_per_km]:
            if np.isnan(val) or np.isinf(val):
                val = 0.0

        tempo_per_km = 0.0 if (np.isnan(tempo_per_km) or np.isinf(tempo_per_km)) else float(tempo_per_km)
        sic_raw = 0.0 if (np.isnan(sic_raw) or np.isinf(sic_raw)) else float(sic_raw)
        eco_per_km = 0.0 if (np.isnan(eco_per_km) or np.isinf(eco_per_km)) else float(eco_per_km)

        features.append([tempo_per_km, sic_raw, eco_per_km])
        edges_to_cluster.append((u, v, k_edge))

    if not features:
        print("Nessun arco valido trovato per il clustering.")
        return

    features_matrix = np.array(features)
    print(f"    Archi validi per clustering: {len(features)}")

    # 2. Scaling
    print("Standardizzazione dei dati (3D)...")
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_matrix)

    # 3. K-Means
    print(f"Addestramento K-Means con k={k} (3 feature)...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)

    # 4. Assegnazione cluster
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    original_centroids = scaler.inverse_transform(centroids)

    cluster_stats = {i: 0 for i in range(k)}

    print("Assegnazione cluster ID agli archi e salvataggio centroidi...")
    for i, (u, v, k_edge) in enumerate(edges_to_cluster):
        G.edges[u, v, k_edge]["cluster_id"] = int(labels[i])
        cluster_stats[labels[i]] += 1

    # Salva metadati nel grafo
    G.graph["cluster_count"] = k
    for i in range(k):
        G.graph[f"centroide_{i}_tempo_km"] = float(original_centroids[i][0])
        G.graph[f"centroide_{i}_sic"] = float(original_centroids[i][1])
        G.graph[f"centroide_{i}_eco_km"] = float(original_centroids[i][2])

        # Nome semantico
        name = _assign_cluster_name(original_centroids[i])
        G.graph[f"centroide_{i}_nome"] = name

    print(f"\n{'='*60}")
    print(f"  Risultato Clustering AI (K={k}, 3 Feature)")
    print(f"{'='*60}")
    for i in range(k):
        name = G.graph[f"centroide_{i}_nome"]
        print(f"  Cluster {i} [{name}]")
        print(f"    → {cluster_stats[i]} archi")
        print(f"    → Tempo: {original_centroids[i][0]:.2f} min/km")
        print(f"    → Rischio: {original_centroids[i][1]:.3f}")
        print(f"    → CO2: {original_centroids[i][2]:.4f} kg/km")
    print(f"{'='*60}")

    # Salva modelli
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    with open(KMEANS_PATH, 'wb') as f:
        pickle.dump(kmeans, f)
    print(f"Modelli salvati: {SCALER_PATH}, {KMEANS_PATH}")


def main():
    try:
        G = load_graph()
        cluster_edges(G, k=5)
        save_graph(G)
        print("\n✅ Clustering completato con successo.")
    except Exception as e:
        print(f"❌ Errore durante il clustering: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
