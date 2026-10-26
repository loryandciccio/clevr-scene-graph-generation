"""Visualizza un scene graph esportato usando NetworkX e Matplotlib."""

import argparse
import json
import os


def parse_args():
    """Analizza gli argomenti della riga di comando per la visualizzazione del grafo."""
    # Espone selezione del campione, filtro degli archi e destinazione opzionale.
    parser = argparse.ArgumentParser(description="Visualize predicted scene graphs with NetworkX")
    parser.add_argument("--input-json", required=True, help="Path to predicted_scene_graphs JSON")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index inside the JSON list")
    parser.add_argument("--min-edge-score", type=float, default=0.0, help="Filter edges below this score")
    parser.add_argument("--save-path", default="", help="Optional output image path (PNG)")
    return parser.parse_args()


def validate_args(args):
    """Valida gli input della visualizzazione."""
    # Interrompe subito l'esecuzione se input o indice non sono utilizzabili.
    if not os.path.isfile(args.input_json):
        raise FileNotFoundError(f"Input JSON not found: {args.input_json}")
    if args.sample_index < 0:
        raise ValueError("--sample-index must be >= 0")
    if not (0.0 <= args.min_edge_score <= 1.0):
        raise ValueError("--min-edge-score must be in [0, 1]")


def load_sample(path, sample_index):
    """Carica un campione dal JSON del grafo di scena esportato."""
    # Il file deve contenere la lista di grafi prodotta da test.py.
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Input JSON must be a non-empty list of scene graph samples")
    if sample_index >= len(data):
        raise IndexError(f"sample-index {sample_index} out of range (num samples: {len(data)})")

    return data[sample_index], len(data)


def main():
    """Esegue il rendering di un campione di grafo di scena utilizzando NetworkX."""
    args = parse_args()
    validate_args(args)

    try:
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "Visualization requires networkx and matplotlib. Install with: pip install networkx matplotlib"
        ) from exc

    sample, total_samples = load_sample(args.input_json, args.sample_index)

    # DiGraph preserva verso e predicato delle relazioni spaziali.
    graph = nx.DiGraph()

    # Compone un'etichetta leggibile combinando identificatore e attributi.
    for node in sample.get("nodes", []):
        node_id = node["node_id"]
        attrs = node.get("attributes", {})
        pixel_coords = node.get("pixel_coords", {})
        coord_text = ""
        if "x" in pixel_coords and "y" in pixel_coords:
            coord_text = f"\nxy=({pixel_coords['x']:.0f}, {pixel_coords['y']:.0f})"
        label = (
            f"id={node_id}\n"
            f"{attrs.get('color', '?')} {attrs.get('material', '?')}\n"
            f"{attrs.get('size', '?')} {attrs.get('shape', '?')}{coord_text}"
        )
        graph.add_node(node_id, label=label)

    # Inserisce soltanto archi che superano il filtro visuale richiesto.
    for edge in sample.get("edges", []):
        score = float(edge.get("score", 0.0))
        if score < args.min_edge_score:
            continue
        graph.add_edge(
            edge["subject"],
            edge["object"],
            label=f"{edge['predicate']} ({score:.2f})",
        )

    if graph.number_of_nodes() == 0:
        raise ValueError("Selected sample has no active nodes to visualize")

    fig = plt.figure(figsize=(10, 8))
    # Il seed rende stabile il layout tra esecuzioni sullo stesso grafo.
    pos = nx.spring_layout(graph, seed=42)

    nx.draw_networkx_nodes(graph, pos, node_color="#dbeafe", edgecolors="#1e3a8a", node_size=2500)
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", arrowsize=18, width=1.8)

    node_labels = {node_id: graph.nodes[node_id]["label"] for node_id in graph.nodes()}
    edge_labels = nx.get_edge_attributes(graph, "label")

    nx.draw_networkx_labels(graph, pos, labels=node_labels, font_size=8)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=7)

    title = (
        f"Scene Graph Sample {args.sample_index}/{total_samples - 1}"
        f" - image: {sample.get('image_filename', 'N/A')}"
    )
    plt.title(title)
    plt.axis("off")

    # Salva in modalità non interattiva oppure mostra la finestra a schermo.
    if args.save_path:
        plt.savefig(args.save_path, dpi=180, bbox_inches="tight")
        print(f"Saved scene graph visualization to {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
