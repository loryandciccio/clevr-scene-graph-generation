"""Genera grafici PNG a partire dallo storico JSON di addestramento."""

import argparse
import json
import os

import matplotlib.pyplot as plt


def parse_args():
    """Analizza gli argomenti della riga di comando."""
    # Raccoglie sorgente, cartella di output e modalità di visualizzazione.
    parser = argparse.ArgumentParser(description="Plot training curves from training history JSON")
    parser.add_argument("--history-json", required=True, help="Path to training_history.json")
    parser.add_argument("--output-dir", default="plots", help="Directory to save plot images")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    return parser.parse_args()


def plot_metrics(history, output_dir, show_plots):
    """Crea i grafici delle metriche di training."""
    # Prepara la destinazione ed estrae l'asse delle epoche condiviso dai grafici.
    os.makedirs(output_dir, exist_ok=True)
    
    epochs = [entry["epoch"] for entry in history]
    
    # Grafico 1: andamento della funzione obiettivo complessiva.
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [entry["loss"] for entry in history], marker='o', linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    loss_path = os.path.join(output_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"✓ Salvato: {loss_path}")
    if show_plots:
        plt.show()
    plt.close()

    # La localizzazione è opzionale per compatibilità con storici meno recenti.
    if "localization_mae_px" in history[0]:
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, [entry["localization_mae_px"] for entry in history], marker='o', linewidth=2, color='#7c3aed')
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel("MAE (px)", fontsize=12)
        plt.title("Localization MAE", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        loc_path = os.path.join(output_dir, "localization_mae.png")
        plt.savefig(loc_path, dpi=150, bbox_inches='tight')
        print(f"✓ Salvato: {loc_path}")
        if show_plots:
            plt.show()
        plt.close()
    
    # Grafico 2: quattro pannelli per le metriche di presenza degli oggetti.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Object Detection Metrics", fontsize=16, fontweight='bold')
    
    metrics = ["object_accuracy", "object_precision", "object_recall", "object_f1"]
    titles = ["Object Accuracy", "Object Precision", "Object Recall", "Object F1 Score"]
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.plot(epochs, [entry[metric] for entry in history], marker='o', linewidth=2, color='#2563eb')
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    obj_path = os.path.join(output_dir, "object_metrics.png")
    plt.savefig(obj_path, dpi=150, bbox_inches='tight')
    print(f"✓ Salvato: {obj_path}")
    if show_plots:
        plt.show()
    plt.close()
    
    # Grafico 3: quattro pannelli per la predizione delle relazioni.
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Relation Prediction Metrics", fontsize=16, fontweight='bold')
    
    metrics = ["relation_accuracy", "relation_precision", "relation_recall", "relation_f1"]
    titles = ["Relation Accuracy", "Relation Precision", "Relation Recall", "Relation F1 Score"]
    
    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.plot(epochs, [entry[metric] for entry in history], marker='o', linewidth=2, color='#dc2626')
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    rel_path = os.path.join(output_dir, "relation_metrics.png")
    plt.savefig(rel_path, dpi=150, bbox_inches='tight')
    print(f"✓ Salvato: {rel_path}")
    if show_plots:
        plt.show()
    plt.close()
    
    # Grafico 4: accuratezza aggregata della classificazione degli attributi.
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, [entry["attribute_accuracy"] for entry in history], marker='o', linewidth=2, color='#16a34a')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Attribute Classification Accuracy", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    attr_path = os.path.join(output_dir, "attribute_accuracy.png")
    plt.savefig(attr_path, dpi=150, bbox_inches='tight')
    print(f"✓ Salvato: {attr_path}")
    if show_plots:
        plt.show()
    plt.close()
    
    # Grafico 5: confronto delle metriche principali sulla stessa figura.
    plt.figure(figsize=(14, 8))
    plt.plot(epochs, [entry["object_accuracy"] for entry in history], marker='o', linewidth=2, label='Object Accuracy')
    plt.plot(epochs, [entry["object_f1"] for entry in history], marker='s', linewidth=2, label='Object F1')
    plt.plot(epochs, [entry["relation_accuracy"] for entry in history], marker='^', linewidth=2, label='Relation Accuracy')
    plt.plot(epochs, [entry["relation_f1"] for entry in history], marker='d', linewidth=2, label='Relation F1')
    plt.plot(epochs, [entry["attribute_accuracy"] for entry in history], marker='*', linewidth=2, label='Attribute Accuracy')
    
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.title("All Metrics Overview", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])
    plt.tight_layout()
    overview_path = os.path.join(output_dir, "all_metrics_overview.png")
    plt.savefig(overview_path, dpi=150, bbox_inches='tight')
    print(f"✓ Salvato: {overview_path}")
    if show_plots:
        plt.show()
    plt.close()


def main():
    """Esegue il plotting delle curve di training."""
    args = parse_args()
    
    # Valida il file prima di tentare il parsing JSON.
    if not os.path.isfile(args.history_json):
        raise FileNotFoundError(f"Training history JSON not found: {args.history_json}")
    
    # Carica lo storico e richiede una lista non vuota di record per epoca.
    with open(args.history_json, "r", encoding="utf-8") as f:
        history = json.load(f)
    
    if not isinstance(history, list) or len(history) == 0:
        raise ValueError("Training history must be a non-empty list of epoch metrics")
    
    print(f"\n📊 Plotting training curves from {len(history)} epochs...")
    # Delega la creazione coordinata di tutte le figure.
    plot_metrics(history, args.output_dir, args.show)
    print(f"\n✅ Tutti i grafici sono stati salvati in: {args.output_dir}/")
    print(f"\nGrafici creati:")
    print(f"  - training_loss.png")
    print(f"  - object_metrics.png (4 subplot)")
    print(f"  - relation_metrics.png (4 subplot)")
    print(f"  - attribute_accuracy.png")
    print(f"  - all_metrics_overview.png (vista completa)")
    if "localization_mae_px" in history[0]:
        print(f"  - localization_mae.png")


if __name__ == "__main__":
    main()
