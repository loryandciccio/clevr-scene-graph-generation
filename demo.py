#!/usr/bin/env python3
"""
Script di demo per presentare il progetto al professore.
Genera tutti i risultati necessari per la presentazione.
"""
import argparse
import json
import os
import subprocess
import sys


def run_command(cmd_list, description):
    """Esegue un comando e stampa il risultato."""
    print(f"\n{'='*60}")
    print(f"📌 {description}")
    print(f"{'='*60}")
    print(f"Comando: {' '.join(cmd_list)}\n")
    
    # Esegue senza shell e restituisce un booleano per permettere alla demo di continuare.
    try:
        result = subprocess.run(cmd_list, check=True, capture_output=False, text=True)
        print(f"✅ Completato con successo\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Errore durante l'esecuzione: {e}\n")
        return False


def main():
    """Esegue valutazione, generazione dei grafici e rendering degli esempi."""
    parser = argparse.ArgumentParser(description="Demo script per la presentazione del progetto")
    parser.add_argument("--model-path", required=True, help="Percorso al modello addestrato (.pth)")
    parser.add_argument("--skip-test", action="store_true", help="Salta la fase di testing")
    parser.add_argument("--skip-plots", action="store_true", help="Salta la generazione dei grafici")
    parser.add_argument("--num-examples", type=int, default=3, help="Numero di esempi di scene graph da visualizzare")
    # Le opzioni consentono di riutilizzare artefatti già generati.
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🎓 DEMO PROGETTO: Image to Scene Graph (CLEVR)")
    print("="*60)
    
    if not os.path.isfile(args.model_path):
        print(f"❌ Errore: Modello non trovato in {args.model_path}")
        sys.exit(1)
    
    # Estrai il percorso della run directory dal model_path
    # Tutti gli artefatti vengono collocati accanto al checkpoint selezionato.
    run_dir = os.path.dirname(args.model_path)
    history_json = os.path.join(run_dir, "training_history.json")
    predicted_json = os.path.join(run_dir, "predicted_scene_graphs.json")
    plots_dir = os.path.join(run_dir, "plots")
    examples_dir = os.path.join(run_dir, "examples")
    python_bin = sys.executable
    overall_success = True
    
    # Step 1: Testing e generazione dei grafi
    # Il test produce sia predizioni decodificate sia metriche quantitative.
    if not args.skip_test:
        success = run_command(
            [python_bin, "test.py", "--model-path", args.model_path, "--export-json", predicted_json],
            "STEP 1: Valutazione del modello sul validation set"
        )
        if not success:
            overall_success = False
            print("⚠️  La valutazione è fallita. Continuo con le altre fasi...\n")
    
    # Step 2: Generazione dei grafici di training
    # I grafici richiedono lo storico, che può non essere incluso nelle run versionate.
    if not args.skip_plots and os.path.isfile(history_json):
        success = run_command(
            [python_bin, "plot_training.py", "--history-json", history_json, "--output-dir", plots_dir],
            "STEP 2: Generazione dei grafici di training"
        )
        if not success:
            overall_success = False
            print("⚠️  La generazione dei grafici è fallita. Continuo...\n")
    elif not args.skip_plots:
        overall_success = False
        print(f"\n⚠️  Training history non trovato in {history_json}")
        print("   Salta la generazione dei grafici di training.\n")
    
    # Step 3: Visualizzazione di esempi di Scene Graph
    # Crea un'immagine separata per i primi N grafi esportati.
    if os.path.isfile(predicted_json):
        os.makedirs(examples_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"📌 STEP 3: Visualizzazione di {args.num_examples} esempi di Scene Graph")
        print(f"{'='*60}\n")
        
        for i in range(args.num_examples):
            output_path = os.path.join(examples_dir, f"scene_graph_{i}.png")
            success = run_command(
                [
                    python_bin, "visualize_scene_graph.py",
                    "--input-json", predicted_json,
                    "--sample-index", str(i),
                    "--save-path", output_path
                ],
                f"Visualizzazione esempio {i+1}/{args.num_examples}"
            )
            if not success:
                overall_success = False
                print(f"⚠️  Visualizzazione esempio {i} fallita\n")
    else:
        overall_success = False
        print(f"\n⚠️  {predicted_json} non trovato.")
        print("   Esegui prima: python test.py --model-path <path>\n")
    
    # Step 4: Riepilogo dei file generati
    print("\n" + "="*60)
    print("📊 RIEPILOGO FILE GENERATI PER LA PRESENTAZIONE")
    print("="*60)
    
    # Verifica esplicitamente tutti gli artefatti attesi dalla presentazione.
    files_to_check = [
        (os.path.join(run_dir, "predicted_scene_graphs_metrics.json"), "Metriche quantitative di valutazione"),
        (os.path.join(plots_dir, "training_loss.png"), "Grafico della loss durante training"),
        (os.path.join(plots_dir, "object_metrics.png"), "Metriche di Object Detection"),
        (os.path.join(plots_dir, "relation_metrics.png"), "Metriche di Relation Prediction"),
        (os.path.join(plots_dir, "attribute_accuracy.png"), "Accuracy di Attribute Classification"),
        (os.path.join(plots_dir, "all_metrics_overview.png"), "Vista completa di tutte le metriche"),
        (os.path.join(examples_dir, "scene_graph_0.png"), "Esempio 1 di Scene Graph visualizzato"),
        (os.path.join(examples_dir, "scene_graph_1.png"), "Esempio 2 di Scene Graph visualizzato"),
        (os.path.join(examples_dir, "scene_graph_2.png"), "Esempio 3 di Scene Graph visualizzato"),
    ]
    
    print()
    missing_files = []
    for file_path, description in files_to_check:
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
            print(f"✅ {file_path:45s} - {description} ({size_str})")
        else:
            missing_files.append(file_path)
            print(f"⚠️  {file_path:45s} - {description} (NON TROVATO)")

    # Qualsiasi artefatto mancante rende non riuscita la pipeline complessiva.
    if missing_files:
        overall_success = False

    print("\n" + "="*60)
    print("✅ DEMO COMPLETATA!" if overall_success else "⚠️  DEMO COMPLETATA CON PROBLEMI")
    print("="*60)
    if overall_success:
        print("\nI file generati sono pronti per la presentazione.")
        print("Puoi ora mostrare al professore:")
        print(f"  1. I grafici di training evolution ({plots_dir})")
        print(f"  2. Le metriche quantitative ({run_dir})")
        print(f"  3. Gli esempi visivi di Scene Graph ({examples_dir})")
        print()
    else:
        print("\nLa demo ha rilevato problemi o artefatti mancanti.")
        print("Controlla il log sopra prima di considerare il run concluso.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
