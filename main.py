"""Fornisce un punto di ingresso unico per training, test e visualizzazione."""

import argparse
import glob
import os
import subprocess
import sys


# Ancora tutti gli script alla directory del progetto, non alla cwd del chiamante.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def run_script(script_name, script_args, python_executable):
    """Esegue uno script del progetto con argomenti extra opzionali.
    
    Args:
        script_name: Nome del file Python da eseguire (es. "train.py")
        script_args: Lista di argomenti da passare allo script
        python_executable: Percorso dell'interprete Python da usare
    
    Raises:
        FileNotFoundError: Se lo script non esiste
        subprocess.CalledProcessError: Se lo script fallisce durante l'esecuzione
    """
    # Risolve e valida lo script prima di avviare un nuovo processo.
    script_path = os.path.join(PROJECT_ROOT, script_name)
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Script not found: {script_path}")

    # Usa una lista di argomenti per evitare interpretazione da parte della shell.
    command = [python_executable, script_path, *script_args]
    print(f"\n[RUN] {' '.join(command)}")
    subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def has_flag(script_args, flag_name):
    """Controlla se una flag e gia presente nella lista degli argomenti."""
    # Supporta sia "--flag" sia la forma compatta "--flag=valore".
    for arg in script_args:
        if arg == flag_name or arg.startswith(f"{flag_name}="):
            return True
    return False


def get_flag_value(script_args, flag_name):
    """Estrae il valore associato a una flag in formato '--flag value' o '--flag=value'."""
    # Cerca entrambe le sintassi accettate da argparse.
    for idx, arg in enumerate(script_args):
        if arg == flag_name and idx + 1 < len(script_args):
            return script_args[idx + 1]
        if arg.startswith(f"{flag_name}="):
            return arg.split("=", 1)[1]
    return ""


def set_flag_if_missing(script_args, flag_name, flag_value):
    """Aggiunge una flag solo se non e gia stata fornita esplicitamente."""
    # La scelta esplicita dell'utente ha sempre precedenza sul valore automatico.
    if has_flag(script_args, flag_name):
        return list(script_args)
    return [*script_args, flag_name, flag_value]


def find_latest_run_dir():
    """Trova la cartella runs/run_* piu recente."""
    # Considera esclusivamente directory conformi alla convenzione delle run.
    run_dirs = [path for path in glob.glob(os.path.join(PROJECT_ROOT, "runs", "run_*")) if os.path.isdir(path)]
    if not run_dirs:
        return ""
    # L'ordinamento per modifica è più robusto del solo timestamp nel nome.
    run_dirs.sort(key=os.path.getmtime, reverse=True)
    return run_dirs[0]


def find_preferred_checkpoint(run_dir):
    """Preferisce il best checkpoint; in fallback usa l'ultimo checkpoint disponibile."""
    if not run_dir:
        return ""

    # Preferisce il checkpoint scelto dalla metrica di validazione.
    best_candidates = sorted(glob.glob(os.path.join(run_dir, "best_*.pth")))
    if best_candidates:
        return best_candidates[0]

    # Ripiega su qualsiasi checkpoint disponibile nella run.
    model_candidates = sorted(glob.glob(os.path.join(run_dir, "*.pth")))
    if model_candidates:
        return model_candidates[0]

    return ""


def prepare_test_args(base_args, run_dir):
    """Completa automaticamente model-path ed export-json per i test."""
    # Copia la lista per non mutare gli argomenti ricevuti dal chiamante.
    test_args = list(base_args)
    checkpoint_path = find_preferred_checkpoint(run_dir)
    if checkpoint_path:
        test_args = set_flag_if_missing(test_args, "--model-path", checkpoint_path)

    # Colloca le predizioni accanto al checkpoint quando una run è disponibile.
    default_export = os.path.join(run_dir or PROJECT_ROOT, "predicted_scene_graphs.json")
    test_args = set_flag_if_missing(test_args, "--export-json", default_export)
    return test_args


def prepare_visualize_args(base_args, export_json_path):
    """Completa automaticamente l'input JSON per la visualizzazione."""
    # Completa soltanto l'input mancante lasciando intatte le opzioni esplicite.
    visualize_args = list(base_args)
    if export_json_path:
        visualize_args = set_flag_if_missing(visualize_args, "--input-json", export_json_path)
    return visualize_args


def parse_args():
    """Analizza gli argomenti della CLI per l'orchestratore della pipeline.
    
    Questo script coordina l'esecuzione dei vari componenti della pipeline:
    - train: Addestra il modello
    - test: Valuta il modello su un set di test
    - visualize: Visualizza i grafi di scena predetti
    - all: Esegue training, testing e opzionalmente visualizzazione in sequenza
    
    Returns:
        argparse.Namespace: Argomenti parsati dalla linea di comando
    """
    # La modalità seleziona quale script specializzato eseguire.
    parser = argparse.ArgumentParser(description="Unified launcher for CLEVR scene-graph pipeline")
    parser.add_argument(
        "mode",
        choices=["train", "test", "visualize", "all"],
        help="Execution mode",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for child scripts",
    )
    parser.add_argument(
        "--train-arg",
        action="append",
        default=[],
        help="Argument to forward to train.py (repeat multiple times)",
    )
    parser.add_argument(
        "--test-arg",
        action="append",
        default=[],
        help="Argument to forward to test.py (repeat multiple times)",
    )
    parser.add_argument(
        "--visualize-arg",
        action="append",
        default=[],
        help="Argument to forward to visualize_scene_graph.py (repeat multiple times)",
    )
    parser.add_argument(
        "--with-visualize",
        action="store_true",
        help="When mode=all, run visualization after test",
    )
    return parser.parse_args()


def main():
    """Esegue le fasi selezionate della pipeline di Scene Graph Generation.
    
    Questo è il punto d'ingresso principale che permette di orchestrare
    l'esecuzione dei vari script del progetto in modo coordinato.
    Ogni modalità (train/test/visualize/all) delega l'esecuzione agli
    script specializzati corrispondenti.
    """
    # Analizza una sola volta la CLI e propaga gli argomenti alla fase scelta.
    args = parse_args()

    try:
        # Ogni ramo prepara gli eventuali default prima di delegare allo script.
        if args.mode == "train":
            run_script("train.py", args.train_arg, args.python)

        elif args.mode == "test":
            run_dir = find_latest_run_dir()
            test_args = prepare_test_args(args.test_arg, run_dir)
            run_script("test.py", test_args, args.python)

        elif args.mode == "visualize":
            run_dir = find_latest_run_dir()
            export_json_path = get_flag_value(args.visualize_arg, "--input-json")
            if not export_json_path and run_dir:
                export_json_path = os.path.join(run_dir, "predicted_scene_graphs.json")
            visualize_args = prepare_visualize_args(args.visualize_arg, export_json_path)
            run_script("visualize_scene_graph.py", visualize_args, args.python)

        elif args.mode == "all":
            run_script("train.py", args.train_arg, args.python)
            run_dir = find_latest_run_dir()
            test_args = prepare_test_args(args.test_arg, run_dir)
            run_script("test.py", test_args, args.python)
            if args.with_visualize:
                export_json_path = get_flag_value(test_args, "--export-json")
                visualize_args = prepare_visualize_args(args.visualize_arg, export_json_path)
                run_script("visualize_scene_graph.py", visualize_args, args.python)

    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Pipeline failed with exit code {exc.returncode}") from exc


if __name__ == "__main__":
    main()
