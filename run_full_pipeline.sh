#!/bin/bash
set -euo pipefail

# Risolve la directory dello script e la usa come root operativa stabile.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Definisce valori predefiniti sovrascrivibili dalla riga di comando.
GPU_ID="0"
NUM_EXAMPLES="3"
PYTHON_BIN="python3"
TRAIN_ARGS=()

# Analizza le opzioni dello script e inoltra le restanti a train.py.
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --num-examples)
            NUM_EXAMPLES="$2"
            shift 2
            ;;
        --python)
            PYTHON_BIN="$2"
            shift 2
            ;;
        --)
            shift
            TRAIN_ARGS+=("$@")
            break
            ;;
        *)
            TRAIN_ARGS+=("$1")
            shift
            ;;
    esac
done

# Limita la visibilità alla GPU scelta e forza Matplotlib in modalità headless.
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$SCRIPT_DIR/.cache/matplotlib}"
mkdir -p "$MPLCONFIGDIR"

echo "[INFO] Project root: $SCRIPT_DIR"
echo "[INFO] Using GPU index: $GPU_ID"
echo "[INFO] Python executable: $PYTHON_BIN"
echo "[INFO] Starting training..."

# Avvia il training conservando esattamente la tokenizzazione degli argomenti.
"$PYTHON_BIN" train.py "${TRAIN_ARGS[@]}"

# Recupera la run appena prodotta secondo la convenzione temporale dei nomi.
RUN_DIR="$(find runs -maxdepth 1 -type d -name 'run_*' | sort | tail -n1)"
if [ -z "$RUN_DIR" ]; then
    echo "[ERROR] No run directory created under runs/."
    exit 1
fi

# Preferisce il best checkpoint e ripiega sul modello dell'ultima epoca.
BEST_CHECKPOINT="$(find "$RUN_DIR" -maxdepth 1 -type f -name 'best_*.pth' | sort | head -n1)"
if [ -z "$BEST_CHECKPOINT" ]; then
    BEST_CHECKPOINT="$RUN_DIR/scene_graph_model.pth"
fi

if [ ! -f "$BEST_CHECKPOINT" ]; then
    echo "[ERROR] Checkpoint not found: $BEST_CHECKPOINT"
    exit 1
fi

echo "[INFO] Training completed. Run directory: $RUN_DIR"
echo "[INFO] Using checkpoint for demo/test: $BEST_CHECKPOINT"
echo "[INFO] Starting post-training demo pipeline..."

# Genera valutazione, grafici disponibili ed esempi a partire dal checkpoint.
"$PYTHON_BIN" demo.py --model-path "$BEST_CHECKPOINT" --num-examples "$NUM_EXAMPLES"

echo "[INFO] Full pipeline completed successfully."
echo "[INFO] Outputs available in: $RUN_DIR"
