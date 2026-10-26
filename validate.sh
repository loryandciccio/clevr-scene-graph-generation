#!/bin/bash
# Script di validazione pre-consegna
# Verifica che tutti i componenti del progetto siano funzionanti

echo "🔍 VALIDAZIONE PRE-CONSEGNA DEL PROGETTO"
echo "========================================"
echo ""

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ] && [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
fi
if [ -z "$PYTHON_BIN" ] && [ -x ".venv/bin/python" ]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python"
fi
if [ -z "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# Colori per output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Contatori
passed=0
failed=0
artifacts_ready=0
latest_run=""
if [ -d "runs" ]; then
    latest_run=$(find runs -maxdepth 1 -type d -name 'run_*' | sort | tail -n1)
fi

# Test 1: Verifica esistenza file essenziali
echo "Test 1: Verifica file essenziali..."
required_files=("config.py" "dataset.py" "model.py" "train.py" "test.py" "visualize_scene_graph.py" "plot_training.py" "demo.py" "main.py" "README.md" "requirements.txt")

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
        ((passed++))
    else
        echo -e "  ${RED}✗${NC} $file MANCANTE"
        ((failed++))
    fi
done

# Test 2: Compilazione Python
echo ""
echo "Test 2: Compilazione dei file Python..."
for pyfile in *.py; do
    if "$PYTHON_BIN" -m py_compile "$pyfile" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $pyfile compila correttamente"
        ((passed++))
    else
        echo -e "  ${RED}✗${NC} $pyfile ha errori di sintassi"
        ((failed++))
    fi
done

# Test 3: Verifica import
echo ""
echo "Test 3: Verifica import dei moduli..."
modules=("torch" "torchvision" "PIL" "tqdm" "networkx" "matplotlib")
for module in "${modules[@]}"; do
    if "$PYTHON_BIN" -c "import $module" 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $module installato"
        ((passed++))
    else
        echo -e "  ${RED}✗${NC} $module NON installato"
        ((failed++))
    fi
done

# Test 4: Verifica struttura CLEVR (opzionale)
echo ""
echo "Test 4: Verifica dataset CLEVR..."
if [ -d "CLEVR_v1.0" ]; then
    echo -e "  ${GREEN}✓${NC} Directory CLEVR_v1.0 presente"
    ((passed++))
    
    if [ -d "CLEVR_v1.0/images" ]; then
        echo -e "  ${GREEN}✓${NC} CLEVR_v1.0/images presente"
        ((passed++))
    else
        echo -e "  ${YELLOW}⚠${NC} CLEVR_v1.0/images assente"
    fi
    
    if [ -d "CLEVR_v1.0/scenes" ]; then
        echo -e "  ${GREEN}✓${NC} CLEVR_v1.0/scenes presente"
        ((passed++))
    else
        echo -e "  ${YELLOW}⚠${NC} CLEVR_v1.0/scenes assente"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Directory CLEVR_v1.0 non trovata (normale se dataset è altrove)"
fi

# Test 5: Verifica modello addestrato
echo ""
echo "Test 5: Verifica modello addestrato..."
if [ -n "$latest_run" ] && find "$latest_run" -maxdepth 1 -type f \( -name '*.pth' \) | grep -q .; then
    echo -e "  ${GREEN}✓${NC} Run rilevata: $latest_run"
    ((passed++))

    best_checkpoint=$(find "$latest_run" -maxdepth 1 -type f -name 'best_*.pth' | sort | head -n1)
    final_checkpoint=$(find "$latest_run" -maxdepth 1 -type f -name '*.pth' | sort | head -n1)

    if [ -n "$best_checkpoint" ]; then
        size=$(du -h "$best_checkpoint" | cut -f1)
        echo -e "  ${GREEN}✓${NC} Best checkpoint: $best_checkpoint ($size)"
        ((passed++))
        artifacts_ready=1
    elif [ -n "$final_checkpoint" ]; then
        size=$(du -h "$final_checkpoint" | cut -f1)
        echo -e "  ${YELLOW}⚠${NC} Solo checkpoint finale trovato: $final_checkpoint ($size)"
    fi
else
    echo -e "  ${YELLOW}⚠${NC} Nessun checkpoint trovato in runs/run_* (devi ancora addestrare)"
fi

# Test 6: Verifica output predetti
echo ""
echo "Test 6: Verifica output di test..."
if [ -n "$latest_run" ] && [ -f "$latest_run/predicted_scene_graphs.json" ]; then
    echo -e "  ${GREEN}✓${NC} $latest_run/predicted_scene_graphs.json presente"
    ((passed++))
elif [ -f "predicted_scene_graphs.json" ]; then
    echo -e "  ${GREEN}✓${NC} predicted_scene_graphs.json presente"
    ((passed++))
else
    echo -e "  ${YELLOW}⚠${NC} predicted_scene_graphs.json non trovato (esegui test.py)"
fi

# Test 7: Verifica documentazione
echo ""
echo "Test 7: Verifica documentazione..."
doc_files=("README.md")
for doc in "${doc_files[@]}"; do
    if [ -f "$doc" ]; then
        echo -e "  ${GREEN}✓${NC} $doc presente"
        ((passed++))
    else
        echo -e "  ${RED}✗${NC} $doc MANCANTE"
        ((failed++))
    fi
done

# Riepilogo finale
echo ""
echo "========================================"
echo "📊 RISULTATI VALIDAZIONE"
echo "========================================"
echo -e "${GREEN}Passati: $passed${NC}"
if [ $failed -gt 0 ]; then
    echo -e "${RED}Falliti: $failed${NC}"
fi
echo ""

if [ $failed -eq 0 ] && [ $artifacts_ready -eq 1 ]; then
    echo -e "${GREEN}🎉 PROGETTO PRONTO PER TEST FINALE E DEMO!${NC}"
    echo ""
    echo "Prossimi passi:"
    echo "1. Esegui: $PYTHON_BIN test.py"
    echo "2. Se vuoi la demo completa: $PYTHON_BIN demo.py --model-path $best_checkpoint"
    echo "3. Verifica che plots/ ed examples/ siano stati generati"
    echo "4. Rivedi PRESENTAZIONE.md per preparare la demo"
elif [ $failed -eq 0 ]; then
    echo -e "${GREEN}✅ CODICE PRONTO PER L'ADDESTRAMENTO${NC}"
    echo ""
    echo "Prossimi passi:"
    echo "1. Avvia il training: $PYTHON_BIN train.py"
    echo "2. Poi esegui il test: $PYTHON_BIN test.py"
    echo "3. Genera i grafici: $PYTHON_BIN plot_training.py --history-json runs/run_YYYYMMDD_HHMMSS/training_history.json"
else
    echo -e "${RED}⚠️  CI SONO PROBLEMI DA RISOLVERE${NC}"
    echo "Controlla i file mancanti o gli errori sopra."
fi

echo ""
