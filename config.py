"""Definisce percorsi, iperparametri e valori predefiniti dell'intero progetto."""

import os
from dataclasses import dataclass

# Calcola la root una sola volta per rendere i percorsi indipendenti dalla cwd.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# Compone il percorso predefinito del dataset locale.
DEFAULT_CLEVR_ROOT = os.path.join(PROJECT_ROOT, "CLEVR_v1.0")


@dataclass(frozen=True)
class DataConfig:
    """Configurazione del dataset e dei dati di input."""

    # Directory che contiene le immagini dei diversi split.
    img_dir: str = os.getenv("CLEVR_IMG_DIR", os.path.join(DEFAULT_CLEVR_ROOT, "images"))
    # JSON generico usato come fallback dai componenti di lettura.
    scene_json: str = os.getenv(
        "CLEVR_SCENE_JSON",
        os.path.join(DEFAULT_CLEVR_ROOT, "scenes", "CLEVR_val_scenes.json"),
    )
    # Annotazioni specifiche dello split di addestramento.
    train_scene_json: str = os.getenv(
        "CLEVR_TRAIN_SCENE_JSON",
        os.path.join(DEFAULT_CLEVR_ROOT, "scenes", "CLEVR_train_scenes.json"),
    )
    # Annotazioni specifiche dello split di validazione.
    val_scene_json: str = os.getenv(
        "CLEVR_VAL_SCENE_JSON",
        os.path.join(DEFAULT_CLEVR_ROOT, "scenes", "CLEVR_val_scenes.json"),
    )
    # Numero massimo di oggetti, e quindi di slot, per scena.
    max_objects: int = 10


@dataclass(frozen=True)
class TrainConfig:
    """Iperparametri di addestramento e percorsi di output."""

    # Dimensione massima, batching e durata del training.
    max_scenes: int = 70000
    batch_size: int = 16
    epochs: int = 80
    # Configurazione dell'ottimizzazione.
    lr: float = 1e-4
    num_workers: int = 8
    persistent_workers: bool = True
    prefetch_factor: int = 4
    # Configurazione del modulo di ragionamento e del backbone.
    reasoning_layers: int = 2
    reasoning_backend: str = os.getenv("SG_REASONING_BACKEND", "dense")
    backbone_init: str = os.getenv("SG_BACKBONE_INIT", "auto")
    # Nome del checkpoint finale creato dentro la directory della run.
    model_path: str = "scene_graph_model.pth"
    # Opzioni di addestramento aggiuntive esposte alla CLI
    use_amp: bool = False
    weight_decay: float = 1e-4
    coord_loss_weight: float = 1.0
    grad_clip_norm: float = 1.0
    scheduler: str = "cosine"
    step_size: int = 30
    gamma: float = 0.1
    early_stop_patience: int = 8
    val_relation_threshold: float = 0.3
    slot_matching_enabled: bool = True


@dataclass(frozen=True)
class EvalConfig:
    """Parametri di valutazione, esportazione e decodifica."""

    # Intervallo opzionale di scene da valutare.
    split_start: int = 0
    split_end: int = None
    # Parametri del DataLoader di valutazione.
    batch_size: int = 8
    num_workers: int = 4
    persistent_workers: bool = True
    prefetch_factor: int = 2
    # Architettura da ricostruire prima di caricare il checkpoint.
    reasoning_layers: int = 2
    reasoning_backend: str = os.getenv("SG_REASONING_BACKEND", "dense")
    backbone_init: str = os.getenv("SG_BACKBONE_INIT", "auto")
    # Configurazione dell'esportazione dei grafi decodificati.
    export_json: str = "predicted_scene_graphs.json"
    export_samples: int = 15000
    # Soglie di decisione per oggetti e relazioni.
    object_threshold: float = 0.5
    relation_threshold: float = 0.5
    slot_matching_enabled: bool = True
    # Intervallo della ricerca automatica della soglia relazionale.
    sweep_relation_thresholds: bool = True
    sweep_min: float = 0.05
    sweep_max: float = 0.95
    sweep_steps: int = 10


@dataclass(frozen=True)
class ProjectConfig:
    """Contenitore di configurazione di alto livello per il progetto."""

    data: DataConfig = DataConfig()
    train: TrainConfig = TrainConfig()
    eval: EvalConfig = EvalConfig()


# Espone un'unica configurazione immutabile condivisa dagli script.
CONFIG = ProjectConfig()
