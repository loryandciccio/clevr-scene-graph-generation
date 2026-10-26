# CLEVR Image-to-Scene-Graph Generation

An academic computer-vision project that predicts structured scene graphs from CLEVR images. Instead of answering CLEVR questions, the system estimates visible object instances, their attributes, approximate 2D image positions, and directed spatial relations.

Developed by Lorenzo Cozzolino, in collaboration with Andrea Nardoni.

## Scope and outputs

For each RGB image, the implemented model produces predictions for **up to 10 object slots**:

- Object presence.
- Attributes: color, material, shape, size.
- Normalized image-plane center coordinates (converted to pixels in reporting).
- Four directed pairwise relations: `left`, `right`, `front`, and `behind`.

This is **scene-graph prediction**, not natural-language visual question answering. Output object centers are not bounding boxes or segmentation masks.

## Implemented architecture

```text
RGB image, resized to 224 × 224
          │
ResNet34 feature extractor
          │  7 × 7 × 512
1 × 1 projection + soft positional embedding
          │
Slot Attention (10 × 256-dimensional slots; 3 iterations)
          │
Graph reasoning:
  dense message passing [default], OR
  graph attention with PyTorch Geometric [optional]
          │
Presence / attribute / coordinate / directed relation heads
          │
Permuted-slot matching and scene-graph decoding
```

The `model.py` implementation includes ResNet34, soft positional embeddings, iterative Slot Attention, a dense graph-reasoning network and an optional PyG/GAT backend. The default configuration uses **two** reasoning layers. The two backends are alternatives, not stages applied together.

`train.py` implements masked attribute/relation losses, localization loss, permutation-aware assignment of predicted slots to ground-truth objects, validation after epochs, gradient clipping, configurable scheduler, early stopping on validation relation F1, optional CUDA mixed precision, and multi-GPU Distributed Data Parallel support. Matching uses SciPy's Hungarian algorithm when available, otherwise a dynamic-programming fallback.

`test.py` evaluates the model and optionally sweeps the relation decision threshold on its evaluation split. **Selecting a threshold on the same split used for reporting performance is not an independent test**.

## Technology

Python, PyTorch, TorchVision, Pillow, SciPy, NetworkX, Matplotlib and tqdm; PyTorch Geometric is optional. Exact reproducibility may depend on the installed PyTorch/TorchVision versions, hardware and original training configuration. This repository does not claim production readiness.

## Setup

Python 3.10+ is the intended project range. In a dedicated environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For the **optional** PyG backend:

```bash
python -m pip install -r requirements-pyg.txt
```

Install a PyTorch/TorchVision build compatible with your operating system and accelerator. Consult the upstream PyTorch/PyG instructions if necessary. The original dependencies are intentionally not pinned; versions are not claimed to have been reproduced.

## CLEVR dataset and image inputs

Download **CLEVR v1.0** from the [official project page](https://cs.stanford.edu/people/jcjohns/clevr/). Its published data license is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): retain creator attribution, license information and any modification notices when redistributing dataset items.

The data is **not packaged here**. Use this layout:

```text
CLEVR_v1.0/
├── images/
│   ├── train/
│   └── val/
└── scenes/
    ├── CLEVR_train_scenes.json
    └── CLEVR_val_scenes.json
```

Paths can be supplied with `--img-dir`, `--train-scene-json` and `--val-scene-json` for training, or `--scene-json` for evaluation. Environment defaults are defined in `config.py`: `CLEVR_IMG_DIR`, `CLEVR_TRAIN_SCENE_JSON`, `CLEVR_VAL_SCENE_JSON` and `CLEVR_SCENE_JSON`.

Image files are expected to match the filenames recorded in the CLEVR scene annotations. This repository does not provide synthetic images advertised as CLEVR samples.

## Training and checkpoints

To train from randomly initialized weights **without an automatic pretrained-weight download**:

```bash
python train.py --backbone-init random --max-scenes 1000 --epochs 2 --batch-size 8 --num-workers 0
```

This is a small-scale *training command*, not a reproduction of the published run or an accuracy benchmark. For full training, configure the split, hardware, training budget and backbone explicitly. The default `--backbone-init auto` uses ResNet34 ImageNet weights only when already cached; otherwise it falls back to random weights. **Explicit** `--backbone-init imagenet` can retrieve model weights from TorchVision over the network. TorchVision cautions that pretrained weights may have additional license obligations.

Training saves a last-epoch `scene_graph_model.pth`, best-validation `best_scene_graph_model.pth`, run metrics/history and generated plots below `runs/run_<timestamp>/`. The original pretrained/training checkpoints are intentionally **not** included in this portfolio copy. No hosted download of the project's trained weights has been verified. Use a checkpoint produced by your own authorized training run, or one obtained with independent permission.

Only load checkpoints from sources you trust: the code reads PyTorch state dictionaries, and older PyTorch deserialization formats require care.

## Evaluation on CLEVR validation

With an authorized trained checkpoint and CLEVR validation data installed:

```bash
python test.py \
  --model-path runs/run_YYYYMMDD_HHMMSS/best_scene_graph_model.pth \
  --img-dir CLEVR_v1.0/images \
  --scene-json CLEVR_v1.0/scenes/CLEVR_val_scenes.json \
  --backbone-init random \
  --no-sweep-relation-thresholds \
  --relation-threshold 0.45 \
  --num-workers 0 \
  --export-json predicted_scene_graphs.json
```

Use the checkpoint's **actual** slot count, backend and layer configuration. `--backbone-init random` suppresses unnecessary pretrained downloads before the trained state dictionary is loaded. The command above is conditional on local inputs; it has not been asserted to reproduce the historical run.

The archived `results/validation_metrics_20260506.json` was copied unchanged from `runs/run_20260506_133445/predicted_scene_graphs_metrics.json` in the historical research run. Its reported `results_a` values are:

| CLEVR validation metric | Value |
| --- | ---: |
| Object presence precision / recall / F1 | 0.9467 / 0.9349 / **0.9408** |
| Relation precision / recall / F1 | 0.8241 / 0.8672 / **0.8451** |
| Attribute accuracy | 0.9496 |
| Localization mean absolute error | 28.39 px |
| Relation threshold used | 0.45 |

The same artifact records a threshold-sweep maximum relation F1 of **0.8456** at approximately 0.45, different from the `results_a` value. These are **historical reported validation metrics**, not independently rerun here, nor results on the hidden CLEVR test split. The artifact does not establish a separate untouched test set or a full, independently reproducible training environment.

## Visualization-only example (no model inference)

The `examples/synthetic_scene_graph.json` fixture is **hand-authored synthetic graph data**, not a CLEVR image, model prediction or evaluation evidence. It supports a local, offline demonstration of the graph renderer:

```bash
python visualize_scene_graph.py \
  --input-json examples/synthetic_scene_graph.json \
  --sample-index 0 \
  --save-path synthetic_graph.png
```

To produce **actual model predictions**, first obtain CLEVR images and an authorized trained checkpoint, run `test.py`, then visualize its exported JSON. `demo.py` coordinates evaluation, available history plots and example graph renderings, but requires a checkpoint. It does **not** implement standalone zero-shot inference on arbitrary uploaded images.

## Project commands

```bash
python main.py --help
python train.py --help
python test.py --help
python demo.py --help
bash validate.sh
```

The shell scripts `run_full_pipeline.sh` and `validate.sh` remain available. `run_full_pipeline.sh` launches real training before its demo stage; it is neither free nor lightweight. Do not run it as a quick preview.

## Repository layout

```text
config.py                 Configuration and defaults
dataset.py                CLEVR scenes/images to tensor targets
model.py                  Visual backbone, slots, graph reasoning, heads
train.py                  Training and validation
test.py                   Evaluation, threshold sweep, prediction export
main.py                   CLI orchestration
demo.py                   Post-training demo orchestration
visualize_scene_graph.py  NetworkX-based graph rendering
plot_training.py          Training metric plots
requirements*.txt         Dependencies (PyG optional)
examples/                 Labeled synthetic *graph* fixture only
results/                  Historical CLEVR validation metrics JSON
docs/                     Source and rights notes
MANIFEST.json             Files and SHA-256 checksums for this snapshot
```

## Limitations and ethics

The model is designed around CLEVR's synthetic scene distribution; performance on real photographs and arbitrary object categories is not established. Slot count is capped by configuration, object centers are approximate, the spatial relationship semantics are CLEVR-specific and absent dataset/checkpoint files prevent out-of-the-box model inference. Training and evaluation may consume substantial compute. No authentication credentials or original training logs are shipped.

The **entire repository has no newly assigned blanket open-source license**. Rights in jointly authored code and project checkpoints require confirmation with their respective holders before public redistribution. The dataset is separately CC BY 4.0, dataset-generation software separately BSD-licensed, and third-party publications retain their own rights. See [docs/SOURCES_AND_RIGHTS.md](docs/SOURCES_AND_RIGHTS.md).

## References

- Justin Johnson et al., [*CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning*](https://arxiv.org/abs/1612.06890), CVPR 2017.
- [Official CLEVR dataset, images and scenes](https://cs.stanford.edu/people/jcjohns/clevr/).
- [CLEVR dataset generation code](https://github.com/facebookresearch/clevr-dataset-gen) (BSD license applies to its code).
- [TorchVision pre-trained weights guidance](https://docs.pytorch.org/vision/stable/models.html).
