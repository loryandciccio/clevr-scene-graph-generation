"""Valuta un checkpoint CLEVR, ottimizza le soglie ed esporta scene graph."""

import argparse
import glob
import json
import os
from copy import deepcopy
from functools import lru_cache

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import CONFIG
from dataset import ClevrRelationDataset
from model import VisualRelationModel

try:
    # SciPy offre il solver ungherese veloce; il progetto mantiene un fallback puro.
    from scipy.optimize import linear_sum_assignment
except Exception:
    linear_sum_assignment = None


def matching_method_name(slot_matching_enabled):
    """Nome del solver/mode usato per il matching dei target sugli slot."""
    if not slot_matching_enabled:
        return "disabled"
    return "scipy_hungarian" if linear_sum_assignment is not None else "dp_fallback"


def relation_pair_mask(object_mask):
    """Costruisce una maschera di coppie valide dalla maschera degli oggetti a livello di slot."""
    # Il prodotto esterno conserva solo coppie in cui entrambi gli slot sono attivi.
    pair_mask = object_mask.unsqueeze(1) * object_mask.unsqueeze(2)
    diagonal = torch.eye(pair_mask.size(-1), device=pair_mask.device).unsqueeze(0)
    return pair_mask * (1.0 - diagonal)


def attribute_accuracy(logits, labels):
    """Calcola l'accuratezza degli attributi a livello di slot sulle etichette valide."""
    # Esclude gli slot vuoti, marcati dal dataset con l'indice sentinella -100.
    valid_mask = labels != -100
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
    preds = logits.argmax(dim=-1)
    correct = ((preds == labels) & valid_mask).float().sum()
    total = valid_mask.float().sum()
    return correct, total


def coordinate_error_sums(pred_coords, target_coords, object_mask, width_px, height_px):
    """Somma errore assoluto in pixel e conteggio delle coordinate valide."""
    # Riporta le coordinate normalizzate alla risoluzione originale dell'immagine.
    scale = pred_coords.new_tensor([width_px, height_px]).view(1, 1, 2)
    error_px = (pred_coords - target_coords).abs() * scale
    valid_mask = object_mask.unsqueeze(-1)
    error_sum = (error_px * valid_mask).sum()
    value_count = valid_mask.sum() * pred_coords.size(-1)
    return error_sum, value_count


def binary_counts(logits, targets, threshold=0.5, mask=None):
    """Calcola TP/FP/FN/TN per le predizioni binarie."""
    # Converte logit e target in vettori booleani confrontabili.
    preds = torch.sigmoid(logits) > threshold
    labels = targets > 0.5

    # Applica la maschera delle coppie valide oppure appiattisce tutti gli elementi.
    if mask is not None:
        valid = mask.bool()
        preds = preds[valid]
        labels = labels[valid]
    else:
        preds = preds.reshape(-1)
        labels = labels.reshape(-1)

    tp = (preds & labels).float().sum().item()
    fp = (preds & (~labels)).float().sum().item()
    fn = ((~preds) & labels).float().sum().item()
    tn = ((~preds) & (~labels)).float().sum().item()
    return tp, fp, fn, tn


def precision_recall_f1(tp, fp, fn):
    """Calcola precisione, recall e F1 da TP/FP/FN."""
    # I denominatori protetti evitano divisioni per zero negli split senza positivi.
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def align_rel_targets(rel_targets, out_rel):
    """Allinea i target relazionali alla forma degli output (B, C, N, N)."""
    # Accetta i layout storici del dataset e li converte nel formato canonico.
    if rel_targets.shape == out_rel.shape:
        return rel_targets
    if rel_targets.dim() == 4 and rel_targets.shape[-1] == out_rel.shape[1] and rel_targets.shape[1] == out_rel.shape[2]:
        return rel_targets.permute(0, 3, 1, 2).contiguous()
    if rel_targets.dim() == 3 and rel_targets.shape[0] == out_rel.shape[1] and rel_targets.shape[1] == out_rel.shape[2]:
        return rel_targets.unsqueeze(0).contiguous()
    if rel_targets.dim() == 4 and rel_targets.shape[1] == out_rel.shape[2] and rel_targets.shape[-1] == out_rel.shape[1]:
        return rel_targets.permute(0, 3, 1, 2).contiguous()
    return None


def _min_cost_assignment_dp(cost_matrix):
    """Fallback esatto O(N^2 2^N) per assegnamento minimo (K righe, N colonne)."""
    # Ogni riga è un oggetto target e ogni colonna uno slot candidato.
    rows, cols = cost_matrix.shape
    if rows == 0:
        return []

    # Il fallback opera su valori Python per usare efficacemente la cache ricorsiva.
    cost = cost_matrix.detach().cpu().tolist()

    @lru_cache(maxsize=None)
    def solve(row_idx, used_mask):
        """Risolve ricorsivamente il sottoproblema tramite memoizzazione."""
        if row_idx == rows:
            return 0.0, ()

        best_cost = float("inf")
        best_path = ()
        # Prova ogni colonna non ancora occupata e conserva il percorso meno costoso.
        for c in range(cols):
            if (used_mask >> c) & 1:
                continue
            tail_cost, tail_path = solve(row_idx + 1, used_mask | (1 << c))
            total_cost = cost[row_idx][c] + tail_cost
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = (c,) + tail_path
        return best_cost, best_path

    _, assignment = solve(0, 0)
    return list(assignment)


def solve_min_cost_assignment(cost_matrix):
    """Ritorna colonne assegnate per ogni riga (matching 1-1 a costo minimo)."""
    rows, cols = cost_matrix.shape
    if rows == 0:
        return []
    if rows > cols:
        raise ValueError(f"Cannot assign {rows} rows to only {cols} columns")

    # Sostituisce NaN e inf per impedire il fallimento dei solver di assegnamento.
    invalid_entries = ~torch.isfinite(cost_matrix)
    if invalid_entries.any():
        cost_matrix = cost_matrix.masked_fill(invalid_entries, 1e6)

    # Preferisce SciPy e ricostruisce la lista ordinata per indice di riga.
    if linear_sum_assignment is not None:
        row_ids, col_ids = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        assignment = [None] * rows
        for r, c in zip(row_ids.tolist(), col_ids.tolist()):
            assignment[r] = c
        if any(c is None for c in assignment):
            raise RuntimeError("Invalid assignment from Hungarian solver")
        return assignment

    return _min_cost_assignment_dp(cost_matrix)


def permutation_align_targets(outputs, object_mask, attr_color, attr_material, attr_shape, attr_size, pixel_coords, relations):
    """Allinea i target GT all'ordine degli slot predetti tramite matching ottimo per-sample."""
    # Alloca copie vuote dei target che verranno riordinate sugli slot predetti.
    batch_size, num_slots = object_mask.shape
    device = object_mask.device

    aligned_object_mask = torch.zeros_like(object_mask)
    aligned_attr_color = torch.full_like(attr_color, fill_value=-100)
    aligned_attr_material = torch.full_like(attr_material, fill_value=-100)
    aligned_attr_shape = torch.full_like(attr_shape, fill_value=-100)
    aligned_attr_size = torch.full_like(attr_size, fill_value=-100)
    aligned_pixel_coords = torch.zeros_like(pixel_coords)
    aligned_relations = torch.zeros_like(relations)

    # Stacca e limita i logit: il matching non deve propagare gradienti né NaN.
    safe_attr_color = torch.nan_to_num(outputs["attr_color"].detach().float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(
        min=-30.0,
        max=30.0,
    )
    safe_attr_material = torch.nan_to_num(
        outputs["attr_material"].detach().float(),
        nan=0.0,
        posinf=30.0,
        neginf=-30.0,
    ).clamp(min=-30.0, max=30.0)
    safe_attr_shape = torch.nan_to_num(outputs["attr_shape"].detach().float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(
        min=-30.0,
        max=30.0,
    )
    safe_attr_size = torch.nan_to_num(outputs["attr_size"].detach().float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(
        min=-30.0,
        max=30.0,
    )
    safe_object_logits = torch.nan_to_num(outputs["object_logits"].detach().float(), nan=0.0, posinf=30.0, neginf=-30.0).clamp(
        min=-30.0,
        max=30.0,
    )

    # Trasforma i logit in costi negativi di classificazione confrontabili.
    logp_color = F.log_softmax(safe_attr_color, dim=-1)
    logp_material = F.log_softmax(safe_attr_material, dim=-1)
    logp_shape = F.log_softmax(safe_attr_shape, dim=-1)
    logp_size = F.log_softmax(safe_attr_size, dim=-1)
    obj_presence_cost = F.binary_cross_entropy_with_logits(
        safe_object_logits,
        torch.ones_like(safe_object_logits),
        reduction="none",
    )

    # Risolve un assegnamento indipendente per ciascuna scena del batch.
    for b in range(batch_size):
        gt_indices = torch.nonzero(object_mask[b] > 0.5, as_tuple=False).squeeze(-1)
        num_gt = int(gt_indices.numel())
        if num_gt == 0:
            continue

        gt_color = attr_color[b, gt_indices]
        gt_material = attr_material[b, gt_indices]
        gt_shape = attr_shape[b, gt_indices]
        gt_size = attr_size[b, gt_indices]

        # Ogni riga somma costo di presenza e costi dei quattro attributi.
        costs = []
        for i in range(num_gt):
            row_cost = (
                obj_presence_cost[b]
                - logp_color[b, :, gt_color[i]]
                - logp_material[b, :, gt_material[i]]
                - logp_shape[b, :, gt_shape[i]]
                - logp_size[b, :, gt_size[i]]
            )
            costs.append(row_cost)

        cost_matrix = torch.stack(costs, dim=0)
        # Mappa ogni oggetto reale a uno slot distinto del modello.
        assigned_slots = solve_min_cost_assignment(cost_matrix)
        slot_idx = torch.tensor(assigned_slots, device=device, dtype=torch.long)

        aligned_object_mask[b, slot_idx] = 1.0
        aligned_attr_color[b, slot_idx] = gt_color
        aligned_attr_material[b, slot_idx] = gt_material
        aligned_attr_shape[b, slot_idx] = gt_shape
        aligned_attr_size[b, slot_idx] = gt_size
        aligned_pixel_coords[b, slot_idx] = pixel_coords[b, gt_indices]

        # Riordina entrambi gli assi delle relazioni con la medesima permutazione.
        gt_rel = relations[b][:, gt_indices][:, :, gt_indices]
        aligned_relations[b][:, slot_idx[:, None], slot_idx[None, :]] = gt_rel

    return (
        aligned_object_mask,
        aligned_attr_color,
        aligned_attr_material,
        aligned_attr_shape,
        aligned_attr_size,
        aligned_pixel_coords,
        aligned_relations,
    )


def parse_args():
    """Analizza gli argomenti della riga di comando."""
    parser = argparse.ArgumentParser(description="Evaluate scene graph model on CLEVR")
    parser.add_argument("--img-dir", default=CONFIG.data.img_dir, help="Directory with CLEVR images")
    parser.add_argument(
        "--scene-json",
        default=CONFIG.data.val_scene_json or CONFIG.data.scene_json,
        help="CLEVR validation/test scenes JSON file",
    )
    parser.add_argument(
        "--scene-json-b",
        default="",
        help="Optional second scenes JSON for compositional shift test (A->B)",
    )
    parser.add_argument("--model-path", default=CONFIG.train.model_path, help="Model checkpoint path")
    parser.add_argument("--split-start", type=int, default=CONFIG.eval.split_start, help="Start index for evaluation subset")
    parser.add_argument("--split-end", type=int, default=CONFIG.eval.split_end, help="End index for evaluation subset")
    parser.add_argument("--max-objects", type=int, default=CONFIG.data.max_objects, help="Maximum number of slots/objects")
    parser.add_argument("--batch-size", type=int, default=CONFIG.eval.batch_size, help="Evaluation batch size")
    parser.add_argument("--num-workers", type=int, default=CONFIG.eval.num_workers, help="DataLoader workers")
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        default=CONFIG.eval.persistent_workers,
        help="Keep DataLoader workers alive for the whole evaluation",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=CONFIG.eval.prefetch_factor,
        help="Batches prefetched by each DataLoader worker",
    )
    parser.add_argument(
        "--reasoning-layers",
        type=int,
        default=CONFIG.eval.reasoning_layers,
        help="Number of graph message-passing layers",
    )
    parser.add_argument(
        "--reasoning-backend",
        type=str,
        default=CONFIG.eval.reasoning_backend,
        choices=["dense", "pyg"],
        help="Graph reasoning backend: dense or pyg",
    )
    parser.add_argument(
        "--backbone-init",
        type=str,
        default=CONFIG.eval.backbone_init,
        choices=["auto", "imagenet", "random"],
        help="How to initialize the ResNet34 backbone before loading the checkpoint",
    )
    parser.add_argument("--export-json", default=CONFIG.eval.export_json, help="Output JSON for sample predictions")
    parser.add_argument("--export-samples", type=int, default=CONFIG.eval.export_samples, help="Number of predicted graphs to export")
    parser.add_argument("--object-threshold", type=float, default=CONFIG.eval.object_threshold, help="Object confidence threshold")
    parser.add_argument("--relation-threshold", type=float, default=CONFIG.eval.relation_threshold, help="Relation confidence threshold")
    parser.add_argument(
        "--sweep-relation-thresholds",
        dest="sweep_relation_thresholds",
        action="store_true",
        help="Esegue uno sweep della soglia relazionale su validation e usa la migliore (per relation_f1)",
    )
    parser.add_argument(
        "--no-sweep-relation-thresholds",
        dest="sweep_relation_thresholds",
        action="store_false",
        help="Disabilita lo sweep e usa solo --relation-threshold",
    )
    parser.add_argument("--sweep-min", type=float, default=CONFIG.eval.sweep_min, help="Soglia minima nello sweep relazionale")
    parser.add_argument("--sweep-max", type=float, default=CONFIG.eval.sweep_max, help="Soglia massima nello sweep relazionale")
    parser.add_argument("--sweep-steps", type=int, default=CONFIG.eval.sweep_steps, help="Numero di soglie equispaziate nello sweep")
    parser.add_argument(
        "--enable-slot-matching",
        dest="disable_slot_matching",
        action="store_false",
        help="Abilita il riallineamento permutation-aware slot-oggetto nelle metriche di test",
    )
    parser.add_argument(
        "--disable-slot-matching",
        dest="disable_slot_matching",
        action="store_true",
        help="Disabilita il riallineamento permutation-aware slot-oggetto nelle metriche di test",
    )
    parser.set_defaults(
        sweep_relation_thresholds=CONFIG.eval.sweep_relation_thresholds,
        disable_slot_matching=(not CONFIG.eval.slot_matching_enabled),
    )
    return parser.parse_args()


def find_latest_checkpoint(runs_root="runs"):
    """Trova automaticamente il checkpoint migliore piu recente in runs/."""
    # Cerca run valide e le ordina dalla più recente alla meno recente.
    candidate_runs = sorted(glob.glob(os.path.join(runs_root, "run_*")))
    candidate_runs = [path for path in candidate_runs if os.path.isdir(path)]
    if not candidate_runs:
        return None

    candidate_runs.sort(key=os.path.getmtime, reverse=True)
    # Per ogni run preferisce il best checkpoint rispetto al checkpoint finale.
    for run_dir in candidate_runs:
        best_candidates = sorted(glob.glob(os.path.join(run_dir, "best_*.pth")))
        if best_candidates:
            return best_candidates[0]

        model_candidates = sorted(glob.glob(os.path.join(run_dir, "*.pth")))
        if model_candidates:
            return model_candidates[0]

    return None


def validate_args(args):
    """Valida gli argomenti di runtime e i percorsi richiesti."""
    # Valida prima percorsi e poi vincoli numerici delle opzioni.
    if not args.img_dir or not args.scene_json:
        raise ValueError(
            "Provide --img-dir and --scene-json or set CLEVR_IMG_DIR with CLEVR_VAL_SCENE_JSON/CLEVR_SCENE_JSON."
        )
    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    if not os.path.isfile(args.scene_json):
        raise FileNotFoundError(f"Scene JSON not found: {args.scene_json}")
    if args.scene_json_b and not os.path.isfile(args.scene_json_b):
        raise FileNotFoundError(f"Compositional scene JSON (B) not found: {args.scene_json_b}")
    if not os.path.isfile(args.model_path):
        auto_checkpoint = find_latest_checkpoint()
        if auto_checkpoint is not None and args.model_path == CONFIG.train.model_path:
            args.model_path = auto_checkpoint
        else:
            raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if args.reasoning_layers < 0:
        raise ValueError("--reasoning-layers must be >= 0")
    if args.reasoning_backend not in {"dense", "pyg"}:
        raise ValueError("--reasoning-backend must be one of: dense, pyg")
    if args.backbone_init not in {"auto", "imagenet", "random"}:
        raise ValueError("--backbone-init must be one of: auto, imagenet, random")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if not (0.0 < args.sweep_min < 1.0 and 0.0 < args.sweep_max < 1.0):
        raise ValueError("--sweep-min and --sweep-max must be in (0, 1)")
    if args.sweep_min >= args.sweep_max:
        raise ValueError("--sweep-min must be < --sweep-max")
    if args.sweep_steps < 2:
        raise ValueError("--sweep-steps must be >= 2")


def build_export_path(base_path, suffix):
    """Costruisce un percorso di esportazione aggiungendo un suffisso prima dell'estensione."""
    # Preserva l'estensione esistente o usa JSON quando è assente.
    root, ext = os.path.splitext(base_path)
    extension = ext or ".json"
    return f"{root}_{suffix}{extension}"


def print_results(results, title="TEST RESULTS"):
    """Stampa un riepilogo formattato dei risultati."""
    print(f"\n=== {title} ===")
    print(
        f"Relation Acc/P/R/F1: {results['relation_accuracy']:.4f}/"
        f"{results['relation_precision']:.4f}/"
        f"{results['relation_recall']:.4f}/"
        f"{results['relation_f1']:.4f}"
    )
    print(
        f"Object Acc/P/R/F1  : {results['object_accuracy']:.4f}/"
        f"{results['object_precision']:.4f}/"
        f"{results['object_recall']:.4f}/"
        f"{results['object_f1']:.4f}"
    )
    print(f"Attribute Accuracy : {results['attribute_accuracy']:.4f}")
    print(f"Localization MAE  : {results['localization_mae_px']:.2f} px")


def evaluate_split(model, device, args, scene_json, export_json=None):
    """Valuta il modello su uno split e restituisce le metriche e il conteggio dei grafi esportati."""
    # Costruisce lo split ed eventualmente ne seleziona una fetta contigua.
    dataset = ClevrRelationDataset(args.img_dir, scene_json, max_objects=args.max_objects)
    if args.split_start != 0 or args.split_end is not None:
        dataset.scenes = dataset.scenes[args.split_start:args.split_end]

    # Le opzioni di prefetch sono valide soltanto in presenza di worker processi.
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        **loader_kwargs,
    )

    # Inizializza accumulatori scalari per relazioni, oggetti, attributi e coordinate.
    rel_correct = rel_total = 0.0
    rel_tp = rel_fp = rel_fn = 0.0
    obj_correct = obj_total = 0.0
    obj_tp = obj_fp = obj_fn = 0.0
    attr_correct = attr_total = 0.0
    loc_error_sum = loc_value_count = 0.0
    exported_graphs = []

    # Disabilita l'autograd e mantiene il modello nella modalità impostata dal chiamante.
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Testing"):
            # Trasferisce input e supervisioni numeriche sul dispositivo di calcolo.
            images = images.to(device)
            object_mask = targets["object_mask"].to(device)
            relations = targets["relations"].to(device)
            attr_color = targets["attr_color"].to(device)
            attr_material = targets["attr_material"].to(device)
            attr_shape = targets["attr_shape"].to(device)
            attr_size = targets["attr_size"].to(device)
            pixel_coords = targets["pixel_coords"].to(device)

            # Esegue una singola inferenza per il batch corrente.
            outputs = model(images)

            # Uniforma il layout relazionale prima di metriche e matching.
            relations = align_rel_targets(relations, outputs["relations"])
            if relations is None:
                raise RuntimeError(
                    f"Unexpected relations shape {targets['relations'].shape}; cannot align with model output {outputs['relations'].shape}"
                )

            # Riallinea i target perché l'ordine degli slot non è semanticamente fisso.
            if not args.disable_slot_matching:
                (
                    object_mask,
                    attr_color,
                    attr_material,
                    attr_shape,
                    attr_size,
                    pixel_coords,
                    relations,
                ) = permutation_align_targets(
                    outputs,
                    object_mask,
                    attr_color,
                    attr_material,
                    attr_shape,
                    attr_size,
                    pixel_coords,
                    relations,
                )

            # Calcola accuracy e conteggi binari solo sulle coppie di oggetti reali.
            pair_mask = relation_pair_mask(object_mask)
            rel_preds = (torch.sigmoid(outputs["relations"]) > args.relation_threshold).float()
            rel_correct += ((rel_preds == relations).float() * pair_mask.unsqueeze(1)).sum().item()
            rel_total += (pair_mask.sum() * outputs["relations"].size(1)).item()

            rel_mask = pair_mask.unsqueeze(1).expand_as(outputs["relations"])
            rel_batch_tp, rel_batch_fp, rel_batch_fn, _ = binary_counts(
                outputs["relations"],
                relations,
                threshold=args.relation_threshold,
                mask=rel_mask,
            )
            rel_tp += rel_batch_tp
            rel_fp += rel_batch_fp
            rel_fn += rel_batch_fn

            # Valuta separatamente la presenza di un oggetto in ciascuno slot.
            obj_preds = (torch.sigmoid(outputs["object_logits"]) > args.object_threshold).float()
            obj_correct += (obj_preds == object_mask).float().sum().item()
            obj_total += object_mask.numel()
            obj_batch_tp, obj_batch_fp, obj_batch_fn, _ = binary_counts(
                outputs["object_logits"],
                object_mask,
                threshold=args.object_threshold,
            )
            obj_tp += obj_batch_tp
            obj_fp += obj_batch_fp
            obj_fn += obj_batch_fn

            # Aggrega le classificazioni dei quattro attributi in un'unica accuracy.
            for key, labels in [
                ("attr_color", attr_color),
                ("attr_material", attr_material),
                ("attr_shape", attr_shape),
                ("attr_size", attr_size),
            ]:
                c, t = attribute_accuracy(outputs[key], labels)
                attr_correct += c.item()
                attr_total += t.item()

            # Accumula l'errore assoluto medio dei centri direttamente in pixel.
            loc_error, loc_count = coordinate_error_sums(
                outputs["pixel_coords"],
                pixel_coords,
                object_mask,
                width_px=ClevrRelationDataset.ORIGINAL_IMAGE_WIDTH,
                height_px=ClevrRelationDataset.ORIGINAL_IMAGE_HEIGHT,
            )
            loc_error_sum += loc_error.item()
            loc_value_count += loc_count.item()

            # Decodifica soltanto il numero di esempi richiesto, senza alterare le metriche.
            if len(exported_graphs) < args.export_samples:
                batch_size = images.size(0)
                for i in range(batch_size):
                    if len(exported_graphs) >= args.export_samples:
                        break
                    sample_outputs = {
                        "object_logits": outputs["object_logits"][i].cpu(),
                        "relations": outputs["relations"][i].cpu(),
                        "attr_color": outputs["attr_color"][i].cpu(),
                        "attr_material": outputs["attr_material"][i].cpu(),
                        "attr_shape": outputs["attr_shape"][i].cpu(),
                        "attr_size": outputs["attr_size"][i].cpu(),
                        "pixel_coords": outputs["pixel_coords"][i].cpu(),
                    }
                    graph = decode_scene_graph(
                        sample_outputs,
                        object_threshold=args.object_threshold,
                        relation_threshold=args.relation_threshold,
                    )
                    graph["image_filename"] = targets["image_filename"][i]
                    exported_graphs.append(graph)

    # Converte gli accumulatori globali nelle metriche finali dello split.
    rel_precision, rel_recall, rel_f1 = precision_recall_f1(rel_tp, rel_fp, rel_fn)
    obj_precision, obj_recall, obj_f1 = precision_recall_f1(obj_tp, obj_fp, obj_fn)

    results = {
        "relation_accuracy": rel_correct / max(rel_total, 1.0),
        "relation_precision": rel_precision,
        "relation_recall": rel_recall,
        "relation_f1": rel_f1,
        "object_accuracy": obj_correct / max(obj_total, 1.0),
        "object_precision": obj_precision,
        "object_recall": obj_recall,
        "object_f1": obj_f1,
        "attribute_accuracy": attr_correct / max(attr_total, 1.0),
        "localization_mae_px": loc_error_sum / max(loc_value_count, 1.0),
    }

    # L'esportazione è opzionale durante lo sweep per evitare file intermedi.
    if export_json:
        with open(export_json, "w", encoding="utf-8") as file:
            json.dump(exported_graphs, file, indent=2)

    return results, len(exported_graphs)


def relation_threshold_sweep(model, device, args, scene_json):
    """Esegue uno sweep di soglia relazionale e restituisce il miglior valore per relation_f1."""
    # Costruisce una griglia uniforme includendo entrambi gli estremi.
    if args.sweep_steps == 1:
        thresholds = [args.sweep_min]
    else:
        step = (args.sweep_max - args.sweep_min) / float(args.sweep_steps - 1)
        thresholds = [args.sweep_min + i * step for i in range(args.sweep_steps)]

    sweep_rows = []
    best_row = None

    # Rivaluta lo split senza esportare grafi e conserva la soglia con F1 maggiore.
    for threshold in thresholds:
        sweep_args = deepcopy(args)
        sweep_args.relation_threshold = float(threshold)
        sweep_args.export_samples = 0

        results, _ = evaluate_split(
            model=model,
            device=device,
            args=sweep_args,
            scene_json=scene_json,
            export_json=None,
        )
        row = {
            "relation_threshold": float(threshold),
            "relation_accuracy": results["relation_accuracy"],
            "relation_precision": results["relation_precision"],
            "relation_recall": results["relation_recall"],
            "relation_f1": results["relation_f1"],
        }
        sweep_rows.append(row)
        if best_row is None or row["relation_f1"] > best_row["relation_f1"]:
            best_row = row

    return best_row, sweep_rows


def decode_scene_graph(sample_outputs, object_threshold=0.5, relation_threshold=0.5):
    """Decodifica gli output del modello in un dizionario del grafo di scena."""
    # Converte in probabilità i due insiemi di predizioni multi-label.
    object_scores = torch.sigmoid(sample_outputs["object_logits"])
    relation_scores = torch.sigmoid(sample_outputs["relations"])

    # Seleziona gli slot attivi e decodifica gli attributi tramite argmax.
    object_flags = object_scores > object_threshold
    color_ids = sample_outputs["attr_color"].argmax(dim=-1)
    material_ids = sample_outputs["attr_material"].argmax(dim=-1)
    shape_ids = sample_outputs["attr_shape"].argmax(dim=-1)
    size_ids = sample_outputs["attr_size"].argmax(dim=-1)
    pixel_coords = sample_outputs["pixel_coords"]

    # Crea un nodo serializzabile per ogni slot sopra soglia.
    nodes = []
    for idx in range(object_scores.shape[0]):
        if not object_flags[idx]:
            continue
        nodes.append(
            {
                "node_id": int(idx),
                "score": float(object_scores[idx].item()),
                "pixel_coords": {
                    "x": float(pixel_coords[idx, 0].item() * ClevrRelationDataset.ORIGINAL_IMAGE_WIDTH),
                    "y": float(pixel_coords[idx, 1].item() * ClevrRelationDataset.ORIGINAL_IMAGE_HEIGHT),
                },
                "attributes": {
                    "color": ClevrRelationDataset.COLOR_VOCAB[int(color_ids[idx].item())],
                    "material": ClevrRelationDataset.MATERIAL_VOCAB[int(material_ids[idx].item())],
                    "shape": ClevrRelationDataset.SHAPE_VOCAB[int(shape_ids[idx].item())],
                    "size": ClevrRelationDataset.SIZE_VOCAB[int(size_ids[idx].item())],
                },
            }
        )

    # Valuta ogni relazione per tutte le coppie ordinate di nodi attivi.
    edges = []
    active_nodes = [node["node_id"] for node in nodes]
    for rel_idx, rel_name in enumerate(ClevrRelationDataset.RELATION_NAMES):
        for subject in active_nodes:
            for obj in active_nodes:
                if subject == obj:
                    continue
                score = float(relation_scores[rel_idx, subject, obj].item())
                if score > relation_threshold:
                    edges.append(
                        {
                            "subject": int(subject),
                            "predicate": rel_name,
                            "object": int(obj),
                            "score": score,
                        }
                    )

    return {"nodes": nodes, "edges": edges}


def main():
    """Esegue la valutazione del modello e l'esportazione opzionale del grafo."""
    # Valida la configurazione prima di allocare modello e dataset.
    args = parse_args()
    validate_args(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Ricostruisce esattamente l'architettura richiesta dal checkpoint.
    model = VisualRelationModel(
        num_slots=args.max_objects,
        reasoning_layers=args.reasoning_layers,
        reasoning_backend=args.reasoning_backend,
        backbone_init=args.backbone_init,
    ).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model mismatch. Ensure --reasoning-layers, --reasoning-backend, --backbone-init, and --max-objects "
            "match the training configuration."
        ) from exc
    model.eval()
    print(f"[INFO] Using checkpoint: {args.model_path}")
    print(f"[INFO] Backbone init resolved to: {model.backbone.backbone_init}")

    # Lo sweep opzionale calibra la soglia sullo split A prima del test finale.
    sweep_best = None
    sweep_rows = None
    if args.sweep_relation_thresholds:
        print("\n=== RELATION THRESHOLD SWEEP (A->A) ===")
        sweep_best, sweep_rows = relation_threshold_sweep(
            model=model,
            device=device,
            args=args,
            scene_json=args.scene_json,
        )
        for row in sweep_rows:
            print(
                f"thr={row['relation_threshold']:.3f} | "
                f"P/R/F1={row['relation_precision']:.4f}/{row['relation_recall']:.4f}/{row['relation_f1']:.4f}"
            )
        args.relation_threshold = sweep_best["relation_threshold"]
        print(
            f"[SWEEP] Miglior soglia relazionale: {args.relation_threshold:.3f} "
            f"(F1={sweep_best['relation_f1']:.4f})"
        )

    # Valuta ed esporta il dataset principale con la soglia definitiva.
    results_a, exported_count_a = evaluate_split(
        model=model,
        device=device,
        args=args,
        scene_json=args.scene_json,
        export_json=args.export_json,
    )
    print_results(results_a, title="TEST RESULTS (A->A)")
    print(f"Exported {exported_count_a} predicted scene graphs to {args.export_json}")

    # Un secondo JSON misura lo scarto composizionale senza slicing dello split B.
    if args.scene_json_b:
        args_b = deepcopy(args)
        args_b.split_start = 0
        args_b.split_end = None

        export_json_b = build_export_path(args.export_json, "B")
        results_b, exported_count_b = evaluate_split(
            model=model,
            device=device,
            args=args_b,
            scene_json=args.scene_json_b,
            export_json=export_json_b,
        )
        print_results(results_b, title="TEST RESULTS (A->B)")
        print(f"Esportati {exported_count_b} grafi di scena predetti in {export_json_b}")

        print("\n=== COMPOSITIONAL GAP (A->B minus A->A) ===")
        compositional_gap = {}
        for key in [
            "relation_accuracy",
            "relation_f1",
            "object_accuracy",
            "object_f1",
            "attribute_accuracy",
        ]:
            gap = results_b[key] - results_a[key]
            print(f"{key:20s}: {gap:+.4f}")
            compositional_gap[key] = gap

        results_a["compositional_gap"] = compositional_gap

    # Salva metriche, configurazione del matching e dettagli dello sweep insieme.
    metrics_export_path = build_export_path(args.export_json, "metrics")
    with open(metrics_export_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "results_a": results_a,
                "results_b": results_b if args.scene_json_b else None,
                "slot_matching_enabled": (not args.disable_slot_matching),
                "slot_matching_method": matching_method_name(not args.disable_slot_matching),
                "relation_threshold_used": args.relation_threshold,
                "relation_threshold_sweep_best": sweep_best,
                "relation_threshold_sweep_rows": sweep_rows,
                "backbone_init_resolved": model.backbone.backbone_init,
            },
            f,
            indent=4,
        )
    print(f"\nMetriche di test salvate con successo in: {metrics_export_path}")


if __name__ == "__main__":
    main()
