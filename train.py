"""Addestra e valida il modello image-to-scene-graph sul dataset CLEVR."""

import argparse
import contextlib
import datetime
import json
import math
import os
import subprocess
import sys
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.amp as amp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from config import CONFIG
from dataset import ClevrRelationDataset
from model import VisualRelationModel

try:
    # SciPy accelera il matching; in sua assenza viene usato il solver DP interno.
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
    # Conserva solo coppie di slot reali e rimuove gli auto-archi diagonali.
    pair_mask = object_mask.unsqueeze(1) * object_mask.unsqueeze(2)
    diagonal = torch.eye(pair_mask.size(-1), device=pair_mask.device).unsqueeze(0)
    return pair_mask * (1.0 - diagonal)


def masked_relation_bce_loss(logits, targets, pair_mask):
    """Calcola la perdita BCE solo sulle coppie di oggetti valide."""
    # Calcola la BCE elemento per elemento prima della maschera strutturale.
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked_loss = loss * pair_mask.unsqueeze(1)
    denominator = pair_mask.sum() * logits.size(1)
    return masked_loss.sum() / denominator.clamp_min(1.0)


def masked_relation_accuracy(logits, targets, pair_mask):
    """Calcola l'accuratezza delle relazioni solo sulle coppie di oggetti valide."""
    # Converte i logit in decisioni binarie e ignora le coppie non valide.
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = ((preds == targets).float() * pair_mask.unsqueeze(1)).sum()
    total = pair_mask.sum() * logits.size(1)
    return correct, total


def attribute_accuracy(logits, labels):
    """Calcola l'accuratezza degli attributi a livello di slot sulle etichette valide."""
    # -100 identifica gli slot vuoti esclusi dalla classificazione.
    valid_mask = labels != -100
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)

    preds = logits.argmax(dim=-1)
    correct = ((preds == labels) & valid_mask).float().sum()
    total = valid_mask.float().sum()
    return correct, total


def masked_coordinate_loss(pred_coords, target_coords, object_mask):
    """Loss di regressione sui centri immagine solo per gli slot associati a oggetti reali."""
    # Smooth L1 limita l'influenza degli errori di localizzazione più grandi.
    coord_loss = F.smooth_l1_loss(pred_coords, target_coords, reduction="none")
    valid_mask = object_mask.unsqueeze(-1)
    denominator = valid_mask.sum() * pred_coords.size(-1)
    return (coord_loss * valid_mask).sum() / denominator.clamp_min(1.0)


def coordinate_error_sums(pred_coords, target_coords, object_mask, width_px, height_px):
    """Somma errore assoluto in pixel e conteggio delle coordinate valide."""
    # Riporta l'errore normalizzato alla risoluzione originale CLEVR.
    scale = pred_coords.new_tensor([width_px, height_px]).view(1, 1, 2)
    error_px = (pred_coords - target_coords).abs() * scale
    valid_mask = object_mask.unsqueeze(-1)
    error_sum = (error_px * valid_mask).sum()
    value_count = valid_mask.sum() * pred_coords.size(-1)
    return error_sum, value_count


def find_nonfinite_output_keys(outputs):
    """Ritorna i nomi degli output che contengono NaN/Inf."""
    # Ispeziona solo i valori tensoriali nel dizionario degli output.
    bad_keys = []
    for key, value in outputs.items():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            bad_keys.append(key)
    return bad_keys


def sanitize_logits_for_matching(logits, clamp_value=30.0):
    """Sanitizza i logits usati solo nel matching per evitare NaN/Inf nel solver."""
    # Il clamp stabilizza il solver senza modificare i logit usati dalle loss.
    return torch.nan_to_num(logits.detach().float(), nan=0.0, posinf=clamp_value, neginf=-clamp_value).clamp(
        min=-clamp_value,
        max=clamp_value,
    )


def restore_checkpoint_into_model(model, checkpoint_path, device):
    """Ricarica un checkpoint dentro il modello corrente."""
    if not checkpoint_path or not os.path.isfile(checkpoint_path):
        return False

    # Carica sul dispositivo corrente e gestisce l'eventuale wrapper DDP.
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    target_model = model.module if hasattr(model, "module") else model
    target_model.load_state_dict(state_dict)
    return True


def binary_counts(logits, targets, threshold=0.5, mask=None):
    """Calcola TP/FP/FN/TN per le predizioni binarie."""
    # Applica soglia e maschera prima della matrice di confusione.
    preds = torch.sigmoid(logits) > threshold
    labels = targets > 0.5

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
    # Protegge i denominatori per i batch senza esempi positivi.
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def align_rel_targets(rel_targets, out_rel):
    """Allinea i target delle relazioni alla forma degli output (B, C, N, N)."""
    # Normalizza i layout supportati nel formato canonico (B, C, N, N).
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
    rows, cols = cost_matrix.shape
    if rows == 0:
        return []

    # Converte in valori Python per memoizzare i sottoproblemi ricorsivi.
    cost = cost_matrix.detach().cpu().tolist()

    @lru_cache(maxsize=None)
    def solve(row_idx, used_mask):
        """Risolve ricorsivamente il sottoproblema tramite memoizzazione."""
        if row_idx == rows:
            return 0.0, ()

        best_cost = float("inf")
        best_path = ()
        # Prova ogni slot libero e conserva la continuazione meno costosa.
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

    # Sostituisce valori non finiti prima del solver scelto.
    invalid_entries = ~torch.isfinite(cost_matrix)
    if invalid_entries.any():
        cost_matrix = cost_matrix.masked_fill(invalid_entries, 1e6)

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
    # Prealloca target vuoti nell'ordine degli slot predetti.
    batch_size, num_slots = object_mask.shape
    device = object_mask.device

    aligned_object_mask = torch.zeros_like(object_mask)
    aligned_attr_color = torch.full_like(attr_color, fill_value=-100)
    aligned_attr_material = torch.full_like(attr_material, fill_value=-100)
    aligned_attr_shape = torch.full_like(attr_shape, fill_value=-100)
    aligned_attr_size = torch.full_like(attr_size, fill_value=-100)
    aligned_pixel_coords = torch.zeros_like(pixel_coords)
    aligned_relations = torch.zeros_like(relations)

    # Il matching opera su copie staccate e numericamente limitate.
    safe_attr_color = sanitize_logits_for_matching(outputs["attr_color"])
    safe_attr_material = sanitize_logits_for_matching(outputs["attr_material"])
    safe_attr_shape = sanitize_logits_for_matching(outputs["attr_shape"])
    safe_attr_size = sanitize_logits_for_matching(outputs["attr_size"])
    safe_object_logits = sanitize_logits_for_matching(outputs["object_logits"])

    logp_color = F.log_softmax(safe_attr_color, dim=-1)
    logp_material = F.log_softmax(safe_attr_material, dim=-1)
    logp_shape = F.log_softmax(safe_attr_shape, dim=-1)
    logp_size = F.log_softmax(safe_attr_size, dim=-1)
    obj_presence_cost = F.binary_cross_entropy_with_logits(
        safe_object_logits,
        torch.ones_like(safe_object_logits),
        reduction="none",
    )

    # Risolve una permutazione indipendente per ogni scena.
    for b in range(batch_size):
        gt_indices = torch.nonzero(object_mask[b] > 0.5, as_tuple=False).squeeze(-1)
        num_gt = int(gt_indices.numel())
        if num_gt == 0:
            continue

        gt_color = attr_color[b, gt_indices]
        gt_material = attr_material[b, gt_indices]
        gt_shape = attr_shape[b, gt_indices]
        gt_size = attr_size[b, gt_indices]

        # Combina costo di presenza e costi dei quattro attributi.
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
        assigned_slots = solve_min_cost_assignment(cost_matrix)
        slot_idx = torch.tensor(assigned_slots, device=device, dtype=torch.long)

        aligned_object_mask[b, slot_idx] = 1.0
        aligned_attr_color[b, slot_idx] = gt_color
        aligned_attr_material[b, slot_idx] = gt_material
        aligned_attr_shape[b, slot_idx] = gt_shape
        aligned_attr_size[b, slot_idx] = gt_size
        aligned_pixel_coords[b, slot_idx] = pixel_coords[b, gt_indices]

        # Applica la medesima permutazione a entrambi gli assi relazionali.
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


def evaluate_model(model, val_loader, device, relation_threshold=0.5, use_slot_matching=True, distributed=False):
    """Valuta il modello sul validation set restituendo metriche aggregate."""
    # Disabilita i comportamenti specifici del training durante la validazione.
    model.eval()

    rel_correct = rel_total = 0.0
    rel_tp = rel_fp = rel_fn = 0.0
    obj_correct = obj_total = 0.0
    obj_tp = obj_fp = obj_fn = 0.0
    attr_correct = attr_total = 0.0
    loc_error_sum = loc_value_count = 0.0

    # Evita di costruire il grafo autograd mentre si aggregano le metriche.
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            object_mask = targets["object_mask"].to(device)
            relations = targets["relations"].to(device)
            attr_color = targets["attr_color"].to(device)
            attr_material = targets["attr_material"].to(device)
            attr_shape = targets["attr_shape"].to(device)
            attr_size = targets["attr_size"].to(device)
            pixel_coords = targets["pixel_coords"].to(device)

            # Produce le predizioni e uniforma il layout delle relazioni.
            outputs = model(images)
            relations = align_rel_targets(relations, outputs["relations"])
            if relations is None:
                raise RuntimeError(
                    f"Unexpected relations shape during validation; cannot align with model output {outputs['relations'].shape}"
                )

            # Compensa l'ordine arbitrario degli slot object-centric.
            if use_slot_matching:
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

            # Valuta relazioni soltanto fra coppie di oggetti reali.
            pair_mask = relation_pair_mask(object_mask)
            rel_preds = (torch.sigmoid(outputs["relations"]) > relation_threshold).float()
            rel_correct += ((rel_preds == relations).float() * pair_mask.unsqueeze(1)).sum().item()
            rel_total += (pair_mask.sum() * outputs["relations"].size(1)).item()

            rel_mask = pair_mask.unsqueeze(1).expand_as(outputs["relations"])
            rel_batch_tp, rel_batch_fp, rel_batch_fn, _ = binary_counts(
                outputs["relations"],
                relations,
                threshold=relation_threshold,
                mask=rel_mask,
            )
            rel_tp += rel_batch_tp
            rel_fp += rel_batch_fp
            rel_fn += rel_batch_fn

            obj_preds = (torch.sigmoid(outputs["object_logits"]) > 0.5).float()
            obj_correct += (obj_preds == object_mask).float().sum().item()
            obj_total += object_mask.numel()
            obj_batch_tp, obj_batch_fp, obj_batch_fn, _ = binary_counts(
                outputs["object_logits"],
                object_mask,
                threshold=0.5,
            )
            obj_tp += obj_batch_tp
            obj_fp += obj_batch_fp
            obj_fn += obj_batch_fn

            for key, labels in [
                ("attr_color", attr_color),
                ("attr_material", attr_material),
                ("attr_shape", attr_shape),
                ("attr_size", attr_size),
            ]:
                c, t = attribute_accuracy(outputs[key], labels)
                attr_correct += c.item()
                attr_total += t.item()

            loc_error, loc_count = coordinate_error_sums(
                outputs["pixel_coords"],
                pixel_coords,
                object_mask,
                width_px=ClevrRelationDataset.ORIGINAL_IMAGE_WIDTH,
                height_px=ClevrRelationDataset.ORIGINAL_IMAGE_HEIGHT,
            )
            loc_error_sum += loc_error.item()
            loc_value_count += loc_count.item()

    # Somma gli accumulatori fra rank per ottenere metriche globali identiche.
    if distributed:
        totals = torch.tensor(
            [
                rel_correct,
                rel_total,
                rel_tp,
                rel_fp,
                rel_fn,
                obj_correct,
                obj_total,
                obj_tp,
                obj_fp,
                obj_fn,
                attr_correct,
                attr_total,
                loc_error_sum,
                loc_value_count,
            ],
            device=device,
            dtype=torch.float64,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        (
            rel_correct,
            rel_total,
            rel_tp,
            rel_fp,
            rel_fn,
            obj_correct,
            obj_total,
            obj_tp,
            obj_fp,
            obj_fn,
            attr_correct,
            attr_total,
            loc_error_sum,
            loc_value_count,
        ) = totals.tolist()

    rel_prec, rel_rec, rel_f1 = precision_recall_f1(rel_tp, rel_fp, rel_fn)
    obj_prec, obj_rec, obj_f1 = precision_recall_f1(obj_tp, obj_fp, obj_fn)
    return {
        "relation_accuracy": rel_correct / max(rel_total, 1.0),
        "relation_precision": rel_prec,
        "relation_recall": rel_rec,
        "relation_f1": rel_f1,
        "object_accuracy": obj_correct / max(obj_total, 1.0),
        "object_precision": obj_prec,
        "object_recall": obj_rec,
        "object_f1": obj_f1,
        "attribute_accuracy": attr_correct / max(attr_total, 1.0),
        "localization_mae_px": loc_error_sum / max(loc_value_count, 1.0),
    }


def parse_args():
    """Analizza gli argomenti della riga di comando."""
    parser = argparse.ArgumentParser(description="Train an image-to-scene-graph model on CLEVR")
    parser.add_argument("--img-dir", default=CONFIG.data.img_dir, help="Directory with CLEVR images")
    parser.add_argument(
        "--scene-json",
        default=CONFIG.data.train_scene_json or CONFIG.data.scene_json,
        help="CLEVR training scenes JSON file",
    )
    parser.add_argument(
        "--val-scene-json",
        default=CONFIG.data.val_scene_json,
        help="CLEVR validation scenes JSON file used for per-epoch evaluation",
    )
    parser.add_argument("--max-scenes", type=int, default=CONFIG.train.max_scenes, help="Maximum number of training scenes")
    parser.add_argument("--max-objects", type=int, default=CONFIG.data.max_objects, help="Maximum number of slots/objects")
    parser.add_argument("--batch-size", type=int, default=CONFIG.train.batch_size, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=CONFIG.train.epochs, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=CONFIG.train.lr, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=CONFIG.train.num_workers, help="DataLoader workers")
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        default=CONFIG.train.persistent_workers,
        help="Keep DataLoader workers alive across epochs",
    )
    parser.add_argument(
        "--no-persistent-workers",
        dest="persistent_workers",
        action="store_false",
        help="Disable DataLoader persistent workers",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=CONFIG.train.prefetch_factor,
        help="Batches prefetched by each DataLoader worker",
    )
    parser.add_argument(
        "--reasoning-layers",
        type=int,
        default=CONFIG.train.reasoning_layers,
        help="Number of graph message-passing layers",
    )
    parser.add_argument(
        "--reasoning-backend",
        type=str,
        default=CONFIG.train.reasoning_backend,
        choices=["dense", "pyg"],
        help="Graph reasoning backend: dense or pyg",
    )
    parser.add_argument(
        "--backbone-init",
        type=str,
        default=CONFIG.train.backbone_init,
        choices=["auto", "imagenet", "random"],
        help="How to initialize the ResNet34 backbone",
    )
    parser.add_argument("--model-path", default=CONFIG.train.model_path, help="Output model path")
    # torch.distributed.run potrebbe iniettare --local_rank; accettalo per compatibilità.
    parser.add_argument("--local_rank", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=CONFIG.train.use_amp,
        help="Enable mixed precision (AMP)",
    )
    parser.add_argument(
        "--no-use-amp",
        dest="use_amp",
        action="store_false",
        help="Disable mixed precision (AMP)",
    )
    parser.add_argument("--weight-decay", type=float, default=CONFIG.train.weight_decay, help="Weight decay for optimizer")
    parser.add_argument(
        "--coord-loss-weight",
        type=float,
        default=CONFIG.train.coord_loss_weight,
        help="Peso della loss di localizzazione dei centri immagine",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=CONFIG.train.grad_clip_norm,
        help="Norma massima del gradiente (0 per disabilitare il clipping)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default=CONFIG.train.scheduler,
        choices=["none", "cosine", "step"],
        help="LR scheduler to use",
    )
    parser.add_argument("--step-size", type=int, default=CONFIG.train.step_size, help="StepLR step size (if using step scheduler)")
    parser.add_argument("--gamma", type=float, default=CONFIG.train.gamma, help="StepLR gamma (if using step scheduler)")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=CONFIG.train.early_stop_patience,
        help="Numero di epoche senza miglioramenti su relation_f1 (validation) prima dello stop",
    )
    parser.add_argument(
        "--val-relation-threshold",
        type=float,
        default=CONFIG.train.val_relation_threshold,
        help="Soglia sigmoid per metriche relazionali in validation",
    )
    parser.add_argument(
        "--enable-slot-matching",
        dest="disable_slot_matching",
        action="store_false",
        help="Abilita il riallineamento permutation-aware slot-oggetto",
    )
    parser.add_argument(
        "--disable-slot-matching",
        dest="disable_slot_matching",
        action="store_true",
        help="Disabilita il riallineamento permutation-aware slot-oggetto (matching ottimo)",
    )
    parser.set_defaults(disable_slot_matching=(not CONFIG.train.slot_matching_enabled))
    return parser.parse_args()


def validate_args(args):
    """Valida gli argomenti di runtime e i percorsi richiesti."""
    if not args.img_dir or not args.scene_json:
        raise ValueError(
            "Provide --img-dir and --scene-json or set CLEVR_IMG_DIR with CLEVR_TRAIN_SCENE_JSON/CLEVR_SCENE_JSON."
        )
    if not os.path.isdir(args.img_dir):
        raise FileNotFoundError(f"Image directory not found: {args.img_dir}")
    if not os.path.isfile(args.scene_json):
        raise FileNotFoundError(f"Scene JSON not found: {args.scene_json}")
    if args.val_scene_json and not os.path.isfile(args.val_scene_json):
        raise FileNotFoundError(f"Validation Scene JSON not found: {args.val_scene_json}")
    if args.reasoning_layers < 0:
        raise ValueError("--reasoning-layers must be >= 0")
    if args.reasoning_backend not in {"dense", "pyg"}:
        raise ValueError("--reasoning-backend must be one of: dense, pyg")
    if args.backbone_init not in {"auto", "imagenet", "random"}:
        raise ValueError("--backbone-init must be one of: auto, imagenet, random")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be >= 0")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be >= 1")
    if args.coord_loss_weight < 0.0:
        raise ValueError("--coord-loss-weight must be >= 0")
    if args.grad_clip_norm < 0.0:
        raise ValueError("--grad-clip-norm must be >= 0")


def build_amp_components(enabled):
    """Costruisce GradScaler e autocast compatibili con diverse versioni di PyTorch."""
    if not enabled:
        return None, contextlib.nullcontext

    try:
        scaler = amp.GradScaler(device_type="cuda")
    except TypeError:
        try:
            scaler = amp.GradScaler()
        except Exception:
            from torch.cuda.amp import GradScaler

            scaler = GradScaler()

    def autocast_context():
        """Restituisce il contesto autocast compatibile con PyTorch installato."""
        try:
            return amp.autocast(device_type="cuda")
        except TypeError:
            try:
                return amp.autocast()
            except Exception:
                from torch.cuda.amp import autocast

                return autocast()

    return scaler, autocast_context


def main():
    """Esegue l'addestramento del modello."""
    args = parse_args()
    validate_args(args)

    # Rileva se l'esecuzione avviene sotto torch.distributed (torchrun) tramite WORLD_SIZE/LOCAL_RANK
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))

    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main_process = (not distributed) or (dist.get_rank() == 0)

    train_dataset = ClevrRelationDataset(
        args.img_dir,
        args.scene_json,
        max_objects=args.max_objects,
        max_scenes=args.max_scenes,
    )
    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = args.persistent_workers
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(not distributed),
        sampler=(DistributedSampler(train_dataset) if distributed else None),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        **loader_kwargs,
    )

    val_loader = None
    val_sampler = None
    if args.val_scene_json:
        val_loader_kwargs = {}
        if args.num_workers > 0:
            # Evita di mantenere vivi i worker di validation tra epoche:
            # con DDP questo riduce i rallentamenti/asimmetrie al passaggio epoch.
            val_loader_kwargs["persistent_workers"] = False
            val_loader_kwargs["prefetch_factor"] = args.prefetch_factor

        val_dataset = ClevrRelationDataset(
            args.img_dir,
            args.val_scene_json,
            max_objects=args.max_objects,
        )
        if distributed:
            val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=False)

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            **val_loader_kwargs,
        )

    model = VisualRelationModel(
        num_slots=args.max_objects,
        reasoning_layers=args.reasoning_layers,
        reasoning_backend=args.reasoning_backend,
        backbone_init=args.backbone_init,
    ).to(device)
    if distributed:
        model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
    else:
        model = model.to(device)

    if is_main_process:
        source_model = model.module if hasattr(model, "module") else model
        print(f"[INFO] Backbone init resolved to: {source_model.backbone.backbone_init}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Pianificatore
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    elif args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    else:
        scheduler = None

    bce_object = nn.BCEWithLogitsLoss()
    ce_attr = nn.CrossEntropyLoss(ignore_index=-100)

    amp_enabled = args.use_amp and device.type == "cuda"
    if args.use_amp and device.type != "cuda" and is_main_process:
        print("[WARN] AMP richiesto ma CUDA non disponibile: AMP disattivato.")

    scaler, autocast_context = build_amp_components(amp_enabled)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"runs/run_{timestamp}"
    model_save_path = os.path.join(run_dir, os.path.basename(args.model_path))
    best_model_save_path = os.path.join(run_dir, "best_" + os.path.basename(args.model_path))
    history_path = os.path.join(run_dir, "training_history.json")
    metrics_path = os.path.join(run_dir, "metrics_final.json")
    best_metrics_path = os.path.join(run_dir, "metrics_best.json")
    if is_main_process:
        os.makedirs(run_dir, exist_ok=True)

    warmup_epochs = 5
    warmup_steps = warmup_epochs * len(train_loader)
    current_step = 0
    
    # Lista per memorizzare la storia delle metriche per ogni epoca
    training_history = []
    best_val_rel_f1 = -1.0
    best_epoch = 0
    best_metrics = None
    epochs_without_improvement = 0
    stop_training = False
    stop_reason = None

    for epoch in range(args.epochs):
        model.train()
        if is_main_process:
            print(f"\nEpoch {epoch + 1}/{args.epochs}")

        epoch_loss = 0.0
        rel_correct = rel_total = 0.0
        rel_tp = rel_fp = rel_fn = 0.0
        obj_correct = obj_total = 0.0
        obj_tp = obj_fp = obj_fn = 0.0
        attr_correct = attr_total = 0.0
        loc_error_sum = loc_value_count = 0.0
        epoch_batches = 0.0
        last_loss = None
        last_rel_batch_correct = torch.tensor(0.0, device=device)
        last_rel_batch_total = torch.tensor(1.0, device=device)
        last_grad_norm = None
        epoch_aborted = False

        # Se distribuito, imposta l'epoca sul campionatore per il mescolamento
        if distributed:
            train_loader.sampler.set_epoch(epoch)

        progress = tqdm(train_loader, desc="Training", leave=False) if is_main_process else train_loader
        for batch_idx, (images, targets) in enumerate(progress):
            # Riscaldamento lineare
            if current_step < warmup_steps:
                lr_scale = min(1.0, float(current_step + 1) / warmup_steps)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_scale * args.lr
            current_step += 1

            images = images.to(device)

            object_mask = targets["object_mask"].to(device)
            attr_color = targets["attr_color"].to(device)
            attr_material = targets["attr_material"].to(device)
            attr_shape = targets["attr_shape"].to(device)
            attr_size = targets["attr_size"].to(device)
            pixel_coords = targets["pixel_coords"].to(device)
            relations = targets["relations"].to(device)

            optimizer.zero_grad()

            if amp_enabled:
                with autocast_context():
                    outputs = model(images)

                    bad_output_keys = find_nonfinite_output_keys(outputs)
                    if bad_output_keys:
                        stop_reason = (
                            f"Non-finite model outputs detected at epoch {epoch + 1}, batch {batch_idx + 1}: "
                            + ", ".join(bad_output_keys)
                        )
                        epoch_aborted = True
                        stop_training = True
                        break

                    # Allinea i target agli output
                    aligned = align_rel_targets(relations, outputs["relations"])
                    if aligned is None:
                        if is_main_process:
                            print(f"[ERROR] could not align relations shape {relations.shape} to output {outputs['relations'].shape}")
                        raise RuntimeError(f"Unexpected relations shape {relations.shape}; cannot align with model output {outputs['relations'].shape}")
                    relations = aligned

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

                    pair_mask = relation_pair_mask(object_mask)

                    loss_object = bce_object(outputs["object_logits"], object_mask)
                    loss_color = ce_attr(outputs["attr_color"].reshape(-1, 8), attr_color.reshape(-1))
                    loss_material = ce_attr(outputs["attr_material"].reshape(-1, 2), attr_material.reshape(-1))
                    loss_shape = ce_attr(outputs["attr_shape"].reshape(-1, 3), attr_shape.reshape(-1))
                    loss_size = ce_attr(outputs["attr_size"].reshape(-1, 2), attr_size.reshape(-1))
                    loss_coord = masked_coordinate_loss(outputs["pixel_coords"], pixel_coords, object_mask)

                    # Calcola pos_weight per canale di relazione basato su questo batch (mascherato)
                    rel_mask = pair_mask.unsqueeze(1).bool()
                    rel_targets = relations
                    positives = ((rel_targets > 0.5) & rel_mask).sum(dim=(0, 2, 3)).float()
                    total = rel_mask.sum(dim=(0, 2, 3)).float()
                    negatives = total - positives
                    pos_weight = torch.ones(positives.shape, device=device)
                    nonzero = positives > 0
                    pos_weight[nonzero] = (negatives[nonzero] / positives[nonzero]).clamp(min=1.0)

                    rel_loss_raw = F.binary_cross_entropy_with_logits(outputs["relations"], relations, reduction="none")
                    # applica la pesatura per canale (trasmessa a BxCxNxN)
                    channel_weight = pos_weight.view(1, -1, 1, 1)
                    rel_loss_raw = rel_loss_raw * channel_weight
                    loss_rel = (rel_loss_raw * rel_mask.float()).sum() / (rel_mask.float().sum() * outputs["relations"].size(1)).clamp_min(1.0)
                    loss = (
                        loss_rel
                        + loss_object
                        + loss_color
                        + loss_material
                        + loss_shape
                        + loss_size
                        + args.coord_loss_weight * loss_coord
                    )

                if not torch.isfinite(loss):
                    stop_reason = f"Non-finite loss detected at epoch {epoch + 1}, batch {batch_idx + 1}."
                    epoch_aborted = True
                    stop_training = True
                    optimizer.zero_grad(set_to_none=True)
                    break
            else:
                outputs = model(images)

                bad_output_keys = find_nonfinite_output_keys(outputs)
                if bad_output_keys:
                    stop_reason = (
                        f"Non-finite model outputs detected at epoch {epoch + 1}, batch {batch_idx + 1}: "
                        + ", ".join(bad_output_keys)
                    )
                    epoch_aborted = True
                    stop_training = True
                    break

                # Allinea i target agli output
                aligned = align_rel_targets(relations, outputs["relations"])
                if aligned is None:
                    if is_main_process:
                        print(f"[ERROR] could not align relations shape {relations.shape} to output {outputs['relations'].shape}")
                    raise RuntimeError(f"Unexpected relations shape {relations.shape}; cannot align with model output {outputs['relations'].shape}")
                relations = aligned

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

                pair_mask = relation_pair_mask(object_mask)

                loss_object = bce_object(outputs["object_logits"], object_mask)
                loss_color = ce_attr(outputs["attr_color"].reshape(-1, 8), attr_color.reshape(-1))
                loss_material = ce_attr(outputs["attr_material"].reshape(-1, 2), attr_material.reshape(-1))
                loss_shape = ce_attr(outputs["attr_shape"].reshape(-1, 3), attr_shape.reshape(-1))
                loss_size = ce_attr(outputs["attr_size"].reshape(-1, 2), attr_size.reshape(-1))
                loss_coord = masked_coordinate_loss(outputs["pixel_coords"], pixel_coords, object_mask)

                # Calcola pos_weight per canale di relazione basato su questo batch (mascherato)
                rel_mask = pair_mask.unsqueeze(1).bool()
                rel_targets = relations
                positives = ((rel_targets > 0.5) & rel_mask).sum(dim=(0, 2, 3)).float()
                total = rel_mask.sum(dim=(0, 2, 3)).float()
                negatives = total - positives
                pos_weight = torch.ones(positives.shape, device=device)
                nonzero = positives > 0
                pos_weight[nonzero] = (negatives[nonzero] / positives[nonzero]).clamp(min=1.0)

                rel_loss_raw = F.binary_cross_entropy_with_logits(outputs["relations"], relations, reduction="none")
                channel_weight = pos_weight.view(1, -1, 1, 1)
                rel_loss_raw = rel_loss_raw * channel_weight
                loss_rel = (rel_loss_raw * rel_mask.float()).sum() / (rel_mask.float().sum() * outputs["relations"].size(1)).clamp_min(1.0)
                loss = (
                    loss_rel
                    + loss_object
                    + loss_color
                    + loss_material
                    + loss_shape
                    + loss_size
                    + args.coord_loss_weight * loss_coord
                )

                if not torch.isfinite(loss):
                    stop_reason = f"Non-finite loss detected at epoch {epoch + 1}, batch {batch_idx + 1}."
                    epoch_aborted = True
                    stop_training = True
                    optimizer.zero_grad(set_to_none=True)
                    break

            if amp_enabled:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                else:
                    grad_norm = torch.tensor(0.0, device=device)
                if not math.isfinite(float(grad_norm)):
                    stop_reason = f"Non-finite gradient norm detected at epoch {epoch + 1}, batch {batch_idx + 1}."
                    epoch_aborted = True
                    stop_training = True
                    optimizer.zero_grad(set_to_none=True)
                    break
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip_norm > 0.0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
                else:
                    grad_norm = torch.tensor(0.0, device=device)
                if not math.isfinite(float(grad_norm)):
                    stop_reason = f"Non-finite gradient norm detected at epoch {epoch + 1}, batch {batch_idx + 1}."
                    epoch_aborted = True
                    stop_training = True
                    optimizer.zero_grad(set_to_none=True)
                    break
                optimizer.step()

            epoch_loss += loss.item()
            epoch_batches += 1.0
            last_loss = loss
            last_grad_norm = float(grad_norm)

            rel_batch_correct, rel_batch_total = masked_relation_accuracy(outputs["relations"], relations, pair_mask)
            rel_correct += rel_batch_correct.item()
            rel_total += rel_batch_total.item()
            last_rel_batch_correct = rel_batch_correct
            last_rel_batch_total = rel_batch_total

            rel_mask = pair_mask.unsqueeze(1).expand_as(outputs["relations"])
            rel_batch_tp, rel_batch_fp, rel_batch_fn, _ = binary_counts(
                outputs["relations"],
                relations,
                threshold=0.5,
                mask=rel_mask,
            )
            rel_tp += rel_batch_tp
            rel_fp += rel_batch_fp
            rel_fn += rel_batch_fn

            obj_preds = (torch.sigmoid(outputs["object_logits"]) > 0.5).float()
            obj_correct += (obj_preds == object_mask).float().sum().item()
            obj_total += object_mask.numel()
            obj_batch_tp, obj_batch_fp, obj_batch_fn, _ = binary_counts(
                outputs["object_logits"],
                object_mask,
                threshold=0.5,
            )
            obj_tp += obj_batch_tp
            obj_fp += obj_batch_fp
            obj_fn += obj_batch_fn

            for key, labels in [
                ("attr_color", attr_color),
                ("attr_material", attr_material),
                ("attr_shape", attr_shape),
                ("attr_size", attr_size),
            ]:
                c, t = attribute_accuracy(outputs[key], labels)
                attr_correct += c.item()
                attr_total += t.item()

            loc_error, loc_count = coordinate_error_sums(
                outputs["pixel_coords"].detach(),
                pixel_coords,
                object_mask,
                width_px=ClevrRelationDataset.ORIGINAL_IMAGE_WIDTH,
                height_px=ClevrRelationDataset.ORIGINAL_IMAGE_HEIGHT,
            )
            loc_error_sum += loc_error.item()
            loc_value_count += loc_count.item()

            # registra la norma del gradiente ogni 100 passi
            if (batch_idx % 100 == 0) and is_main_process:
                grad_log = f"Grad norm: {last_grad_norm:.4f}"
            else:
                grad_log = None

        if epoch_aborted:
            if is_main_process and stop_reason:
                print(f"[WARN] {stop_reason}")
                if restore_checkpoint_into_model(model, best_model_save_path, device):
                    print(f"[WARN] Restored best checkpoint from {best_model_save_path} before stopping.")
                else:
                    print("[WARN] No best checkpoint available to restore; keeping current model weights.")
            if distributed:
                stop_tensor = torch.tensor(1 if stop_training else 0, device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
            break

        if is_main_process:
            postfix = {
                "loss": f"{(last_loss.item() if last_loss is not None else 0.0):.4f}",
                "rel_acc": f"{(last_rel_batch_correct / last_rel_batch_total.clamp_min(1.0)).item():.4f}",
            }
            if grad_log is not None:
                postfix["grad_norm"] = grad_log
            progress.set_postfix(**postfix)

        if distributed:
            train_totals = torch.tensor(
                [
                    epoch_loss,
                    epoch_batches,
                    rel_correct,
                    rel_total,
                    rel_tp,
                    rel_fp,
                    rel_fn,
                    obj_correct,
                    obj_total,
                    obj_tp,
                    obj_fp,
                    obj_fn,
                    attr_correct,
                    attr_total,
                    loc_error_sum,
                    loc_value_count,
                ],
                device=device,
                dtype=torch.float64,
            )
            dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            (
                epoch_loss,
                epoch_batches,
                rel_correct,
                rel_total,
                rel_tp,
                rel_fp,
                rel_fn,
                obj_correct,
                obj_total,
                obj_tp,
                obj_fp,
                obj_fn,
                attr_correct,
                attr_total,
                loc_error_sum,
                loc_value_count,
            ) = train_totals.tolist()

        avg_loss = epoch_loss / max(epoch_batches, 1.0)
        rel_acc = rel_correct / max(rel_total, 1.0)
        rel_prec, rel_rec, rel_f1 = precision_recall_f1(rel_tp, rel_fp, rel_fn)
        obj_acc = obj_correct / max(obj_total, 1.0)
        obj_prec, obj_rec, obj_f1 = precision_recall_f1(obj_tp, obj_fp, obj_fn)
        att_acc = attr_correct / max(attr_total, 1.0)
        loc_mae_px = loc_error_sum / max(loc_value_count, 1.0)
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate_model(
                model,
                val_loader,
                device,
                relation_threshold=args.val_relation_threshold,
                use_slot_matching=(not args.disable_slot_matching),
                distributed=distributed,
            )

            if is_main_process:
                print(
                    f"Val  Relation Acc/P/R/F1: {val_metrics['relation_accuracy']:.4f}/{val_metrics['relation_precision']:.4f}/"
                    f"{val_metrics['relation_recall']:.4f}/{val_metrics['relation_f1']:.4f} | "
                    f"Val Object F1: {val_metrics['object_f1']:.4f} | Val Attr Acc: {val_metrics['attribute_accuracy']:.4f} | "
                    f"Val Loc MAE(px): {val_metrics['localization_mae_px']:.2f}"
                )

                if val_metrics["relation_f1"] > best_val_rel_f1:
                    best_val_rel_f1 = val_metrics["relation_f1"]
                    best_epoch = epoch + 1
                    epochs_without_improvement = 0
                    source_model = model.module if hasattr(model, "module") else model
                    torch.save(source_model.state_dict(), best_model_save_path)
                    best_metrics = {
                        "epoch": best_epoch,
                        "train_loss": avg_loss,
                        "train_relation_f1": rel_f1,
                        "slot_matching_enabled": (not args.disable_slot_matching),
                        "slot_matching_method": matching_method_name(not args.disable_slot_matching),
                        "val_relation_accuracy": val_metrics["relation_accuracy"],
                        "val_relation_precision": val_metrics["relation_precision"],
                        "val_relation_recall": val_metrics["relation_recall"],
                        "val_relation_f1": val_metrics["relation_f1"],
                        "val_object_f1": val_metrics["object_f1"],
                        "val_attribute_accuracy": val_metrics["attribute_accuracy"],
                        "val_localization_mae_px": val_metrics["localization_mae_px"],
                    }
                    print(f"[BEST] Nuovo best checkpoint salvato (epoch {best_epoch}) in {best_model_save_path}")
                else:
                    epochs_without_improvement += 1

                if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                    stop_training = True
                    print(
                        f"[EARLY STOP] Nessun miglioramento di relation_f1 in validation da {epochs_without_improvement} epoche."
                    )

            if distributed:
                stop_tensor = torch.tensor(1 if (stop_training and is_main_process) else 0, device=device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())

        if is_main_process:
            print(
                f"Avg Loss: {avg_loss:.4f} | "
                f"Relation Acc/P/R/F1: {rel_acc:.4f}/{rel_prec:.4f}/{rel_rec:.4f}/{rel_f1:.4f} | "
                f"Object Acc/P/R/F1: {obj_acc:.4f}/{obj_prec:.4f}/{obj_rec:.4f}/{obj_f1:.4f} | "
                f"Attr Acc: {att_acc:.4f} | Loc MAE(px): {loc_mae_px:.2f}"
            )
            
            # Salva le metriche dell'epoca corrente nella storia
            epoch_metrics = {
                "epoch": epoch + 1,
                "loss": avg_loss,
                "relation_accuracy": rel_acc,
                "relation_precision": rel_prec,
                "relation_recall": rel_rec,
                "relation_f1": rel_f1,
                "object_accuracy": obj_acc,
                "object_precision": obj_prec,
                "object_recall": obj_rec,
                "object_f1": obj_f1,
                "attribute_accuracy": att_acc,
                "localization_mae_px": loc_mae_px,
            }
            if val_metrics is not None:
                epoch_metrics.update({
                    "val_relation_accuracy": val_metrics["relation_accuracy"],
                    "val_relation_precision": val_metrics["relation_precision"],
                    "val_relation_recall": val_metrics["relation_recall"],
                    "val_relation_f1": val_metrics["relation_f1"],
                    "val_object_accuracy": val_metrics["object_accuracy"],
                    "val_object_precision": val_metrics["object_precision"],
                    "val_object_recall": val_metrics["object_recall"],
                    "val_object_f1": val_metrics["object_f1"],
                    "val_attribute_accuracy": val_metrics["attribute_accuracy"],
                    "val_localization_mae_px": val_metrics["localization_mae_px"],
                })
            training_history.append(epoch_metrics)

        if scheduler is not None:
            scheduler.step()

        if distributed:
            # Sincronizza tutti i rank prima di iniziare l'epoca successiva.
            dist.barrier()

        if stop_training:
            break

    if is_main_process:
        source_model = model.module if hasattr(model, "module") else model
        torch.save(source_model.state_dict(), model_save_path)
        print(f"Modello salvato in {model_save_path}")
        
        # Salva la storia completa del training (tutte le epoche)
        history_path = os.path.join(run_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(training_history, f, indent=4)
        print(f"Storia del training salvata in {history_path}")

        latest_epoch_metrics = training_history[-1] if training_history else None
        if latest_epoch_metrics is not None:
            final_metrics = {
                "epoch": latest_epoch_metrics["epoch"],
                "loss": latest_epoch_metrics["loss"],
                "relation_accuracy": latest_epoch_metrics["relation_accuracy"],
                "relation_precision": latest_epoch_metrics["relation_precision"],
                "relation_recall": latest_epoch_metrics["relation_recall"],
                "relation_f1": latest_epoch_metrics["relation_f1"],
                "object_accuracy": latest_epoch_metrics["object_accuracy"],
                "object_precision": latest_epoch_metrics["object_precision"],
                "object_recall": latest_epoch_metrics["object_recall"],
                "object_f1": latest_epoch_metrics["object_f1"],
                "attribute_accuracy": latest_epoch_metrics["attribute_accuracy"],
                "localization_mae_px": latest_epoch_metrics["localization_mae_px"],
                "slot_matching_enabled": (not args.disable_slot_matching),
                "slot_matching_method": matching_method_name(not args.disable_slot_matching),
                "best_epoch": best_epoch,
                "best_val_relation_f1": best_val_rel_f1,
                "backbone_init_resolved": source_model.backbone.backbone_init,
                "stopped_early": bool(stop_training),
                "stop_reason": stop_reason,
            }
        else:
            final_metrics = {
                "loss": None,
                "relation_accuracy": None,
                "relation_precision": None,
                "relation_recall": None,
                "relation_f1": None,
                "object_accuracy": None,
                "object_precision": None,
                "object_recall": None,
                "object_f1": None,
                "attribute_accuracy": None,
                "localization_mae_px": None,
                "slot_matching_enabled": (not args.disable_slot_matching),
                "slot_matching_method": matching_method_name(not args.disable_slot_matching),
                "best_epoch": best_epoch,
                "best_val_relation_f1": best_val_rel_f1,
                "backbone_init_resolved": source_model.backbone.backbone_init,
                "stopped_early": bool(stop_training),
                "stop_reason": stop_reason,
            }
        metrics_path = os.path.join(run_dir, "metrics_final.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(final_metrics, f, indent=4)
        print(f"Metriche finali salvate in {metrics_path}")

        if best_metrics is not None:
            with open(best_metrics_path, "w", encoding="utf-8") as f:
                json.dump(best_metrics, f, indent=4)
            print(f"Metriche best validation salvate in {best_metrics_path}")

        # Genera automaticamente i grafici di training dalla history appena salvata.
        plot_script_path = os.path.join(os.path.dirname(__file__), "plot_training.py")
        plots_output_dir = os.path.join(run_dir, "plots")
        if os.path.isfile(plot_script_path):
            try:
                subprocess.run(
                    [
                        sys.executable,
                        plot_script_path,
                        "--history-json",
                        history_path,
                        "--output-dir",
                        plots_output_dir,
                    ],
                    check=True,
                )
                print(f"Grafici di training salvati in {plots_output_dir}")
            except subprocess.CalledProcessError as exc:
                print(f"[WARN] Impossibile generare i grafici di training (exit code {exc.returncode}).")
        else:
            print("[WARN] plot_training.py non trovato: grafici non generati automaticamente.")

    if distributed:
        dist.barrier()
        # distrugge in modo pulito il gruppo di processi per evitare avvisi NCCL all'uscita
        try:
            dist.destroy_process_group()
        except Exception:
            pass

    if stop_reason and best_epoch == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
