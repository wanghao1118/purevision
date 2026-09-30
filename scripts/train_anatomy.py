from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from train_anatomy_base import balanced_patch_indices, geometry_loss

from purevision.anatomy import ANATOMY_NAMES, regular_simplex_targets
from purevision.anatomy_dataset import AnatomyPatchDataset, collate_anatomy_batch
from purevision.anatomy_lesion import (
    balanced_lung_status_indices,
    cosine_distillation_loss,
    focused_anchor_supervised_contrastive_loss,
    hierarchical_group_labels_from_targets,
    hierarchical_lung_targets,
    hierarchical_status_centroid_loss,
    lesion_parent_patch_labels,
    memory_teacher_relative_status_gap_loss,
    sampled_nonlesion_indices,
    teacher_relative_anchor_hard_negative_loss,
    teacher_relative_parent_containment_loss,
    teacher_relative_parent_pair_separation_loss,
    teacher_relative_status_gap_loss,
)
from purevision.anatomy_lesion_dataset import (
    AnatomyLesionPatchDataset,
    collate_anatomy_lesion_batch,
)
from purevision.anatomy_model import (
    build_anatomy_patch_encoder,
    load_anatomy_patch_encoder,
    set_trainable_anatomy_layers,
)
from purevision.losses import (
    GeometryMemoryBank,
    dynamic_geometry_losses,
    supervised_contrastive_loss,
)
from purevision.protocol import validate_dataset_protocol, write_dataset_companion_zh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Incrementally train R33 with within-lung lesion hierarchy"
    )
    parser.add_argument("--config", required=True, type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--initialize-from", type=Path)
    group.add_argument("--resume", type=Path)
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def seed_everything(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cosine_schedule(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def next_cycled(
    iterator: Iterator[dict[str, Any]], loader: DataLoader
) -> tuple[dict[str, Any], Iterator[dict[str, Any]]]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def hierarchy_loss(
    normalized_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    side_masks: torch.Tensor,
    lesion_masks: torch.Tensor,
    lung_classes: torch.Tensor,
    parent_prototypes: torch.Tensor,
    memory: GeometryMemoryBank,
    teacher_memory: GeometryMemoryBank,
    training: dict[str, Any],
    *,
    generator: torch.Generator | None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:
    indices, parents, statuses = balanced_lung_status_indices(
        side_masks,
        lesion_masks,
        lung_classes,
        per_status=int(training["hierarchy_patches_per_status_per_slice"]),
        generator=generator,
    )
    embeddings = normalized_tokens.reshape(-1, normalized_tokens.shape[-1])[indices]
    teacher_embeddings = teacher_tokens.reshape(-1, teacher_tokens.shape[-1])[indices]
    targets = hierarchical_lung_targets(
        parent_prototypes,
        parents,
        statuses,
        status_offset=float(training["hierarchy_status_offset"]),
    )
    memory_embeddings, memory_targets = memory.tensors()
    triplet, pairwise, triplets = dynamic_geometry_losses(
        embeddings,
        targets,
        memory_embeddings=memory_embeddings,
        memory_targets=memory_targets,
        triplet_margin=float(training["hierarchy_triplet_margin"]),
        ideal_gap_min=float(training["hierarchy_ideal_gap_min"]),
        triplets_per_anchor=int(training["hierarchy_triplets_per_anchor"]),
        generator=generator,
    )
    centroid, _, _ = hierarchical_status_centroid_loss(
        embeddings,
        targets,
        parents,
        statuses,
        memory_embeddings=memory_embeddings,
        memory_targets=memory_targets,
        minimum_cosine_distance=float(
            training.get("hierarchy_minimum_status_cosine_distance", 0.45)
        ),
        compact_weight=float(training.get("hierarchy_centroid_compact_weight", 0.1)),
    )
    if bool(training.get("hierarchy_use_global_teacher_memory", False)):
        memory_teacher_embeddings, memory_teacher_targets = teacher_memory.tensors()
        teacher_gap, _, _ = memory_teacher_relative_status_gap_loss(
            embeddings,
            teacher_embeddings,
            targets,
            parents,
            statuses,
            memory_teacher_embeddings=memory_teacher_embeddings,
            memory_targets=memory_teacher_targets,
            improvement_margin=float(training.get("hierarchy_teacher_gap_margin", 0.0)),
        )
    else:
        teacher_gap, _, _ = teacher_relative_status_gap_loss(
            embeddings,
            teacher_embeddings,
            parents,
            statuses,
            improvement_margin=float(training.get("hierarchy_teacher_gap_margin", 0.0)),
        )
    groups = (parents - 1) * 2 + statuses
    memory_groups = (
        hierarchical_group_labels_from_targets(memory_targets)
        if memory_targets is not None
        else None
    )
    contrastive = supervised_contrastive_loss(
        embeddings,
        groups,
        memory_embeddings=memory_embeddings,
        memory_labels=memory_groups,
        temperature=float(training.get("hierarchy_contrastive_temperature", 0.1)),
    )
    parent_containment, _, _ = teacher_relative_parent_containment_loss(
        embeddings,
        teacher_embeddings,
        parents,
        maximum_similarity_drop=float(
            training.get("hierarchy_parent_similarity_maximum_drop", 0.0)
        ),
    )
    total = (
        float(training["hierarchy_triplet_loss_weight"]) * triplet
        + float(training["hierarchy_pairwise_loss_weight"]) * pairwise
        + float(training.get("hierarchy_centroid_loss_weight", 0.0)) * centroid
        + float(training.get("hierarchy_teacher_gap_loss_weight", 0.0)) * teacher_gap
        + float(training.get("hierarchy_contrastive_loss_weight", 0.0)) * contrastive
        + float(training.get("hierarchy_parent_containment_loss_weight", 0.0))
        * parent_containment
    )
    memory.add(embeddings, targets)
    teacher_memory.add(teacher_embeddings, targets)
    return (
        total,
        triplet,
        pairwise,
        centroid,
        teacher_gap,
        contrastive,
        parent_containment,
        triplets,
        int(indices.numel()),
    )


def reduce_sums(values: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def make_loaders(
    config: dict[str, Any], rank: int, world_size: int
) -> tuple[
    dict[str, DataLoader],
    dict[str, DistributedSampler | WeightedRandomSampler | None],
]:
    data = config["data"]
    training = config["training"]
    loaders: dict[str, DataLoader] = {}
    samplers: dict[str, DistributedSampler | WeightedRandomSampler | None] = {}
    for split in ("train", "val"):
        lesion_dataset = AnatomyLesionPatchDataset(
            config["paths"]["lesion_processed"],
            split,
            image_size=int(data["image_size"]),
            patch_size=int(data["patch_size"]),
            manifest_name=str(data.get("lesion_manifest_name", "metadata.csv")),
        )
        anatomy_dataset = AnatomyPatchDataset(
            config["paths"]["anatomy_processed"],
            split,
            image_size=int(data["image_size"]),
            patch_size=int(data["patch_size"]),
            manifest_name=str(data.get("anatomy_manifest_name", "metadata.csv")),
        )
        shuffle = split == "train"
        if world_size > 1:
            lesion_sampler = DistributedSampler(
                lesion_dataset, world_size, rank, shuffle=shuffle
            )
        elif shuffle and bool(training.get("balance_lung_sides", False)):
            side_counts = lesion_dataset.frame["side"].value_counts()
            weights = lesion_dataset.frame["side"].map(
                {side: 1.0 / count for side, count in side_counts.items()}
            )
            lesion_sampler = WeightedRandomSampler(
                torch.as_tensor(weights.to_numpy(), dtype=torch.double),
                num_samples=len(lesion_dataset),
                replacement=True,
            )
        else:
            lesion_sampler = None
        anatomy_sampler = (
            DistributedSampler(anatomy_dataset, world_size, rank, shuffle=shuffle)
            if world_size > 1
            else None
        )
        samplers[f"lesion_{split}"] = lesion_sampler
        samplers[f"anatomy_{split}"] = anatomy_sampler
        loaders[f"lesion_{split}"] = DataLoader(
            lesion_dataset,
            batch_size=int(training["lesion_per_device_batch_size"]),
            sampler=lesion_sampler,
            shuffle=shuffle and lesion_sampler is None,
            drop_last=split == "train",
            num_workers=int(training["num_workers"]),
            pin_memory=True,
            collate_fn=collate_anatomy_lesion_batch,
        )
        loaders[f"anatomy_{split}"] = DataLoader(
            anatomy_dataset,
            batch_size=int(training["anatomy_per_device_batch_size"]),
            sampler=anatomy_sampler,
            shuffle=shuffle and anatomy_sampler is None,
            drop_last=split == "train",
            num_workers=int(training["num_workers"]),
            pin_memory=True,
            collate_fn=collate_anatomy_batch,
        )
    return loaders, samplers


def batch_losses(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    lesion_batch: dict[str, Any],
    anatomy_batch: dict[str, Any],
    parent_prototypes: torch.Tensor,
    hierarchy_memory: GeometryMemoryBank,
    hierarchy_teacher_memory: GeometryMemoryBank,
    anatomy_memory: GeometryMemoryBank,
    parent_memory: GeometryMemoryBank,
    vessel_lung_memory: GeometryMemoryBank,
    vessel_lung_teacher_memory: GeometryMemoryBank,
    lesion_vessel_memory: GeometryMemoryBank,
    lesion_vessel_teacher_memory: GeometryMemoryBank,
    training: dict[str, Any],
    device: torch.device,
    *,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, dict[str, float]]:
    lesion_pixels = lesion_batch["pixel_values"].to(device, non_blocking=True)
    anatomy_pixels = anatomy_batch["pixel_values"].to(device, non_blocking=True)
    side_masks = lesion_batch["side_patch_mask"].to(device, non_blocking=True)
    lesion_masks = lesion_batch["lesion_patch_mask"].to(device, non_blocking=True)
    lung_classes = lesion_batch["lung_class"].to(device, non_blocking=True)
    anatomy_labels = anatomy_batch["patch_labels"].to(device, non_blocking=True)

    combined = torch.cat((lesion_pixels, anatomy_pixels), dim=0)
    outputs = model(combined)["normalized_embedding"]
    lesion_count = lesion_pixels.shape[0]
    lesion_tokens = outputs[:lesion_count]
    anatomy_tokens = outputs[lesion_count:]
    vessel_lung_weight = float(training.get("vessel_lung_loss_weight", 0.0))
    lesion_vessel_weight = float(training.get("lesion_vessel_hard_loss_weight", 0.0))
    with torch.no_grad():
        if vessel_lung_weight or lesion_vessel_weight:
            teacher_outputs = teacher(combined)["normalized_embedding"]
            teacher_tokens = teacher_outputs[:lesion_count]
            teacher_anatomy_tokens = teacher_outputs[lesion_count:]
        else:
            teacher_tokens = teacher(lesion_pixels)["normalized_embedding"]
            teacher_anatomy_tokens = None

    (
        hierarchy,
        hierarchy_triplet,
        hierarchy_pairwise,
        hierarchy_centroid,
        hierarchy_teacher_gap,
        hierarchy_contrastive,
        hierarchy_parent_containment,
        triplets,
        hierarchy_selected,
    ) = hierarchy_loss(
        lesion_tokens,
        teacher_tokens,
        side_masks,
        lesion_masks,
        lung_classes,
        parent_prototypes,
        hierarchy_memory,
        hierarchy_teacher_memory,
        training,
        generator=generator,
    )
    distill_indices = sampled_nonlesion_indices(
        lesion_masks,
        per_slice=int(training["distillation_patches_per_slice"]),
        generator=generator,
    )
    distillation = cosine_distillation_loss(
        lesion_tokens, teacher_tokens, distill_indices
    )
    (
        anatomy,
        anatomy_triplet,
        anatomy_pairwise,
        _,
        anatomy_triplets,
        anatomy_selected,
    ) = geometry_loss(
        anatomy_tokens,
        anatomy_labels,
        parent_prototypes,
        anatomy_memory,
        training,
        generator=generator,
    )
    parent_weight = float(training.get("parent_replay_loss_weight", 0.0))
    if parent_weight or vessel_lung_weight:
        parent_tokens = torch.cat((lesion_tokens, anatomy_tokens), dim=0)
        parent_labels = torch.cat(
            (lesion_parent_patch_labels(side_masks, lung_classes), anatomy_labels),
            dim=0,
        )
    if parent_weight:
        (
            parent_replay,
            _,
            _,
            _,
            parent_triplets,
            parent_selected,
        ) = geometry_loss(
            parent_tokens,
            parent_labels,
            parent_prototypes,
            parent_memory,
            training,
            generator=generator,
        )
    else:
        parent_replay = anatomy.sum() * 0.0
        parent_triplets = 0
        parent_selected = 0
    if vessel_lung_weight:
        assert teacher_anatomy_tokens is not None
        teacher_parent_tokens = torch.cat(
            (teacher_tokens, teacher_anatomy_tokens), dim=0
        )
        vessel_lung_indices = balanced_patch_indices(
            parent_labels,
            classes=4,
            per_class=int(training.get("vessel_lung_patches_per_class_per_slice", 32)),
            generator=generator,
        )
        flat_parent_labels = parent_labels.reshape(-1)
        vessel_lung_labels = flat_parent_labels[vessel_lung_indices]
        targeted = vessel_lung_labels.gt(0)
        vessel_lung_indices = vessel_lung_indices[targeted]
        vessel_lung_labels = vessel_lung_labels[targeted]
        flat_parent_tokens = parent_tokens.reshape(-1, parent_tokens.shape[-1])
        flat_teacher_parent_tokens = teacher_parent_tokens.reshape(
            -1, teacher_parent_tokens.shape[-1]
        )
        vessel_lung_embeddings = flat_parent_tokens[vessel_lung_indices]
        vessel_lung_teacher_embeddings = flat_teacher_parent_tokens[vessel_lung_indices]
        memory_embeddings, memory_targets = vessel_lung_memory.tensors()
        memory_teacher_embeddings, _ = vessel_lung_teacher_memory.tensors()
        memory_labels = None if memory_targets is None else memory_targets.argmax(dim=1)
        vessel_lung_pair, _, _, vessel_lung_comparisons = (
            teacher_relative_parent_pair_separation_loss(
                vessel_lung_embeddings,
                vessel_lung_teacher_embeddings,
                vessel_lung_labels,
                memory_student_embeddings=memory_embeddings,
                memory_teacher_embeddings=memory_teacher_embeddings,
                memory_labels=memory_labels,
                class_pairs=((1, 3), (2, 3)),
                improvement_margin=float(
                    training.get("vessel_lung_improvement_margin", 0.0)
                ),
            )
        )
        vessel_contrastive = focused_anchor_supervised_contrastive_loss(
            vessel_lung_embeddings,
            vessel_lung_labels,
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
            anchor_class=3,
            negative_classes=(1, 2),
            temperature=float(
                training.get("vessel_anchor_contrastive_temperature", 0.1)
            ),
        )
        vessel_lung = (
            float(training.get("vessel_parent_pair_weight", 1.0)) * vessel_lung_pair
            + float(training.get("vessel_anchor_contrastive_weight", 0.0))
            * vessel_contrastive
        )
        vessel_lung_targets = parent_prototypes[vessel_lung_labels]
        vessel_lung_memory.add(vessel_lung_embeddings, vessel_lung_targets)
        vessel_lung_teacher_memory.add(
            vessel_lung_teacher_embeddings, vessel_lung_targets
        )
    else:
        vessel_lung = anatomy.sum() * 0.0
        vessel_lung_comparisons = 0
    if lesion_vessel_weight:
        assert teacher_anatomy_tokens is not None
        target_per_class = int(
            training.get("lesion_vessel_patches_per_class_per_slice", 32)
        )
        lesion_indices, lesion_parents, lesion_statuses = balanced_lung_status_indices(
            side_masks,
            lesion_masks,
            lung_classes,
            per_status=target_per_class,
            generator=generator,
        )
        lesion_keep = lesion_statuses.eq(1)
        lesion_indices = lesion_indices[lesion_keep]
        lesion_parents = lesion_parents[lesion_keep]
        vessel_indices = balanced_patch_indices(
            anatomy_labels,
            classes=4,
            per_class=target_per_class,
            generator=generator,
        )
        flat_anatomy_labels = anatomy_labels.reshape(-1)
        vessel_indices = vessel_indices[flat_anatomy_labels[vessel_indices].eq(3)]
        flat_lesion_tokens = lesion_tokens.reshape(-1, lesion_tokens.shape[-1])
        flat_teacher_lesion_tokens = teacher_tokens.reshape(
            -1, teacher_tokens.shape[-1]
        )
        flat_anatomy_tokens = anatomy_tokens.reshape(-1, anatomy_tokens.shape[-1])
        flat_teacher_anatomy_tokens = teacher_anatomy_tokens.reshape(
            -1, teacher_anatomy_tokens.shape[-1]
        )
        lesion_vessel_embeddings = torch.cat(
            (flat_lesion_tokens[lesion_indices], flat_anatomy_tokens[vessel_indices]),
            dim=0,
        )
        lesion_vessel_teacher_embeddings = torch.cat(
            (
                flat_teacher_lesion_tokens[lesion_indices],
                flat_teacher_anatomy_tokens[vessel_indices],
            ),
            dim=0,
        )
        lesion_vessel_labels = torch.cat(
            (lesion_parents, torch.full_like(vessel_indices, 3)), dim=0
        )
        lesion_vessel_memory_embeddings, lesion_vessel_memory_targets = (
            lesion_vessel_memory.tensors()
        )
        lesion_vessel_teacher_memory_embeddings, _ = (
            lesion_vessel_teacher_memory.tensors()
        )
        lesion_vessel_memory_labels = (
            None
            if lesion_vessel_memory_targets is None
            else lesion_vessel_memory_targets.argmax(dim=1)
        )
        vessel_anchor_hard, _, _, lesion_vessel_comparisons = (
            teacher_relative_anchor_hard_negative_loss(
                lesion_vessel_embeddings,
                lesion_vessel_teacher_embeddings,
                lesion_vessel_labels,
                memory_student_embeddings=lesion_vessel_memory_embeddings,
                memory_teacher_embeddings=lesion_vessel_teacher_memory_embeddings,
                memory_labels=lesion_vessel_memory_labels,
                anchor_class=3,
                negative_classes=(1, 2),
                improvement_margin=float(
                    training.get("lesion_vessel_improvement_margin", 0.0)
                ),
                hard_negatives=int(training.get("lesion_vessel_hard_negatives", 16)),
            )
        )
        shared_lesion_anchor_weight = float(
            training.get("lesion_vessel_lesion_anchor_weight", 0.0)
        )
        lesion_anchor_weights = {
            1: float(
                training.get(
                    "lesion_vessel_left_lesion_anchor_weight",
                    shared_lesion_anchor_weight,
                )
            ),
            2: float(
                training.get(
                    "lesion_vessel_right_lesion_anchor_weight",
                    shared_lesion_anchor_weight,
                )
            ),
        }
        if any(lesion_anchor_weights.values()):
            lesion_anchor_hard = vessel_anchor_hard.sum() * 0.0
            for anchor_class, anchor_weight in lesion_anchor_weights.items():
                if not anchor_weight:
                    continue
                term, _, _, comparisons = teacher_relative_anchor_hard_negative_loss(
                    lesion_vessel_embeddings,
                    lesion_vessel_teacher_embeddings,
                    lesion_vessel_labels,
                    memory_student_embeddings=lesion_vessel_memory_embeddings,
                    memory_teacher_embeddings=lesion_vessel_teacher_memory_embeddings,
                    memory_labels=lesion_vessel_memory_labels,
                    anchor_class=anchor_class,
                    negative_classes=(3,),
                    improvement_margin=float(
                        training.get("lesion_vessel_improvement_margin", 0.0)
                    ),
                    hard_negatives=int(
                        training.get("lesion_vessel_hard_negatives", 16)
                    ),
                )
                lesion_anchor_hard = (
                    lesion_anchor_hard + 0.5 * float(anchor_weight) * term
                )
                lesion_vessel_comparisons += comparisons
        else:
            lesion_anchor_hard = vessel_anchor_hard.sum() * 0.0
        vessel_anchor_contrastive = focused_anchor_supervised_contrastive_loss(
            lesion_vessel_embeddings,
            lesion_vessel_labels,
            memory_embeddings=lesion_vessel_memory_embeddings,
            memory_labels=lesion_vessel_memory_labels,
            anchor_class=3,
            negative_classes=(1, 2),
            temperature=float(
                training.get("lesion_vessel_anchor_contrastive_temperature", 0.1)
            ),
        )
        lesion_vessel = (
            float(training.get("lesion_vessel_hard_component_weight", 1.0))
            * vessel_anchor_hard
            + float(training.get("lesion_vessel_anchor_contrastive_weight", 0.0))
            * vessel_anchor_contrastive
            + lesion_anchor_hard
        )
        lesion_vessel_targets = parent_prototypes[lesion_vessel_labels]
        lesion_vessel_memory.add(lesion_vessel_embeddings, lesion_vessel_targets)
        lesion_vessel_teacher_memory.add(
            lesion_vessel_teacher_embeddings, lesion_vessel_targets
        )
    else:
        lesion_vessel = anatomy.sum() * 0.0
        lesion_vessel_comparisons = 0
    total = (
        float(training["anatomy_replay_loss_weight"]) * anatomy
        + float(training["distillation_loss_weight"]) * distillation
        + float(training["hierarchy_loss_weight"]) * hierarchy
        + parent_weight * parent_replay
        + vessel_lung_weight * vessel_lung
        + lesion_vessel_weight * lesion_vessel
    )
    return total, {
        "loss": float(total.detach()),
        "anatomy": float(anatomy.detach()),
        "anatomy_triplet": float(anatomy_triplet.detach()),
        "anatomy_pairwise": float(anatomy_pairwise.detach()),
        "distillation": float(distillation.detach()),
        "hierarchy": float(hierarchy.detach()),
        "hierarchy_triplet": float(hierarchy_triplet.detach()),
        "hierarchy_pairwise": float(hierarchy_pairwise.detach()),
        "hierarchy_centroid": float(hierarchy_centroid.detach()),
        "hierarchy_teacher_gap": float(hierarchy_teacher_gap.detach()),
        "hierarchy_contrastive": float(hierarchy_contrastive.detach()),
        "hierarchy_parent_containment": float(hierarchy_parent_containment.detach()),
        "parent_replay": float(parent_replay.detach()),
        "vessel_lung": float(vessel_lung.detach()),
        "lesion_vessel": float(lesion_vessel.detach()),
        "hierarchy_selected": float(hierarchy_selected),
        "anatomy_selected": float(anatomy_selected),
        "parent_selected": float(parent_selected),
        "triplets": float(triplets + anatomy_triplets + parent_triplets),
        "vessel_lung_comparisons": float(vessel_lung_comparisons),
        "lesion_vessel_comparisons": float(lesion_vessel_comparisons),
    }


def aggregate_metrics(totals: torch.Tensor, world_size: int) -> dict[str, float]:
    totals = reduce_sums(totals, world_size)
    batches = totals[15].clamp_min(1.0)
    return {
        "loss": float((totals[0] / batches).item()),
        "anatomy": float((totals[1] / batches).item()),
        "anatomy_triplet": float((totals[2] / batches).item()),
        "anatomy_pairwise": float((totals[3] / batches).item()),
        "distillation": float((totals[4] / batches).item()),
        "hierarchy": float((totals[5] / batches).item()),
        "hierarchy_triplet": float((totals[6] / batches).item()),
        "hierarchy_pairwise": float((totals[7] / batches).item()),
        "hierarchy_centroid": float((totals[8] / batches).item()),
        "hierarchy_teacher_gap": float((totals[9] / batches).item()),
        "hierarchy_contrastive": float((totals[10] / batches).item()),
        "hierarchy_parent_containment": float((totals[11] / batches).item()),
        "parent_replay": float((totals[12] / batches).item()),
        "vessel_lung": float((totals[13] / batches).item()),
        "lesion_vessel": float((totals[14] / batches).item()),
        "batches": int(totals[15].item()),
        "hierarchy_selected": int(totals[16].item()),
        "anatomy_selected": int(totals[17].item()),
        "parent_selected": int(totals[18].item()),
        "triplets": int(totals[19].item()),
        "vessel_lung_comparisons": int(totals[20].item()),
        "lesion_vessel_comparisons": int(totals[21].item()),
    }


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    teacher: torch.nn.Module,
    lesion_loader: DataLoader,
    anatomy_loader: DataLoader,
    parent_prototypes: torch.Tensor,
    training: dict[str, Any],
    device: torch.device,
    world_size: int,
) -> dict[str, float]:
    model.eval()
    teacher.eval()
    hierarchy_memory = GeometryMemoryBank(
        int(training["hierarchy_memory_bank_capacity"])
    )
    hierarchy_teacher_memory = GeometryMemoryBank(
        int(training["hierarchy_memory_bank_capacity"])
    )
    anatomy_memory = GeometryMemoryBank(int(training["memory_bank_capacity"]))
    parent_memory = GeometryMemoryBank(int(training["memory_bank_capacity"]))
    vessel_lung_memory = GeometryMemoryBank(int(training["memory_bank_capacity"]))
    vessel_lung_teacher_memory = GeometryMemoryBank(
        int(training["memory_bank_capacity"])
    )
    lesion_vessel_memory = GeometryMemoryBank(int(training["memory_bank_capacity"]))
    lesion_vessel_teacher_memory = GeometryMemoryBank(
        int(training["memory_bank_capacity"])
    )
    totals = torch.zeros(22, dtype=torch.float64, device=device)
    anatomy_iterator = iter(anatomy_loader)
    for lesion_batch in lesion_loader:
        anatomy_batch, anatomy_iterator = next_cycled(anatomy_iterator, anatomy_loader)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, metrics = batch_losses(
                model,
                teacher,
                lesion_batch,
                anatomy_batch,
                parent_prototypes,
                hierarchy_memory,
                hierarchy_teacher_memory,
                anatomy_memory,
                parent_memory,
                vessel_lung_memory,
                vessel_lung_teacher_memory,
                lesion_vessel_memory,
                lesion_vessel_teacher_memory,
                training,
                device,
                generator=None,
            )
        totals += torch.tensor(
            [
                metrics["loss"],
                metrics["anatomy"],
                metrics["anatomy_triplet"],
                metrics["anatomy_pairwise"],
                metrics["distillation"],
                metrics["hierarchy"],
                metrics["hierarchy_triplet"],
                metrics["hierarchy_pairwise"],
                metrics["hierarchy_centroid"],
                metrics["hierarchy_teacher_gap"],
                metrics["hierarchy_contrastive"],
                metrics["hierarchy_parent_containment"],
                metrics["parent_replay"],
                metrics["vessel_lung"],
                metrics["lesion_vessel"],
                1.0,
                metrics["hierarchy_selected"],
                metrics["anatomy_selected"],
                metrics["parent_selected"],
                metrics["triplets"],
                metrics["vessel_lung_comparisons"],
                metrics["lesion_vessel_comparisons"],
            ],
            dtype=torch.float64,
            device=device,
        )
    return aggregate_metrics(totals, world_size)


def checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    validation: dict[str, float],
    initialization: dict[str, Any],
) -> dict[str, Any]:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    return {
        "model": {
            key: value.detach().cpu() for key, value in unwrapped.state_dict().items()
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "validation": validation,
        "initialization": initialization,
    }


def atomic_torch_save(state: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    torch.save(state, temporary)
    os.replace(temporary, path)


def link_checkpoint(source: Path, destination: Path) -> None:
    destination.unlink(missing_ok=True)
    os.link(source, destination)


@contextmanager
def checkpoint_write_lock(path: str | None) -> Iterator[None]:
    if path is None or os.name != "posix":
        yield
        return
    import fcntl

    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_dataset_protocol(config, verify_files=True)
    rank, local_rank, world_size = setup_distributed()
    seed_everything(int(config["seed"]), rank)
    device = torch.device("cuda", local_rank)
    training = config["training"]
    output = Path(config["paths"]["output"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_dataset_companion_zh(output, config)
        (output / "config.yaml").write_text(
            args.config.read_text(encoding="utf-8"), encoding="utf-8"
        )

    loaders, samplers = make_loaders(config, rank, world_size)
    initialization_path = config["paths"].get("initialization_checkpoint")
    if initialization_path is None:
        initialization_path = config["paths"].get("r33_checkpoint")
    if initialization_path is None:
        raise KeyError("paths.initialization_checkpoint is required")
    initialization_checkpoint = Path(initialization_path)
    expected_sha_value = config["initialization"].get("checkpoint_sha256")
    if expected_sha_value is None:
        expected_sha_value = config["initialization"].get("r33_checkpoint_sha256")
    if expected_sha_value is None:
        raise KeyError("initialization.checkpoint_sha256 is required")
    expected_sha = str(expected_sha_value)
    actual_sha = file_sha256(initialization_checkpoint)
    if expected_sha == "auto_record":
        expected_sha = actual_sha
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"Initialization checkpoint SHA mismatch: {actual_sha} != {expected_sha}"
        )
    teacher_source = str(
        config["initialization"].get("teacher_source", "checkpoint")
    )
    teacher_checkpoint: Path | None
    if teacher_source == "pretrained_medgemma_vision_tower":
        teacher_checkpoint = None
        teacher_reference = Path(config["paths"]["model"]) / "config.json"
        teacher_sha = file_sha256(teacher_reference)
        expected_teacher_sha = str(
            config["initialization"].get(
                "teacher_model_config_sha256", "auto_record"
            )
        )
    elif teacher_source == "checkpoint":
        teacher_checkpoint = Path(
            config["paths"].get("teacher_checkpoint", initialization_checkpoint)
        )
        teacher_reference = teacher_checkpoint
        teacher_sha = file_sha256(teacher_checkpoint)
        expected_teacher_sha = str(
            config["initialization"].get("teacher_checkpoint_sha256", teacher_sha)
        )
    else:
        raise ValueError(f"Unsupported teacher source: {teacher_source}")
    if expected_teacher_sha == "auto_record":
        expected_teacher_sha = teacher_sha
    if teacher_sha != expected_teacher_sha:
        raise RuntimeError(
            f"Teacher reference SHA mismatch: {teacher_sha} != {expected_teacher_sha}"
        )
    initialization = {
        "checkpoint": str(initialization_checkpoint),
        "sha256": actual_sha,
        "teacher_source": teacher_source,
        "teacher_reference": str(teacher_reference),
        "teacher_sha256": teacher_sha,
        "model_parameters_only": True,
        "optimizer_restored": args.resume is not None,
    }

    model = build_anatomy_patch_encoder(
        config["paths"]["model"], dtype=torch.bfloat16, gradient_checkpointing=True
    )
    parameter_counts = set_trainable_anatomy_layers(
        model, training.get("trainable_vision_layers")
    )
    model.to(device)
    if teacher_checkpoint is None:
        teacher = build_anatomy_patch_encoder(
            config["paths"]["model"],
            dtype=torch.bfloat16,
            gradient_checkpointing=False,
        ).to(device)
    else:
        teacher = load_anatomy_patch_encoder(
            config["paths"]["model"], teacher_checkpoint, dtype=torch.bfloat16
        ).to(device)
    teacher.requires_grad_(False)
    teacher.eval()

    if args.initialize_from is not None:
        if args.initialize_from.resolve() != initialization_checkpoint.resolve():
            raise RuntimeError(
                "--initialize-from must be the configured initialization checkpoint"
            )
        checkpoint = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        model.load_state_dict(checkpoint["model"], strict=True)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    accumulation = int(training["gradient_accumulation_steps"])
    steps_per_epoch = math.ceil(len(loaders["lesion_train"]) / accumulation)
    total_steps = int(training["epochs"]) * steps_per_epoch
    warmup_steps = round(total_steps * float(training["warmup_fraction"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_schedule(
            step, warmup_steps=warmup_steps, total_steps=total_steps
        ),
    )
    start_epoch = 1
    global_step = 0
    best_loss = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        unwrapped = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )
        unwrapped.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint.get("validation", {}).get("loss", best_loss))

    parent_prototypes = torch.from_numpy(
        regular_simplex_targets(len(ANATOMY_NAMES))
    ).to(device)
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + rank)
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        for sampler in samplers.values():
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
        model.train()
        teacher.eval()
        hierarchy_memory = GeometryMemoryBank(
            int(training["hierarchy_memory_bank_capacity"])
        )
        hierarchy_teacher_memory = GeometryMemoryBank(
            int(training["hierarchy_memory_bank_capacity"])
        )
        anatomy_memory = GeometryMemoryBank(int(training["memory_bank_capacity"]))
        parent_memory = GeometryMemoryBank(int(training["memory_bank_capacity"]))
        vessel_lung_memory = GeometryMemoryBank(int(training["memory_bank_capacity"]))
        vessel_lung_teacher_memory = GeometryMemoryBank(
            int(training["memory_bank_capacity"])
        )
        lesion_vessel_memory = GeometryMemoryBank(int(training["memory_bank_capacity"]))
        lesion_vessel_teacher_memory = GeometryMemoryBank(
            int(training["memory_bank_capacity"])
        )
        totals = torch.zeros(22, dtype=torch.float64, device=device)
        anatomy_iterator = iter(loaders["anatomy_train"])
        train_loader = loaders["lesion_train"]
        remainder = len(train_loader) % accumulation
        for batch_index, lesion_batch in enumerate(train_loader, start=1):
            anatomy_batch, anatomy_iterator = next_cycled(
                anatomy_iterator, loaders["anatomy_train"]
            )
            sync_step = batch_index % accumulation == 0 or batch_index == len(
                train_loader
            )
            sync_context = (
                nullcontext()
                if sync_step or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss, metrics = batch_losses(
                        model,
                        teacher,
                        lesion_batch,
                        anatomy_batch,
                        parent_prototypes,
                        hierarchy_memory,
                        hierarchy_teacher_memory,
                        anatomy_memory,
                        parent_memory,
                        vessel_lung_memory,
                        vessel_lung_teacher_memory,
                        lesion_vessel_memory,
                        lesion_vessel_teacher_memory,
                        training,
                        device,
                        generator=generator,
                    )
                    divisor = (
                        remainder
                        if remainder and batch_index > len(train_loader) - remainder
                        else accumulation
                    )
                    scaled_loss = loss / divisor
                scaled_loss.backward()
            totals += torch.tensor(
                [
                    metrics["loss"],
                    metrics["anatomy"],
                    metrics["anatomy_triplet"],
                    metrics["anatomy_pairwise"],
                    metrics["distillation"],
                    metrics["hierarchy"],
                    metrics["hierarchy_triplet"],
                    metrics["hierarchy_pairwise"],
                    metrics["hierarchy_centroid"],
                    metrics["hierarchy_teacher_gap"],
                    metrics["hierarchy_contrastive"],
                    metrics["hierarchy_parent_containment"],
                    metrics["parent_replay"],
                    metrics["vessel_lung"],
                    metrics["lesion_vessel"],
                    1.0,
                    metrics["hierarchy_selected"],
                    metrics["anatomy_selected"],
                    metrics["parent_selected"],
                    metrics["triplets"],
                    metrics["vessel_lung_comparisons"],
                    metrics["lesion_vessel_comparisons"],
                ],
                dtype=torch.float64,
                device=device,
            )
            if sync_step:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(training["max_grad_norm"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if rank == 0 and global_step % int(training["log_interval_steps"]) == 0:
                    progress = {
                        "epoch": epoch,
                        "batch": batch_index,
                        "batches_per_epoch": len(train_loader),
                        "global_step": global_step,
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    (output / "progress.json").write_text(
                        json.dumps(progress, indent=2), encoding="utf-8"
                    )
                    print(json.dumps(progress), flush=True)

        train_metrics = aggregate_metrics(totals, world_size)
        validation = evaluate(
            model,
            teacher,
            loaders["lesion_val"],
            loaders["anatomy_val"],
            parent_prototypes,
            training,
            device,
            world_size,
        )
        if rank == 0:
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": scheduler.get_last_lr()[0],
                "train": train_metrics,
                "val": validation,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            state = checkpoint_state(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                config,
                validation,
                initialization,
            )
            with checkpoint_write_lock(training.get("checkpoint_lock_path")):
                last_checkpoint = output / "last.pt"
                atomic_torch_save(state, last_checkpoint)
                if bool(training.get("save_each_epoch", False)):
                    link_checkpoint(last_checkpoint, output / f"epoch{epoch:02d}.pt")
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    link_checkpoint(last_checkpoint, output / "best.pt")
            (output / "status.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_validation_loss": best_loss,
                        "parameter_counts": parameter_counts,
                        "world_size": world_size,
                        "initialization": initialization,
                        "peak_cuda_memory_allocated_mib": round(
                            torch.cuda.max_memory_allocated(device) / (1024**2), 2
                        ),
                        "peak_cuda_memory_reserved_mib": round(
                            torch.cuda.max_memory_reserved(device) / (1024**2), 2
                        ),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        (output / "training.complete").touch()
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
