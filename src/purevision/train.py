from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from torch.utils.tensorboard import SummaryWriter

from .attribute_2d import ATTRIBUTE_NAMES, Attribute2DSpace
from .attributes import ORDINAL_SOURCE_FIELDS
from .auxiliary_probe import (
    AuxiliaryProbeBank,
    AuxiliaryProbeLoss,
    EncoderWithAuxiliaryProbes,
    PCAProjection,
    compute_auxiliary_probe_loss,
    fit_pca_projection,
    load_pca_projections,
    serialize_pca_projections,
)
from .config import load_config, validate_input_contract, with_path_overrides
from .dataset import LIDCLesionDataset, collate_lesion_batch
from .losses import (
    EncoderWithLearnableGeometry,
    GeometryMemoryBank,
    LearnableAttributeGeometry,
    compute_loss,
)
from .model import MedGemmaLesionEncoder, build_model, set_trainable_vision_layers
from .protocol import validate_dataset_protocol, write_dataset_companion_zh
from .target_space import build_target_space


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune MedGemma's vision tower")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--processed-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--model-path")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Load only encoder weights and start a fresh optimizer/schedule at epoch 1.",
    )
    return parser.parse_args()


def setup_distributed() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    return rank, local_rank, world_size


def seed_everything(seed: int, rank: int) -> None:
    seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DistributedEvalSampler(Sampler[int]):


    def __init__(self, dataset: LIDCLesionDataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        return max(0, (len(self.dataset) - self.rank + self.world_size - 1) // self.world_size)


def build_loader(
    dataset: LIDCLesionDataset,
    config: dict[str, Any],
    rank: int,
    world_size: int,
    shuffle: bool,
) -> tuple[DataLoader, Sampler[int] | None]:
    sampler: Sampler[int] | None = None
    if world_size > 1:
        if shuffle:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=True
            )
        else:
            sampler = DistributedEvalSampler(dataset, rank, world_size)
    loader = DataLoader(
        dataset,
        batch_size=int(config["per_device_batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=int(config["num_workers"]),
        pin_memory=True,
        persistent_workers=int(config["num_workers"]) > 0,
        drop_last=False,
        collate_fn=collate_lesion_batch,
    )
    return loader, sampler


def move_model_inputs(
    batch: dict[str, Any], device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    return (
        [value.to(device, non_blocking=True) for value in batch["pixel_values"]],
        [value.to(device, non_blocking=True) for value in batch["patch_mask"]],
        [
            value.to(device, non_blocking=True)
            for value in batch["global_position_ids"]
        ],
    )


def cosine_schedule(warmup_steps: int, total_steps: int):
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return schedule


def delayed_schedule(schedule, delay_steps: int):
    def shifted(step: int) -> float:
        return schedule(max(0, step - delay_steps))

    return shifted


def reduce_metrics(values: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


def batch_attribute_labels(
    batch: dict[str, Any],
    size_edges: tuple[float, ...],
    device: torch.device,
    *,
    include_location: bool = False,
) -> torch.Tensor:
    ordinal = batch["ordinal_levels"].to(device, non_blocking=True).long() - 1
    calcification = batch["calcification"].to(device, non_blocking=True).long()
    diameter = batch["diameter"].to(device, non_blocking=True)
    size = torch.where(
        diameter <= size_edges[1],
        torch.zeros_like(diameter, dtype=torch.long),
        torch.where(
            diameter <= size_edges[2],
            torch.ones_like(diameter, dtype=torch.long),
            torch.full_like(diameter, 2, dtype=torch.long),
        ),
    )
    columns = [ordinal, calcification[:, None], size[:, None]]
    if include_location:
        columns.append(batch["location"].to(device, non_blocking=True).long()[:, None])
    return torch.cat(columns, dim=1)


def batch_attribute_measurements(
    batch: dict[str, Any], labels: torch.Tensor
) -> torch.Tensor:

    values = labels.float().clone()
    size_index = ATTRIBUTE_NAMES.index("size")
    values[:, size_index] = batch["diameter"].to(
        labels.device, non_blocking=True
    ).float()
    return values


def relation_uses_location(training_config: dict[str, Any]) -> bool:
    relation_config = training_config.get("attribute_relation_supervision", {})
    return bool(relation_config.get("enabled", False)) and any(
        str(task.get("target_attribute")) == "location"
        for task in relation_config.get("tasks", [])
    )


def build_attribute_loss_context(
    dataset: LIDCLesionDataset,
    attribute_space: Attribute2DSpace,
    device: torch.device,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    labels: list[np.ndarray] = [
        np.asarray(
            [
                attribute_space.ordinal_level(name, float(value)) - 1
                for value in dataset.frame[source].to_numpy()
            ],
            dtype=np.int64,
        )
        for name, source in attribute_space.ordinal_source_fields.items()
    ]
    labels.append(
        np.asarray(
            [
                attribute_space.calcification_index(value)
                for value in dataset.frame[
                    attribute_space.calcification_column
                ].to_numpy(dtype=np.int64)
            ],
            dtype=np.int64,
        )
    )
    diameter = dataset.frame["diameter_mm_mean"].to_numpy(dtype=np.float32)
    labels.append(
        np.where(
            diameter <= attribute_space.size_edges[1],
            0,
            np.where(diameter <= attribute_space.size_edges[2], 1, 2),
        ).astype(np.int64)
    )

    prototypes: list[torch.Tensor] = []
    class_weights: list[torch.Tensor] = []
    for name, values in zip(ATTRIBUTE_NAMES, labels):
        classes = int(attribute_space.level_counts[name])
        counts = np.bincount(values, minlength=classes).astype(np.float32)
        if np.any(counts == 0):
            raise ValueError(f"Training split has an empty class for {name}: {counts}")
        weights = len(values) / (classes * counts)
        prototypes.append(
            torch.as_tensor(
                attribute_space.prototypes[name], dtype=torch.float32, device=device
            )
        )
        class_weights.append(torch.as_tensor(weights, dtype=torch.float32, device=device))
    return tuple(prototypes), tuple(class_weights)


def pca_alignment_targets(
    sample_targets: torch.Tensor,
    labels: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    if sample_targets.ndim != 2 or sample_targets.shape[1] != 2:
        raise ValueError("PCA alignment targets must have shape [samples, 2]")
    if labels.shape != (len(sample_targets),):
        raise ValueError("PCA alignment labels must have shape [samples]")
    if mode == "sample":
        return sample_targets
    if mode != "class_center":
        raise ValueError(
            "pca_alignment_target_mode must be 'sample' or 'class_center'"
        )
    result = torch.empty_like(sample_targets)
    for class_index in labels.unique(sorted=True):
        mask = labels == class_index
        result[mask] = sample_targets[mask].mean(dim=0)
    return result


def effective_pca_distance_weight(
    auxiliary_config: dict[str, Any], epoch: int | None
) -> float:
    stabilization_epochs = int(
        auxiliary_config.get("probe_stabilization_epochs", 0)
    )
    if stabilization_epochs < 0:
        raise ValueError("probe_stabilization_epochs must be non-negative")
    if epoch is not None and epoch <= stabilization_epochs:
        return 0.0
    return float(auxiliary_config["pca_distance_loss_weight"])


def should_refresh_training_pca(
    auxiliary_config: dict[str, Any], epoch: int, has_projection: bool
) -> bool:
    if not has_projection:
        return True
    stabilization_epochs = int(
        auxiliary_config.get("probe_stabilization_epochs", 0)
    )
    if bool(auxiliary_config.get("freeze_pca_after_stabilization", False)):
        return epoch == stabilization_epochs + 1
    refresh_epochs = int(auxiliary_config.get("pca_refresh_epochs", 1))
    if refresh_epochs < 1:
        raise ValueError("pca_refresh_epochs must be at least 1")
    return (epoch - 1) % refresh_epochs == 0


@torch.no_grad()
def fit_training_pca_projections(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    world_size: int,
    *,
    seed: int,
    size_edges: tuple[float, ...],
    class_weights: tuple[torch.Tensor, ...],
    alignment: str,
    alignment_target_mode: str = "sample",
    fit_weighting: str = "none",
    approximation_rank: int = 2,
    niter: int = 4,
) -> dict[str, PCAProjection]:
    model.eval()
    local: dict[str, list[torch.Tensor]] = {name: [] for name in ATTRIBUTE_NAMES}
    local_targets: list[torch.Tensor] = []
    local_labels: list[torch.Tensor] = []
    for batch in loader:
        pixel_values, patch_mask, position_ids = move_model_inputs(batch, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values, patch_mask, position_ids)
        geometry_hiddens = outputs.get("auxiliary_geometry_hidden")
        if geometry_hiddens is None:
            raise ValueError("PCA fitting requires training-time geometry features")
        for name in ATTRIBUTE_NAMES:
            local[name].append(geometry_hiddens[name].float().cpu())
        local_targets.append(batch["ideal_target"].float())
        local_labels.append(
            batch_attribute_labels(batch, size_edges, torch.device("cpu"))
        )

    local_values = {
        **{name: torch.cat(local[name]) for name in ATTRIBUTE_NAMES},
        "__targets": torch.cat(local_targets),
        "__labels": torch.cat(local_labels),
    }
    if world_size > 1:
        gathered: list[dict[str, torch.Tensor] | None] = [None] * world_size
        dist.all_gather_object(gathered, local_values)
        all_values = {
            name: torch.cat(
                [payload[name] for payload in gathered if payload is not None]
            )
            for name in (*ATTRIBUTE_NAMES, "__targets", "__labels")
        }
    else:
        all_values = local_values

    if alignment not in {"none", "similarity"}:
        raise ValueError("PCA alignment must be 'none' or 'similarity'")
    if fit_weighting not in {"none", "class_balanced"}:
        raise ValueError(
            "pca_fit_weighting must be 'none' or 'class_balanced'"
        )
    target_blocks = all_values["__targets"].reshape(
        len(all_values["__targets"]), len(ATTRIBUTE_NAMES), 2
    )
    projections: dict[str, PCAProjection] = {}
    for index, name in enumerate(ATTRIBUTE_NAMES):
        labels = all_values["__labels"][:, index].long()
        sample_weights = class_weights[index].detach().cpu()[labels]
        targets = pca_alignment_targets(
            target_blocks[:, index], labels, alignment_target_mode
        )
        projections[name] = fit_pca_projection(
            all_values[name].to(device, non_blocking=True),
            seed=seed + index,
            alignment_targets=(
                targets.to(device, non_blocking=True)
                if alignment == "similarity"
                else None
            ),
            alignment_weights=(
                sample_weights.to(device, non_blocking=True)
                if alignment == "similarity"
                else None
            ),
            pca_weights=(
                sample_weights.to(device, non_blocking=True)
                if fit_weighting == "class_balanced"
                else None
            ),
            approximation_rank=approximation_rank,
            niter=niter,
        )
    model.train()
    return projections


def _loss_metric_values(
    encoder_loss: Any,
    auxiliary_loss: AuxiliaryProbeLoss | None,
    count: int,
) -> list[float]:
    auxiliary_total = 0.0 if auxiliary_loss is None else auxiliary_loss.total.item()
    combined_total = encoder_loss.total.item() + auxiliary_total
    values = [
        combined_total * count,
        encoder_loss.total.item() * count,
        encoder_loss.coordinate.item() * count,
        encoder_loss.triplet.item() * count,
        encoder_loss.pairwise.item() * count,
        encoder_loss.prototype_compact.item() * count,
        encoder_loss.prototype_margin.item() * count,
        auxiliary_total * count,
        0.0 if auxiliary_loss is None else auxiliary_loss.classification.item() * count,
        0.0 if auxiliary_loss is None else auxiliary_loss.pca_distance.item() * count,
        0.0 if auxiliary_loss is None else auxiliary_loss.target_position.item() * count,
        0.0 if auxiliary_loss is None else auxiliary_loss.contrastive.item() * count,
        float(count),
        float(encoder_loss.triplets),
    ]
    for name in ATTRIBUTE_NAMES:
        if auxiliary_loss is None:
            values.extend((0.0, 0.0, 0.0, 0.0))
        else:
            values.extend(
                (
                    auxiliary_loss.per_attribute[name]["classification"].item() * count,
                    auxiliary_loss.per_attribute[name]["pca_distance"].item() * count,
                    auxiliary_loss.per_attribute[name]["target_position"].item()
                    * count,
                    auxiliary_loss.per_attribute[name]["contrastive"].item()
                    * count,
                )
            )
    return values


def _loss_metrics(totals: torch.Tensor) -> dict[str, Any]:
    denominator = max(1.0, totals[12].item())
    result: dict[str, Any] = {
        "total": totals[0].item() / denominator,
        "encoder_total": totals[1].item() / denominator,
        "coordinate": totals[2].item() / denominator,
        "triplet": totals[3].item() / denominator,
        "pairwise": totals[4].item() / denominator,
        "prototype_compact": totals[5].item() / denominator,
        "prototype_margin": totals[6].item() / denominator,
        "auxiliary_total": totals[7].item() / denominator,
        "auxiliary_classification": totals[8].item() / denominator,
        "auxiliary_pca_distance": totals[9].item() / denominator,
        "auxiliary_target_position": totals[10].item() / denominator,
        "auxiliary_contrastive": totals[11].item() / denominator,
        "samples": int(totals[12].item()),
        "triplets": int(totals[13].item()),
    }
    result["auxiliary_by_attribute"] = {
        name: {
            "classification": totals[14 + index * 4].item() / denominator,
            "pca_distance": totals[15 + index * 4].item() / denominator,
            "target_position": totals[16 + index * 4].item() / denominator,
            "contrastive": totals[17 + index * 4].item() / denominator,
        }
        for index, name in enumerate(ATTRIBUTE_NAMES)
    }
    return result


@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    training_config: dict[str, Any],
    device: torch.device,
    world_size: int,
    attribute_prototypes: tuple[torch.Tensor, ...] | None = None,
    attribute_class_weights: tuple[torch.Tensor, ...] | None = None,
    size_edges: tuple[float, ...] | None = None,
    pca_projections: dict[str, PCAProjection] | None = None,
    epoch: int | None = None,
) -> dict[str, Any]:
    model.eval()
    bank = GeometryMemoryBank(
        capacity=int(training_config.get("memory_bank_capacity", 1024))
    )
    totals = torch.zeros(
        14 + 4 * len(ATTRIBUTE_NAMES), device=device, dtype=torch.float64
    )
    auxiliary_config = training_config.get("auxiliary_probe", {})
    for batch in loader:
        pixel_values, patch_mask, position_ids = move_model_inputs(batch, device)
        ideal = batch["ideal_target"].to(device, non_blocking=True)
        labels = (
            batch_attribute_labels(
                batch,
                size_edges,
                device,
                include_location=relation_uses_location(training_config),
            )
            if size_edges is not None
            else None
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(pixel_values, patch_mask, position_ids)
            loss = compute_loss(
                outputs,
                ideal,
                bank,
                training_config,
                attribute_labels=labels,
                attribute_measurements=(
                    batch_attribute_measurements(batch, labels)
                    if labels is not None
                    else None
                ),
                attribute_prototypes=attribute_prototypes,
                attribute_class_weights=attribute_class_weights,
            )
            auxiliary_loss = None
            if pca_projections is not None:
                if labels is None or attribute_prototypes is None or attribute_class_weights is None:
                    raise ValueError("Auxiliary probe loss requires full attribute context")
                auxiliary_loss = compute_auxiliary_probe_loss(
                    outputs["auxiliary_probes"],
                    labels[:, : len(ATTRIBUTE_NAMES)],
                    ideal,
                    attribute_prototypes,
                    attribute_class_weights,
                    pca_projections,
                    classification_weight=float(
                        auxiliary_config["classification_loss_weight"]
                    ),
                    pca_distance_weight=effective_pca_distance_weight(
                        auxiliary_config, epoch
                    ),
                    distance_margin=float(auxiliary_config["distance_margin"]),
                    target_position_weight=float(
                        auxiliary_config.get("target_position_loss_weight", 0.0)
                    ),
                    label_smoothing=float(
                        auxiliary_config.get("label_smoothing", 0.0)
                    ),
                    attribute_weights=auxiliary_config.get("attribute_weights"),
                    geometry_hiddens=outputs.get("auxiliary_geometry_hidden"),
                    contrastive_weight=float(
                        auxiliary_config.get("contrastive_loss_weight", 0.0)
                    ),
                    contrastive_temperature=float(
                        auxiliary_config.get("contrastive_temperature", 0.1)
                    ),
                    negative_aggregation=str(
                        auxiliary_config.get("negative_aggregation", "mean")
                    ),
                    attribute_distance_margins=auxiliary_config.get(
                        "attribute_distance_margins"
                    ),
                    classification_reduction=str(
                        auxiliary_config.get("classification_reduction", "mean")
                    ),
                )
        count = len(pixel_values)
        totals += torch.tensor(
            _loss_metric_values(loss, auxiliary_loss, count),
            device=device,
            dtype=torch.float64,
        )
    totals = reduce_metrics(totals, world_size)
    model.train()
    return _loss_metrics(totals)


def save_checkpoint(
    path: Path,
    model: MedGemmaLesionEncoder,
    optimizer: AdamW,
    scheduler: LambdaLR,
    epoch: int,
    global_step: int,
    validation: dict[str, Any],
    config: dict[str, Any],
    include_optimizer: bool,
    auxiliary_probes: AuxiliaryProbeBank | None = None,
    pca_projections: dict[str, PCAProjection] | None = None,
    learnable_geometry: LearnableAttributeGeometry | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model": {
            key: value.detach().cpu() for key, value in model.state_dict().items()
        },
        "epoch": epoch,
        "global_step": global_step,
        "validation": validation,
        "config": config,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
        payload["scheduler"] = scheduler.state_dict()
    if auxiliary_probes is not None:
        payload["auxiliary_probes"] = {
            key: value.detach().cpu()
            for key, value in auxiliary_probes.state_dict().items()
        }
    if pca_projections is not None:
        payload["training_pca_projections"] = serialize_pca_projections(
            pca_projections
        )
    if learnable_geometry is not None:
        payload["learnable_geometry"] = {
            key: value.detach().cpu()
            for key, value in learnable_geometry.state_dict().items()
        }
    torch.save(payload, temporary)
    temporary.replace(path)


def snapshot_model(model: MedGemmaLesionEncoder) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def save_model_snapshot(
    path: Path,
    state: dict[str, torch.Tensor],
    epoch: int,
    global_step: int,
    validation: dict[str, Any],
    config: dict[str, Any],
    learnable_geometry_state: dict[str, torch.Tensor] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload: dict[str, Any] = {
        "model": state,
        "epoch": epoch,
        "global_step": global_step,
        "validation": validation,
        "config": config,
    }
    if learnable_geometry_state is not None:
        payload["learnable_geometry"] = learnable_geometry_state
    torch.save(payload, temporary)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.resume is not None and args.initialize_from is not None:
        raise ValueError("--resume and --initialize-from cannot be used together")
    config = with_path_overrides(
        load_config(args.config),
        processed=args.processed_dir,
        output=args.output_dir,
        model=args.model_path,
    )
    validate_dataset_protocol(config, verify_files=True)
    training_config = config["training"]
    validate_input_contract(config)
    if args.epochs is not None:
        training_config["epochs"] = args.epochs
    rank, local_rank, world_size = setup_distributed()
    seed_everything(int(config["seed"]), rank)
    device = torch.device("cuda", local_rank)
    output_dir = Path(config["paths"]["output"])
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_dataset_companion_zh(output_dir, config)
        (output_dir / "resolved_config.json").write_text(
            json.dumps(config, indent=2), encoding="utf-8"
        )

    attribute_space = build_target_space(config["attribute_space"])
    train_dataset = LIDCLesionDataset(
        config["paths"]["processed"],
        "train",
        config["attribute_space"],
        config["data"]["image_size"],
        config["data"]["patch_size"],
        config["data"].get("input_mode", "global"),
        bool(training_config.get("random_dihedral_augmentation", False)),
    )
    val_dataset = LIDCLesionDataset(
        config["paths"]["processed"],
        "val",
        config["attribute_space"],
        config["data"]["image_size"],
        config["data"]["patch_size"],
        config["data"].get("input_mode", "global"),
    )
    train_loader, train_sampler = build_loader(
        train_dataset, training_config, rank, world_size, True
    )
    val_loader, val_sampler = build_loader(
        val_dataset, training_config, rank, world_size, False
    )
    attribute_prototypes: tuple[torch.Tensor, ...] | None = None
    attribute_class_weights: tuple[torch.Tensor, ...] | None = None
    size_edges: tuple[float, ...] | None = None
    if isinstance(attribute_space, Attribute2DSpace):
        attribute_prototypes, attribute_class_weights = build_attribute_loss_context(
            train_dataset, attribute_space, device
        )
        size_edges = attribute_space.size_edges

    auxiliary_config = training_config.get("auxiliary_probe", {})
    auxiliary_enabled = bool(auxiliary_config.get("enabled", False))
    center_config = training_config.get("learnable_center_supervision", {})
    center_enabled = bool(center_config.get("enabled", False))
    if auxiliary_enabled and center_enabled:
        raise ValueError(
            "Auxiliary probe training and learnable center supervision cannot be enabled together"
        )
    pca_loader: DataLoader | None = None
    if auxiliary_enabled:
        if not isinstance(attribute_space, Attribute2DSpace):
            raise ValueError("Auxiliary attribute probes require per_attribute_2d targets")
        pca_dataset = LIDCLesionDataset(
            config["paths"]["processed"],
            "train",
            config["attribute_space"],
            config["data"]["image_size"],
            config["data"]["patch_size"],
            config["data"].get("input_mode", "global"),
            False,
        )
        pca_loader, _ = build_loader(
            pca_dataset, training_config, rank, world_size, False
        )

    encoder = build_model(
        config["paths"]["model"],
        attribute_space.dimension,


        dtype=torch.float32,
        gradient_checkpointing=bool(training_config["gradient_checkpointing"]),
    ).to(device)
    parameter_counts = set_trainable_vision_layers(
        encoder, training_config.get("trainable_vision_layers")
    )
    if args.initialize_from is not None:
        initialization = torch.load(
            args.initialize_from, map_location="cpu", weights_only=False
        )
        encoder.load_state_dict(initialization["model"], strict=True)
        if rank == 0:
            (output_dir / "initialization.json").write_text(
                json.dumps(
                    {
                        "checkpoint": str(args.initialize_from),
                        "source_epoch": int(initialization.get("epoch", -1)),
                        "loaded": ["model"],
                        "not_loaded": ["optimizer", "scheduler", "global_step"],
                        "start_epoch": 1,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    auxiliary_probes: AuxiliaryProbeBank | None = None
    learnable_geometry: LearnableAttributeGeometry | None = None
    training_system: torch.nn.Module = encoder
    if auxiliary_enabled:
        auxiliary_probes = AuxiliaryProbeBank(
            int(encoder.vision_encoder.config.hidden_size),
            {
                name: int(attribute_space.level_counts[name])
                for name in ATTRIBUTE_NAMES
            },
            dropout=float(auxiliary_config.get("dropout", 0.0)),
            first_layer_initialization=str(
                auxiliary_config.get("first_layer_initialization", "default")
            ),
        ).to(device)
        training_system = EncoderWithAuxiliaryProbes(
            encoder,
            auxiliary_probes,
            geometry_updates_probe=bool(
                auxiliary_config.get("geometry_updates_probe", True)
            ),
            classification_updates_encoder=bool(
                auxiliary_config.get("classification_updates_encoder", True)
            ),
            geometry_uses_embedding=bool(
                auxiliary_config.get("geometry_uses_embedding", False)
            ),
        ).to(device)
        parameter_counts["auxiliary_probe_parameters"] = sum(
            parameter.numel() for parameter in auxiliary_probes.parameters()
        )
    elif center_enabled:
        target_attribute = str(center_config["target_attribute"])
        if target_attribute not in ATTRIBUTE_NAMES:
            raise ValueError(f"Unknown target attribute: {target_attribute}")
        learnable_geometry = LearnableAttributeGeometry(
            int(encoder.vision_encoder.config.hidden_size),
            int(attribute_space.level_counts[target_attribute]),
            ordinal=target_attribute != "calcification",
            seed=int(config["seed"]) + ATTRIBUTE_NAMES.index(target_attribute),
            initial_spacing=float(center_config.get("initial_spacing", 0.5)),
        ).to(device)
        training_system = EncoderWithLearnableGeometry(
            encoder, learnable_geometry
        ).to(device)
        parameter_counts["learnable_geometry_parameters"] = sum(
            parameter.numel() for parameter in learnable_geometry.parameters()
        )
    if rank == 0:
        print(json.dumps({"parameter_counts": parameter_counts}), flush=True)
    model = training_system
    if world_size > 1:
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )
    unwrapped_system = model.module if isinstance(model, DistributedDataParallel) else model
    if isinstance(unwrapped_system, EncoderWithAuxiliaryProbes):
        encoder = unwrapped_system.encoder
        auxiliary_probes = unwrapped_system.probes
    elif isinstance(unwrapped_system, EncoderWithLearnableGeometry):
        encoder = unwrapped_system.encoder
        learnable_geometry = unwrapped_system.geometry
    else:
        encoder = unwrapped_system
    parameter_groups: list[dict[str, Any]] = [
        {
            "params": [
                parameter for parameter in encoder.parameters() if parameter.requires_grad
            ],
            "lr": float(training_config["learning_rate"]),
        }
    ]
    if auxiliary_probes is not None:
        parameter_groups.append(
            {
                "params": list(auxiliary_probes.parameters()),
                "lr": float(
                    auxiliary_config.get(
                        "learning_rate", training_config["learning_rate"]
                    )
                ),
                "weight_decay": float(
                    auxiliary_config.get(
                        "weight_decay", training_config["weight_decay"]
                    )
                ),
            }
        )
    if learnable_geometry is not None:
        parameter_groups.append(
            {
                "params": list(learnable_geometry.parameters()),
                "lr": float(
                    center_config.get(
                        "learning_rate", training_config["learning_rate"]
                    )
                ),
                "weight_decay": float(
                    center_config.get(
                        "weight_decay", training_config["weight_decay"]
                    )
                ),
            }
        )
    optimizer = AdamW(
        parameter_groups, weight_decay=float(training_config["weight_decay"])
    )
    accumulation = int(training_config["gradient_accumulation_steps"])
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    training_epochs = int(training_config["epochs"])
    scheduler_epochs = int(
        training_config.get("scheduler_total_epochs", training_epochs)
    )
    stabilization_epochs = int(
        auxiliary_config.get("probe_stabilization_epochs", 0)
    )
    delay_encoder_scheduler = (
        auxiliary_enabled
        and stabilization_epochs > 0
        and not bool(
            auxiliary_config.get(
                "stabilization_advances_encoder_scheduler", True
            )
        )
    )
    scheduled_training_epochs = scheduler_epochs + (
        stabilization_epochs if delay_encoder_scheduler else 0
    )
    if scheduled_training_epochs < training_epochs:
        raise ValueError(
            "scheduler_total_epochs cannot cover the configured training epochs"
        )
    total_steps = optimizer_steps_per_epoch * scheduler_epochs
    warmup_steps = int(total_steps * float(training_config["warmup_fraction"]))
    encoder_schedule = cosine_schedule(warmup_steps, total_steps)
    if delay_encoder_scheduler:
        delay_steps = optimizer_steps_per_epoch * stabilization_epochs
        probe_total_steps = total_steps + delay_steps
        probe_warmup_steps = int(
            probe_total_steps * float(training_config["warmup_fraction"])
        )
        scheduler = LambdaLR(
            optimizer,
            [
                delayed_schedule(encoder_schedule, delay_steps),
                cosine_schedule(probe_warmup_steps, probe_total_steps),
            ],
        )
    else:
        scheduler = LambdaLR(optimizer, encoder_schedule)
    writer = SummaryWriter(output_dir / "tensorboard") if rank == 0 else None
    history_path = output_dir / "history.jsonl"
    best_validation = float("inf")
    global_step = 0
    start_epoch = 1
    checkpoint_window = int(training_config.get("checkpoint_window_epochs", 0))
    last_checkpoint_interval = int(
        training_config.get("last_checkpoint_interval", 1)
    )
    window_best_validation = float("inf")
    window_best_epoch = 0
    window_best_step = 0
    window_best_metrics: dict[str, float] | None = None
    window_best_state: dict[str, torch.Tensor] | None = None
    window_best_geometry_state: dict[str, torch.Tensor] | None = None
    window_records: list[dict[str, Any]] = []
    defer_best_checkpoint = bool(
        training_config.get("defer_best_checkpoint", False)
    )
    deferred_best_state: dict[str, torch.Tensor] | None = None
    deferred_best_geometry_state: dict[str, torch.Tensor] | None = None
    deferred_best_epoch = 0
    deferred_best_step = 0
    deferred_best_metrics: dict[str, Any] | None = None
    pca_projections: dict[str, PCAProjection] | None = None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        encoder.load_state_dict(checkpoint["model"], strict=True)
        if auxiliary_probes is not None:
            auxiliary_probes.load_state_dict(
                checkpoint["auxiliary_probes"], strict=True
            )
            if "training_pca_projections" in checkpoint:
                pca_projections = load_pca_projections(
                    checkpoint["training_pca_projections"], device
                )
        if learnable_geometry is not None:
            learnable_geometry.load_state_dict(
                checkpoint["learnable_geometry"], strict=True
            )
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["epoch"]) + 1
        resumed_before_encoder_training = bool(
            auxiliary_config.get("checkpoint_after_stabilization_only", False)
        ) and int(checkpoint["epoch"]) <= int(
            auxiliary_config.get("probe_stabilization_epochs", 0)
        )
        best_validation = (
            float("inf")
            if resumed_before_encoder_training
            else float(checkpoint["validation"]["total"])
        )

    for epoch in range(start_epoch, int(training_config["epochs"]) + 1):
        started = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if isinstance(val_sampler, DistributedSampler):
            val_sampler.set_epoch(epoch)
        if auxiliary_enabled and should_refresh_training_pca(
            auxiliary_config, epoch, pca_projections is not None
        ):
            if pca_loader is None:
                raise RuntimeError("Auxiliary training is missing its PCA loader")
            if size_edges is None or attribute_class_weights is None:
                raise RuntimeError("Auxiliary PCA fitting requires attribute context")
            pca_projections = fit_training_pca_projections(
                model,
                pca_loader,
                device,
                world_size,
                seed=int(config["seed"]) + epoch * len(ATTRIBUTE_NAMES),
                size_edges=size_edges,
                class_weights=attribute_class_weights,
                alignment=str(auxiliary_config.get("pca_alignment", "none")),
                alignment_target_mode=str(
                    auxiliary_config.get(
                        "pca_alignment_target_mode", "sample"
                    )
                ),
                fit_weighting=str(
                    auxiliary_config.get("pca_fit_weighting", "none")
                ),
                approximation_rank=int(auxiliary_config.get("pca_rank", 2)),
                niter=int(auxiliary_config.get("pca_niter", 4)),
            )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        bank = GeometryMemoryBank(
            capacity=int(training_config.get("memory_bank_capacity", 1024))
        )
        epoch_totals = torch.zeros(
            14 + 4 * len(ATTRIBUTE_NAMES), device=device, dtype=torch.float64
        )

        for batch_index, batch in enumerate(train_loader):
            pixel_values, patch_mask, position_ids = move_model_inputs(batch, device)
            ideal = batch["ideal_target"].to(device, non_blocking=True)
            labels = (
                batch_attribute_labels(
                    batch,
                    size_edges,
                    device,
                    include_location=relation_uses_location(training_config),
                )
                if size_edges is not None
                else None
            )
            should_step = (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader)
            sync_context = contextlib.nullcontext()
            if isinstance(model, DistributedDataParallel) and not should_step:
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(pixel_values, patch_mask, position_ids)
                    loss = compute_loss(
                        outputs,
                        ideal,
                        bank,
                        training_config,
                        attribute_labels=labels,
                        attribute_measurements=(
                            batch_attribute_measurements(batch, labels)
                            if labels is not None
                            else None
                        ),
                        attribute_prototypes=attribute_prototypes,
                        attribute_class_weights=attribute_class_weights,
                    )
                    auxiliary_loss = None
                    if pca_projections is not None:
                        if labels is None or attribute_prototypes is None or attribute_class_weights is None:
                            raise ValueError(
                                "Auxiliary probe loss requires full attribute context"
                            )
                        auxiliary_loss = compute_auxiliary_probe_loss(
                            outputs["auxiliary_probes"],
                            labels[:, : len(ATTRIBUTE_NAMES)],
                            ideal,
                            attribute_prototypes,
                            attribute_class_weights,
                            pca_projections,
                            classification_weight=float(
                                auxiliary_config["classification_loss_weight"]
                            ),
                            pca_distance_weight=effective_pca_distance_weight(
                                auxiliary_config, epoch
                            ),
                            distance_margin=float(
                                auxiliary_config["distance_margin"]
                            ),
                            target_position_weight=float(
                                auxiliary_config.get(
                                    "target_position_loss_weight", 0.0
                                )
                            ),
                            label_smoothing=float(
                                auxiliary_config.get("label_smoothing", 0.0)
                            ),
                            attribute_weights=auxiliary_config.get(
                                "attribute_weights"
                            ),
                            geometry_hiddens=outputs.get(
                                "auxiliary_geometry_hidden"
                            ),
                            contrastive_weight=float(
                                auxiliary_config.get(
                                    "contrastive_loss_weight", 0.0
                                )
                            ),
                            contrastive_temperature=float(
                                auxiliary_config.get(
                                    "contrastive_temperature", 0.1
                                )
                            ),
                            negative_aggregation=str(
                                auxiliary_config.get(
                                    "negative_aggregation", "mean"
                                )
                            ),
                            attribute_distance_margins=auxiliary_config.get(
                                "attribute_distance_margins"
                            ),
                            classification_reduction=str(
                                auxiliary_config.get(
                                    "classification_reduction", "mean"
                                )
                            ),
                        )
                    combined_loss = loss.total + (
                        auxiliary_loss.total
                        if auxiliary_loss is not None
                        else loss.total * 0.0
                    )
                    scaled_loss = combined_loss / accumulation
                scaled_loss.backward()

            count = len(pixel_values)
            epoch_totals += torch.tensor(
                _loss_metric_values(loss, auxiliary_loss, count),
                device=device,
                dtype=torch.float64,
            )
            if should_step:
                freeze_probe_after_epoch = auxiliary_config.get(
                    "freeze_after_epoch"
                )
                freeze_encoder_for_stabilization = (
                    auxiliary_probes is not None
                    and epoch
                    <= int(
                        auxiliary_config.get("probe_stabilization_epochs", 0)
                    )
                    and bool(
                        auxiliary_config.get(
                            "freeze_encoder_during_probe_stabilization", True
                        )
                    )
                )
                if freeze_encoder_for_stabilization:
                    for parameter in encoder.parameters():
                        parameter.grad = None
                if (
                    auxiliary_probes is not None
                    and freeze_probe_after_epoch is not None
                    and epoch > int(freeze_probe_after_epoch)
                ):
                    for parameter in auxiliary_probes.parameters():
                        parameter.grad = None
                max_grad_norm = float(training_config["max_grad_norm"])
                if auxiliary_probes is not None and bool(
                    auxiliary_config.get("separate_gradient_clipping", False)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        encoder.parameters(), max_grad_norm
                    )
                    torch.nn.utils.clip_grad_norm_(
                        auxiliary_probes.parameters(), max_grad_norm
                    )
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        epoch_totals = reduce_metrics(epoch_totals, world_size)
        train_metrics = _loss_metrics(epoch_totals)
        validation = evaluate_loss(
            model,
            val_loader,
            training_config,
            device,
            world_size,
            attribute_prototypes,
            attribute_class_weights,
            size_edges,
            pca_projections,
            epoch,
        )
        stabilization_epochs = int(
            auxiliary_config.get("probe_stabilization_epochs", 0)
        )
        encoder_training_epoch = max(0, epoch - stabilization_epochs)
        record = {
            "epoch": epoch,
            "encoder_training_epoch": encoder_training_epoch,
            "global_step": global_step,
            "seconds": round(time.time() - started, 3),
            "learning_rate": scheduler.get_last_lr()[0],
            "train": train_metrics,
            "validation": validation,
            "training_pca_samples": len(pca_loader.dataset) if pca_loader is not None else 0,
            "probe_stabilization_active": epoch <= int(
                auxiliary_config.get("probe_stabilization_epochs", 0)
            ),
            "effective_pca_distance_loss_weight": effective_pca_distance_weight(
                auxiliary_config, epoch
            ) if auxiliary_enabled else 0.0,
            "training_pca_explained_variance": (
                {
                    name: pca_projections[name]
                    .explained_variance_ratio.detach()
                    .cpu()
                    .tolist()
                    for name in ATTRIBUTE_NAMES
                }
                if pca_projections is not None
                else None
            ),
        }
        if rank == 0:
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            if writer is not None:
                for section in ("train", "validation"):
                    for key in (
                        "total",
                        "coordinate",
                        "triplet",
                        "pairwise",
                        "prototype_compact",
                        "prototype_margin",
                        "encoder_total",
                        "auxiliary_total",
                        "auxiliary_classification",
                        "auxiliary_pca_distance",
                        "auxiliary_target_position",
                        "auxiliary_contrastive",
                    ):
                        writer.add_scalar(f"{section}/{key}", record[section][key], epoch)
                writer.add_scalar("train/learning_rate", record["learning_rate"], epoch)
            window_checkpoint_eligible = encoder_training_epoch > 0
            if (
                checkpoint_window
                and window_checkpoint_eligible
                and validation["total"] < window_best_validation
            ):
                window_best_validation = validation["total"]
                window_best_epoch = epoch
                window_best_step = global_step
                window_best_metrics = dict(validation)
                window_best_state = snapshot_model(encoder)
                window_best_geometry_state = (
                    snapshot_model(learnable_geometry)
                    if learnable_geometry is not None
                    else None
                )
            if bool(training_config.get("save_last_checkpoint", True)) and (
                epoch % last_checkpoint_interval == 0
                or epoch == int(training_config["epochs"])
            ):
                save_checkpoint(
                    output_dir / "last.pt",
                    encoder,
                    optimizer,
                    scheduler,
                    epoch,
                    global_step,
                    validation,
                    config,
                    True,
                    auxiliary_probes,
                    pca_projections,
                    learnable_geometry,
                )
            best_checkpoint_eligible = not bool(
                auxiliary_config.get(
                    "checkpoint_after_stabilization_only", False
                )
            ) or epoch > int(
                auxiliary_config.get("probe_stabilization_epochs", 0)
            )
            if best_checkpoint_eligible and validation["total"] < best_validation:
                best_validation = validation["total"]
                if defer_best_checkpoint:
                    deferred_best_state = snapshot_model(encoder)
                    deferred_best_geometry_state = (
                        snapshot_model(learnable_geometry)
                        if learnable_geometry is not None
                        else None
                    )
                    deferred_best_epoch = epoch
                    deferred_best_step = global_step
                    deferred_best_metrics = dict(validation)
                else:
                    save_checkpoint(
                        output_dir / "best.pt",
                        encoder,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        validation,
                        config,
                        False,
                        auxiliary_probes,
                        pca_projections,
                        learnable_geometry,
                    )
            if checkpoint_window and window_checkpoint_eligible and (
                encoder_training_epoch % checkpoint_window == 0
                or epoch == int(training_config["epochs"])
            ):
                if window_best_state is None or window_best_metrics is None:
                    raise RuntimeError("Checkpoint window ended without a model snapshot")
                window_start = (
                    (encoder_training_epoch - 1) // checkpoint_window
                ) * checkpoint_window + 1
                window_end = min(
                    window_start + checkpoint_window - 1,
                    int(training_config["epochs"]) - stabilization_epochs,
                )
                checkpoint_path = (
                    output_dir
                    / "checkpoints"
                    / f"window_{window_start:03d}_{window_end:03d}_best.pt"
                )
                save_model_snapshot(
                    checkpoint_path,
                    window_best_state,
                    window_best_epoch,
                    window_best_step,
                    window_best_metrics,
                    config,
                    window_best_geometry_state,
                )
                window_records.append(
                    {
                        "window_start": window_start,
                        "window_end": window_end,
                        "best_epoch": window_best_epoch,
                        "best_encoder_training_epoch": (
                            window_best_epoch - stabilization_epochs
                        ),
                        "global_step": window_best_step,
                        "validation": window_best_metrics,
                        "checkpoint": str(checkpoint_path.relative_to(output_dir)),
                    }
                )
                (output_dir / "checkpoint_windows.json").write_text(
                    json.dumps(window_records, indent=2), encoding="utf-8"
                )
                window_best_validation = float("inf")
                window_best_epoch = 0
                window_best_step = 0
                window_best_metrics = None
                window_best_state = None
                window_best_geometry_state = None
        if world_size > 1:
            dist.barrier()

    if rank == 0 and defer_best_checkpoint:
        if deferred_best_state is None or deferred_best_metrics is None:
            raise RuntimeError("Deferred best checkpoint has no model snapshot")
        save_model_snapshot(
            output_dir / "best.pt",
            deferred_best_state,
            deferred_best_epoch,
            deferred_best_step,
            deferred_best_metrics,
            config,
            deferred_best_geometry_state,
        )
    if world_size > 1:
        dist.barrier()

    if writer is not None:
        writer.close()
    if world_size > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
