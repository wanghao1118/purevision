from __future__ import annotations

import argparse
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from purevision.anatomy import ANATOMY_NAMES, regular_simplex_targets
from purevision.anatomy_dataset import AnatomyPatchDataset, collate_anatomy_batch
from purevision.anatomy_model import (
    build_anatomy_patch_encoder,
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
        description="Fine-tune MedGemma for patch anatomy geometry"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
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


def cosine_schedule(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def balanced_patch_indices(
    labels: torch.Tensor,
    *,
    classes: int,
    per_class: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, patches]")
    batch, patches = labels.shape
    selected: list[torch.Tensor] = []
    for batch_index in range(batch):
        for class_index in range(classes):
            local = torch.nonzero(
                labels[batch_index].eq(class_index), as_tuple=False
            ).flatten()
            if local.numel() == 0:
                continue
            if local.numel() > per_class:
                if generator is None:
                    positions = (
                        torch.linspace(
                            0,
                            local.numel() - 1,
                            per_class,
                            device=labels.device,
                        )
                        .round()
                        .long()
                    )
                    local = local[positions]
                else:
                    order = torch.randperm(
                        local.numel(), device=labels.device, generator=generator
                    )[:per_class]
                    local = local[order]
            selected.append(batch_index * patches + local)
    if not selected:
        raise RuntimeError("No labeled patches were selected")
    return torch.cat(selected)


def geometry_loss(
    normalized_tokens: torch.Tensor,
    labels: torch.Tensor,
    prototypes: torch.Tensor,
    memory_bank: GeometryMemoryBank,
    training: dict[str, Any],
    *,
    classes: int,
    generator: torch.Generator | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    selected = balanced_patch_indices(
        labels,
        classes=classes,
        per_class=int(training["patches_per_class_per_slice"]),
        generator=generator,
    )
    flat_tokens = normalized_tokens.reshape(-1, normalized_tokens.shape[-1])
    flat_labels = labels.reshape(-1)
    embeddings = flat_tokens[selected]
    selected_labels = flat_labels[selected]
    targets = prototypes[selected_labels]
    memory_embeddings, memory_targets = memory_bank.tensors()
    triplet, pairwise, triplets = dynamic_geometry_losses(
        embeddings,
        targets,
        memory_embeddings=memory_embeddings,
        memory_targets=memory_targets,
        triplet_margin=float(training["triplet_margin"]),
        ideal_gap_min=float(training["ideal_gap_min"]),
        triplets_per_anchor=int(training["triplets_per_anchor"]),
        generator=generator,
    )
    contrastive_weight = float(training.get("supervised_contrastive_loss_weight", 0.0))
    if contrastive_weight:
        memory_labels = None if memory_targets is None else memory_targets.argmax(dim=1)
        contrastive = supervised_contrastive_loss(
            embeddings,
            selected_labels,
            memory_embeddings=memory_embeddings,
            memory_labels=memory_labels,
            temperature=float(training.get("supervised_contrastive_temperature", 0.1)),
        )
    else:
        contrastive = embeddings.sum() * 0.0
    total = (
        float(training["triplet_loss_weight"]) * triplet
        + float(training["pairwise_loss_weight"]) * pairwise
        + contrastive_weight * contrastive
    )
    memory_bank.add(embeddings, targets)
    return total, triplet, pairwise, contrastive, triplets, int(selected.numel())


def reduce_sums(values: torch.Tensor, world_size: int) -> torch.Tensor:
    if world_size > 1:
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return values


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    prototypes: torch.Tensor,
    training: dict[str, Any],
    device: torch.device,
    world_size: int,
    classes: int,
) -> dict[str, float]:
    model.eval()
    memory = GeometryMemoryBank(capacity=int(training["memory_bank_capacity"]))
    totals = torch.zeros(6, dtype=torch.float64, device=device)
    for batch in loader:
        pixels = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["patch_labels"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(pixels)
            if outputs["normalized_embedding"].shape[:2] != labels.shape:
                raise RuntimeError(
                    "Vision token grid and pseudo-label grid are not aligned: "
                    f"{outputs['normalized_embedding'].shape[:2]} vs {labels.shape}"
                )
            loss, triplet, pairwise, contrastive, triplets, selected = geometry_loss(
                outputs["normalized_embedding"],
                labels,
                prototypes,
                memory,
                training,
                classes=classes,
                generator=None,
            )
        totals += torch.tensor(
            [
                float(loss.item()) * selected,
                float(triplet.item()) * selected,
                float(pairwise.item()) * selected,
                float(contrastive.item()) * selected,
                float(selected),
                float(triplets),
            ],
            dtype=torch.float64,
            device=device,
        )
    totals = reduce_sums(totals, world_size)
    denominator = totals[4].clamp_min(1.0)
    return {
        "loss": float((totals[0] / denominator).item()),
        "triplet": float((totals[1] / denominator).item()),
        "pairwise": float((totals[2] / denominator).item()),
        "contrastive": float((totals[3] / denominator).item()),
        "selected_patches": int(totals[4].item()),
        "triplets": int(totals[5].item()),
    }


def checkpoint_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    global_step: int,
    config: dict[str, Any],
    validation: dict[str, float],
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
    }


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_dataset_protocol(config, verify_files=True)
    rank, local_rank, world_size = setup_distributed()
    seed_everything(int(config["seed"]), rank)
    device = torch.device("cuda", local_rank)
    data = config["data"]
    training = config["training"]
    anatomy_names = tuple(
        data.get("anatomy_names", data.get("anatomy_classes", ANATOMY_NAMES))
    )
    if len(anatomy_names) < 2 or len(set(anatomy_names)) != len(anatomy_names):
        raise ValueError("data.anatomy_names must contain at least two unique names")
    source_label_values = data.get("source_label_values")
    if source_label_values is not None and len(source_label_values) != len(
        anatomy_names
    ):
        raise ValueError("data.source_label_values must align with anatomy names")
    output = Path(config["paths"]["output"])
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_dataset_companion_zh(output, config)
        (output / "config.yaml").write_text(
            args.config.read_text(encoding="utf-8"), encoding="utf-8"
        )

    train_dataset = AnatomyPatchDataset(
        config["paths"]["processed"],
        "train",
        image_size=int(data["image_size"]),
        patch_size=int(data["patch_size"]),
        manifest_name=str(data.get("manifest_name", "metadata.csv")),
        image_cache_root=data.get("image_cache_root"),
        source_label_values=source_label_values,
    )
    val_dataset = AnatomyPatchDataset(
        config["paths"]["processed"],
        "val",
        image_size=int(data["image_size"]),
        patch_size=int(data["patch_size"]),
        manifest_name=str(data.get("manifest_name", "metadata.csv")),
        image_cache_root=data.get("image_cache_root"),
        source_label_values=source_label_values,
    )
    train_sampler = (
        DistributedSampler(train_dataset, world_size, rank, shuffle=True)
        if world_size > 1
        else None
    )
    val_sampler = (
        DistributedSampler(val_dataset, world_size, rank, shuffle=False)
        if world_size > 1
        else None
    )
    loader_kwargs = {
        "batch_size": int(training["per_device_batch_size"]),
        "num_workers": int(training["num_workers"]),
        "pin_memory": True,
        "collate_fn": collate_anatomy_batch,
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        sampler=val_sampler,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    model = build_anatomy_patch_encoder(
        config["paths"]["model"], dtype=torch.bfloat16, gradient_checkpointing=True
    )
    parameter_counts = set_trainable_anatomy_layers(
        model, training.get("trainable_vision_layers")
    )
    parameter_dtype = str(training.get("parameter_dtype", "bfloat16"))
    if parameter_dtype not in {"float32", "bfloat16"}:
        raise ValueError("training.parameter_dtype must be float32 or bfloat16")
    model.to(device=device, dtype=getattr(torch, parameter_dtype))
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
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
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
        if (output / "best.pt").exists():
            previous_best = torch.load(output / "best.pt", map_location="cpu", weights_only=False, mmap=True)
            best_loss = min(best_loss, float(previous_best["validation"]["loss"]))
            del previous_best

    prototypes = torch.from_numpy(regular_simplex_targets(len(anatomy_names))).to(
        device
    )
    memory = GeometryMemoryBank(capacity=int(training["memory_bank_capacity"]))
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + rank)
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        memory.clear()
        totals = torch.zeros(6, dtype=torch.float64, device=device)
        remainder = len(train_loader) % accumulation
        for batch_index, batch in enumerate(train_loader, start=1):
            pixels = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["patch_labels"].to(device, non_blocking=True)
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
                    outputs = model(pixels)
                    if outputs["normalized_embedding"].shape[:2] != labels.shape:
                        raise RuntimeError(
                            "Vision token grid and pseudo-label grid are not aligned: "
                            f"{outputs['normalized_embedding'].shape[:2]} vs {labels.shape}"
                        )
                    loss, triplet, pairwise, contrastive, triplets, selected = (
                        geometry_loss(
                            outputs["normalized_embedding"],
                            labels,
                            prototypes,
                            memory,
                            training,
                            classes=len(anatomy_names),
                            generator=generator,
                        )
                    )
                    accumulation_divisor = (
                        remainder
                        if remainder and batch_index > len(train_loader) - remainder
                        else accumulation
                    )
                    scaled_loss = loss / accumulation_divisor
                scaled_loss.backward()
            totals += torch.tensor(
                [
                    float(loss.item()) * selected,
                    float(triplet.item()) * selected,
                    float(pairwise.item()) * selected,
                    float(contrastive.item()) * selected,
                    float(selected),
                    float(triplets),
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
                log_interval = int(training.get("log_interval_steps", 50))
                if rank == 0 and global_step % log_interval == 0:
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

        totals = reduce_sums(totals, world_size)
        denominator = totals[4].clamp_min(1.0)
        train_metrics = {
            "loss": float((totals[0] / denominator).item()),
            "triplet": float((totals[1] / denominator).item()),
            "pairwise": float((totals[2] / denominator).item()),
            "contrastive": float((totals[3] / denominator).item()),
            "selected_patches": int(totals[4].item()),
            "triplets": int(totals[5].item()),
        }
        validation = evaluate(
            model,
            val_loader,
            prototypes,
            training,
            device,
            world_size,
            len(anatomy_names),
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
                model, optimizer, scheduler, epoch, global_step, config, validation
            )
            torch.save(state, output / "last.pt")
            if validation["loss"] < best_loss:
                best_loss = validation["loss"]
                torch.save(state, output / "best.pt")
            (output / "status.json").write_text(
                json.dumps(
                    {
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_validation_loss": best_loss,
                        "parameter_counts": parameter_counts,
                        "world_size": world_size,
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
