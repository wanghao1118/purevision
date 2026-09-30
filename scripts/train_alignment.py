from __future__ import annotations

import argparse
import json
import math
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from purevision.alignment import (
    AlignmentTargetBank,
    build_target_groups,
    group_alignment_loss,
    group_matching_cosine_loss,
    load_native_patch_aligner,
    pathology_alignment_loss,
    pathology_matching_cosine_loss,
)
from purevision.alignment_cache import (
    AnatomyAlignmentCache,
    LesionAlignmentCache,
)
from purevision.medgemma_text import (
    encode_text_descriptions,
    load_medgemma_text_encoder,
)
from purevision.protocol import validate_dataset_protocol, write_dataset_companion_zh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one shared MedGemma align layer for R39 anatomy and R30 pathology"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--matching-cosine-loss-weight", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--epochs", type=int)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def inference_autocast(device: torch.device):
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )


def cosine_schedule(step: int, *, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def load_or_build_target_bank(
    config: dict[str, Any], device: torch.device
) -> AlignmentTargetBank:
    paths = config["paths"]
    groups, texts = build_target_groups(config["text_targets"])
    cache_path = Path(paths["text_target_cache"])
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        if tuple(payload["texts"]) != texts:
            raise RuntimeError(
                "Text target cache was built from different descriptions; remove or rename it"
            )
        embeddings = payload["embeddings"].float()
    else:
        text_model, tokenizer = load_medgemma_text_encoder(paths["model"])
        embeddings = encode_text_descriptions(
            text_model,
            tokenizer,
            texts,
            device=device,
            batch_size=int(config["text_encoder"].get("batch_size", 16)),
            max_length=int(config["text_encoder"].get("max_length", 96)),
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "embeddings": embeddings,
                "texts": texts,
                "pooling": "last_non_padding_token",
                "model": str(paths["model"]),
            },
            cache_path,
        )
        del text_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return AlignmentTargetBank(embeddings, groups).to(device)


def make_loaders(
    paths: dict[str, Any], training: dict[str, Any]
) -> dict[str, DataLoader]:
    workers = int(training.get("num_workers", 0))
    loaders: dict[str, DataLoader] = {}
    for split in ("train", "val"):
        anatomy_batch_size = int(
            training["anatomy_batch_size"]
            if split == "train"
            else training.get("evaluation_batch_size", 4096)
        )
        lesion_batch_size = int(
            training["lesion_batch_size"]
            if split == "train"
            else training.get("evaluation_batch_size", 4096)
        )
        loaders[f"anatomy_{split}"] = DataLoader(
            AnatomyAlignmentCache(Path(paths["anatomy_cache"]) / split),
            batch_size=anatomy_batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=True,
            drop_last=split == "train",
        )
        loaders[f"lesion_{split}"] = DataLoader(
            LesionAlignmentCache(Path(paths["lesion_cache"]) / split),
            batch_size=lesion_batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=True,
            drop_last=split == "train",
        )
    return loaders


def balanced_class_weights(values: np.ndarray, classes: int) -> torch.Tensor:
    labels = np.asarray(values, dtype=np.int64).reshape(-1)
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    if len(counts) != classes or np.any(counts == 0):
        raise RuntimeError(f"Cannot balance classes with counts {counts.tolist()}")
    weights = labels.size / (classes * counts)
    weights /= weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def build_class_weights(
    loaders: dict[str, DataLoader],
    target_bank: AlignmentTargetBank,
    attribute_names: list[str],
    device: torch.device,
    anatomy_weight_source: str = "anatomy",
) -> dict[str, torch.Tensor]:
    anatomy_dataset = loaders["anatomy_train"].dataset
    lesion_dataset = loaders["lesion_train"].dataset
    _, anatomy_group = target_bank.group("anatomy")
    if anatomy_weight_source == "anatomy":
        anatomy_labels = np.asarray(anatomy_dataset.labels)
    elif anatomy_weight_source == "combined_anatomy_and_lesion":
        anatomy_labels = np.concatenate(
            (
                np.asarray(anatomy_dataset.labels, dtype=np.int64),
                np.asarray(lesion_dataset.anatomy_labels, dtype=np.int64),
            )
        )
    else:
        raise ValueError(f"Unknown anatomy class-weight source: {anatomy_weight_source}")
    result = {
        "anatomy": balanced_class_weights(
            anatomy_labels, anatomy_group.classes
        ).to(device)
    }
    for index, name in enumerate(attribute_names):
        _, group = target_bank.group(name)
        result[name] = balanced_class_weights(
            lesion_dataset.attribute_labels[:, index], group.classes
        ).to(device)
    return result


def apply_class_weight_multipliers(
    class_weights: dict[str, torch.Tensor],
    target_bank: AlignmentTargetBank,
    configured: dict[str, dict[str, float]] | None,
) -> dict[str, torch.Tensor]:
    if not configured:
        return class_weights
    adjusted = dict(class_weights)
    for group_name, label_multipliers in configured.items():
        if group_name not in adjusted:
            raise ValueError(f"Unknown class-weight group: {group_name}")
        _, group = target_bank.group(group_name)
        unknown = set(label_multipliers) - set(group.labels)
        if unknown:
            raise ValueError(
                f"Unknown labels for {group_name}: {sorted(unknown)}"
            )
        factors = torch.tensor(
            [float(label_multipliers.get(label, 1.0)) for label in group.labels],
            device=adjusted[group_name].device,
            dtype=adjusted[group_name].dtype,
        )
        if bool((factors <= 0).any()):
            raise ValueError("Class-weight multipliers must be positive")
        values = adjusted[group_name] * factors
        adjusted[group_name] = values / values.mean()
    return adjusted


def compute_batch_losses(
    aligner: torch.nn.Module,
    target_bank: AlignmentTargetBank,
    anatomy_batch: dict[str, torch.Tensor],
    lesion_batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    device: torch.device,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    training = config["training"]
    temperature = float(training["temperature"])
    anatomy_embedding = anatomy_batch["embedding"].to(device, non_blocking=True)
    anatomy_label = anatomy_batch["label"].to(device, non_blocking=True)
    lesion_anatomy = lesion_batch["anatomy_embedding"].to(device, non_blocking=True)
    lesion_pathology = lesion_batch["pathology_embedding"].to(
        device, non_blocking=True
    )
    lesion_anatomy_label = lesion_batch["anatomy_label"].to(
        device, non_blocking=True
    )
    attribute_labels = lesion_batch["attribute_labels"].to(
        device, non_blocking=True
    )

    aligned_anatomy = aligner(anatomy_embedding)
    aligned_lesion_anatomy = aligner(lesion_anatomy)
    aligned_lesion_pathology = aligner(lesion_pathology)
    anatomy_loss, _ = group_alignment_loss(
        aligned_anatomy,
        anatomy_label,
        target_bank,
        "anatomy",
        temperature=temperature,
        class_weights=None if class_weights is None else class_weights["anatomy"],
    )
    lesion_anatomy_loss, _ = group_alignment_loss(
        aligned_lesion_anatomy,
        lesion_anatomy_label,
        target_bank,
        "anatomy",
        temperature=temperature,
        class_weights=None if class_weights is None else class_weights["anatomy"],
    )
    pathology_loss, attribute_losses = pathology_alignment_loss(
        aligned_lesion_pathology,
        attribute_labels,
        target_bank,
        config["pathology_attribute_order"],
        temperature=temperature,
        attribute_weights=training.get("pathology_attribute_weights"),
        class_weights=class_weights,
    )
    anatomy_cosine_loss, _ = group_matching_cosine_loss(
        aligned_anatomy,
        anatomy_label,
        target_bank,
        "anatomy",
        class_weights=None if class_weights is None else class_weights["anatomy"],
    )
    lesion_anatomy_cosine_loss, _ = group_matching_cosine_loss(
        aligned_lesion_anatomy,
        lesion_anatomy_label,
        target_bank,
        "anatomy",
        class_weights=None if class_weights is None else class_weights["anatomy"],
    )
    pathology_cosine_loss, attribute_cosine_losses = pathology_matching_cosine_loss(
        aligned_lesion_pathology,
        attribute_labels,
        target_bank,
        config["pathology_attribute_order"],
        attribute_weights=training.get("pathology_attribute_weights"),
        class_weights=class_weights,
    )
    cosine_weight = float(training.get("matching_cosine_loss_weight", 0.0))
    if cosine_weight < 0:
        raise ValueError("matching_cosine_loss_weight must be non-negative")
    anatomy_objective = anatomy_loss + cosine_weight * anatomy_cosine_loss
    lesion_anatomy_objective = (
        lesion_anatomy_loss + cosine_weight * lesion_anatomy_cosine_loss
    )
    pathology_objective = pathology_loss + cosine_weight * pathology_cosine_loss
    total = (
        float(training.get("anatomy_loss_weight", 1.0)) * anatomy_objective
        + float(training.get("lesion_anatomy_loss_weight", 1.0))
        * lesion_anatomy_objective
        + float(training.get("pathology_loss_weight", 1.0)) * pathology_objective
    )
    components = {
        "total": total,
        "anatomy": anatomy_loss,
        "lesion_anatomy": lesion_anatomy_loss,
        "pathology": pathology_loss,
        "cosine_loss_anatomy": anatomy_cosine_loss,
        "cosine_loss_lesion_anatomy": lesion_anatomy_cosine_loss,
        "cosine_loss_pathology": pathology_cosine_loss,
        **{f"pathology_{name}": value for name, value in attribute_losses.items()},
        **{
            f"cosine_loss_pathology_{name}": value
            for name, value in attribute_cosine_losses.items()
        },
    }
    return total, components


@torch.no_grad()
def evaluate(
    aligner: torch.nn.Module,
    target_bank: AlignmentTargetBank,
    anatomy_loader: DataLoader,
    lesion_loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    class_weights: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    aligner.eval()
    training = config["training"]
    temperature = float(training["temperature"])
    anatomy_sum = 0.0
    anatomy_cosine_sum = 0.0
    anatomy_count = 0
    for batch in anatomy_loader:
        embedding = batch["embedding"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with inference_autocast(device):
            aligned = aligner(embedding)
            loss, _ = group_alignment_loss(
                aligned,
                labels,
                target_bank,
                "anatomy",
                temperature=temperature,
                class_weights=(
                    None if class_weights is None else class_weights["anatomy"]
                ),
            )
            cosine_loss, _ = group_matching_cosine_loss(
                aligned,
                labels,
                target_bank,
                "anatomy",
                class_weights=(
                    None if class_weights is None else class_weights["anatomy"]
                ),
            )
        anatomy_sum += float(loss.item()) * len(labels)
        anatomy_cosine_sum += float(cosine_loss.item()) * len(labels)
        anatomy_count += len(labels)

    lesion_sums: dict[str, float] = {}
    lesion_count = 0
    for batch in lesion_loader:
        anatomy_embedding = batch["anatomy_embedding"].to(
            device, non_blocking=True
        )
        pathology_embedding = batch["pathology_embedding"].to(
            device, non_blocking=True
        )
        anatomy_labels = batch["anatomy_label"].to(device, non_blocking=True)
        attribute_labels = batch["attribute_labels"].to(device, non_blocking=True)
        with inference_autocast(device):
            aligned_anatomy = aligner(anatomy_embedding)
            aligned_pathology = aligner(pathology_embedding)
            anatomy_loss, _ = group_alignment_loss(
                aligned_anatomy,
                anatomy_labels,
                target_bank,
                "anatomy",
                temperature=temperature,
                class_weights=(
                    None if class_weights is None else class_weights["anatomy"]
                ),
            )
            pathology_loss, attribute_losses = pathology_alignment_loss(
                aligned_pathology,
                attribute_labels,
                target_bank,
                config["pathology_attribute_order"],
                temperature=temperature,
                attribute_weights=training.get("pathology_attribute_weights"),
                class_weights=class_weights,
            )
            anatomy_cosine_loss, _ = group_matching_cosine_loss(
                aligned_anatomy,
                anatomy_labels,
                target_bank,
                "anatomy",
                class_weights=(
                    None if class_weights is None else class_weights["anatomy"]
                ),
            )
            pathology_cosine_loss, attribute_cosine_losses = (
                pathology_matching_cosine_loss(
                    aligned_pathology,
                    attribute_labels,
                    target_bank,
                    config["pathology_attribute_order"],
                    attribute_weights=training.get("pathology_attribute_weights"),
                    class_weights=class_weights,
                )
            )
        count = len(anatomy_labels)
        values = {
            "lesion_anatomy": anatomy_loss,
            "pathology": pathology_loss,
            "cosine_loss_lesion_anatomy": anatomy_cosine_loss,
            "cosine_loss_pathology": pathology_cosine_loss,
            **{f"pathology_{name}": value for name, value in attribute_losses.items()},
            **{
                f"cosine_loss_pathology_{name}": value
                for name, value in attribute_cosine_losses.items()
            },
        }
        for name, value in values.items():
            lesion_sums[name] = lesion_sums.get(name, 0.0) + float(value.item()) * count
        lesion_count += count
    if not anatomy_count or not lesion_count:
        raise RuntimeError("Validation alignment caches must both be non-empty")
    result = {
        "anatomy": anatomy_sum / anatomy_count,
        "cosine_loss_anatomy": anatomy_cosine_sum / anatomy_count,
        **{name: value / lesion_count for name, value in lesion_sums.items()},
    }
    cosine_weight = float(training.get("matching_cosine_loss_weight", 0.0))
    if cosine_weight < 0:
        raise ValueError("matching_cosine_loss_weight must be non-negative")
    result["total"] = (
        float(training.get("anatomy_loss_weight", 1.0))
        * (result["anatomy"] + cosine_weight * result["cosine_loss_anatomy"])
        + float(training.get("lesion_anatomy_loss_weight", 1.0))
        * (
            result["lesion_anatomy"]
            + cosine_weight * result["cosine_loss_lesion_anatomy"]
        )
        + float(training.get("pathology_loss_weight", 1.0))
        * (result["pathology"] + cosine_weight * result["cosine_loss_pathology"])
    )
    return result


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_dataset_protocol(config, verify_files=True)
    if args.output is not None:
        config["paths"]["output"] = str(args.output)
    if args.matching_cosine_loss_weight is not None:
        config["training"]["matching_cosine_loss_weight"] = (
            args.matching_cosine_loss_weight
        )
    if args.learning_rate is not None:
        config["training"]["learning_rate"] = args.learning_rate
    if args.temperature is not None:
        config["training"]["temperature"] = args.temperature
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    seed_everything(int(config["seed"]))
    if not torch.cuda.is_available():
        raise RuntimeError("Patch-text alignment training requires a CUDA device")
    device = torch.device(str(config.get("device", "cuda:0")))
    output = Path(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    write_dataset_companion_zh(output, config)
    (output / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    target_bank = load_or_build_target_bank(config, device)


    aligner = load_native_patch_aligner(
        config["paths"]["model"], dtype=torch.float32
    ).to(device)
    loaders = make_loaders(config["paths"], config["training"])
    attribute_names = list(config["pathology_attribute_order"])
    class_weights = (
        build_class_weights(
            loaders,
            target_bank,
            attribute_names,
            device,
            str(config["training"].get("anatomy_class_weight_source", "anatomy")),
        )
        if str(config["training"].get("class_weight", "balanced")) == "balanced"
        else None
    )
    if class_weights is not None:
        class_weights = apply_class_weight_multipliers(
            class_weights,
            target_bank,
            config["training"].get("class_weight_multipliers"),
        )

    training = config["training"]
    optimizer = torch.optim.AdamW(
        aligner.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    step_source = str(training.get("steps_per_epoch_source", "lesion"))
    if step_source not in {"anatomy", "lesion"}:
        raise ValueError("steps_per_epoch_source must be anatomy or lesion")
    steps_per_epoch = len(loaders[f"{step_source}_train"])
    if steps_per_epoch <= 0:
        raise RuntimeError(f"No complete batches in {step_source} training cache")
    total_steps = int(training["epochs"]) * steps_per_epoch
    warmup_steps = int(round(float(training["warmup_fraction"]) * total_steps))
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
        aligner.load_state_dict(checkpoint["aligner"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_loss = float(checkpoint["best_validation_loss"])

    log_path = output / "metrics.jsonl"
    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        aligner.train()
        anatomy_iterator = cycle(loaders["anatomy_train"])
        lesion_iterator = cycle(loaders["lesion_train"])
        sums: dict[str, float] = {}
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, components = compute_batch_losses(
                    aligner,
                    target_bank,
                    next(anatomy_iterator),
                    next(lesion_iterator),
                    config,
                    device,
                    class_weights,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                aligner.parameters(), float(training["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
            global_step += 1
            for name, value in components.items():
                sums[name] = sums.get(name, 0.0) + float(value.item())

        validation = evaluate(
            aligner,
            target_bank,
            loaders["anatomy_val"],
            loaders["lesion_val"],
            config,
            device,
            class_weights,
        )
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": scheduler.get_last_lr()[0],
            "train": {name: value / steps_per_epoch for name, value in sums.items()},
            "validation": validation,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        state = {
            "aligner": {key: value.detach().cpu() for key, value in aligner.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_validation_loss": min(best_loss, validation["total"]),
            "text_target_embeddings": target_bank.embeddings.detach().cpu(),
            "text_target_metadata": target_bank.metadata(),
            "class_weights": (
                {name: values.detach().cpu() for name, values in class_weights.items()}
                if class_weights is not None
                else None
            ),
            "config": config,
        }
        torch.save(state, output / "last.pt")
        if validation["total"] < best_loss:
            best_loss = validation["total"]
            state["best_validation_loss"] = best_loss
            torch.save(state, output / "best.pt")
        print(json.dumps(metrics), flush=True)
    (output / "training.complete").write_text("complete\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
