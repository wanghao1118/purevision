from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


def _map_text_checkpoint_key(full_key: str, target_keys: set[str]) -> str | None:
    prefixes = (
        "model.language_model.model.",
        "model.language_model.",
        "language_model.model.",
        "language_model.",
    )
    for prefix in prefixes:
        if full_key.startswith(prefix):
            candidate = full_key[len(prefix) :]
            if candidate in target_keys:
                return candidate
    return None


def load_medgemma_text_encoder(
    model_path: str | Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[nn.Module, object]:

    from safetensors import safe_open
    from transformers import AutoConfig, AutoTokenizer, Gemma3TextModel

    root = Path(model_path)
    full_config = AutoConfig.from_pretrained(root, local_files_only=True)
    model = Gemma3TextModel(full_config.text_config)
    target_keys = set(model.state_dict())
    index = json.loads(
        (root / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    mapped: dict[str, str] = {}
    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for full_key, shard in index["weight_map"].items():
        target_key = _map_text_checkpoint_key(full_key, target_keys)
        if target_key is not None:
            mapped[target_key] = full_key
            by_shard[shard].append((full_key, target_key))
    missing = sorted(target_keys - set(mapped))
    if missing:
        raise RuntimeError(
            f"MedGemma text checkpoint is missing {len(missing)} tensors; first: "
            f"{missing[:5]}"
        )

    state_dict: dict[str, torch.Tensor] = {}
    for shard, pairs in by_shard.items():
        with safe_open(root / shard, framework="pt", device="cpu") as handle:
            for full_key, target_key in pairs:
                state_dict[target_key] = handle.get_tensor(full_key)
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Text checkpoint mismatch: {incompatible}")
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model.requires_grad_(False).eval()
    return model.to(dtype=dtype), tokenizer


@torch.no_grad()
def encode_text_descriptions(
    model: nn.Module,
    tokenizer: object,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 96,
) -> torch.Tensor:

    if not texts:
        raise ValueError("At least one text description is required")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    model = model.to(device)
    outputs: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        hidden = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        positions = torch.arange(hidden.shape[1], device=device)[None, :]
        last_indices = positions.masked_fill(attention_mask.eq(0), -1).argmax(dim=1)
        batch_indices = torch.arange(hidden.shape[0], device=device)
        pooled = hidden[batch_indices, last_indices]
        outputs.append(F.normalize(pooled.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def encode_text_token_sequences(
    model: nn.Module,
    tokenizer: object,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 96,
) -> tuple[list[torch.Tensor], list[list[int]]]:









    if not texts or any(not str(text) for text in texts):
        raise ValueError("At least one non-empty text description is required")
    if batch_size <= 0 or max_length <= 1:
        raise ValueError("batch_size must be positive and max_length must exceed one")
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is None:
        raise ValueError("The tokenizer must define a BOS token")

    model = model.to(device)
    sequences: list[torch.Tensor] = []
    token_ids: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        lengths = attention_mask.sum(dim=1)
        if int(lengths.max()) > max_length:
            raise ValueError(
                "A text description exceeds max_length; truncation is forbidden"
            )
        hidden = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        for row in range(len(input_ids)):
            valid = attention_mask[row].bool()
            ids = input_ids[row, valid]
            values = hidden[row, valid.to(device)]
            if len(ids) < 2 or int(ids[0]) != int(bos_token_id):
                raise ValueError(
                    "Expected one leading BOS followed by source-text tokens"
                )
            ids = ids[1:]
            values = F.normalize(values[1:].float(), dim=-1).cpu()
            if len(values) != len(ids):
                raise RuntimeError("Text token IDs and hidden states became misaligned")
            sequences.append(values)
            token_ids.append([int(value) for value in ids.tolist()])
    return sequences, token_ids


@torch.no_grad()
def encode_input_embedding_sequences(
    model: nn.Module,
    tokenizer: object,
    texts: Sequence[str],
    *,
    device: torch.device,
    batch_size: int = 16,
    max_length: int = 96,
) -> tuple[list[torch.Tensor], list[list[int]]]:

    if not texts or any(not str(text) for text in texts):
        raise ValueError("At least one non-empty text description is required")
    if batch_size <= 0 or max_length <= 0:
        raise ValueError("batch_size and max_length must be positive")
    embeddings = model.get_input_embeddings().to(device)
    sequences: list[torch.Tensor] = []
    token_ids: list[list[int]] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            add_special_tokens=False,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        lengths = attention_mask.sum(dim=1)
        if int(lengths.max()) > max_length:
            raise ValueError(
                "A text description exceeds max_length; truncation is forbidden"
            )
        hidden = embeddings(input_ids.to(device))
        for row in range(len(input_ids)):
            valid = attention_mask[row].bool()
            ids = input_ids[row, valid]
            values = hidden[row, valid.to(device)].float().cpu()
            if not len(ids):
                raise ValueError("A text description produced no input tokens")
            sequences.append(values)
            token_ids.append([int(value) for value in ids.tolist()])
    return sequences, token_ids
