from __future__ import annotations

from pathlib import Path

import torch
import yaml

from purevision.losses import (
    attribute_relation_losses,
    continuous_attribute_relation_loss,
)
from purevision.protocol import validate_dataset_protocol
from purevision.protocol import validate_weight_protocol


ROOT = Path(__file__).resolve().parents[1]


def read_config(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text(encoding="utf-8"))


def nested_strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from nested_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from nested_strings(item)
    elif isinstance(value, str):
        yield value


def test_continuous_size_geometry_matches_equation() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    measurements = torch.tensor([0.0, 1.0])
    loss = continuous_attribute_relation_loss(
        embeddings,
        measurements,
        memory_embeddings=None,
        memory_measurements=None,
        distance_scale=1.5,
    )
    learned = torch.sqrt(torch.tensor(2.0))
    expected = torch.nn.functional.smooth_l1_loss(learned, torch.tensor(1.5))
    assert torch.allclose(loss, expected)


def test_categorical_geometry_has_no_ordinal_rank_term() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]])
    rank, pair, count = attribute_relation_losses(
        embeddings,
        torch.tensor([0, 0, 1]),
        torch.tensor([1.0, 2.0]),
        memory_embeddings=None,
        memory_labels=None,
        ordinal=False,
        maximum_distance=1.5,
        order_margin=0.1,
    )
    assert rank.item() == 0.0
    assert pair.item() >= 0.0
    assert count == 0


def test_training_and_inference_configs_match_paper_constants() -> None:
    local = read_config("01_phenotype_local.yaml")
    global_config = read_config("02_phenotype_global.yaml")
    anatomy = read_config("04_anatomy_final.yaml")
    alignment = read_config("05_alignment.yaml")
    inference = read_config("06_inference.yaml")
    for config in (local, global_config, anatomy, alignment, inference):
        validate_dataset_protocol(config)

    for config in (local, global_config):
        training = config["training"]
        assert training["learning_rate"] == 5e-7
        assert training["epochs"] == 30
        assert training["memory_bank_capacity"] == 1024
        size = next(
            task
            for task in training["attribute_relation_supervision"]["tasks"]
            if task["target_attribute"] == "size"
        )
        assert size["kind"] == "continuous"
        assert size["distance_scale"] == 1.5

    anatomy_training = anatomy["training"]
    assert anatomy_training["anatomy_replay_loss_weight"] == 4.0
    assert anatomy_training["parent_replay_loss_weight"] == 0.55
    assert anatomy_training["hierarchy_loss_weight"] == 4.0
    assert anatomy_training["distillation_loss_weight"] == 10.0
    assert anatomy_training["vessel_lung_loss_weight"] == 1.0
    assert anatomy_training["vessel_lung_improvement_margin"] == 0.01
    assert anatomy_training["vessel_anchor_contrastive_weight"] == 0.05
    assert anatomy_training["vessel_anchor_contrastive_temperature"] == 0.1

    assert alignment["align_layer"]["shared_between_encoders"] is True
    assert alignment["training"]["temperature"] == 0.07
    assert alignment["training"]["learning_rate"] == 2e-5
    assert alignment["training"]["epochs"] == 30

    inference_config = inference["inference"]
    assert inference_config["window_size"] == 5
    assert inference_config["selected_patches"] == 8
    assert inference_config["semantic_temperature"] == 0.125
    assert inference_config["native_visual_tokens_retained"] is True
    assert inference_config["inference_masks"] is False
    assert inference_config["lora"] is False
    lowered = " ".join(nested_strings(inference)).lower()
    assert "hard_centroid" not in lowered
    assert "tsne" not in lowered


def test_reproduction_smoke_binds_medgemma15_and_frozen_checkpoints() -> None:
    config = read_config("07_medgemma15_reproduction_smoke.yaml")
    protocol = validate_weight_protocol(
        config,
        required_roles=(
            "backbone_config",
            "backbone_index",
            "phenotype_checkpoint",
            "anatomy_checkpoint",
            "alignment_checkpoint",
        ),
    )
    assert protocol["backbone_id"] == "google/medgemma-1.5-4b-it"
    assert protocol["files"]["phenotype_checkpoint"]["sha256"].startswith("1a98")
    assert protocol["files"]["anatomy_checkpoint"]["sha256"].startswith("c8df")
    assert protocol["files"]["alignment_checkpoint"]["sha256"].startswith("4901")
    assert config["comparison_protocol"]["smoke_only"] is True
    assert config["comparison_protocol"]["full_benchmark_claim"] is False
