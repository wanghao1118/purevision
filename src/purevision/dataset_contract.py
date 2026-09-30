from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


CONTRACT_VERSION = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_choices(choices: Any, owner: str) -> tuple[str, ...]:
    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError(f"{owner} 至少需要两个候选类别")
    ids = []
    for item in choices:
        if not isinstance(item, dict) or not item.get("id") or not item.get("text"):
            raise ValueError(f"{owner} 候选类别必须含 id 和 text")
        if " / " not in str(item.get("display_zh_en", "")):
            raise ValueError(f"{owner} 类别显示名必须中英并列")
        ids.append(str(item["id"]))
    if len(set(ids)) != len(ids):
        raise ValueError(f"{owner} 候选 ID 重复")
    return tuple(ids)


def validate_contract(contract: Mapping[str, Any]) -> None:
    if contract.get("schema_version") != CONTRACT_VERSION:
        raise ValueError("不支持的数据集规范版本")
    dataset = contract.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("缺少数据集身份与来源")
    for key in (
        "dataset_id", "release", "source_root", "label_provenance_zh",
        "split_manifest", "split_manifest_sha256", "disease_display_zh_en",
    ):
        if not dataset.get(key):
            raise ValueError(f"数据集来源缺少 {key}")
    if len(str(dataset["split_manifest_sha256"])) != 64:
        raise ValueError("划分清单 SHA-256 无效")
    if " / " not in str(dataset["disease_display_zh_en"]):
        raise ValueError("疾病类别显示名必须中英并列")
    anatomy = contract.get("anatomy")
    if not isinstance(anatomy, dict):
        raise ValueError("缺少解剖类别")
    labels = _check_choices(anatomy.get("choices"), "解剖")
    lesions = tuple(anatomy.get("lesion_labels", ()))
    if not lesions or any(label not in labels for label in lesions):
        raise ValueError("病灶类别必须属于解剖候选类别")
    if anatomy.get("mask_encoding") not in {"class_id", "independent_channel_bitset"}:
        raise ValueError("未知解剖掩码编码")
    dimensions = contract.get("phenotypes")
    if not isinstance(dimensions, list) or not dimensions:
        raise ValueError("至少需要一个表型维度")
    names = []
    for dimension in dimensions:
        name = str(dimension.get("id", ""))
        if not name or " / " not in str(dimension.get("display_zh_en", "")):
            raise ValueError("表型维度必须有 ID 和中英显示名")
        if " / " not in str(dimension.get("question_zh_en", "")):
            raise ValueError(f"表型题干必须中英并列：{name}")
        if dimension.get("kind") not in {"continuous", "ordinal", "categorical"}:
            raise ValueError(f"表型类型无效：{name}")
        _check_choices(dimension.get("choices"), name)
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("表型维度 ID 重复")
    grounding = contract.get("grounding")
    if not isinstance(grounding, dict) or type(grounding.get("grid_size")) is not int or grounding["grid_size"] < 2:
        raise ValueError("定位网格规格无效")
    for key in ("require_single_cell", "benchmark_grounding_eligible"):
        if key in grounding and not isinstance(grounding[key], bool):
            raise ValueError(f"定位规范 {key} 必须为布尔值")
    if grounding.get("benchmark_grounding_eligible", True):
        source = contract.get("source", {})
        if not isinstance(source, dict) or not (source.get("lesion_mask_field") or source.get("lesion_mask_template")):
            raise ValueError("可定位数据集缺少参考掩码字段")


def load_contract(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_contract(contract)
    return contract


def label_groups(contract: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    validate_contract(contract)
    groups = {
        "anatomy": tuple(str(item["id"]) for item in contract["anatomy"]["choices"])
    }
    groups.update({
        str(dimension["id"]): tuple(str(item["id"]) for item in dimension["choices"])
        for dimension in contract["phenotypes"]
    })
    return groups


def display_catalog(contract: Mapping[str, Any]) -> dict[str, Any]:
    validate_contract(contract)
    return {
        "疾病": contract["dataset"]["disease_display_zh_en"],
        "解剖": {
            item["id"]: item["display_zh_en"]
            for item in contract["anatomy"]["choices"]
        },
        "表型": {
            dimension["id"]: {
                "名称": dimension["display_zh_en"],
                "类别": {
                    item["id"]: item["display_zh_en"]
                    for item in dimension["choices"]
                },
            }
            for dimension in contract["phenotypes"]
        },
    }


def text_targets(contract: Mapping[str, Any]) -> dict[str, list[dict[str, str]]]:
    validate_contract(contract)
    groups = {"anatomy": contract["anatomy"]["choices"]}
    groups.update({dimension["id"]: dimension["choices"] for dimension in contract["phenotypes"]})
    return {
        name: [{"label": str(item["id"]), "text": str(item["text"])} for item in choices]
        for name, choices in groups.items()
    }


def validate_target_bank(contract: Mapping[str, Any], target_groups: Mapping[str, Any]) -> None:
    for name, expected in label_groups(contract).items():
        if name not in target_groups:
            raise ValueError(f"对齐 checkpoint 缺少词表组：{name}")
        actual = tuple(str(value) for value in target_groups[name])
        if actual != expected:
            raise ValueError(f"对齐 checkpoint 词表与数据集不一致：{name}")
