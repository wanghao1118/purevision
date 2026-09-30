from __future__ import annotations

import bisect
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image

from .dataset_contract import validate_contract


def field_value(row: Mapping[str, Any], field: str) -> Any:
    value: Any = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def source_rows(contract: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    root = Path(contract["dataset"]["processed_root"])
    source = contract["source"]
    path = root / source["manifest"]
    if source["format"] == "csv":
        with path.open(newline="", encoding="utf-8") as handle:
            yield from csv.DictReader(handle)
    elif source["format"] == "jsonl":
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
    else:
        raise ValueError("不支持的来源清单格式")


def resolve_path(root: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("来源清单包含数据集目录外路径")
    return path


def mapped_label(row: Mapping[str, Any], dimension: Mapping[str, Any]) -> str | None:
    value = field_value(row, str(dimension["source_field"]))
    if value is None or value == "":
        return None
    labels = [str(choice["id"]) for choice in dimension["choices"]]
    if "value_labels" in dimension:
        if not isinstance(value, list) or len(value) != len(dimension["value_labels"]):
            raise ValueError(f"多标签字段形状不匹配：{dimension['id']}")
        selected = sorted(
            str(dimension["value_labels"][index])
            for index, flag in enumerate(value) if flag
        )
        label = "+".join(selected) if selected else None
    elif "value_groups" in dimension:
        matches = [index for index, group in enumerate(dimension["value_groups"]) if int(value) in group]
        label = labels[matches[0]] if len(matches) == 1 else None
    elif "cut_points" in dimension:
        label = labels[bisect.bisect_left(dimension["cut_points"], float(value))]
    else:
        label = str(value)
    if label is not None and label not in labels:
        raise ValueError(f"来源标签不在候选词表中：{dimension['id']}={label}")
    return label


def grid_cell(mask_path: Path, grid_size: int) -> str | None:
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L")) > 0
    if not mask.any():
        return None
    rows, cols = np.nonzero(mask)
    cells = (rows * grid_size // mask.shape[0]) * grid_size + cols * grid_size // mask.shape[1]
    unique = np.unique(cells)
    if len(unique) != 1:
        return None
    index = int(unique[0])
    return f"r{index // grid_size + 1}c{index % grid_size + 1}"


def normalized_test_records(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    validate_contract(contract)
    source = contract["source"]
    root = Path(contract["dataset"]["processed_root"])
    patients: dict[str, str] = {}
    if source.get("split_from_patient_lists"):
        payload = json.loads(Path(contract["dataset"]["split_manifest"]).read_text(encoding="utf-8"))
        for split, members in payload["splits"].items():
            for patient in members:
                if patient in patients:
                    raise ValueError(f"患者跨划分重复：{patient}")
                patients[str(patient)] = str(split)
    seen_samples = set()
    records = []
    for row in source_rows(contract):
        sample_id = str(row["sample_id"])
        patient_id = str(field_value(row, source["patient_field"]))
        split = (
            patients.get(patient_id)
            if source.get("split_from_patient_lists")
            else str(field_value(row, source["split_field"]))
        )
        if split not in {"train", "val", "test"}:
            raise ValueError(f"无效患者划分：{sample_id}")
        old_split = patients.setdefault(patient_id, split)
        if old_split != split:
            raise ValueError(f"患者跨划分：{patient_id}")
        if sample_id in seen_samples:
            raise ValueError(f"重复样本：{sample_id}")
        seen_samples.add(sample_id)
        if split != "test":
            continue
        image_value = (
            source["image_template"].format(sample_id=sample_id)
            if "image_template" in source else field_value(row, source["image_field"])
        )
        mask_value = (
            source["lesion_mask_template"].format(sample_id=sample_id)
            if "lesion_mask_template" in source else field_value(row, source.get("lesion_mask_field", ""))
        )
        image = resolve_path(root, image_value)
        mask = resolve_path(root, mask_value)
        if image is None or not image.is_file():
            raise FileNotFoundError(f"缺少测试图像：{sample_id}")
        if mask is not None and not mask.is_file():
            raise FileNotFoundError(f"缺少病灶掩码：{sample_id}")
        eligible = contract["grounding"].get("benchmark_grounding_eligible", True)
        cell = grid_cell(mask, int(contract["grounding"]["grid_size"])) if eligible and mask else None
        records.append({
            "sample_id": sample_id,
            "patient_id": patient_id,
            "split": split,
            "image_path": str(image),
            "lesion_mask_path": str(mask) if mask else None,
            "grounding_cell": cell,
            "phenotypes": {
                dimension["id"]: mapped_label(row, dimension)
                for dimension in contract["phenotypes"]
            },
        })
    return records


def _balanced_sample(records: list[dict], target: int, key, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        label = key(record)
        if label is not None:
            buckets[str(label)].append(record)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    selected = []
    labels = sorted(buckets)
    while len(selected) < target and labels:
        for label in tuple(labels):
            if len(selected) >= target:
                break
            if buckets[label]:
                selected.append(buckets[label].pop())
            else:
                labels.remove(label)
    return selected


def build_questions(
    contract: Mapping[str, Any], records: list[dict], *,
    phenotype_target: int, grounding_target: int, seed: int,
) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    questions = []
    counts = {}
    grid_size = int(contract["grounding"]["grid_size"])
    cells = [f"r{row}c{col}" for row in range(1, grid_size + 1) for col in range(1, grid_size + 1)]
    if grounding_target and not contract["grounding"].get("benchmark_grounding_eligible", True):
        raise ValueError("此数据集没有可验证的病灶 mask，不能生成正式 grounding 题")
    selected = _balanced_sample(records, grounding_target, lambda row: row["grounding_cell"], seed)
    counts["grounding"] = len(selected)
    for row in selected:
        answer = row["grounding_cell"]
        options = rng.sample([cell for cell in cells if cell != answer], 3) + [answer]
        rng.shuffle(options)
        questions.append({
            "question_id": f"{row['sample_id']}:grounding",
            "sample_id": row["sample_id"], "task": "grounding",
            "question_zh_en": f"将图像均分为 {grid_size}×{grid_size} 网格，病灶位于哪个单元？ / Divide the image into a {grid_size}x{grid_size} grid. Which cell contains the lesion?",
            "answer": answer, "options": options,
            "option_display_zh_en": {
                cell: f"第 {cell[1:cell.index('c')]} 行第 {cell[cell.index('c') + 1:]} 列 / row {cell[1:cell.index('c')]}, column {cell[cell.index('c') + 1:]}"
                for cell in options
            },
        })
    for offset, dimension in enumerate(contract["phenotypes"], start=1):
        name = dimension["id"]
        choices = [choice["id"] for choice in dimension["choices"]]
        if phenotype_target and len(choices) < 4:
            raise ValueError(f"四选一题需要至少四个候选：{name}")
        selected = _balanced_sample(records, phenotype_target, lambda row: row["phenotypes"][name], seed + offset)
        counts[name] = len(selected)
        display = {choice["id"]: choice["display_zh_en"] for choice in dimension["choices"]}
        for row in selected:
            answer = row["phenotypes"][name]
            options = rng.sample([choice for choice in choices if choice != answer], 3) + [answer]
            rng.shuffle(options)
            questions.append({
                "question_id": f"{row['sample_id']}:{name}",
                "sample_id": row["sample_id"], "task": name,
                "question_zh_en": dimension["question_zh_en"],
                "answer": answer, "options": options,
                "option_display_zh_en": {choice: display[choice] for choice in options},
            })
    return questions, counts


def build_rrg_cases(contract: Mapping[str, Any], records: list[dict], target: int, seed: int) -> list[dict]:
    if target <= 0:
        return []
    candidates = (
        [record for record in records if record["grounding_cell"] is not None]
        if contract["grounding"].get("benchmark_grounding_eligible", True)
        else records
    )
    groups = [dimension["id"] for dimension in contract["phenotypes"]]
    def stratum(row: dict) -> str | None:
        for group in groups:
            label = row["phenotypes"][group]
            if label is not None:
                return f"{group}:{label}"
        return None
    return _balanced_sample(
        candidates, target,
        stratum,
        seed,
    )


def score_vqa(questions: list[dict], predictions: Mapping[str, Any], phenotype_groups: list[str]) -> dict:
    by_task = {}
    for task in ["grounding", *phenotype_groups]:
        selected = [question for question in questions if question["task"] == task]
        correct = sum(
            predictions.get(question["question_id"]) == question["answer"]
            for question in selected
        )
        by_task[task] = {
            "questions": len(selected),
            "correct": correct,
            "accuracy": correct / len(selected) if selected else None,
        }
    values = [by_task[name]["accuracy"] for name in phenotype_groups if by_task[name]["accuracy"] is not None]
    return {
        "grounding_accuracy": by_task["grounding"]["accuracy"],
        "phenotype_macro_accuracy": sum(values) / len(values) if values else None,
        "by_task": by_task,
    }


def score_rrg(references: list[dict], predictions: Mapping[str, Mapping[str, Any]], groups: list[str]) -> dict:
    grounding = [row for row in references if row.get("grid_cell") is not None]
    grounding_correct = sum(
        predictions.get(row["sample_id"], {}).get("grid_cell") == row["grid_cell"]
        for row in grounding
    )
    by_group = {}
    for group in (name for name in groups if name != "grid_cell"):
        rows = [row for row in references if row.get(group) is not None]
        classes = sorted({row[group] for row in rows})
        class_accuracy = {}
        for label in classes:
            matching = [row for row in rows if row[group] == label]
            correct = sum(predictions.get(row["sample_id"], {}).get(group) == label for row in matching)
            class_accuracy[label] = correct / len(matching)
        by_group[group] = {
            "samples": len(rows),
            "class_balanced_accuracy": sum(class_accuracy.values()) / len(class_accuracy) if classes else None,
            "class_accuracy": class_accuracy,
        }
    values = [item["class_balanced_accuracy"] for item in by_group.values() if item["class_balanced_accuracy"] is not None]
    return {
        "grounding_accuracy": grounding_correct / len(grounding) if grounding else None,
        "grounding_samples": len(grounding),
        "by_dimension": by_group,
        "phenotype_macro_accuracy": sum(values) / len(values) if values else None,
    }
