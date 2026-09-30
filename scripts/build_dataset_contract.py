from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from purevision.dataset_contract import sha256, validate_contract


def source_choices(root: Path, source: dict) -> list[dict]:
    if source["kind"] == "observed_multilabel":
        labels = tuple(source["labels"])
        manifest = root / source["path"]
        observed = set()
        with manifest.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                values = row["source_labels"]["multilabel"].get(source["group"])
                if values is None:
                    continue
                if len(values) != len(labels):
                    raise ValueError(f"多标签长度不匹配：{source['group']}")
                selected = tuple(sorted(labels[index] for index, flag in enumerate(values) if flag))
                if selected:
                    observed.add(selected)
        return [
            {
                "id": "+".join(selected),
                "text": " and ".join(source["text_en"][label] for label in selected),
                "display_zh_en": "、".join(source["text_zh"][label] for label in selected)
                + " / " + " and ".join(source["text_en"][label] for label in selected),
            }
            for selected in sorted(observed)
        ]
    if source["kind"] == "alignment_config":
        config_path = Path(__file__).resolve().parents[1] / source["path"]
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return [
            {
                "id": str(item["label"]),
                "text": str(item["text"]),
                "display_zh_en": source["display_zh_en"][str(item["label"])],
            }
            for item in payload["text_targets"][source["group"]]
        ]
    payload = json.loads((root / source["path"]).read_text(encoding="utf-8"))
    if source["kind"] == "knee_dimension":
        dimension = next(
            item for item in payload["attributes"] if item["group"] == source["group"]
        )
        return [
            {
                "id": str(choice["label"]),
                "text": str(choice["text"]),
                "display_zh_en": source["display_zh_en"][str(choice["label"])],
            }
            for choice in dimension["choices"]
        ]
    if source["kind"] == "knee_anatomy":
        return [
            {
                "id": str(channel["npz_key"]),
                "text": str(channel["name"]),
                "display_zh_en": source["display_zh_en"][str(channel["npz_key"])],
                "bit_value": int(channel["bit_value"]),
            }
            for channel in payload["channels"]
        ]
    raise ValueError(f"未知来源类别类型：{source['kind']}")


def build(recipe: dict, root: Path) -> dict:
    if not root.is_dir():
        raise FileNotFoundError(root)
    dataset = dict(recipe["dataset"])
    split_manifest = root / recipe["source"]["split_manifest"]
    if not split_manifest.is_file():
        raise FileNotFoundError(split_manifest)
    dataset.update({
        "processed_root": str(root.resolve()),
        "split_manifest": str(split_manifest.resolve()),
        "split_manifest_sha256": sha256(split_manifest),
    })
    anatomy = dict(recipe["anatomy"])
    if "choices_source" in anatomy:
        anatomy["choices"] = source_choices(root, anatomy.pop("choices_source"))
    phenotypes = []
    for dimension in recipe["phenotypes"]:
        value = dict(dimension)
        if "choices_source" in value:
            choice_source = value.pop("choices_source")
            value["choices"] = source_choices(root, choice_source)
            if choice_source["kind"] == "observed_multilabel":
                value["value_labels"] = choice_source["labels"]
        phenotypes.append(value)
    contract = {
        "schema_version": 1,
        "dataset": dataset,
        "source": recipe["source"],
        "anatomy": anatomy,
        "phenotypes": phenotypes,
        "grounding": recipe["grounding"],
        "construction_stage_zh": recipe["construction_stage_zh"],
    }
    validate_contract(contract)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    recipe = yaml.safe_load(args.recipe.read_text(encoding="utf-8"))
    contract = build(recipe, args.dataset_root)
    output = args.output or args.dataset_root / "dataset_contract.json"
    if output.exists():
        raise FileExistsError(f"规范已存在，不覆盖冻结记录：{output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"数据集规范已生成：{output}")


if __name__ == "__main__":
    main()
