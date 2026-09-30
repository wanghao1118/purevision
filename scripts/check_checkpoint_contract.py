from __future__ import annotations

import argparse
import json
from pathlib import Path

from purevision.dataset_contract import display_catalog, load_contract, sha256, validate_target_bank
from purevision.pipeline import load_alignment_target_bank


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--alignment-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.dataset_contract)
    bank = load_alignment_target_bank(args.alignment_checkpoint, device="cpu")
    validate_target_bank(contract, {group.name: group.labels for group in bank.groups})
    dataset = contract["dataset"]
    result = {
        "核查状态": "通过",
        "数据集ID": dataset["dataset_id"],
        "发布版": dataset["release"],
        "来源根目录": dataset["source_root"],
        "标签来源": dataset["label_provenance_zh"],
        "划分清单": dataset["split_manifest"],
        "划分清单SHA256": dataset["split_manifest_sha256"],
        "规范SHA256": sha256(args.dataset_contract),
        "对齐checkpointSHA256": sha256(args.alignment_checkpoint),
        "候选组": {group.name: len(group.labels) for group in bank.groups},
        "类别中英对照": display_catalog(contract),
        "核查说明": "仅检查冻结文本目标的类别 ID 与顺序；不执行原始模型标准推理、mask 条件池化、监督质心评估或零样本分类，也不使用 t-SNE 距离。",
    }
    if args.output:
        if args.output.exists():
            raise FileExistsError("不覆盖既有核查记录")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
