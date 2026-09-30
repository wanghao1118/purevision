from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from purevision.benchmark import normalized_test_records
from purevision.dataset_contract import display_catalog, load_contract, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    contract = load_contract(args.dataset_contract)
    records = normalized_test_records(contract)
    dataset = contract["dataset"]
    report = {
        "数据集ID": dataset["dataset_id"],
        "发布版": dataset["release"],
        "来源根目录": dataset["source_root"],
        "处理后根目录": dataset["processed_root"],
        "标签来源": dataset["label_provenance_zh"],
        "划分清单": dataset["split_manifest"],
        "划分清单SHA256": dataset["split_manifest_sha256"],
        "规范SHA256": sha256(args.dataset_contract),
        "类别中英对照": display_catalog(contract),
        "测试样本数": len(records),
        "测试患者数": len({row["patient_id"] for row in records}),
        "可定位参考样本数": sum(row["grounding_cell"] is not None for row in records),
        "定位参考掩码角色": contract["grounding"].get("lesion_mask_role"),
        "定位网格规则": "单格阳性筛选" if contract["grounding"].get("require_single_cell", True) else "阳性像素最多的网格；并列时按行列顺序取首格",
        "表型有效标签数": {
            dimension["id"]: dict(Counter(
                row["phenotypes"][dimension["id"]]
                for row in records if row["phenotypes"][dimension["id"]] is not None
            ))
            for dimension in contract["phenotypes"]
        },
        "核查说明": "仅验证清单、患者级划分、标签映射和定位网格规则；不进行模型训练或推理、mask 条件池化、监督质心评估、零样本分类，也不测量 t-SNE 簇间距离。",
    }
    result = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        if args.output.exists():
            raise FileExistsError("不覆盖既有核查记录")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    print(result)


if __name__ == "__main__":
    main()
