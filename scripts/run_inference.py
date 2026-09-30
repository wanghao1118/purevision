from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers
import yaml
from PIL import Image

from purevision.pipeline import PureVisionPipeline
from purevision.protocol import (
    sha256_file,
    validate_dataset_protocol,
    validate_weight_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="运行论文一致的 PureVision R60 软融合与冻结解码流程"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--instruction")
    prompt.add_argument("--instruction-file", type=Path)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--embedding-output",
        type=Path,
        help="保存对齐前的表型与解剖 patch embedding，供审计复核",
    )
    parser.add_argument("--sample-id")
    parser.add_argument("--split", choices=("train", "val", "test"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    dataset = validate_dataset_protocol(config, verify_files=True)
    weights = validate_weight_protocol(
        config,
        required_roles=(
            "backbone_config",
            "backbone_index",
            "phenotype_checkpoint",
            "anatomy_checkpoint",
            "alignment_checkpoint",
        ),
        verify_files=True,
    )
    instruction = (
        args.instruction
        if args.instruction is not None
        else args.instruction_file.read_text(encoding="utf-8")
    )
    runtime = config.get("runtime", {})
    device = torch.device(runtime.get("device", "cuda:0"))
    pipeline = PureVisionPipeline(config)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with Image.open(args.image) as opened:
        result = pipeline.run(
            opened.convert("RGB"),
            instruction,
            max_new_tokens=args.max_new_tokens,
        )
    selection = result.fusion.selection
    peak_gpu_mib = (
        round(torch.cuda.max_memory_allocated(device) / 1024 / 1024, 2)
        if device.type == "cuda"
        else None
    )
    embedding_artifact = None
    if args.embedding_output is not None:
        args.embedding_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": 1,
                "sample_id": args.sample_id,
                "dataset_id": dataset["dataset_id"],
                "dataset_release": dataset["release"],
                "split_manifest_sha256": dataset["split_manifest_sha256"],
                "phenotype_patch_embeddings": result.phenotype_patch_embeddings,
                "anatomy_patch_embeddings": result.anatomy_patch_embeddings,
                "说明_zh": (
                    "两个张量均为独立视觉塔输出、共享对齐器之前的原始 patch embedding；"
                    "形状为 [4096, 1152]。"
                ),
            },
            args.embedding_output,
        )
        embedding_artifact = {
            "path": str(args.embedding_output.resolve()),
            "sha256": sha256_file(args.embedding_output),
            "phenotype_shape": list(result.phenotype_patch_embeddings.shape),
            "anatomy_shape": list(result.anatomy_patch_embeddings.shape),
            "stage": "共享对齐器之前",
        }
    record = {
        "schema_version": 2,
        "记录类型": "PureVision R60 推理结果",
        "实验编号": str(config.get("experiment_id", "未设置")),
        "dataset_id": dataset["dataset_id"],
        "dataset_release": dataset["release"],
        "source_root": dataset["source_root"],
        "label_provenance_zh": dataset["label_provenance_zh"],
        "split_manifest": dataset["split_manifest"],
        "split_manifest_sha256": dataset["split_manifest_sha256"],
        "sample_id": args.sample_id,
        "sample_split": args.split,
        "类别中英对照": config.get("display_names_zh_en"),
        "权重协议": weights,
        "已加载权重角色": [
            "backbone_config",
            "backbone_index",
            "phenotype_checkpoint",
            "anatomy_checkpoint",
            "alignment_checkpoint",
        ],
        "流程说明_zh": (
            "MedGemma 1.5 原始预训练权重分别初始化表型与解剖视觉塔；完整图像经已训练的"
            "独立表型视觉编码器和解剖视觉编码器提取；共享对齐器定位；"
            "5x5 局部候选内选取 8 个 patch；按病灶分数和组内标准化概率执行软语义融合；"
            "保留原始视觉 token，并在任务指令前插入融合 token；冻结 MedGemma 1.5 "
            "decoder 生成答案。该流程不是标准未修改推理、mask-conditioned pooling、"
            "监督质心评估或 hard-centroid 分类。"
        ),
        "inference_masks_used": False,
        "native_visual_tokens_retained": True,
        "lora_used": False,
        "image_path": str(args.image.resolve()),
        "image_sha256": sha256_file(args.image),
        "instruction": instruction,
        "selected_patch_indices": list(selection.patch_indices),
        "selected_patch_weights": list(selection.patch_weights),
        "estimated_grid_location": {
            "row": selection.center_row,
            "column": selection.center_column,
        },
        "semantic_probabilities": result.semantic_probabilities(),
        "原始特征证据": embedding_artifact,
        "generated_text": result.text,
        "runtime": {
            "device": str(device),
            "dtype": str(runtime.get("dtype", "bfloat16")),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_gpu_mib": peak_gpu_mib,
        },
    }
    serialized = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is None:
        print(serialized)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"推理结果已写入：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
