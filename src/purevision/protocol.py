from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_file(path: str | Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def validate_dataset_protocol(
    config: Mapping[str, Any], *, verify_files: bool = False
) -> dict[str, Any]:
    protocol = config.get("dataset")
    if not isinstance(protocol, dict):
        raise ValueError("config must contain a dataset protocol mapping")
    required = (
        "dataset_id",
        "release",
        "source_root",
        "label_provenance_zh",
        "split_manifest",
        "split_manifest_sha256",
    )
    missing = [key for key in required if not str(protocol.get(key, "")).strip()]
    if missing:
        raise ValueError(f"dataset protocol is missing required fields: {missing}")
    split_keys = tuple(protocol.get("split_keys", ()))
    if split_keys != ("train", "val", "test"):
        raise ValueError("dataset.split_keys must be [train, val, test]")
    expected_hash = str(protocol["split_manifest_sha256"]).lower()
    if len(expected_hash) != 64 or any(
        character not in "0123456789abcdef" for character in expected_hash
    ):
        raise ValueError("dataset.split_manifest_sha256 must be a SHA-256 digest")
    if verify_files:
        manifest = Path(protocol["split_manifest"])
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        actual_hash = sha256_file(manifest)
        if actual_hash != expected_hash:
            raise ValueError(
                f"split manifest hash mismatch: expected {expected_hash}, "
                f"found {actual_hash}"
            )
    return dict(protocol)


def validate_weight_protocol(
    config: Mapping[str, Any],
    *,
    required_roles: tuple[str, ...] = (),
    verify_files: bool = False,
) -> dict[str, Any]:
    protocol = config.get("weights")
    if not isinstance(protocol, dict):
        raise ValueError("config must contain a weights protocol mapping")
    required = ("backbone_id", "backbone_release", "files")
    missing = [key for key in required if not protocol.get(key)]
    if missing:
        raise ValueError(f"weights protocol is missing required fields: {missing}")
    files = protocol["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("weights.files must be a non-empty list")

    by_role: dict[str, dict[str, str]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("each weights.files entry must be a mapping")
        role = str(entry.get("role", "")).strip()
        path = str(entry.get("path", "")).strip()
        expected_hash = str(entry.get("sha256", "")).strip().lower()
        if not role or not path:
            raise ValueError("weight file entries require role and path")
        if role in by_role:
            raise ValueError(f"duplicate weight file role: {role}")
        if len(expected_hash) != 64 or any(
            character not in "0123456789abcdef" for character in expected_hash
        ):
            raise ValueError(f"weights.files[{role}].sha256 must be a SHA-256 digest")
        if verify_files:
            source = Path(path)
            if not source.is_file():
                raise FileNotFoundError(source)
            actual_hash = sha256_file(source)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"weight file hash mismatch for {role}: expected {expected_hash}, "
                    f"found {actual_hash}"
                )
        by_role[role] = {
            "path": path,
            "sha256": expected_hash,
            **(
                {"source_experiment_id": str(entry["source_experiment_id"])}
                if entry.get("source_experiment_id")
                else {}
            ),
        }

    missing_roles = [role for role in required_roles if role not in by_role]
    if missing_roles:
        raise ValueError(f"weights protocol is missing required roles: {missing_roles}")
    return {
        "backbone_id": str(protocol["backbone_id"]),
        "backbone_release": str(protocol["backbone_release"]),
        "files": by_role,
    }


def write_dataset_companion_zh(
    output_dir: str | Path, config: Mapping[str, Any]
) -> Path:
    protocol = validate_dataset_protocol(config, verify_files=True)
    payload = {
        "schema_version": 1,
        "记录名称": "数据集归属与划分侧录",
        "实验编号": str(config.get("experiment_id", "未设置")),
        "数据集ID": protocol["dataset_id"],
        "数据集名称_中英": protocol.get("dataset_name_zh_en"),
        "数据集release": protocol["release"],
        "源根目录": protocol["source_root"],
        "标签来源": protocol["label_provenance_zh"],
        "划分清单": protocol["split_manifest"],
        "划分清单SHA256": protocol["split_manifest_sha256"],
        "划分键": protocol["split_keys"],
        "说明": "该侧录只记录数据归属与固定划分，不改变权重、标签、检查点或历史审计证据。",
    }
    output = Path(output_dir) / "DATASET_INDEX_ZH.json"
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output
