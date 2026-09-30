from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import load_config, with_path_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create leakage-free patient splits")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--processed-dir")
    return parser.parse_args()


def create_patient_splits(
    frame: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    if not np.isclose(train_fraction + val_fraction + test_fraction, 1.0):
        raise ValueError("Split fractions must sum to one")
    if np.allclose((train_fraction, val_fraction, test_fraction), (0.7, 0.1, 0.2)):
        size_bin = pd.cut(
            frame["diameter_mm_mean"],
            bins=[3.0, 6.0, 10.0, np.inf],
            labels=["small", "medium", "large"],
            include_lowest=False,
        ).astype(str)
        strata = frame["position"].astype(str) + "__" + size_bin
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        remaining_indices, test_indices = next(
            outer.split(frame, strata, groups=frame["patient_id"])
        )
        remaining = frame.iloc[remaining_indices].reset_index(drop=True)
        remaining_strata = strata.iloc[remaining_indices].reset_index(drop=True)
        inner = StratifiedGroupKFold(n_splits=8, shuffle=True, random_state=seed + 1)
        train_local, val_local = next(
            inner.split(
                remaining,
                remaining_strata,
                groups=remaining["patient_id"],
            )
        )
        return {
            "train": sorted(remaining.iloc[train_local]["patient_id"].unique().tolist()),
            "val": sorted(remaining.iloc[val_local]["patient_id"].unique().tolist()),
            "test": sorted(frame.iloc[test_indices]["patient_id"].unique().tolist()),
        }

    patients = np.asarray(sorted(frame["patient_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(patients)
    count = len(patients)
    train_end = round(count * train_fraction)
    val_end = train_end + round(count * val_fraction)
    return {
        "train": patients[:train_end].tolist(),
        "val": patients[train_end:val_end].tolist(),
        "test": patients[val_end:].tolist(),
    }


def split_report(frame: pd.DataFrame, splits: dict[str, list[str]]) -> dict[str, object]:
    report: dict[str, object] = {}
    seen: set[str] = set()
    for split, patients in splits.items():
        overlap = seen.intersection(patients)
        if overlap:
            raise ValueError(f"Patient leakage into {split}: {sorted(overlap)[:5]}")
        seen.update(patients)
        subset = frame[frame["patient_id"].isin(patients)]
        report[split] = {
            "patients": len(patients),
            "nodules": len(subset),
            "positions": dict(Counter(subset["position"])),
            "size_bins": dict(
                Counter(
                    pd.cut(
                        subset["diameter_mm_mean"],
                        bins=[3.0, 6.0, 10.0, np.inf],
                        labels=["small", "medium", "large"],
                    ).astype(str)
                )
            ),
        }
    if seen != set(frame["patient_id"].unique()):
        raise ValueError("Not every patient was assigned to a split")
    return report


def main() -> int:
    args = parse_args()
    config = with_path_overrides(load_config(args.config), processed=args.processed_dir)
    processed = Path(config["paths"]["processed"])
    frame = pd.read_csv(processed / "metadata.csv")
    data = config["data"]
    splits = create_patient_splits(
        frame,
        float(data["train_fraction"]),
        float(data["val_fraction"]),
        float(data["test_fraction"]),
        int(config["seed"]),
    )
    payload = {
        "seed": int(config["seed"]),
        "splits": splits,
        "report": split_report(frame, splits),
    }
    (processed / "splits.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    reverse = {
        patient: split for split, patients in splits.items() for patient in patients
    }
    frame["split"] = frame["patient_id"].map(reverse)
    frame.to_csv(processed / "metadata_with_splits.csv", index=False)
    print(json.dumps(payload["report"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
