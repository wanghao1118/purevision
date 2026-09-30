import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from build_knee_native_subset import read_json, read_jsonl, require, sha256, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prior, output = args.prior_build.resolve(), args.output.resolve()
    require(prior != output and prior not in output.parents and output not in prior.parents, "Separate output required")
    with (prior / "build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = read_json(prior / "build_status.json")
        status.update(state="stopped_superseded_by_strict_2d", stopped_unix=time.time(),
                      completed=len(list((prior / "records").glob("*.json"))),
                      reason="User explicitly requires only a single 2D input slice; 3D worker terminated",
                      estimated_remaining_hours=None)
        write_json(prior / "build_status.json", status)
    config = read_json(prior / "build_config.json")
    source = Path(config["source"])
    plan = read_jsonl(prior / "provenance/build_plan.jsonl")
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / "input.complete").exists(), "Input bundle already complete")
    rows = []
    for i, item in enumerate(plan):
        row = item["row"]
        sid = row["sample_id"]
        relative = row["views"]["full"]["image_path"]
        src = source / relative
        require(sha256(src) == row["sha256"][relative], "Source image changed")
        destination = output / "images" / f"{sid}.png"
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(src, destination)
        with Image.open(destination) as image:
            require(image.size == (160, 384), "Unexpected native slice size")
        sample = {k: row[k] for k in ("sample_id", "patient_id", "split", "dimensions", "label_scope", "geometry", "intensity_window")}
        sample.update(image_path=destination.relative_to(output).as_posix(), image_sha256=sha256(destination),
                      manual_path=None, manual_sha256=None)
        if item["manual"]:
            record = read_json(prior / "records" / f"{sid}.json")
            require(all(s["source_type"] == "manual" for s in record["annotation_sources"].values()), "Partial manual case requires explicit availability channels")
            relative = record["views"]["full"]["anatomy_channels_path"]
            require(sha256(prior / relative) == record["sha256"][relative], "Manual export changed")
            target = output / "manual" / f"{sid}.npz"
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(prior / relative, target)
            sample.update(manual_path=target.relative_to(output).as_posix(), manual_sha256=sha256(target),
                          manual_sources=record["annotation_sources"])
            with np.load(target, allow_pickle=False) as archive:
                require(all(archive[k].dtype == bool and archive[k].shape == (384, 160) for k in archive.files), "Wrong manual array")
        rows.append(sample)
        if (i + 1) % 1000 == 0:
            print(json.dumps({"copied": i + 1}), flush=True)
    write_jsonl(output / "samples.jsonl", rows)
    write_json(output / "dimensions.json", read_json(source / "dimensions.json"))
    write_json(output / "anatomy_labels.json", read_json(prior / "anatomy_labels.json"))
    files = [p for p in output.rglob("*") if p.is_file()]
    summary = {"samples": len(rows), "manual_samples": sum(r["manual_path"] is not None for r in rows),
               "manual_by_original_split": dict(Counter(r["split"] for r in rows if r["manual_path"])),
               "manual_patients_by_split": {s: len({r["patient_id"] for r in rows if r["split"] == s and r["manual_path"]}) for s in ("train", "val", "test")},
               "manual_grade_counts": dict(Counter(str(r["dimensions"]["medial_meniscus_medial_extrusion"]["value"]) for r in rows if r["manual_path"])),
               "native_image_shape_hw": [384, 160], "crop_shape_hw": [128, 128],
               "input_files_bytes": sum(p.stat().st_size for p in files), "spatial_resize": False,
               "inference_input": "only one grayscale 2D image; no neighboring slices or predicted-mask prompts",
               "manual_priority": True, "prior_3d_inference_stopped": True, "inherited_slice_selection": True}
    write_json(output / "input_summary.json", summary)
    write_json(output / "input.complete", {"status": "complete", "samples_sha256": sha256(output / "samples.jsonl"),
                                          "note": "2D input bundle only, NOT a finalized segmented dataset"})
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
