import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CHECKPOINT_SHA = "bb63556a712255531da10e9342202571c1f3fd247e5756b7c0ed160af2ad47c2"
TISSUES = {"femur": [7], "tibia": [8], "patella": [9], "femur_cartilage": [2],
           "tibia_cartilage": [3, 4], "patella_cartilage": [1], "lateral_meniscus": [6], "medial_meniscus": [5]}
COLORS = {"femur": (245, 193, 49), "tibia": (41, 180, 140), "patella": (230, 125, 65),
          "femur_cartilage": (162, 124, 255), "tibia_cartilage": (65, 175, 232),
          "patella_cartilage": (180, 180, 240), "lateral_meniscus": (255, 135, 60), "medial_meniscus": (238, 55, 145)}


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_image(path):
    with Image.open(path) as im:
        return np.array(im)


def crop(array, box):
    x0, y0, x1, y1 = box
    out = np.zeros((y1-y0, x1-x0), dtype=array.dtype)
    a, b, c, d = max(0, x0), max(0, y0), min(array.shape[1], x1), min(array.shape[0], y1)
    if a < c and b < d:
        out[b-y0:d-y0, a-x0:c-x0] = array[b:d, a:c]
    return out


def to_model(array):
    if array.ndim != 2 or array.shape != (384, 160):
        raise ValueError("Exactly one native 384x160 2D plane required")
    native = array.astype(np.float32)
    mean, std = float(native.mean()), float(native.std())
    if not np.isfinite(native).all() or std < 1e-8:
        raise ValueError("Invalid native intensities")
    oriented = ((native - mean) / (std + 1e-8)).T[::-1, :]
    padded = np.pad(oriented, ((0, 0), (64, 64)), constant_values=-mean / (std + 1e-8))
    return np.ascontiguousarray(padded[None, None], dtype=np.float32)


def from_model(labels):
    if labels.shape != (160, 512):
        raise ValueError("Wrong model label plane")
    return np.ascontiguousarray(labels[:, 64:448][::-1, :].T, dtype=np.uint8)


def load_model(path, backend="tensorflow"):
    if checksum(path) != CHECKPOINT_SHA:
        raise ValueError("Released checkpoint checksum mismatch")
    if backend == "tensorflow":
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        devices = tf.config.list_physical_devices("GPU")
        if not devices:
            raise RuntimeError("TensorFlow cannot see the A100 GPU")
        tf.config.set_logical_device_configuration(devices[0], [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
        model = tf.keras.models.load_model(path, compile=False)
        if tuple(model.input_shape) != (None, 1, 160, 512) or tuple(model.output_shape) != (None, 10, 160, 512):
            raise ValueError("Unexpected native TensorFlow model shapes")
        return model
    os.environ["KERAS_BACKEND"] = "torch"
    import torch
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.055)
    torch.backends.cudnn.benchmark = False
    import keras

    class LegacyBatchNormalization(keras.layers.BatchNormalization):
        def __init__(self, axis=-1, **kwargs):
            if isinstance(axis, (tuple, list)):
                if len(axis) != 1:
                    raise ValueError("Unsupported multi-axis legacy batch normalization")
                axis = axis[0]
            super().__init__(axis=axis, **kwargs)

    class LegacyConv2DTranspose(keras.layers.Conv2DTranspose):
        def __init__(self, groups=1, **kwargs):
            if groups != 1:
                raise ValueError("Grouped transpose convolution is unsupported")
            super().__init__(**kwargs)

    with keras.device("cpu"):
        model = keras.models.load_model(path, compile=False,
                                       custom_objects={"BatchNormalization": LegacyBatchNormalization,
                                                       "Conv2DTranspose": LegacyConv2DTranspose})
    if tuple(model.input_shape) != (None, 1, 160, 512) or tuple(model.output_shape) != (None, 10, 160, 512):
        raise ValueError(f"Unexpected model shapes: {model.input_shape}, {model.output_shape}")
    model.to("cuda")
    model.eval()
    return model


def dice(a, b):
    denominator = int(a.sum()) + int(b.sum())
    return 2 * int((a & b).sum()) / denominator if denominator else None


def solid(masks):
    rgb = np.zeros((*next(iter(masks.values())).shape, 3), np.uint8)
    counts = np.zeros(rgb.shape[:2], np.uint8)
    for tissue, mask in masks.items():
        rgb[mask] = COLORS[tissue]
        counts += mask
    rgb[counts > 1] = 255
    return rgb


def preview(gray, pred, manual, row, output):
    box = row["geometry"]["crop_box_native_xyxy"]
    image = crop(gray, box)
    predicted = solid({t: crop(np.isin(pred, ids), box) for t, ids in TISSUES.items()})
    reference = solid({t: crop(manual[t], box) for t in TISSUES})
    overlay = np.repeat(image[..., None], 3, axis=2)
    foreground = np.any(predicted, axis=2)
    overlay[foreground] = np.rint(0.5 * overlay[foreground] + 0.5 * predicted[foreground]).astype(np.uint8)
    canvas = Image.new("RGB", (624, 214), (248, 249, 250))
    draw = ImageDraw.Draw(canvas)
    font_path = Path("/usr/share/fonts/dejavu/DejaVuSans.ttf")
    try:
        font = ImageFont.truetype(str(font_path), 12) if font_path.exists() else ImageFont.load_default()
    except (ImportError, OSError):
        font = ImageFont.load_default_imagefont()
    draw.text((12, 8), row["sample_id"], fill="black", font=font)
    for i, (title, arr) in enumerate(zip(("Native MRI", "2D model mask", "2D filled overlay", "Manual reference"), (image, predicted, overlay, reference))):
        x = 12 + i * 153
        draw.text((x, 30), title, fill="black", font=font)
        canvas.paste(Image.fromarray(arr).convert("RGB"), (x, 50))
    draw.text((12, 190), "Single slice only; original pixels; padding only; candidate, not expert-approved.", fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--manifest", default="samples_native_manual.jsonl")
    p.add_argument("--backend", choices=("torch", "tensorflow"), default="tensorflow")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in (args.input / args.manifest).read_text().splitlines() if line]
    if args.limit:
        rows = rows[:args.limit]
    if args.backend == "torch":
        import torch
    model = load_model(args.model, args.backend)
    results, times = [], []
    for i, row in enumerate(rows):
        if checksum(args.input / row["image_path"]) != row["image_sha256"]:
            raise ValueError("Source image changed")
        if checksum(args.input / row["native_path"]) != row["native_sha256"]:
            raise ValueError("Source native intensities changed")
        with np.load(args.input / row["native_path"], allow_pickle=False) as f:
            array = f["intensity"]
        gray = read_image(args.input / row["image_path"])
        if args.backend == "torch":
            data = torch.from_numpy(to_model(array)).to("cuda")
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                probabilities = model(data, training=False)
                if not bool(torch.isfinite(probabilities).all()):
                    raise ValueError("Nonfinite model output")
                labels = probabilities.argmax(dim=1)[0].cpu().numpy()
            torch.cuda.synchronize()
        else:
            data = to_model(array)
            started = time.perf_counter()
            probabilities = model(data, training=False).numpy()
            if not np.isfinite(probabilities).all():
                raise ValueError("Nonfinite model output")
            labels = probabilities.argmax(axis=1)[0]
        elapsed = time.perf_counter() - started
        times.append(elapsed)
        predicted = from_model(labels)
        directory = args.output / "predictions"
        directory.mkdir(exist_ok=True)
        Image.fromarray(predicted).save(directory / f"{row['sample_id']}.png")
        record = {"sample_id": row["sample_id"], "seconds": elapsed, "targets": {}, "model_only_input": "single_native_2d_intensity_array"}
        if row["manual_path"]:
            if checksum(args.input / row["manual_path"]) != row["manual_sha256"]:
                raise ValueError("Manual reference changed")
            with np.load(args.input / row["manual_path"], allow_pickle=False) as f:
                manual = {t: f[t] for t in TISSUES}
            for tissue, ids in TISSUES.items():
                binary = np.isin(predicted, ids)
                box = row["geometry"]["crop_box_native_xyxy"]
                record["targets"][tissue] = {"dice_full": dice(binary, manual[tissue]),
                                             "dice_crop": dice(crop(binary, box), crop(manual[tissue], box)),
                                             "manual_pixels_full": int(manual[tissue].sum()),
                                             "predicted_pixels_full": int(binary.sum())}
            preview(gray, predicted, manual, row, args.output / "previews" / f"{row['sample_id']}.png")
        results.append(record)
        write_json(args.output / "progress.json", {"completed": i+1, "expected": len(rows), "latest": record})
        print(json.dumps({"case": i+1, "sample_id": row["sample_id"], "seconds": elapsed}), flush=True)
    averages = {}
    for tissue in TISSUES:
        averages[tissue] = {}
        for metric in ("dice_full", "dice_crop"):
            values = [r["targets"][tissue][metric] for r in results if tissue in r["targets"] and r["targets"][tissue][metric] is not None]
            averages[tissue][metric] = {"n": len(values), "mean": float(np.mean(values)) if values else None}
    write_json(args.output / "evaluation.json", {"cases": len(results), "summary": averages,
               "steady_state_mean_seconds": float(np.mean(times[1:])) if len(times) > 1 else None,
               "source": "aagatti/dosma_bones/coronal_best_model.h5", "revision": "ff5e0b5d1bf8b850ee6258210e05418889d7133f",
               "checkpoint_sha256": CHECKPOINT_SHA, "backend": args.backend, "input_is_strictly_2d": True,
               "spatial_resize": False, "input_padding_si": [64, 64], "normalization": "single-slice mean/std",
               "postprocessing": "none", "manual_labels_used_for_fitting_or_prompts": False,
               "cautions": ["Feasibility check, not an independent clinical validation.", "Upstream training membership unknown.",
                            "Mostly grade-0 manual references; severe extrusion accuracy not established.",
                            "Only model-input padding, never scale interpolation; padding differs from upstream volume resize preprocessing."],
               "results": results})
    print(json.dumps({"summary": averages, "steady_state_seconds": float(np.mean(times[1:])) if len(times) > 1 else None}), flush=True)


if __name__ == "__main__":
    main()
