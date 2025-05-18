import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.distance import cdist
from sewar.full_ref import ssim, ergas, rase


def dtw_columns(a, b, threshold=150):
    a = (a < threshold).astype(np.float64).T
    b = (b < threshold).astype(np.float64).T
    distances = cdist(a, b, metric="sqeuclidean")
    previous = np.full(len(b) + 1, np.inf)
    previous[0] = 0.0
    for row in distances:
        current = np.full(len(b) + 1, np.inf)
        for j, cost in enumerate(row, 1):
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return float(np.sqrt(previous[-1]))


def evaluate(output):
    output = Path(output)

    manifest = json.loads((output / "generation.json").read_text())
    rows = []
    for name in manifest["filenames"]:
        with Image.open(output / "images_gold" / name) as img:
            gold = np.asarray(img.convert("L"))
        with Image.open(output / "images_pred" / name) as img:
            pred = np.asarray(img.convert("L"))
        if pred.shape != gold.shape:
            raise ValueError(f"Unmatched shapes for {name}")
        error = float(np.mean((pred.astype(float) - gold.astype(float)) ** 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            values = {
                "filename": name,
                "dtw": dtw_columns(pred, gold),
                "rmse": math.sqrt(error),
                "ssim": float(ssim(gold, pred)[0]),
                "psnr": float(10 * math.log10(255**2 / error)) if error else None,
                "ergas": float(ergas(gold, pred)),
                "rase": float(rase(gold, pred)),
            }

        values = {
            k: (None if isinstance(v, float) and not math.isfinite(v) else v)
            for k, v in values.items()
        }
        rows.append(values)
    if not rows:
        raise ValueError("No generated images to evaluate")
    summary = {
        "count": len(rows),
        "grayscale_metrics": True,
        "pixel_range": [0, 255],
        "dtw_threshold": 150,
        "split": manifest["split"],
        "checkpoint": manifest["checkpoint"],
        "inference_steps": manifest["inference_steps"],
    }
    for key in ["dtw", "rmse", "ssim", "psnr", "ergas", "rase"]:
        valid = [r[key] for r in rows if r[key] is not None]
        summary[key] = float(np.mean(valid)) if valid else None
        summary[key + "_valid_count"] = len(valid)
    (output / "metrics.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (output / "metrics_per_image.json").write_text(
        json.dumps(rows, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return summary
