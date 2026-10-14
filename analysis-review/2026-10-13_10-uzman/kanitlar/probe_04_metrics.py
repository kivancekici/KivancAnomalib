"""Read-only source probes; creates tiny synthetic artifacts only in review folder."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "probe-04"
OUT.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("review_worker", ROOT / "worker/anomalib_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

image_path = OUT / "synthetic-input.png"
Image.new("RGB", (2, 2), (128, 128, 128)).save(image_path)
maps = {
    "low_calibrated": np.array([[0.10, 0.13], [0.17, 0.20]], dtype=np.float32),
    "high_calibrated": np.array([[0.80, 0.83], [0.87, 0.90]], dtype=np.float32),
    "constant_high": np.full((2, 2), 0.90, dtype=np.float32),
}
results = {}
for name, array in maps.items():
    scaled = worker.normalized_map(array)
    results[name] = {
        "original": array.tolist(),
        "after_image_minmax": scaled.tolist(),
        "original_mask_at_060": (array >= 0.60).tolist(),
        "actual_mask_at_060": (scaled >= 0.60).tolist(),
    }

def artifact_case(name, prediction):
    result = worker.write_prediction_artifacts(image_path, prediction, OUT / name, 0.60)
    return {
        "label": result["label"],
        "score_repr": repr(result["anomaly_score"]),
        "score_isfinite": math.isfinite(result["anomaly_score"]),
    }

results["missing_pred_score"] = artifact_case("missing-score", {"anomaly_map": maps["low_calibrated"]})
results["nan_pred_score"] = artifact_case("nan-score", {"anomaly_map": maps["high_calibrated"], "pred_score": float("nan")})
results["ignore_native_label_at_055"] = artifact_case("native-label", {"anomaly_map": maps["high_calibrated"], "pred_score": 0.55, "pred_label": True})

def reject_constant(value):
    raise ValueError(f"Non-standard JSON constant: {value}")

metric = worker.metric_value({"image_AUROC": float("nan")}, ["image_AUROC"])
with contextlib.redirect_stdout(io.StringIO()) as captured:
    worker.emit("result", model="PatchCore", image_auroc=metric)
line = captured.getvalue().strip()
try:
    json.loads(line, parse_constant=reject_constant)
    strict_json_valid = True
except ValueError:
    strict_json_valid = False
results["metric_nan_event"] = {"metric_repr": repr(metric), "emitted": line, "strict_json_valid": strict_json_valid}

# Reproduce the selection expression verbatim, without running training.
unscored = [{"model": "PatchCore", "image_auroc": None}, {"model": "PaDiM", "image_auroc": None}]
scored = [item for item in unscored if item.get("image_auroc") is not None]
winner = max(scored, key=lambda item: item["image_auroc"])["model"] if scored else unscored[0]["model"]
results["winner_without_metrics"] = winner
with_nan = [{"model": "PatchCore", "image_auroc": float("nan")}, {"model": "PaDiM", "image_auroc": 0.99}]
results["winner_with_first_nan"] = max(with_nan, key=lambda item: item["image_auroc"])["model"]

output = json.dumps(results, ensure_ascii=False, indent=2, allow_nan=False)
(OUT.parent / "probe-04-metrics-result.json").write_text(output + "\n", encoding="utf-8")
print(output)
