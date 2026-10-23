from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from worker import anomalib_worker as worker
import numpy as np
from PIL import Image

OUT = Path(__file__).resolve().parent
report = {}

# All modified paths belong to this temporary synthetic fixture.
with tempfile.TemporaryDirectory(dir=OUT, prefix="integration-fixture-") as directory:
    fixture = Path(directory)
    destination = fixture / "managed"
    staging = fixture / "staging"
    destination.mkdir()
    staging.mkdir()
    (destination / "existing.txt").write_text("previous usable asset", encoding="utf-8")
    (staging / "new.txt").write_text("new candidate", encoding="utf-8")
    original_replace = Path.replace

    def fail_promotion(path, target):
        if path == staging:
            raise PermissionError("synthetic promotion failure")
        return original_replace(path, target)

    with mock.patch.object(Path, "replace", fail_promotion):
        try:
            worker._replace_directory(staging, destination)
        except PermissionError as exc:
            report["asset_promotion_failure"] = {
                "error": str(exc),
                "previous_asset_preserved": (destination / "existing.txt").exists(),
                "staged_candidate_preserved": (staging / "new.txt").exists(),
            }

    image = fixture / "input.png"
    Image.new("RGB", (2, 2), (64, 64, 64)).save(image)
    cases = {}
    for name, score in (("missing", None), ("nan", float("nan")), ("infinity", float("inf"))):
        prediction = {
            "anomaly_map": np.array([[0.1, 0.12], [0.15, 0.2]], dtype=np.float32),
            "pred_score": score,
        }
        result = worker.write_prediction_artifacts(image, prediction, fixture / name, 0.6)
        stream = io.StringIO()
        with redirect_stdout(stream):
            worker.emit("inference", **result)

        def reject_nonfinite(token):
            raise ValueError(token)

        valid_json = True
        try:
            json.loads(stream.getvalue(), parse_constant=reject_nonfinite)
        except ValueError:
            valid_json = False
        cases[name] = {
            "anomaly_score": str(result["anomaly_score"]),
            "label": result["label"],
            "standards_compliant_json": valid_json,
        }
    report["invalid_prediction_scores"] = cases

print(json.dumps(report, indent=2, allow_nan=False))
(OUT / "integration-results.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
