"""Read-only worker diagnostic: demo environment check skips ML imports."""

import builtins
import contextlib
import importlib.util
import io
import json
from pathlib import Path

root = Path(__file__).resolve().parents[3]
worker_path = root / "worker" / "anomalib_worker.py"
spec = importlib.util.spec_from_file_location("operator_review_worker", worker_path)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)

real_import = builtins.__import__
blocked_attempts = []


def guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in {"anomalib", "torch"}:
        blocked_attempts.append(name)
        raise ImportError(f"Review probe intentionally rejects {name}")
    return real_import(name, *args, **kwargs)


capture = io.StringIO()
builtins.__import__ = guarded_import
try:
    with contextlib.redirect_stdout(capture):
        worker.run_demo({"mode": "check", "demo": True})
finally:
    builtins.__import__ = real_import

events = [json.loads(line) for line in capture.getvalue().splitlines()]
assert events == [{"event": "environment", "ok": True, "demo": True,
                   "message": "Demo worker ready. Configure an Anomalib Python environment for real jobs."}]
assert blocked_attempts == []
print(json.dumps({"probe": "demo_environment_check", "passed": True,
                  "anomalib_torch_import_attempts": blocked_attempts,
                  "worker_event": events[0]}, ensure_ascii=False, indent=2))
