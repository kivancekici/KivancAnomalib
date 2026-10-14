"""Sentetik hata enjeksiyonu; model, kamera, ağ veya uygulama GUI'si açmaz."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import uuid

workspace = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("review_worker_08", workspace / "worker/anomalib_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
evidence = Path(__file__).resolve().parent
probe_root = evidence / ("rol08-sentetik-" + uuid.uuid4().hex[:8])
probe_root.mkdir()
results = {"kind": "mock_fault_injection_no_model_or_hardware", "probe_root": str(probe_root)}

def ensure_probe_path(path):
    resolved = path.resolve()
    assert resolved.is_relative_to(probe_root.resolve()), str(resolved)
    return resolved

# 1. Yeni klasorun rename'i hata verdiginde eski veri kalmiyor.
destination = ensure_probe_path(probe_root / "rename-failure" / "destination")
staging = ensure_probe_path(probe_root / "rename-failure" / "staging")
destination.mkdir(parents=True)
staging.mkdir()
(destination / "previous-ready.txt").write_text("previous", encoding="utf-8")
(staging / "new.txt").write_text("candidate", encoding="utf-8")
original_replace = Path.replace
def fail_staging_replace(self, target):
    ensure_probe_path(self)
    ensure_probe_path(Path(target))
    if self == staging:
        raise PermissionError("synthetic rename denied")
    return original_replace(self, target)
with patch.object(Path, "replace", fail_staging_replace):
    try:
        worker._replace_directory(staging, destination)
    except PermissionError as exc:
        error = str(exc)
assert not destination.exists()
assert staging.exists()
results["replacement_failure"] = {"error": error, "previous_destination_exists": destination.exists(), "staging_exists": staging.exists()}

# 2. HF import checksum/format kontrolu promotion sonrasinda yapiliyor.
asset_root = ensure_probe_path(probe_root / "import-after-validation")
definition = {"repo_id": "synthetic/review", "filename": "model.safetensors", "display_name": "Synthetic review asset"}
source_repo = ensure_probe_path(probe_root / "synthetic-source" / "models--synthetic--review")
source_repo.mkdir(parents=True)
source_file = source_repo / "model.safetensors"
source_file.write_bytes(b"synthetically-invalid-candidate")
managed_repo = ensure_probe_path(worker._hf_repo_directory(worker._managed_hf_cache(asset_root), definition["repo_id"]))
managed_repo.mkdir(parents=True)
(managed_repo / "previous-ready.txt").write_text("previous", encoding="utf-8")
managed_file = managed_repo / "model.safetensors"
def fake_cached_file(cache_dir, ignored_definition):
    if cache_dir == source_repo.parent:
        return source_file, "synthetic-revision"
    return managed_file, "synthetic-revision"
with patch.object(worker, "_shared_hf_cache", return_value=source_repo.parent), patch.object(worker, "_hf_cached_file", fake_cached_file), patch.object(worker, "_safetensors_validation_error", return_value="synthetic corrupt format"):
    try:
        worker._import_hf_asset("synthetic", definition, asset_root)
    except RuntimeError as exc:
        error = str(exc)
assert managed_file.read_bytes() == source_file.read_bytes()
assert not (managed_repo / "previous-ready.txt").exists()
results["promotion_before_validation"] = {"error": error, "old_marker_exists": (managed_repo / "previous-ready.txt").exists(), "new_invalid_candidate_promoted": managed_file.exists()}

# 3. Bos output_root, Path('') == Path('.') oldugu icin reddedilmiyor.
input_file = ensure_probe_path(probe_root / "synthetic-input.bin")
input_file.write_bytes(b"synthetic")
backend = SimpleNamespace(model_name="PaDiM", backend_name="Synthetic", device="cpu", device_name="Mock", model_parameters=worker.normalize_model_parameters("PaDiM", {}), predict=lambda image: ({"synthetic": True}, 1.0))
session = worker.PersistentInferenceSession.__new__(worker.PersistentInferenceSession)
session.backend = backend
session.metadata = {"path": "synthetic", "format": "synthetic", "fingerprint": "synthetic"}
captured = {}
def fake_artifacts(image, prediction, output, threshold):
    captured["output_root"] = str(output)
    return {"anomaly_score": 0.0, "label": "normal"}
with patch.object(worker, "write_prediction_artifacts", fake_artifacts):
    session.predict({"image_path": str(input_file)})
assert captured["output_root"] == "."
results["missing_output_root"] = {"accepted": True, "resolved_argument": captured["output_root"], "artifact_write_mocked": True}

# 4. Pozitif kontrol: bozuk komutlar servis dongusunu oldurmuyor.
class FakeSession:
    def __init__(self, *ignored):
        pass
    def unload(self):
        pass
fake_input = io.StringIO("invalid-json\n" + json.dumps({"request_id": "unknown-1", "command": "unknown"}) + "\n" + json.dumps({"request_id": "shutdown-1", "command": "shutdown"}) + "\n")
fake_output = io.StringIO()
with patch.object(worker, "configure_offline_environment"), patch.object(worker, "install_offline_network_guard"), patch.object(worker, "PersistentInferenceSession", FakeSession), patch("sys.stdin", fake_input), contextlib.redirect_stdout(fake_output):
    exit_code = worker.run_inference_service(str(asset_root), str(ensure_probe_path(probe_root / "mock-session")))
events = [json.loads(line) for line in fake_output.getvalue().splitlines()]
assert [event["event"] for event in events] == ["session_ready", "error", "error", "session_stopped"]
assert events[2]["request_id"] == "unknown-1"
assert exit_code == 0
results["malformed_command_recovery"] = {"exit_code": exit_code, "event_order": [event["event"] for event in events], "unknown_request_correlated": events[2]["request_id"]}

output = evidence / "08-dayaniklilik-probe.json"
output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(results, indent=2, ensure_ascii=False))
