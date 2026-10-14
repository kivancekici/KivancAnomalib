"""Salt okunur kod sözleşmesi kontrolleri; gerçek gecikme/GPU benchmark değildir."""
import contextlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image

root = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("review_worker_07", root / "worker/anomalib_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
result = {"type": "mock_contract_probe_not_physical_latency_benchmark"}

# Aynı varsayılan iş yükü: görüntü okuma 15 sanal ms, forward 5 sanal ms.
# Saat ilerletme kontrollüdür; sleep veya model yükleme/eğitim yoktur.
clock = SimpleNamespace(value=0.0)
def clock_now():
    return clock.value
def fake_engine_predict(**kwargs):
    clock.value += 0.020
    return [{"anomaly_map": [[0.0]]}]
checkpoint = worker._CheckpointInferenceBackend.__new__(worker._CheckpointInferenceBackend)
checkpoint.model_name = "PaDiM"
checkpoint.device = "cpu"
checkpoint.model = object()
checkpoint.torch = SimpleNamespace(inference_mode=contextlib.nullcontext)
checkpoint.engine = SimpleNamespace(predict=fake_engine_predict)
with patch.object(worker.time, "perf_counter", clock_now):
    _, checkpoint_ms = checkpoint.predict(Path("unused-image.png"))

clock.value = 0.0
def fake_image_open(path):
    clock.value += 0.015
    return Image.new("RGB", (8, 8))
def fake_forward(image):
    clock.value += 0.005
    return {"anomaly_map": np.zeros((8, 8), dtype=np.float32)}
onnx = worker._OnnxInferenceBackend.__new__(worker._OnnxInferenceBackend)
onnx.np = np
onnx.input_size = (8, 8)
onnx.sigma_override_supported = False
onnx._run_array = fake_forward
with patch.object(worker.time, "perf_counter", clock_now), patch.object(Image, "open", fake_image_open):
    _, onnx_ms = onnx.predict(Path("unused-image.png"))
assert abs(checkpoint_ms - 20.0) < 1e-6
assert abs(onnx_ms - 5.0) < 1e-6
result["timer_scope"] = {"virtual_checkpoint_ms": round(checkpoint_ms, 3), "virtual_onnx_ms": round(onnx_ms, 3), "note": "Kontrollü sanal saat; fiziksel gecikme değildir. ONNX timer görüntü okumayı dışarıda bırakıyor."}

# ORT run() fallback davranışını taklit et. Worker seçilen provider bilgisini yenilemiyor.
class FakeOrtSession:
    def __init__(self):
        self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    def get_providers(self):
        return self.providers
    def run(self, output_names, feed):
        self.providers = ["CPUExecutionProvider"]
        return [np.zeros((1, 1), dtype=np.float32)]
fallback = worker._OnnxInferenceBackend.__new__(worker._OnnxInferenceBackend)
fallback.np = np
fallback.session = FakeOrtSession()
fallback.input_name = "input"
fallback.output_names = ["anomaly_map"]
fallback.device = "cuda:0"
fallback.backend_name = "ONNX Runtime · CUDAExecutionProvider"
fallback._run_array(np.zeros((8, 8, 3), dtype=np.uint8))
assert fallback.session.get_providers() == ["CPUExecutionProvider"]
assert fallback.device == "cuda:0"
result["fallback_label"] = {"providers_after_mock_run": fallback.session.get_providers(), "worker_device_after_mock_run": fallback.device, "worker_backend_after_mock_run": fallback.backend_name}

# Büyük tensor ayırmadan yeni backend constructor'ı sırasında eski backend'in yaşadığını doğrula.
class OldBackend:
    pass
session = worker.PersistentInferenceSession.__new__(worker.PersistentInferenceSession)
session.backend = OldBackend()
old_ref = weakref.ref(session.backend)
session.session_root = root / "analysis-review/2026-10-13_10-uzman/kanitlar/test-temp"
session.metadata = {}
session.device_cache = {}
session.inspect = lambda path: {"requires_trust": False, "format": "onnx", "path": "unused.onnx", "fingerprint": "mock"}
observed = {}
class CandidateBackend:
    def __init__(self, metadata):
        observed["old_alive_during_candidate_constructor"] = old_ref() is not None
        self.model_name = "PatchCore"
        self.model_parameters = worker.normalize_model_parameters("PatchCore", {})
        self.backend_name = "Mock"
        self.device = "cpu"
        self.device_name = "CPU"
session._release_accelerator_cache = lambda: observed.update(old_alive_at_cleanup=old_ref() is not None)
with patch.object(worker, "_OnnxInferenceBackend", CandidateBackend):
    kind, loaded = session.load({"path": "unused.onnx"})
assert kind == "model_loaded"
assert observed == {"old_alive_during_candidate_constructor": True, "old_alive_at_cleanup": False}
result["replacement_lifetime"] = observed

# Önceden var olan küçük JSON kayıtlarını oku; model dosyası açma yoktur.
real_exports = []
production = []
for path in (root / "build").rglob("*.json"):
    if not (path.name.startswith("export-") or path.name == "production_model.json"):
        continue
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    metrics = record.get("metrics", {})
    if path.name.startswith("export-") and record.get("demo") is False:
        real_exports.append({"path": str(path.relative_to(root)), "duration_seconds": metrics.get("duration_seconds"), "latency_ms": metrics.get("latency_ms")})
    elif path.name == "production_model.json":
        production.append({"path": str(path.relative_to(root)), "checkpoint": record.get("checkpoint"), "latency_ms": metrics.get("latency_ms")})
durations = [r["duration_seconds"] for r in real_exports if isinstance(r["duration_seconds"], (int, float))]
result["existing_record_summary"] = {"real_export_job_count": len(real_exports), "non_null_latency_count": sum(r["latency_ms"] is not None for r in real_exports), "duration_min_seconds": min(durations) if durations else None, "duration_max_seconds": max(durations) if durations else None, "production_manifests": production, "caveat": "Export işinin metrics alanına kopyalanan eğitim+test süresi; inference veya export benchmark değildir. Tekrarlayan işler bağımsız eğitim sayılmaz."}
output = Path(__file__).with_suffix(".json")
output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
