"""Offline review probes: synthetic data and graphs only; no real model loading."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]
os.environ.update(PYTHONDONTWRITEBYTECODE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
TEMP = ROOT / "temp"
TEMP.mkdir(exist_ok=True)
os.environ["TEMP"] = os.environ["TMP"] = str(TEMP)
spec = importlib.util.spec_from_file_location("review_worker", REPO / "worker/anomalib_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker.install_offline_network_guard()

import numpy as np
from PIL import Image
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
import torch

results = {}

image_path = ROOT / "image.png"
Image.new("RGB", (32, 32), (64, 64, 64)).save(image_path)

# A calibrated map must keep its calibration when thresholding, even if constant.
masks = []
for name, values in [("normal", [[.1, .2], [.2, .1]]), ("all_anomalous", [[.9, .9], [.9, .9]])]:
    amap = np.asarray(values, dtype=np.float32)
    output = worker.write_prediction_artifacts(image_path, {"anomaly_map": amap, "pred_score": float(amap.max())}, ROOT / name, .6)
    mask = np.asarray(Image.open(output["predicted_mask_path"]))
    masks.append({"case": name, "input_range": [float(amap.min()), float(amap.max())], "threshold": .6, "expected_fraction": float((amap >= .6).mean()), "actual_fraction": float((mask[..., 0] > 0).mean()), "image_label": output["label"]})
results["mask_threshold"] = masks

# Use the real two preprocessing functions with capture-only runtimes.
scaling = {}
for cls in (worker._OnnxInferenceBackend, worker._OpenVinoOnnxInferenceBackend):
    backend = cls.__new__(cls)
    backend.np = np
    backend.input_name = "input"
    backend.output_names = ["anomaly_map"]
    backend.input = "input"
    backend.outputs = ["anomaly_map"]
    observed = []
    def capture(_outputs, inputs):
        array = inputs["input"]
        observed.append(float(array.max()))
        return [array[:, :1]]
    backend.session = type("CaptureSession", (), {"run": staticmethod(capture)})()
    backend.compiled_model = lambda inputs: {"anomaly_map": capture(None, inputs)[0]}
    for code in (1, 2, 255):
        backend._run_array(np.full((32, 32, 3), code, dtype=np.uint8))
    scaling[cls.__name__] = {"uint8_input_max": [1, 2, 255], "float_input_max": observed, "expected": [1 / 255, 2 / 255, 1]}
results["dark_input_scaling"] = scaling

# Construct a real tiny Gaussian graph with saved calibration scale = 1/4.
sigma = 1
kernel_size = 9
axis = np.arange(kernel_size, dtype=np.float32) - 4
gaussian = np.exp(-(axis ** 2) / 2)
gaussian /= gaussian.sum()
kernel = np.outer(gaussian, gaussian)[None, None].astype(np.float32)
nodes = [
    helper.make_node("ReduceMean", ["input"], ["raw"], axes=[1], keepdims=1, name="patchcore.memory_bank.nearest_neighbors"),
    helper.make_node("Pad", ["raw", "pads"], ["padded"], mode="reflect"),
    helper.make_node("Conv", ["padded", "blur_weights"], ["smoothed"]),
    helper.make_node("Div", ["smoothed", "scale"], ["anomaly_map"]),
    helper.make_node("ReduceMax", ["anomaly_map"], ["pred_score"], axes=[1, 2, 3], keepdims=0),
]
graph = helper.make_graph(nodes, "calibrated_patchcore", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])], [helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, 1, 32, 32]), helper.make_tensor_value_info("pred_score", TensorProto.FLOAT, [1])], [numpy_helper.from_array(np.asarray([0, 0, 4, 4, 0, 0, 4, 4], dtype=np.int64), "pads"), numpy_helper.from_array(kernel, "blur_weights"), numpy_helper.from_array(np.asarray(4, dtype=np.float32), "scale")])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
graph_path = ROOT / "calibrated.onnx"
graph_path.write_bytes(model.SerializeToString())
params = worker.normalize_model_parameters("PatchCore", {"sigma": 1, "image_size": {"width": 32, "height": 32}})
worker.add_patchcore_raw_map_output(graph_path, params)
metadata = worker.inspect_model_artifact(str(graph_path))
with patch.object(torch.cuda, "is_available", return_value=False), patch.object(ort, "preload_dlls", return_value=None):
    backend = worker._OnnxInferenceBackend(metadata)
    direct = backend._run_array(np.asarray(Image.open(image_path)))
    result, _ = backend.predict(image_path, None)
    results["raw_map_replacement"] = {"sigma_override": None, "saved_sigma": 1, "direct_calibrated_mean": float(direct["anomaly_map"].mean()), "backend_mean": float(result["anomaly_map"].mean()), "pred_score_unchanged": bool(np.array_equal(direct["pred_score"], result["pred_score"]))}

# Graph fingerprints must include external tensor bytes.
external_path = ROOT / "external.onnx"
tensor = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "memory_bank")
from onnx.external_data_helper import set_external_data
payload = tensor.raw_data
set_external_data(tensor, location="external.data", offset=0, length=len(payload))
tensor.ClearField("raw_data")
external_data = ROOT / "external.data"
external_data.write_bytes(payload)
graph = helper.make_graph([helper.make_node("Mul", ["input", "memory_bank"], ["anomaly_map"], name="patchcore.nearest_neighbors")], "external_patchcore", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, 1])], [helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, 1, 1, 1])], [tensor])
external_path.write_bytes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10).SerializeToString())
(ROOT / "external.manifest.json").write_text(json.dumps({"model": "PatchCore", "artifact": {"path": external_path.name, "sha256": hashlib.sha256(external_path.read_bytes()).hexdigest()}}))
before_meta = worker.inspect_model_artifact(str(external_path))
def run_external():
    session = ort.InferenceSession(str(external_path), providers=["CPUExecutionProvider"])
    return float(session.run(None, {"input": np.ones((1, 1, 1, 1), dtype=np.float32)})[0].mean())
before_value = run_external()
external_data.write_bytes(np.full((1, 1, 1, 1), 2, dtype=np.float32).tobytes())
after_meta = worker.inspect_model_artifact(str(external_path))
results["external_weight_integrity"] = {"before_output": before_value, "after_output": run_external(), "same_fingerprint": before_meta["fingerprint"] == after_meta["fingerprint"], "manifest_passes_after_mutation": True}

# Exercise real export orchestration while replacing model creation/export.
from anomalib.models.components.base.export_mixin.utils import create_export_root
from anomalib.engine import Engine
import inspect
default_name = inspect.signature(Engine.export).parameters["model_file_name"].default
exports = []
class FakeEngine:
    def __init__(self, **kwargs):
        pass
    def export(self, model, export_type, export_root, model_file_name=default_name, **kwargs):
        destination = create_export_root(export_root, export_type) / (model_file_name + ".pt")
        destination.write_text(model)
        exports.append({"model": model, "path": str(destination), "sha256": worker._sha256(destination)})
        return destination
with patch("anomalib.engine.Engine", FakeEngine), patch.object(worker, "create_model", side_effect=lambda name, params: name), patch.object(worker, "restore_checkpoint_state", return_value=None), patch.object(worker, "preflight_selected_models", side_effect=lambda job, names: {name: worker.normalize_model_parameters(name) for name in names}), patch.object(worker, "configure_anomalib_asset_paths", return_value=None), contextlib.redirect_stdout(io.StringIO()):
    for name in ("PatchCore", "PaDiM"):
        worker.run_real_export({"model": name, "format": "TORCH", "checkpoint": str(ROOT / "unused.ckpt"), "output_root": str(ROOT / "exports"), "asset_root": str(ROOT / "unused-assets")})
final_path = Path(exports[0]["path"])
results["export_collision"] = {"probe": "Fake Engine, real worker orchestration and installed create_export_root/default filename", "exports": exports, "same_path": exports[0]["path"] == exports[1]["path"], "first_artifact_replaced": worker._sha256(final_path) != exports[0]["sha256"], "final_model": final_path.read_text(), "manifest_model": json.loads(final_path.with_name("model.manifest.json").read_text())["model"]}

result_path = ROOT / "results.json"
result_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
print(json.dumps(results, indent=2))
