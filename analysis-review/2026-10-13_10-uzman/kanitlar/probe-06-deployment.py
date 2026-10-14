"""Read-only project audit; synthetic models and mocks only, all writes in review temp."""
from __future__ import annotations

import hashlib
import inspect
import json
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from PIL import Image
from worker import anomalib_worker as worker

EVIDENCE = Path(__file__).resolve().parent
TEMP = EVIDENCE / "test-temp"
TEMP.mkdir(exist_ok=True)
result = {"versions": {"onnx": onnx.__version__, "ort": ort.__version__, "torch": torch.__version__}}
started = time.perf_counter()


def tiny_graph(path, height=256, width=256, output="anomaly_map", dtype=TensorProto.FLOAT, external=False):
    nodes = [helper.make_node("ReduceMean", ["input"], ["mean"], axes=[1], keepdims=1, name="nearest_neighbors")]
    initializer = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float16 if dtype == TensorProto.FLOAT16 else np.float32), name="memory_bank")
    nodes.append(helper.make_node("Add", ["mean", "memory_bank"], [output]))
    graph = helper.make_graph(nodes, "SyntheticAudit", [helper.make_tensor_value_info("input", dtype, [1, 3, height, width])], [helper.make_tensor_value_info(output, dtype, [1, 1, height, width])], [initializer])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    if external:
        # Write the tiny external payload in Python; native ONNX file creation
        # is denied by this desktop sandbox, while ordinary workspace writes work.
        tensor = model.graph.initializer[0]
        payload = tensor.raw_data
        from onnx.external_data_helper import set_external_data
        set_external_data(tensor, location=f"{path.stem}.data", offset=0, length=len(payload))
        tensor.ClearField("raw_data")
        path.with_suffix(".data").write_bytes(payload)
        path.write_bytes(model.SerializeToString())
    else:
        onnx.save_model(model, path)


def metadata(path):
    return worker.inspect_model_artifact(str(path))


with tempfile.TemporaryDirectory(prefix="deployment-06-", dir=TEMP) as directory:
    root = Path(directory)
    external = root / "external.onnx"
    tiny_graph(external, 16, 16, external=True)
    declared_hash = hashlib.sha256(external.read_bytes()).hexdigest()
    (root / "external.manifest.json").write_text(json.dumps({"schema_version": 2, "model": "PatchCore", "artifact": {"path": external.name, "sha256": declared_hash}}), encoding="utf-8")
    first = metadata(external)
    sample = np.zeros((1, 3, 16, 16), dtype=np.float32)
    before = ort.InferenceSession(str(external), providers=["CPUExecutionProvider"]).run(None, {"input": sample})[0]
    external_data = root / "external.data"
    external_data.write_bytes(np.full((1, 1, 1, 1), 2, dtype=np.float32).tobytes())
    second = metadata(external)
    after = ort.InferenceSession(str(external), providers=["CPUExecutionProvider"]).run(None, {"input": sample})[0]
    external_data.unlink()
    missing = metadata(external)
    result["external_integrity"] = {"manifest_sha256": declared_hash, "fingerprint_equal_after_weight_mutation": first["fingerprint"] == second["fingerprint"], "mean_before": float(before.mean()), "mean_after": float(after.mean()), "missing_weight_file_still_inspected": bool(missing["fingerprint"])}

    with patch.object(torch.cuda, "is_available", return_value=False), patch.object(ort, "preload_dlls", return_value=None):
        dimension_path = root / "dimensions.onnx"
        tiny_graph(dimension_path, 384, 512)
        dimension_meta = metadata(dimension_path)
        correct_session = ort.InferenceSession(str(dimension_path), providers=["CPUExecutionProvider"])
        correct_session.run(None, {"input": np.zeros((1, 3, 384, 512), dtype=np.float32)})
        try:
            worker._OnnxInferenceBackend(dimension_meta)
            dimension_error = None
        except Exception as exc:
            dimension_error = f"{type(exc).__name__}: {exc}"
        result["input_shape"] = {"graph_shape": correct_session.get_inputs()[0].shape, "inspected_image_size": dimension_meta["model_parameters"]["PatchCore"]["image_size"], "graph_executes_with_correct_input": True, "worker_error": dimension_error}

        half_path = root / "float16.onnx"
        tiny_graph(half_path, dtype=TensorProto.FLOAT16)
        try:
            worker._OnnxInferenceBackend(metadata(half_path))
            half_error = None
        except Exception as exc:
            half_error = f"{type(exc).__name__}: {exc}"
        result["input_dtype"] = {"graph_type": "tensor(float16)", "worker_error": half_error}

        output_path = root / "outputs.onnx"
        tiny_graph(output_path, output="features")
        session = worker.PersistentInferenceSession(root / "assets", root / "session")
        event, data = session.load({"path": str(output_path)})
        image = root / "image.png"
        Image.new("RGB", (32, 32), (64, 64, 64)).save(image)
        try:
            session.predict({"image_path": str(image), "output_root": str(root / "prediction"), "threshold": 0.6})
            output_error = None
        except Exception as exc:
            output_error = f"{type(exc).__name__}: {exc}"
        result["output_contract"] = {"load_event": event, "backend": data["backend"], "graph_outputs": [item.name for item in session.backend.session.get_outputs()], "first_prediction_error": output_error}
        session.unload()

        conflict_path = root / "conflict.onnx"
        tiny_graph(conflict_path)
        graph_model = worker._detect_model_from_names(["memory_bank", "nearest_neighbors"])
        (root / "conflict.manifest.json").write_text(json.dumps({"schema_version": 2, "model": "PaDiM", "artifact": {"path": conflict_path.name, "sha256": hashlib.sha256(conflict_path.read_bytes()).hexdigest()}}), encoding="utf-8")
        event, data = session.load({"path": str(conflict_path)})
        result["model_identity_conflict"] = {"graph_detected_model": graph_model, "manifest_model": "PaDiM", "load_event": event, "returned_model": data["model"]}
        session.unload()

        raw_path = root / "raw-map.onnx"
        sigma = 1
        axis = np.arange(9, dtype=np.float64) - 4
        gaussian = np.exp(-(axis ** 2) / 2)
        gaussian /= gaussian.sum()
        kernel = np.outer(gaussian, gaussian).astype(np.float32)[None, None]
        raw_graph = helper.make_graph([
            helper.make_node("ReduceMean", ["input"], ["raw_anomaly_map"], axes=[1], keepdims=1, name="nearest_neighbors"),
            helper.make_node("Pad", ["raw_anomaly_map", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "gaussian_kernel"], ["smoothed"]),
            helper.make_node("Mul", ["smoothed", "calibration"], ["anomaly_map"]),
        ], "SyntheticCalibratedPatchCoreContract", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])], [helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, 1, 32, 32]), helper.make_tensor_value_info("raw_anomaly_map", TensorProto.FLOAT, [1, 1, 32, 32])], [numpy_helper.from_array(kernel, name="gaussian_kernel"), numpy_helper.from_array(np.array([0, 0, 4, 4, 0, 0, 4, 4], dtype=np.int64), name="pads"), numpy_helper.from_array(np.array(0.25, dtype=np.float32), name="calibration")])
        raw_model = helper.make_model(raw_graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
        raw_parameters = worker.normalize_model_parameters("PatchCore", {"sigma": 1, "image_size": {"height": 32, "width": 32}})
        worker._set_onnx_metadata(raw_model, worker.ONNX_METADATA_MODEL_PARAMETERS, json.dumps(worker.parameter_snapshot("PatchCore", raw_parameters)))
        worker._set_onnx_metadata(raw_model, worker.ONNX_METADATA_RAW_MAP_OUTPUT, "raw_anomaly_map")
        onnx.save_model(raw_model, raw_path)
        raw_backend = worker._OnnxInferenceBackend(metadata(raw_path))
        exported_values = raw_backend._run_array(np.full((32, 32, 3), 64, dtype=np.uint8))
        worker_values, _ = raw_backend.predict(image)
        result["raw_map_overwrites_calibrated_map"] = {"sigma_override_argument": None, "saved_sigma": 1, "same_sample_rgb_value": 64, "graph_calibrated_map_mean": float(exported_values["anomaly_map"].mean()), "worker_returned_map_mean": float(worker_values["anomaly_map"].mean()), "factor": float(worker_values["anomaly_map"].mean() / exported_values["anomaly_map"].mean()), "calibration_fixture": "pixel_min=0, pixel_max=4, pixel_threshold=2: clip((smoothed-2)/4+.5)=smoothed/4"}

    import anomalib.engine
    from anomalib.deploy import ExportType
    from anomalib.models.components.base.export_mixin.utils import create_export_root
    result["export_default_name"] = inspect.signature(anomalib.engine.Engine.export).parameters["model_file_name"].default
    calls = []

    class FakeEngine:
        def __init__(self, **_kwargs):
            pass

        def export(self, model, export_type, export_root, model_file_name="model", **_kwargs):
            path = create_export_root(export_root, ExportType(export_type)) / f"{model_file_name}.onnx"
            calls.append({"model": model, "path": str(path.relative_to(root))})
            path.write_text(model, encoding="utf-8")
            return path

    params = {name: worker.normalize_model_parameters(name) for name in ("PatchCore", "PaDiM")}
    exports = root / "exports"
    with patch.object(anomalib.engine, "Engine", FakeEngine), patch.object(worker, "preflight_selected_models", side_effect=lambda job, names: {name: params[name] for name in names}), patch.object(worker, "create_model", side_effect=lambda name, _params: name), patch.object(worker, "restore_checkpoint_state"), patch.object(worker, "make_postprocessor_onnx_export_friendly"), patch.object(worker, "add_patchcore_raw_map_output"), patch.object(worker, "embed_onnx_model_parameters"), patch.object(worker, "package_onnx_as_single_file"), patch.object(worker, "emit"), patch.object(torch.cuda, "is_available", return_value=False):
        for name in ("PatchCore", "PaDiM"):
            worker.run_real_export({"model": name, "checkpoint": str(root / f"{name}.ckpt"), "format": "ONNX", "output_root": str(exports), "asset_root": str(root / "assets")})
    exported = exports / "weights" / "onnx" / "model.onnx"
    manifest = json.loads(exported.with_name("model.manifest.json").read_text(encoding="utf-8"))
    result["mock_export_path_collision"] = {"calls": calls, "same_path": calls[0]["path"] == calls[1]["path"], "final_mock_content": exported.read_text(encoding="utf-8"), "final_manifest_model": manifest["model"], "probe_kind": "fake export, actual worker path/manifest logic and actual installed create_export_root"}

result["seconds"] = round(time.perf_counter() - started, 3)
result["limits"] = "Only newly constructed tiny graphs on CPU and a fake exporter; no project model/checkpoint loaded, no GPU execution or camera, no training/download/install."
(EVIDENCE / "probe-06-deployment-result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
