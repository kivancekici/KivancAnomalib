from __future__ import annotations

from contextlib import redirect_stdout
import inspect
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from worker import anomalib_worker as worker


def calibrated_graph(path: Path, *, calibration: bool = True) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    axis = np.arange(9, dtype=np.float32) - 4
    gaussian = np.exp(-(axis**2) / 2)
    gaussian /= gaussian.sum()
    kernel = np.outer(gaussian, gaussian)[None, None].astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["input"], ["raw"], axes=[1], keepdims=1, name="patchcore.memory_bank.nearest_neighbors"),
            helper.make_node("Pad", ["raw", "pads"], ["padded"], mode="reflect"),
            helper.make_node("Conv", ["padded", "kernel"], ["smoothed"]),
            helper.make_node("Div", ["smoothed", "scale"], ["anomaly_map"]),
            helper.make_node("ReduceMax", ["anomaly_map"], ["pred_score"], axes=[1, 2, 3], keepdims=0),
        ],
        "calibrated_patchcore",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
        [helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, 1, 32, 32]), helper.make_tensor_value_info("pred_score", TensorProto.FLOAT, [1])],
        [numpy_helper.from_array(np.asarray([0, 0, 4, 4, 0, 0, 4, 4], dtype=np.int64), "pads"), numpy_helper.from_array(kernel, "kernel"), numpy_helper.from_array(np.asarray(4, dtype=np.float32), "scale")],
    )
    path.write_bytes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10).SerializeToString())
    parameters = worker.normalize_model_parameters("PatchCore", {"sigma": 1, "image_size": {"width": 32, "height": 32}})
    worker.add_patchcore_raw_map_output(path, parameters)
    if calibration:
        worker.embed_onnx_pixel_calibration(
            path,
            SimpleNamespace(post_processor=SimpleNamespace(enable_normalization=True, pixel_min=0.0, pixel_max=4.0, pixel_threshold=2.0)),
        )


def external_graph(path: Path) -> Path:
    from onnx import TensorProto, helper, numpy_helper
    from onnx.external_data_helper import set_external_data

    tensor = numpy_helper.from_array(np.ones((1, 1, 1, 1), dtype=np.float32), "memory_bank")
    payload = tensor.raw_data
    external = path.with_suffix(".data")
    set_external_data(tensor, location=external.name, offset=0, length=len(payload))
    tensor.ClearField("raw_data")
    external.write_bytes(payload)
    graph = helper.make_graph(
        [helper.make_node("Mul", ["input", "memory_bank"], ["anomaly_map"], name="patchcore.nearest_neighbors")],
        "external_patchcore",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 1, 1])],
        [helper.make_tensor_value_info("anomaly_map", TensorProto.FLOAT, [1, 1, 1, 1])],
        [tensor],
    )
    path.write_bytes(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=10).SerializeToString())
    return external


class InferenceRegressionTests(unittest.TestCase):
    def test_calibrated_masks_and_lossless_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            Image.new("RGB", (2, 2)).save(image)
            for index, values in enumerate(([[.1, .2], [.2, .1]], [[.9, .9], [.9, .9]], [[.5999, .6001], [.5999, .6001]])):
                with self.subTest(values=values):
                    array = np.asarray(values, dtype=np.float32)
                    result = worker.write_prediction_artifacts(image, {"anomaly_map": array, "pred_score": float(array.max())}, root / str(index), .6)
                    mask = np.asarray(Image.open(result["predicted_mask_path"]))[..., 0] > 0
                    np.testing.assert_array_equal(mask, array >= .6)
                    self.assertEqual((result["anomaly_map_width"], result["anomaly_map_height"]), (2, 2))
                    stored = np.frombuffer(Path(result["anomaly_map_float_path"]).read_bytes(), dtype="<f4").reshape(2, 2)
                    np.testing.assert_array_equal(stored, array)

    def test_training_preview_emits_same_calibrated_float_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            Image.new("RGB", (2, 2)).save(image)
            array = np.asarray([[.55, .65], [.55, .65]], dtype=np.float32)
            engine = SimpleNamespace(predict=lambda **kwargs: {"anomaly_map": array, "pred_score": .65, "pred_mask": np.ones((2, 2))})
            with mock.patch.object(worker, "dataset_inspection", return_value={"image_path": str(image)}):
                result = worker.create_inspection_artifacts({}, "PatchCore", engine, None, "", root / "model")
            np.testing.assert_array_equal(np.frombuffer(Path(result["anomaly_map_float_path"]).read_bytes(), dtype="<f4").reshape(2, 2), array)
            np.testing.assert_array_equal(np.asarray(Image.open(result["predicted_mask_path"]))[..., 0] > 0, array >= .6)

    def test_invalid_scores_are_rejected_before_any_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            Image.new("RGB", (2, 2)).save(image)
            for index, score in enumerate((None, float("nan"), float("inf"), -float("inf"), [], [.2, .3])):
                destination = root / str(index)
                with self.subTest(score=score), self.assertRaisesRegex(ValueError, "finite scalar pred_score"):
                    worker.write_prediction_artifacts(image, {"anomaly_map": np.zeros((2, 2)), "pred_score": score}, destination, .6)
                self.assertFalse(destination.exists())

    def test_nonfinite_map_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only finite"):
            worker.normalized_map(np.asarray([[0, np.nan], [1, 0]], dtype=np.float32))

    def test_uint8_dark_inputs_are_scaled_in_both_onnx_backends(self) -> None:
        for backend_type in (worker._OnnxInferenceBackend, worker._OpenVinoOnnxInferenceBackend):
            with self.subTest(backend=backend_type.__name__):
                backend = backend_type.__new__(backend_type)
                backend.np = np
                backend.input_name = backend.input = "input"
                backend.output_names = backend.outputs = ["anomaly_map"]
                observed = []
                def capture(inputs):
                    observed.append(inputs["input"].copy())
                    return inputs["input"][:, :1]
                backend.session = SimpleNamespace(run=lambda outputs, inputs: [capture(inputs)])
                backend.compiled_model = lambda inputs: {"anomaly_map": capture(inputs)}
                for code in (1, 2, 255):
                    backend._run_array(np.full((8, 8, 3), code, dtype=np.uint8))
                backend._run_array(np.ones((8, 8, 3), dtype=np.float32))
                np.testing.assert_allclose([float(value.max()) for value in observed], [1 / 255, 2 / 255, 1, 1])

    def test_generic_openvino_dark_inputs(self) -> None:
        from anomalib.deploy import OpenVINOInferencer

        backend = worker._OpenVINOInferenceBackend.__new__(worker._OpenVINOInferenceBackend)
        backend.np = np
        backend.input_size = (8, 8)
        observed = []
        def predict(image):
            prepared = OpenVINOInferencer.pre_process(image)
            observed.append(float(prepared.max()))
            return {"anomaly_map": prepared[:, :1]}
        backend.inferencer = SimpleNamespace(predict=predict)
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.png"
            for code in (1, 2, 255):
                Image.new("RGB", (8, 8), (code, code, code)).save(image)
                backend.predict(image)
        np.testing.assert_allclose(observed, [1 / 255, 2 / 255, 1])

    def test_single_shot_sigma_bounds_and_model_support(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            checkpoint = root / "never-loaded.ckpt"
            Image.new("RGB", (32, 32)).save(image)
            checkpoint.write_bytes(b"fixture; never deserialized")
            parameters = worker.normalize_model_parameters("PatchCore", {"sigma": 1, "image_size": {"width": 32, "height": 32}})
            job = {"model": "PatchCore", "image_path": str(image), "checkpoint": str(checkpoint), "output_root": str(root / "output"), "asset_root": str(root / "assets"), "model_parameters": {"PatchCore": parameters}, "sigma_override": 8}
            with mock.patch.object(worker, "preflight_selected_models", return_value={"PatchCore": parameters}):
                with self.assertRaisesRegex(ValueError, "sigma"):
                    worker.run_real_inference(job)
            with self.assertRaisesRegex(ValueError, "sigma"):
                worker.run_demo_inference(job)
            for function in (worker.run_real_inference, worker.run_demo_inference):
                with self.subTest(function=function.__name__), mock.patch.object(worker, "preflight_selected_models", return_value={"PaDiM": worker.normalize_model_parameters("PaDiM")}):
                    with self.assertRaisesRegex(ValueError, "only supported for PatchCore"):
                        function({**job, "model": "PaDiM", "sigma_override": 1})
            self.assertFalse((root / "output").exists())

    def test_single_shot_applies_sigma_after_restore(self) -> None:
        calls = []
        model = object()
        class FakeEngine:
            def __init__(self, **kwargs):
                pass
            def predict(self, **kwargs):
                self_test.assertIs(kwargs["model"], model)
                self_test.assertIsNone(kwargs["ckpt_path"])
                calls.append("predict")
                return {"anomaly_map": np.full((2, 2), .2, dtype=np.float32), "pred_score": .2}
        self_test = self
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            checkpoint = root / "never-loaded.ckpt"
            Image.new("RGB", (32, 32)).save(image)
            checkpoint.write_bytes(b"fixture; never deserialized")
            parameters = worker.normalize_model_parameters("PatchCore", {"sigma": 1, "image_size": {"width": 32, "height": 32}})
            job = {"model": "PatchCore", "image_path": str(image), "checkpoint": str(checkpoint), "output_root": str(root / "output"), "asset_root": str(root / "assets"), "sigma_override": 2}
            output = io.StringIO()
            with mock.patch.object(worker, "preflight_selected_models", return_value={"PatchCore": parameters}), mock.patch.object(worker, "configure_anomalib_asset_paths"), mock.patch.object(worker, "create_model", return_value=model), mock.patch.object(worker, "restore_checkpoint_state", side_effect=lambda model, path: calls.append("restore")) as restore, mock.patch.object(worker, "configure_patchcore_sigma", side_effect=lambda model, sigma: calls.append(f"sigma:{sigma}")), mock.patch("anomalib.engine.Engine", FakeEngine), redirect_stdout(output):
                worker.run_real_inference(job)
            restore.assert_called_once_with(model, checkpoint)
            self.assertEqual(calls, ["restore", "sigma:2", "predict"])
            result = json.loads(output.getvalue().splitlines()[-1])
            self.assertTrue(result["sigma_overridden"])
            self.assertEqual(result["effective_sigma"], 2)
            self.assertEqual(result["saved_sigma"], 1)

    def test_direct_checkpoint_sigma_bounds(self) -> None:
        backend = worker._CheckpointInferenceBackend.__new__(worker._CheckpointInferenceBackend)
        backend.model_name = "PatchCore"
        backend.model = object()
        backend.input_size = (32, 32)
        backend.saved_sigma = 1
        with mock.patch.object(worker, "configure_patchcore_sigma") as configure:
            with self.assertRaisesRegex(ValueError, "sigma"):
                backend.predict(Path("unused.png"), 8)
            configure.assert_not_called()

    def test_onnx_no_override_and_saved_sigma_preserve_calibration(self) -> None:
        import onnxruntime as ort
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "calibrated.onnx"
            calibrated_graph(path)
            image_path = root / "input.png"
            image_array = np.zeros((32, 32, 3), dtype=np.uint8)
            image_array[8:24, 8:24] = 128
            Image.fromarray(image_array).save(image_path)
            metadata = worker.inspect_model_artifact(str(path))
            self.assertTrue(metadata["sigma_override_supported"])
            with mock.patch.object(torch.cuda, "is_available", return_value=False), mock.patch.object(ort, "preload_dlls", return_value=None):
                backend = worker._OnnxInferenceBackend(metadata)
            direct = backend._run_array(image_array)
            prediction, _ = backend.predict(image_path)
            np.testing.assert_array_equal(prediction["anomaly_map"], direct["anomaly_map"])
            prediction, _ = backend.predict(image_path, 1)
            np.testing.assert_allclose(prediction["anomaly_map"], direct["anomaly_map"], rtol=1e-5, atol=1e-7)
            expected_override = worker.gaussian_blur_anomaly_map(direct[worker.PATCHCORE_RAW_ANOMALY_MAP_OUTPUT], 2) / 4
            prediction, _ = backend.predict(image_path, 2)
            np.testing.assert_allclose(prediction["anomaly_map"], expected_override, rtol=1e-5, atol=1e-7)
            np.testing.assert_array_equal(prediction["pred_score"], direct["pred_score"])
            # Exercise the OpenVINO adapter with the same real graph outputs.
            fallback = worker._OpenVinoOnnxInferenceBackend.__new__(worker._OpenVinoOnnxInferenceBackend)
            fallback.np = np
            fallback.input_size = (32, 32)
            fallback.sigma_override_supported = True
            fallback.pixel_calibration = metadata["pixel_calibration"]
            fallback.raw_anomaly_map_output = worker.PATCHCORE_RAW_ANOMALY_MAP_OUTPUT
            fallback._run_array = backend._run_array
            prediction, _ = fallback.predict(image_path)
            np.testing.assert_array_equal(prediction["anomaly_map"], direct["anomaly_map"])
            prediction, _ = fallback.predict(image_path, 1)
            np.testing.assert_allclose(prediction["anomaly_map"], direct["anomaly_map"], rtol=1e-5, atol=1e-7)

    def test_legacy_raw_graph_without_calibration_rejects_override(self) -> None:
        import onnxruntime as ort
        import torch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "legacy.onnx"
            calibrated_graph(path, calibration=False)
            metadata = worker.inspect_model_artifact(str(path))
            self.assertFalse(metadata["sigma_override_supported"])
            with mock.patch.object(torch.cuda, "is_available", return_value=False), mock.patch.object(ort, "preload_dlls", return_value=None):
                backend = worker._OnnxInferenceBackend(metadata)
            image = root / "input.png"
            Image.new("RGB", (32, 32), (64, 64, 64)).save(image)
            prediction, _ = backend.predict(image)
            self.assertAlmostEqual(float(prediction["anomaly_map"].mean()), 64 / 255 / 4, places=6)
            with self.assertRaisesRegex(ValueError, "not supported"):
                backend.predict(image, 1)

    def test_runtime_sigma_rejects_kernel_larger_than_map(self) -> None:
        with self.assertRaisesRegex(ValueError, "sigma"):
            worker.gaussian_blur_anomaly_map(np.ones((32, 32), dtype=np.float32), 8)

    def test_legacy_raw_graph_keeps_outputs_in_openvino_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "legacy.onnx"
            calibrated_graph(path, calibration=False)
            metadata = worker.inspect_model_artifact(str(path))
            session = worker.PersistentInferenceSession.__new__(worker.PersistentInferenceSession)
            session.backend = None
            session.metadata = {}
            session.device_cache = {metadata["fingerprint"]: "CPU"}
            session.session_root = root
            with mock.patch.object(worker, "_OnnxInferenceBackend", side_effect=RuntimeError("ORT unavailable")), mock.patch.object(worker, "_OpenVINOInferenceBackend", side_effect=AssertionError("generic fallback loses raw output")), mock.patch.object(worker.PersistentInferenceSession, "_release_accelerator_cache"):
                event, loaded = session.load({"path": str(path)})
            self.assertEqual(event, "model_loaded")
            self.assertEqual(loaded["device"], "CPU")
            self.assertFalse(loaded["sigma_override_supported"])
            image = root / "input.png"
            Image.new("RGB", (32, 32), (64, 64, 64)).save(image)
            prediction, _ = session.backend.predict(image)
            self.assertIn(worker.PATCHCORE_RAW_ANOMALY_MAP_OUTPUT, prediction)
            self.assertAlmostEqual(float(prediction["anomaly_map"].mean()), 64 / 255 / 4, places=4)
            with self.assertRaisesRegex(ValueError, "calibration"):
                session.backend.predict(image, 1)

    def test_export_postprocessor_matches_threshold_fallback_and_calibration(self) -> None:
        import torch
        from anomalib.data import InferenceBatch
        from anomalib.post_processing import PostProcessor

        processor = PostProcessor()
        processor.image_min.fill_(0)
        processor.image_max.fill_(4)
        processor.pixel_min.fill_(0)
        processor.pixel_max.fill_(4)
        processor._image_threshold.fill_(2)
        # Pixel threshold intentionally remains NaN; the property uses image threshold.
        predictions = InferenceBatch(pred_score=torch.tensor([1.0]), anomaly_map=torch.tensor([[[[.5, 1.5], [2.5, 3.5]]]]))
        expected = processor(predictions)
        model = SimpleNamespace(post_processor=processor)
        worker.make_postprocessor_onnx_export_friendly(model)
        actual = processor(predictions)
        torch.testing.assert_close(actual.pred_score, expected.pred_score)
        torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)
        torch.testing.assert_close(actual.pred_mask, expected.pred_mask)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            calibrated_graph(path, calibration=False)
            calibration = worker.embed_onnx_pixel_calibration(path, model)
            self.assertEqual(calibration, {"mode": "min_max", "minimum": 0, "maximum": 4, "threshold": 2})
            np.testing.assert_allclose(worker.calibrate_override_map(predictions.anomaly_map.numpy(), calibration), expected.anomaly_map.numpy())

    def test_export_postprocessor_preserves_disabled_normalization(self) -> None:
        import torch
        from anomalib.data import InferenceBatch
        from anomalib.post_processing import PostProcessor

        processor = PostProcessor(enable_normalization=False, enable_thresholding=False)
        predictions = InferenceBatch(pred_score=torch.tensor([2.0]), anomaly_map=torch.ones((1, 1, 2, 2)) * 3)
        model = SimpleNamespace(post_processor=processor)
        expected = processor(predictions)
        worker.make_postprocessor_onnx_export_friendly(model)
        actual = processor(predictions)
        torch.testing.assert_close(actual.pred_score, expected.pred_score)
        torch.testing.assert_close(actual.anomaly_map, expected.anomaly_map)
        self.assertIsNone(actual.pred_mask)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            calibrated_graph(path, calibration=False)
            self.assertEqual(worker.embed_onnx_pixel_calibration(path, model), {"mode": "identity"})

    def test_export_rejects_degenerate_calibration_range(self) -> None:
        from anomalib.post_processing import PostProcessor

        processor = PostProcessor()
        for name in ("image_min", "image_max", "pixel_min", "pixel_max"):
            getattr(processor, name).fill_(1)
        with self.assertRaisesRegex(ValueError, "maximum greater"):
            worker.make_postprocessor_onnx_export_friendly(SimpleNamespace(post_processor=processor))

    def test_external_weights_change_fingerprint_and_fail_saved_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "external.onnx"
            data = external_graph(path)
            initial = worker.inspect_model_artifact(str(path))
            self.assertEqual(len(initial["artifact_files"]), 2)
            manifest = path.with_name("external.manifest.json")
            manifest.write_text(json.dumps({"model": "PatchCore", "artifact": {"path": path.name, "sha256": initial["fingerprint"], "files": initial["artifact_files"]}}))
            self.assertEqual(worker.inspect_model_artifact(str(path))["fingerprint"], initial["fingerprint"])
            data.write_bytes(np.full((1, 1, 1, 1), 2, dtype=np.float32).tobytes())
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                worker.inspect_model_artifact(str(path))
            manifest.unlink()
            self.assertNotEqual(worker.inspect_model_artifact(str(path))["fingerprint"], initial["fingerprint"])

    def test_missing_or_truncated_external_weights_fail_inspection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "external.onnx"
            data = external_graph(path)
            data.write_bytes(b"\0")
            with self.assertRaisesRegex(ValueError, "exceeds|truncated"):
                worker.inspect_model_artifact(str(path))
            data.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "external tensor"):
                worker.inspect_model_artifact(str(path))

    def test_exports_are_unique_and_failure_preserves_first_delivery(self) -> None:
        from anomalib.engine import Engine
        from anomalib.models.components.base.export_mixin.utils import create_export_root

        default_name = inspect.signature(Engine.export).parameters["model_file_name"].default
        fail = False
        class FakeEngine:
            def __init__(self, **kwargs):
                pass
            def export(self, model, export_type, export_root, model_file_name=default_name, **kwargs):
                destination = create_export_root(export_root, export_type) / f"{model_file_name}.pt"
                destination.write_text(model)
                if fail:
                    raise RuntimeError("simulated export failure after artifact creation")
                return destination

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {"model": "PatchCore", "format": "TORCH", "checkpoint": str(root / "unused.ckpt"), "output_root": str(root / "exports"), "asset_root": str(root / "assets")}
            capture = io.StringIO()
            with mock.patch("anomalib.engine.Engine", FakeEngine), mock.patch.object(worker, "create_model", side_effect=lambda name, parameters: name), mock.patch.object(worker, "restore_checkpoint_state"), mock.patch.object(worker, "preflight_selected_models", side_effect=lambda job, names: {name: worker.normalize_model_parameters(name) for name in names}), mock.patch.object(worker, "configure_anomalib_asset_paths"), redirect_stdout(capture):
                worker.run_real_export(job)
                first = json.loads(capture.getvalue().splitlines()[-1])
                first_path = Path(first["path"])
                first_manifest = Path(first["manifest_path"])
                saved = (first_path.read_bytes(), first_manifest.read_bytes())
                fail = True
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    worker.run_real_export({**job, "model": "PaDiM"})
                self.assertEqual((first_path.read_bytes(), first_manifest.read_bytes()), saved)
                self.assertEqual(len(list((root / "exports").iterdir())), 1)
                fail = False
                worker.run_real_export({**job, "model": "PaDiM"})
                second = json.loads(capture.getvalue().splitlines()[-1])
            self.assertNotEqual(first["path"], second["path"])
            self.assertEqual((first_path.read_bytes(), first_manifest.read_bytes()), saved)
            self.assertEqual(Path(second["path"]).read_text(), "PaDiM")
            self.assertEqual(json.loads(Path(second["manifest_path"]).read_text())["model"], "PaDiM")
            self.assertFalse(list((root / "exports").glob("*.staging")))


if __name__ == "__main__":
    unittest.main()
