from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np
from PIL import Image

from worker import anomalib_worker


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "worker" / "anomalib_worker.py"


def write_test_safetensors(path: Path) -> None:
    payload = b"\x00\x00\x80?"
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    padding = (-len(header)) % 8
    header += b" " * padding
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)


def run_job_process(job: dict) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
    with tempfile.TemporaryDirectory() as directory:
        temporary_root = Path(directory)
        prepared = dict(job)
        prepared.setdefault("asset_root", str(temporary_root / "model-assets"))
        prepared.setdefault("workspace_root", str(ROOT))
        job_path = temporary_root / "job.json"
        job_path.write_text(json.dumps(prepared), encoding="utf-8")
        process = subprocess.run(
            [sys.executable, "-u", str(WORKER), "--job", str(job_path)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    events = [
        json.loads(line)
        for line in process.stdout.splitlines()
        if line.strip()
    ]
    return process, events


def run_job(job: dict) -> list[dict]:
    process, events = run_job_process(job)
    if process.returncode:
        raise AssertionError(process.stderr or process.stdout)
    return events


class WorkerTests(unittest.TestCase):
    def test_inspect_checkpoint_detects_patchcore_without_pickle_execution(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.ckpt"
            torch.save(
                {
                    "state_dict": {
                        "model.memory_bank": torch.zeros((2, 3)),
                    },
                    "hyper_parameters": {
                        "backbone": "wide_resnet50_2",
                        "layers": ["layer2", "layer3"],
                        "pre_trained": True,
                        "coreset_sampling_ratio": 0.2,
                        "num_neighbors": 5,
                        "precision": "float32",
                        "sigma": 7,
                        "patch_size": 5,
                        "patch_stride": 2,
                        "image_size": {"width": 512, "height": 384},
                    },
                },
                path,
            )
            inspected = anomalib_worker.inspect_model_artifact(str(path))
            self.assertEqual(inspected["format"], "checkpoint")
            self.assertEqual(inspected["model"], "PatchCore")
            self.assertEqual(
                inspected["model_parameters"]["PatchCore"]["num_neighbors"], 5
            )
            self.assertEqual(
                inspected["model_parameters"]["PatchCore"]["sigma"], 7
            )
            self.assertEqual(
                inspected["model_parameters"]["PatchCore"]["patch_size"], 5
            )
            self.assertEqual(
                inspected["model_parameters"]["PatchCore"]["image_size"],
                {"width": 512, "height": 384},
            )
            self.assertTrue(inspected["sigma_override_supported"])
            self.assertEqual(inspected["saved_sigma"], 7)
            self.assertFalse(inspected["requires_trust"])

    def test_inspect_torch_export_requires_fingerprint_trust(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            path.write_bytes(b"untrusted pickle payload")
            inspected = anomalib_worker.inspect_model_artifact(str(path))
            self.assertEqual(inspected["format"], "torch")
            self.assertTrue(inspected["requires_trust"])
            self.assertEqual(len(inspected["fingerprint"]), 64)

    def test_manifest_resolves_relative_openvino_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.xml").write_text(
                '<net><layer name="memory_bank"/></net>', encoding="utf-8"
            )
            (root / "model.bin").write_bytes(b"weights")
            manifest = root / "model.manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "model": "PatchCore",
                        "artifact": {"format": "openvino", "path": "model.xml"},
                        "model_parameters": {
                            "PatchCore": {
                                "num_neighbors": 7,
                                "image_size": {"width": 640, "height": 480},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            inspected = anomalib_worker.inspect_model_artifact(str(manifest))
            self.assertEqual(inspected["format"], "openvino")
            self.assertEqual(inspected["model"], "PatchCore")
            self.assertEqual(
                inspected["model_parameters"]["PatchCore"]["num_neighbors"], 7
            )
            direct = anomalib_worker.inspect_model_artifact(str(root / "model.xml"))
            self.assertEqual(
                direct["model_parameters"]["PatchCore"]["image_size"],
                {"width": 640, "height": 480},
            )
            (root / "model.bin").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "matching .xml and .bin"):
                anomalib_worker.inspect_model_artifact(str(manifest))

    def test_patchcore_onnx_raw_map_capability_and_external_sigma(self) -> None:
        import onnx
        import onnxruntime as ort
        from onnx import TensorProto, helper, numpy_helper

        sigma = 2
        kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
        axis = np.arange(kernel_size, dtype=np.float64) - (kernel_size - 1) / 2.0
        gaussian = np.exp(-(axis**2) / (2.0 * sigma**2))
        gaussian /= gaussian.sum()
        kernel = np.outer(gaussian, gaussian).astype(np.float32)[None, None]
        padding = kernel_size // 2

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "patchcore.onnx"
            input_value = helper.make_tensor_value_info(
                "input", TensorProto.FLOAT, [1, 1, 16, 16]
            )
            output_value = helper.make_tensor_value_info(
                "anomaly_map", TensorProto.FLOAT, [1, 1, 16, 16]
            )
            graph = helper.make_graph(
                [
                    helper.make_node(
                        "Pad", ["input", "pads"], ["padded"], mode="reflect"
                    ),
                    helper.make_node(
                        "Conv", ["padded", "gaussian_kernel"], ["anomaly_map"]
                    ),
                ],
                "PatchCoreSynthetic",
                [input_value],
                [output_value],
                initializer=[
                    numpy_helper.from_array(
                        np.asarray(
                            [0, 0, padding, padding, 0, 0, padding, padding],
                            dtype=np.int64,
                        ),
                        name="pads",
                    ),
                    numpy_helper.from_array(kernel, name="gaussian_kernel"),
                    numpy_helper.from_array(
                        np.zeros((1, 1), dtype=np.float32), name="memory_bank"
                    ),
                ],
            )
            model = helper.make_model(
                graph,
                opset_imports=[helper.make_opsetid("", 13)],
                ir_version=10,
            )
            onnx.save(model, path)

            legacy = anomalib_worker.inspect_model_artifact(str(path))
            self.assertFalse(legacy["sigma_override_supported"])

            parameters = anomalib_worker.normalize_model_parameters(
                "PatchCore", {"sigma": sigma, "patch_size": 5, "patch_stride": 2}
            )
            anomalib_worker.add_patchcore_raw_map_output(path, parameters)
            inspected = anomalib_worker.inspect_model_artifact(str(path))
            self.assertTrue(inspected["sigma_override_supported"])
            self.assertEqual(
                inspected["raw_anomaly_map_output"],
                anomalib_worker.PATCHCORE_RAW_ANOMALY_MAP_OUTPUT,
            )
            self.assertEqual(
                inspected["model_parameters"]["PatchCore"]["patch_size"], 5
            )

            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            sample = np.random.default_rng(42).random(
                (1, 1, 16, 16), dtype=np.float32
            )
            names = [item.name for item in session.get_outputs()]
            values = dict(
                zip(names, session.run(None, {"input": sample}), strict=True)
            )
            externally_blurred = anomalib_worker.gaussian_blur_anomaly_map(
                values[anomalib_worker.PATCHCORE_RAW_ANOMALY_MAP_OUTPUT], sigma
            )
            np.testing.assert_allclose(
                externally_blurred,
                values["anomaly_map"],
                rtol=1e-4,
                atol=1e-5,
            )

    def test_persistent_session_requires_exact_pt_fingerprint(self) -> None:
        metadata = {
            "path": "model.pt",
            "format": "torch",
            "model": "",
            "model_parameters": {},
            "fingerprint": "abc123",
            "requires_trust": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = anomalib_worker.PersistentInferenceSession(root, root)
            with mock.patch.object(session, "inspect", return_value=metadata), mock.patch(
                "worker.anomalib_worker._TorchExportInferenceBackend"
            ) as constructor:
                event_name, returned = session.load(
                    {"path": "model.pt", "trusted_sha256": "wrong"}
                )
                self.assertEqual(event_name, "trust_required")
                self.assertEqual(returned["fingerprint"], "abc123")
                constructor.assert_not_called()

    def test_export_backends_resize_inputs_to_saved_dimensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "source.png"
            Image.new("RGB", (80, 60), (20, 40, 60)).save(image_path)

            onnx_backend = anomalib_worker._OnnxInferenceBackend.__new__(
                anomalib_worker._OnnxInferenceBackend
            )
            onnx_backend.np = np
            onnx_backend.input_size = (384, 512)
            onnx_backend.sigma_override_supported = False
            onnx_backend.raw_anomaly_map_output = ""
            captured: dict[str, tuple[int, ...]] = {}

            def run_array(image):
                captured["onnx"] = image.shape
                return {"anomaly_map": np.zeros((384, 512), dtype=np.float32)}

            onnx_backend._run_array = run_array
            onnx_backend.predict(image_path)
            self.assertEqual(captured["onnx"], (384, 512, 3))

            class FakeInferencer:
                def predict(self, image):
                    captured["openvino"] = image.shape
                    return {"anomaly_map": np.zeros((384, 512), dtype=np.float32)}

            openvino_backend = anomalib_worker._OpenVINOInferenceBackend.__new__(
                anomalib_worker._OpenVINOInferenceBackend
            )
            openvino_backend.np = np
            openvino_backend.input_size = (384, 512)
            openvino_backend.inferencer = FakeInferencer()
            openvino_backend.predict(image_path)
            self.assertEqual(captured["openvino"], (384, 512, 3))

    def test_persistent_session_reuses_loaded_backend_for_predictions(self) -> None:
        class FakeBackend:
            model_name = "PatchCore"
            model_parameters = anomalib_worker.normalize_model_parameters("PatchCore")
            backend_name = "Fake CUDA"
            device = "cuda:0"
            device_name = "Test GPU"
            saved_sigma = 4
            sigma_override_supported = True

            def __init__(self) -> None:
                self.calls = 0
                self.sigmas = []

            def predict(self, _image_path, sigma_override=None):
                self.calls += 1
                self.sigmas.append(sigma_override)
                return {
                    "anomaly_map": np.arange(16, dtype=np.float32).reshape(4, 4),
                    "pred_score": np.asarray([0.8], dtype=np.float32),
                }, 12.0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.png"
            Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8), mode="RGB").save(
                image_path
            )
            session = anomalib_worker.PersistentInferenceSession(root, root)
            backend = FakeBackend()
            session.backend = backend
            session.metadata = {
                "path": "model.ckpt",
                "format": "checkpoint",
                "fingerprint": "hash",
            }
            for index in range(2):
                command = {
                        "image_path": str(image_path),
                        "threshold": 0.5,
                        "output_root": str(root / f"run-{index}"),
                    }
                if index == 0:
                    command["sigma_override"] = 2
                result = session.predict(command)
                self.assertTrue(result["cache_hit"])
                self.assertEqual(result["device"], "cuda:0")
                self.assertEqual(result["effective_sigma"], 2 if index == 0 else 4)
                self.assertEqual(result["sigma_overridden"], index == 0)
            self.assertEqual(backend.calls, 2)
            self.assertEqual(backend.sigmas, [2, None])

    def test_pth_is_not_treated_as_a_complete_model_export(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pth"
            path.write_bytes(b"weights")
            with self.assertRaisesRegex(ValueError, "Unsupported model file"):
                anomalib_worker.inspect_model_artifact(str(path))

    def test_persistent_worker_protocol_echoes_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.pt"
            model_path.write_bytes(b"untrusted export")
            commands = "\n".join(
                json.dumps(command)
                for command in (
                    {
                        "request_id": "inspect-1",
                        "command": "inspect_model",
                        "path": str(model_path),
                    },
                    {"request_id": "stop-1", "command": "shutdown"},
                )
            ) + "\n"
            process = subprocess.run(
                [
                    sys.executable,
                    "-u",
                    str(WORKER),
                    "--serve",
                    "--asset-root",
                    str(root / "assets"),
                    "--session-root",
                    str(root / "session"),
                ],
                cwd=ROOT,
                input=commands,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(process.returncode, 0, process.stderr)
            events = [json.loads(line) for line in process.stdout.splitlines()]
            self.assertEqual(events[0]["event"], "session_ready")
            self.assertGreater(events[0]["pid"], 0)
            inspected = next(
                event for event in events if event["event"] == "model_inspected"
            )
            self.assertEqual(inspected["request_id"], "inspect-1")
            self.assertTrue(inspected["requires_trust"])
            stopped = next(
                event for event in events if event["event"] == "session_stopped"
            )
            self.assertEqual(stopped["request_id"], "stop-1")

    def test_demo_training_emits_results_and_winner(self) -> None:
        events = run_job(
            {
                "mode": "train",
                "demo": True,
                "models": ["PatchCore", "PaDiM", "EfficientAD"],
                "max_epochs": 1,
            }
        )
        results = [event for event in events if event["event"] == "result"]
        self.assertEqual([item["model"] for item in results],
                         ["PatchCore", "PaDiM", "EfficientAD"])
        finished = next(event for event in events if event["event"] == "finished")
        self.assertEqual(finished["winner"], "PatchCore")

    def test_demo_training_honors_requested_epoch_count(self) -> None:
        events = run_job(
            {
                "mode": "train",
                "demo": True,
                "models": ["PatchCore"],
                "max_epochs": 25,
            }
        )
        progress = [event for event in events if event["event"] == "progress"]
        self.assertEqual(len(progress), 25)
        self.assertEqual(progress[-1]["epoch"], 25)
        self.assertEqual(progress[-1]["max_epochs"], 25)
        self.assertEqual(progress[-1]["message"], "PatchCore: epoch 25/25")

    def test_demo_training_preserves_custom_model_parameters(self) -> None:
        events = run_job(
            {
                "mode": "train",
                "demo": True,
                "models": ["PatchCore", "EfficientAD"],
                "max_epochs": 2,
                "train_batch_size": 16,
                "eval_batch_size": 4,
                "num_workers": 0,
                "accelerator": "cpu",
                "devices": 1,
                "model_parameters": {
                    "PatchCore": {
                        "coreset_sampling_ratio": 0.25,
                        "num_neighbors": 5,
                        "precision": "float16",
                        "sigma": 6,
                        "patch_size": 5,
                        "patch_stride": 2,
                        "image_size": {"width": 512, "height": 384},
                    },
                    "EfficientAD": {
                        "model_size": "m",
                        "lr": 0.0002,
                    },
                },
            }
        )
        results = {
            event["model"]: event
            for event in events
            if event["event"] == "result"
        }
        patchcore = results["PatchCore"]
        self.assertEqual(
            patchcore["model_parameters"]["PatchCore"]["precision"], "float16"
        )
        self.assertEqual(
            patchcore["model_parameters"]["PatchCore"]["backbone"],
            "wide_resnet50_2",
        )
        self.assertEqual(
            patchcore["model_parameters"]["PatchCore"]["sigma"], 6
        )
        self.assertEqual(
            patchcore["model_parameters"]["PatchCore"]["patch_size"], 5
        )
        self.assertEqual(
            patchcore["model_parameters"]["PatchCore"]["patch_stride"], 2
        )
        self.assertEqual(
            patchcore["model_parameters"]["PatchCore"]["image_size"],
            {"width": 512, "height": 384},
        )
        self.assertEqual(patchcore["training_parameters"]["train_batch_size"], 16)
        self.assertEqual(patchcore["training_parameters"]["eval_batch_size"], 4)

        efficient_ad = results["EfficientAD"]
        self.assertEqual(
            efficient_ad["model_parameters"]["EfficientAD"]["model_size"], "m"
        )
        self.assertEqual(
            efficient_ad["training_parameters"]["train_batch_size"], 1
        )
        self.assertEqual(efficient_ad["training_parameters"]["accelerator"], "cpu")

    def test_legacy_demo_job_receives_default_parameter_snapshot(self) -> None:
        events = run_job(
            {
                "mode": "train",
                "demo": True,
                "models": ["PaDiM"],
                "max_epochs": 1,
            }
        )
        result = next(event for event in events if event["event"] == "result")
        parameters = result["model_parameters"]["PaDiM"]
        self.assertEqual(parameters["backbone"], "resnet18")
        self.assertEqual(parameters["layers"], ["layer1", "layer2", "layer3"])
        self.assertIsNone(parameters["n_features"])
        self.assertEqual(
            parameters["image_size"], {"width": 256, "height": 256}
        )

    def test_model_parameter_validation_rejects_unknown_and_invalid_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported PatchCore parameter"):
            anomalib_worker.normalize_model_parameters(
                "PatchCore", {"arbitrary": True}
            )
        with self.assertRaisesRegex(ValueError, "coreset_sampling_ratio"):
            anomalib_worker.normalize_model_parameters(
                "PatchCore", {"coreset_sampling_ratio": 0.0}
            )
        with self.assertRaisesRegex(ValueError, "at least one feature layer"):
            anomalib_worker.normalize_model_parameters(
                "PaDiM", {"layers": []}
            )
        for invalid in (0, 33, 4.0):
            with self.subTest(sigma=invalid), self.assertRaisesRegex(
                ValueError, "sigma"
            ):
                anomalib_worker.normalize_model_parameters(
                    "PatchCore", {"sigma": invalid}
                )
        with self.assertRaisesRegex(ValueError, "odd integer"):
            anomalib_worker.normalize_model_parameters(
                "PatchCore", {"patch_size": 4}
            )
        with self.assertRaisesRegex(ValueError, "patch_stride"):
            anomalib_worker.normalize_model_parameters(
                "PatchCore", {"patch_size": 3, "patch_stride": 4}
            )
        for invalid in (
            [256, 256],
            {"width": 256},
            {"width": True, "height": 256},
            {"width": 512, "height": 2_049},
        ):
            with self.subTest(image_size=invalid), self.assertRaisesRegex(
                ValueError, "image_size"
            ):
                anomalib_worker.normalize_model_parameters(
                    "PatchCore", {"image_size": invalid}
                )
        with self.assertRaisesRegex(ValueError, "image_size.height"):
            anomalib_worker.normalize_model_parameters(
                "EfficientAD",
                {"image_size": {"width": 256, "height": 255}},
            )

        bounded = anomalib_worker.normalize_model_parameters(
            "PatchCore", {"sigma": 32, "patch_size": 31, "patch_stride": 31}
        )
        self.assertEqual(bounded["sigma"], 32)
        self.assertEqual(bounded["patch_size"], 31)
        self.assertEqual(bounded["patch_stride"], 31)

    def test_training_parameter_validation_rejects_ranges_and_unknown_keys(self) -> None:
        billion_plus = anomalib_worker.normalize_training_parameters(
            {"max_epochs": 1_000_000_001}, "PatchCore"
        )
        self.assertEqual(billion_plus["max_epochs"], 1_000_000_001)
        with self.assertRaisesRegex(ValueError, "max_epochs"):
            anomalib_worker.normalize_training_parameters(
                {"max_epochs": 0}, "PatchCore"
            )
        with self.assertRaisesRegex(ValueError, "max_epochs"):
            anomalib_worker.normalize_training_parameters(
                {"max_epochs": 2_000_000_001}, "PatchCore"
            )
        with self.assertRaisesRegex(ValueError, "num_workers"):
            anomalib_worker.normalize_training_parameters(
                {"num_workers": 65}, "PaDiM"
            )
        with self.assertRaisesRegex(ValueError, "Unsupported training parameter"):
            anomalib_worker.training_parameters_from_job(
                {"training_parameters": {"strategy": "ddp"}}, "PatchCore"
            )

    def test_large_demo_epoch_count_uses_bounded_progress_events(self) -> None:
        requested_epochs = 1_000_000_001
        events = run_job(
            {
                "mode": "train",
                "demo": True,
                "models": ["PatchCore"],
                "max_epochs": requested_epochs,
            }
        )
        progress = [event for event in events if event["event"] == "progress"]
        self.assertLessEqual(len(progress), 101)
        self.assertEqual(progress[-1]["epoch"], requested_epochs)
        self.assertEqual(progress[-1]["max_epochs"], requested_epochs)
        result = next(event for event in events if event["event"] == "result")
        self.assertEqual(
            result["training_parameters"]["max_epochs"], requested_epochs
        )

    def test_create_model_forwards_normalized_constructor_arguments(self) -> None:
        captured: dict = {}

        def fake_patchcore(**kwargs):
            captured.update(kwargs)
            return "constructed"

        model = anomalib_worker.create_model(
            "PatchCore",
            {
                "coreset_sampling_ratio": 0.2,
                "num_neighbors": 3,
                "sigma": 6,
                "patch_size": 7,
                "patch_stride": 2,
            },
            constructors={"PatchCore": fake_patchcore},
        )
        self.assertEqual(model, "constructed")
        self.assertEqual(captured["coreset_sampling_ratio"], 0.2)
        self.assertEqual(captured["num_neighbors"], 3)
        self.assertEqual(captured["precision"], "float32")
        self.assertEqual(captured["layers"], ["layer2", "layer3"])
        self.assertNotIn("sigma", captured)
        self.assertNotIn("patch_size", captured)
        self.assertNotIn("patch_stride", captured)

    def test_create_model_configures_patchcore_pooling_and_sigma(self) -> None:
        model = anomalib_worker.create_model(
            "PatchCore",
            {
                "pre_trained": False,
                "sigma": 2,
                "patch_size": 5,
                "patch_stride": 2,
            },
        )
        pooler = model.model.feature_pooler
        self.assertEqual(pooler.kernel_size, 5)
        self.assertEqual(pooler.stride, 2)
        self.assertEqual(pooler.padding, 2)
        self.assertEqual(
            tuple(model.model.anomaly_map_generator.blur.kernel.shape),
            (1, 1, 17, 17),
        )
        self.assertEqual(model.hparams["sigma"], 2)
        self.assertEqual(model.hparams["patch_size"], 5)
        self.assertEqual(model.hparams["patch_stride"], 2)

    def test_create_model_configures_rectangular_preprocessor(self) -> None:
        cases = {
            "PatchCore": {"pre_trained": False},
            "PaDiM": {"pre_trained": False},
            "EfficientAD": {},
        }
        for model_name, parameters in cases.items():
            with self.subTest(model=model_name):
                parameters["image_size"] = {"width": 512, "height": 384}
                model = anomalib_worker.create_model(model_name, parameters)
                self.assertEqual(tuple(model.input_size), (384, 512))
                self.assertEqual(
                    dict(model.hparams["image_size"]),
                    {"width": 512, "height": 384},
                )

    def test_create_model_rejects_unsupported_constructor_fields(self) -> None:
        class NarrowPatchcore:
            def __init__(self, backbone):
                self.backbone = backbone

        with self.assertRaisesRegex(
            RuntimeError, "does not support PatchCore parameter"
        ):
            anomalib_worker.create_model(
                "PatchCore",
                {},
                constructors={"PatchCore": NarrowPatchcore},
            )

    def test_anomalib_version_guard_targets_2_5_and_newer_2_x(self) -> None:
        anomalib_worker.require_supported_anomalib_version("2.5.1")
        anomalib_worker.require_supported_anomalib_version("2.9.0rc1")
        with self.assertRaisesRegex(RuntimeError, "requires Anomalib >=2.5,<3"):
            anomalib_worker.require_supported_anomalib_version("2.4.2")
        with self.assertRaisesRegex(RuntimeError, "requires Anomalib >=2.5,<3"):
            anomalib_worker.require_supported_anomalib_version("3.0.0")

    def test_demo_export_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            events = run_job(
                {
                    "mode": "export",
                    "demo": True,
                    "model": "PatchCore",
                    "format": "OPENVINO",
                    "checkpoint": "demo/patchcore/model.ckpt",
                    "output_root": directory,
                }
            )
            exported = next(event for event in events if event["event"] == "exported")
            manifest_path = Path(exported["path"])
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["model_parameters"], exported["model_parameters"])
            self.assertEqual(
                manifest["training_parameters"], exported["training_parameters"]
            )
            self.assertFalse(
                manifest["capabilities"]["sigma_override"]["supported"]
            )

            onnx_events = run_job(
                {
                    "mode": "export",
                    "demo": True,
                    "model": "PatchCore",
                    "format": "ONNX",
                    "checkpoint": "demo/patchcore/model.ckpt",
                    "output_root": directory,
                }
            )
            onnx_exported = next(
                event for event in onnx_events if event["event"] == "exported"
            )
            onnx_manifest = json.loads(
                Path(onnx_exported["path"]).read_text(encoding="utf-8")
            )
            self.assertTrue(
                onnx_manifest["capabilities"]["sigma_override"]["supported"]
            )
            self.assertEqual(
                onnx_manifest["capabilities"]["sigma_override"]["raw_output"],
                "raw_anomaly_map",
            )

    def test_demo_inference_emits_score_and_visual_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "part.png"
            output_root = root / "inference"
            image = np.zeros((24, 32, 3), dtype=np.uint8)
            image[8:16, 12:20] = (240, 240, 240)
            Image.fromarray(image, mode="RGB").save(image_path)

            events = run_job(
                {
                    "mode": "inference",
                    "demo": True,
                    "model": "PatchCore",
                    "checkpoint": "demo/patchcore/model.ckpt",
                    "image_path": str(image_path),
                    "threshold": 0.60,
                    "sigma_override": 2,
                    "output_root": str(output_root),
                    "model_parameters": {
                        "PatchCore": {
                            "coreset_sampling_ratio": 0.2,
                            "num_neighbors": 4,
                        }
                    },
                }
            )

            result = next(event for event in events if event["event"] == "inference")
            self.assertEqual(result["model"], "PatchCore")
            self.assertEqual(
                result["model_parameters"]["PatchCore"]["num_neighbors"], 4
            )
            self.assertEqual(result["saved_sigma"], 4)
            self.assertEqual(result["effective_sigma"], 2)
            self.assertTrue(result["sigma_overridden"])
            self.assertTrue(result["sigma_override_supported"])
            self.assertIn(result["label"], ("normal", "anomalous"))
            self.assertIsInstance(result["anomaly_score"], float)
            self.assertGreaterEqual(result["latency_ms"], 0.0)
            for key in (
                "anomaly_map_path",
                "heatmap_path",
                "predicted_mask_path",
            ):
                self.assertTrue(Path(result[key]).is_file())

    def test_demo_training_references_selected_dataset_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abnormal = root / "abnormal"
            masks = root / "masks"
            abnormal.mkdir()
            masks.mkdir()
            sample = abnormal / "part-001.png"
            mask = masks / "part-001_mask.png"
            sample.write_bytes(b"sample")
            mask.write_bytes(b"mask")

            events = run_job(
                {
                    "mode": "train",
                    "demo": True,
                    "models": ["PatchCore"],
                    "max_epochs": 1,
                    "dataset_root": str(root),
                    "normal_dir": "normal",
                    "abnormal_dir": "abnormal",
                    "mask_dir": "masks",
                }
            )
            result = next(event for event in events if event["event"] == "result")
            inspection = result["inspection"]
            self.assertTrue(Path(inspection["image_path"]).samefile(sample))
            self.assertTrue(Path(inspection["mask_path"]).samefile(mask))
            self.assertEqual(inspection["sample_count"], 1)

    def test_inspection_artifacts_are_generated_from_prediction(self) -> None:
        class FakeEngine:
            def predict(self, **kwargs):
                self.kwargs = kwargs
                anomaly_map = np.linspace(0.0, 1.0, 6 * 8, dtype=np.float32)
                return [{"anomaly_map": anomaly_map.reshape(1, 6, 8)}]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            abnormal = root / "abnormal"
            masks = root / "masks"
            output = root / "results" / "patchcore"
            abnormal.mkdir()
            masks.mkdir()
            sample = abnormal / "part-001.png"
            mask = masks / "part-001_mask.png"
            Image.new("RGB", (16, 12), (60, 80, 100)).save(sample)
            Image.new("L", (16, 12), 255).save(mask)
            job = {
                "dataset_root": str(root),
                "normal_dir": "normal",
                "abnormal_dir": "abnormal",
                "mask_dir": "masks",
            }

            engine = FakeEngine()
            inspection = anomalib_worker.create_inspection_artifacts(
                job, "PatchCore", engine, object(), "", output
            )

            self.assertTrue(Path(inspection["image_path"]).samefile(sample))
            self.assertTrue(Path(inspection["mask_path"]).samefile(mask))
            for key in (
                "anomaly_map_path",
                "heatmap_path",
                "predicted_mask_path",
            ):
                self.assertTrue(Path(inspection[key]).is_file())
            with Image.open(inspection["heatmap_path"]) as heatmap:
                self.assertEqual(heatmap.size, (16, 12))
            with Image.open(inspection["predicted_mask_path"]) as predicted_mask:
                self.assertEqual(predicted_mask.size, (16, 12))
            self.assertEqual(engine.kwargs["data_path"], str(sample.resolve()))
            self.assertTrue(engine.kwargs["return_predictions"])

    def test_real_check_reports_missing_or_ready_environment(self) -> None:
        events = run_job({"mode": "check", "demo": False})
        environment = next(event for event in events if event["event"] == "environment")
        self.assertIn("ok", environment)
        self.assertIn("message", environment)

    def test_asset_check_emits_complete_status_protocol(self) -> None:
        events = run_job(
            {
                "mode": "assets",
                "action": "check",
                "demo": False,
                "imagenette_dir": "./not-present",
            }
        )
        statuses = [
            event for event in events if event["event"] == "asset_status"
        ]
        self.assertEqual(
            {event["asset_id"] for event in statuses},
            set(anomalib_worker.ASSET_DEFINITIONS),
        )
        for status in statuses:
            self.assertIn(status["state"], {"ready", "missing", "corrupt"})
            self.assertIn("source", status)
            self.assertIn("path", status)
            self.assertIn("revision", status)
        result = next(
            event
            for event in events
            if event["event"] == "asset_result"
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["asset_count"], 4)

    def test_pretrained_false_removes_backbone_preflight_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = {
                "asset_root": str(root / "assets"),
                "workspace_root": str(root),
            }
            for model_name in ("PatchCore", "PaDiM"):
                parameters = anomalib_worker.normalize_model_parameters(
                    model_name,
                    {"pre_trained": False},
                )
                prepared = anomalib_worker._preflight_model_assets(
                    job,
                    model_name,
                    parameters,
                )
                self.assertFalse(prepared["pre_trained"])

    def test_blank_cache_preflight_reports_every_selected_model_asset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            process, events = run_job_process(
                {
                    "mode": "train",
                    "demo": False,
                    "asset_root": str(root / "assets"),
                    "workspace_root": str(root),
                    "models": ["PatchCore", "PaDiM", "EfficientAD"],
                    "output_root": str(root / "output"),
                    "imagenette_dir": str(root / "imagenette"),
                }
            )
        self.assertNotEqual(process.returncode, 0)
        error = next(event for event in events if event["event"] == "error")
        message = error["message"]
        self.assertIn("PatchCore", message)
        self.assertIn("wide_resnet50_2", message)
        self.assertIn("PaDiM", message)
        self.assertIn("resnet18", message)
        self.assertIn("EfficientAD small teacher file is missing", message)
        self.assertIn("ImageNette", message)
        self.assertNotIn('"event": "hello"', process.stdout)

    def test_efficientad_preflight_distinguishes_data_and_teacher_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "assets"
            teacher_root = (
                asset_root
                / anomalib_worker.ASSET_DEFINITIONS[
                    "efficientad_teacher"
                ]["destination"]
            )
            teacher_root.mkdir(parents=True)
            (teacher_root / "pretrained_teacher_small.pth").write_bytes(b"small")
            job = {
                "asset_root": str(asset_root),
                "workspace_root": str(root),
            }
            parameters = anomalib_worker.normalize_model_parameters(
                "EfficientAD",
                {"model_size": "s", "imagenet_dir": str(root / "missing-data")},
            )
            with self.assertRaisesRegex(
                anomalib_worker.OfflineResourceError,
                "ImageNette",
            ) as missing_data:
                anomalib_worker._preflight_model_assets(
                    job,
                    "EfficientAD",
                    parameters,
                )
            self.assertNotIn(
                "small teacher file is missing",
                str(missing_data.exception),
            )

            imagenette = root / "local-imagenette" / "class-a"
            imagenette.mkdir(parents=True)
            (imagenette / "sample.jpg").write_bytes(b"image")
            parameters = anomalib_worker.normalize_model_parameters(
                "EfficientAD",
                {"model_size": "m", "imagenet_dir": str(imagenette.parent)},
            )
            with self.assertRaisesRegex(
                anomalib_worker.OfflineResourceError,
                "medium teacher file is missing",
            ):
                anomalib_worker._preflight_model_assets(
                    job,
                    "EfficientAD",
                    parameters,
                )

    def test_cached_wide_resnet_is_imported_and_revision_is_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared = root / "shared-hub"
            asset_root = root / "managed"
            definition = anomalib_worker.ASSET_DEFINITIONS[
                "patchcore_backbone"
            ]
            revision = "30f73aceaaa1911830a9795b83ab1908dba18719"
            repository = anomalib_worker._hf_repo_directory(
                shared,
                definition["repo_id"],
            )
            (repository / "refs").mkdir(parents=True)
            (repository / "refs" / "main").write_text(
                revision,
                encoding="utf-8",
            )
            snapshot = repository / "snapshots" / revision
            snapshot.mkdir(parents=True)
            write_test_safetensors(snapshot / definition["filename"])
            job = {
                "mode": "assets",
                "action": "download",
                "asset_root": str(asset_root),
                "workspace_root": str(root),
                "asset_ids": ["patchcore_backbone"],
            }
            output = io.StringIO()
            with mock.patch.dict(
                os.environ,
                {"INSPECTION_SHARED_HF_CACHE": str(shared)},
            ), redirect_stdout(output):
                anomalib_worker.run_assets(job)

            events = [
                json.loads(line)
                for line in output.getvalue().splitlines()
                if line.strip()
            ]
            result = next(
                event
                for event in events
                if event["event"] == "asset_result"
            )
            self.assertEqual(result["result"], "imported")
            state = anomalib_worker._asset_state(
                "patchcore_backbone",
                asset_root,
                job,
            )
            self.assertTrue(state["ready"])
            self.assertEqual(state["revision"], revision)
            lock = anomalib_worker._load_asset_lock(asset_root)
            self.assertEqual(
                lock["assets"]["patchcore_backbone"]["revision"],
                revision,
            )

    def test_invalid_managed_safetensors_is_reported_as_corrupt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "managed"
            definition = anomalib_worker.ASSET_DEFINITIONS[
                "patchcore_backbone"
            ]
            repository = anomalib_worker._hf_repo_directory(
                anomalib_worker._managed_hf_cache(asset_root),
                definition["repo_id"],
            )
            revision = "invalid-revision"
            (repository / "refs").mkdir(parents=True)
            (repository / "refs" / "main").write_text(
                revision,
                encoding="utf-8",
            )
            snapshot = repository / "snapshots" / revision
            snapshot.mkdir(parents=True)
            (snapshot / definition["filename"]).write_bytes(b"truncated")
            state = anomalib_worker._asset_state(
                "patchcore_backbone",
                asset_root,
                {
                    "asset_root": str(asset_root),
                    "workspace_root": str(root),
                },
            )
            self.assertFalse(state["ready"])
            self.assertEqual(state["state"], "corrupt")
            self.assertIn("safetensors", state["message"])

    def test_corrupt_download_never_replaces_valid_asset(self) -> None:
        class FakeResponse(io.BytesIO):
            headers = {"Content-Length": "12"}

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.close()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "assets"
            destination = asset_root / "managed" / "teacher"
            destination.mkdir(parents=True)
            valid = destination / "keep.txt"
            valid.write_text("valid", encoding="utf-8")
            definition = {
                "display_name": "Test archive",
                "source_url": "https://example.invalid/test.zip",
                "filename": "test.zip",
                "sha256": "0" * 64,
                "destination": "managed/teacher",
                "required_files": ["required.pth"],
            }
            with mock.patch.object(
                anomalib_worker,
                "urlopen",
                return_value=FakeResponse(b"not-an-archive"),
            ), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Checksum validation failed",
                ):
                    anomalib_worker._download_archive_asset(
                        "efficientad_teacher",
                        definition,
                        asset_root,
                    )
            self.assertEqual(valid.read_text(encoding="utf-8"), "valid")
            self.assertFalse(
                (asset_root / "downloads" / "test.zip.part").exists()
            )

    def test_interrupted_download_removes_partial_and_keeps_valid_asset(self) -> None:
        class InterruptedResponse:
            headers = {"Content-Length": "100"}

            def __init__(self):
                self.read_count = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self, _size):
                self.read_count += 1
                if self.read_count == 1:
                    return b"partial"
                raise OSError("connection interrupted")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset_root = root / "assets"
            destination = asset_root / "managed" / "teacher"
            destination.mkdir(parents=True)
            valid = destination / "keep.txt"
            valid.write_text("valid", encoding="utf-8")
            definition = {
                "display_name": "Interrupted archive",
                "source_url": "https://example.invalid/interrupted.zip",
                "filename": "interrupted.zip",
                "sha256": "",
                "destination": "managed/teacher",
                "required_files": ["required.pth"],
            }
            with mock.patch.object(
                anomalib_worker,
                "urlopen",
                return_value=InterruptedResponse(),
            ), redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(OSError, "connection interrupted"):
                    anomalib_worker._download_archive_asset(
                        "efficientad_teacher",
                        definition,
                        asset_root,
                    )
            self.assertEqual(valid.read_text(encoding="utf-8"), "valid")
            self.assertFalse(
                (
                    asset_root
                    / "downloads"
                    / "interrupted.zip.part"
                ).exists()
            )

    def test_unknown_asset_is_rejected_before_network_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            anomalib_worker,
            "urlopen",
        ) as network:
            with self.assertRaisesRegex(ValueError, "Unknown asset identifier"):
                anomalib_worker.run_assets(
                    {
                        "mode": "assets",
                        "action": "download",
                        "asset_root": str(Path(directory) / "assets"),
                        "workspace_root": directory,
                        "asset_ids": ["not-in-the-manifest"],
                    }
                )
            network.assert_not_called()

    def test_staged_zip_supports_part_suffix_and_rejects_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_archive = root / "asset.zip.part"
            with zipfile.ZipFile(valid_archive, "w") as archive:
                archive.writestr("bundle/file.txt", "ok")
            destination = root / "valid"
            anomalib_worker._safe_extract_archive(valid_archive, destination)
            self.assertEqual(
                (destination / "bundle" / "file.txt").read_text(
                    encoding="utf-8"
                ),
                "ok",
            )

            unsafe_archive = root / "unsafe.zip.part"
            with zipfile.ZipFile(unsafe_archive, "w") as archive:
                archive.writestr("../outside.txt", "bad")
            with self.assertRaisesRegex(RuntimeError, "Unsafe archive member"):
                anomalib_worker._safe_extract_archive(
                    unsafe_archive,
                    root / "unsafe",
                )
            self.assertFalse((root / "outside.txt").exists())

    def test_offline_socket_guard_reports_explicit_resource_error(self) -> None:
        code = (
            "import socket\n"
            "from worker import anomalib_worker as worker\n"
            "worker.install_offline_network_guard()\n"
            "try:\n"
            "    socket.create_connection(('example.com', 443))\n"
            "except Exception as exc:\n"
            "    print(type(exc).__name__ + ': ' + str(exc))\n"
        )
        process = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(process.returncode, 0)
        self.assertIn("OfflineResourceError", process.stdout)
        self.assertIn("Download required public assets", process.stdout)


if __name__ == "__main__":
    unittest.main()
