"""Offline regression coverage for comparable training runs and valid parameters."""

from contextlib import ExitStack, redirect_stdout
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

from PIL import Image

from worker import anomalib_worker as worker


def strict_json(text: str):
    def reject_constant(value: str):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, parse_constant=reject_constant)


class TrainingRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.offline = ExitStack()
        self.offline.enter_context(mock.patch.dict(os.environ, {
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "disabled",
        }))
        self.offline.enter_context(mock.patch.object(
            socket.socket, "connect", side_effect=AssertionError("Test attempted network access")
        ))
        self.addCleanup(self.offline.close)

    def _run_training(self, metrics, *, actual_datamodule=False, checkpoint_exists=True):
        """Exercise the real job/result boundary without fitting any model."""
        import anomalib.engine

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            for name, count in (("normal", 20), ("abnormal", 8)):
                destination = dataset / name
                destination.mkdir(parents=True)
                for index in range(count):
                    Image.new("RGB", (32, 32), (index, 20, 30)).save(
                        destination / f"{index:03}.png"
                    )
            checkpoint = root / "model.ckpt"
            if checkpoint_exists:
                checkpoint.write_bytes(b"isolated-checkpoint-fixture")
            models = list(metrics)
            parameters = {name: worker.normalize_model_parameters(name) for name in models}
            captured_seeds = []
            captured_splits = []
            original_create_datamodule = worker.create_datamodule

            def make_datamodule(job, training_parameters):
                captured_seeds.append(job["split_seed"])
                if actual_datamodule:
                    return original_create_datamodule(job, training_parameters)
                return None

            class StubEngine:
                def __init__(self, **kwargs):
                    self.best_model_path = str(checkpoint)

                def fit(self, *, model, datamodule):
                    if datamodule is not None:
                        datamodule.setup()
                        captured_splits.append({
                            split: sorted(getattr(datamodule, f"{split}_data").samples.image_path)
                            for split in ("train", "val", "test")
                        })

                def test(self, *, model, datamodule):
                    return [metrics[model]]

            job = {
                "dataset_root": str(dataset), "normal_dir": "normal", "abnormal_dir": "abnormal",
                "output_root": str(root / "output"), "asset_root": str(root / "assets"),
                "models": models, "num_workers": 0,
            }
            stream = io.StringIO()
            with mock.patch.object(anomalib.engine, "Engine", StubEngine), \
                 mock.patch.object(worker, "preflight_selected_models", return_value=parameters), \
                 mock.patch.object(worker, "configure_anomalib_asset_paths"), \
                 mock.patch.object(worker, "create_datamodule", side_effect=make_datamodule), \
                 mock.patch.object(worker, "create_model", side_effect=lambda name, params: name), \
                 mock.patch.object(worker, "create_inspection_artifacts", return_value={}), \
                 redirect_stdout(stream):
                worker.run_real_train(job)
            events = [strict_json(line) for line in stream.getvalue().splitlines()]
            snapshot = strict_json((root / "output" / "training-job.json").read_text(encoding="utf-8"))
            return events, snapshot, captured_seeds, captured_splits, job

    def test_models_use_identical_actual_folder_splits_and_bound_provenance(self):
        events, snapshot, seeds, splits, original_job = self._run_training(
            {"PatchCore": {"image_AUROC": 0.9}, "PaDiM": {"image_AUROC": 0.8}},
            actual_datamodule=True,
        )
        self.assertEqual(seeds, [42, 42])
        self.assertEqual(len(splits), 2)
        self.assertEqual(splits[0], splits[1])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertTrue(set(splits[0][left]).isdisjoint(splits[0][right]))
        self.assertNotIn("split_seed", original_job)
        self.assertTrue(snapshot["run_id"])
        self.assertEqual(snapshot["split_seed"], 42)
        expected_hash = hashlib.sha256(b"isolated-checkpoint-fixture").hexdigest()
        for result in (event for event in events if event["event"] == "result"):
            self.assertIs(result["demo"], False)
            self.assertIs(result["completed"], True)
            self.assertEqual(result["checkpoint_sha256"], expected_hash)
            for key in ("dataset_root", "run_id", "split_seed"):
                self.assertEqual(result[key], snapshot[key])
        self.assertEqual(events[-1]["winner"], "PatchCore")
        self.assertIs(events[-1]["comparison_available"], True)

    def test_nonfinite_scores_do_not_beat_a_scored_model_or_break_json(self):
        events, *_ = self._run_training({
            "PatchCore": {"image_AUROC": float("nan"), "pixel_AUROC": float("inf")},
            "PaDiM": {"image_AUROC": 0.99},
        })
        results = [event for event in events if event["event"] == "result"]
        self.assertIsNone(results[0]["image_auroc"])
        self.assertIsNone(results[0]["pixel_auroc"])
        self.assertEqual(events[-1]["winner"], "PaDiM")

    def test_unscored_run_has_no_winner_and_missing_checkpoint_has_no_hash(self):
        events, *_ = self._run_training(
            {"PatchCore": {}, "PaDiM": {"image_AUROC": float("-inf")}},
            checkpoint_exists=False,
        )
        self.assertEqual(events[-1]["winner"], "")
        self.assertIs(events[-1]["comparison_available"], False)
        self.assertTrue(any(event["event"] == "warning" for event in events))
        for result in (event for event in events if event["event"] == "result"):
            self.assertEqual(result["checkpoint_sha256"], "")

    def test_protocol_sanitizes_nested_nonfinite_numbers(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            worker.emit("result", values=[float("nan"), {"score": float("inf")}, 0.25])
        self.assertEqual(strict_json(stream.getvalue())["values"], [None, {"score": None}, 0.25])
        self.assertEqual(worker.metric_value(
            {"image_AUROC": float("nan"), "test_image_AUROC": 0.75},
            ["image_AUROC", "test_image_AUROC"],
        ), 0.75)

    def test_seed_zero_cannot_disable_anomalib_deterministic_split(self):
        for value in (0, -1, True, 2_147_483_648):
            with self.subTest(seed=value), self.assertRaisesRegex(ValueError, "split_seed"):
                worker.split_seed_from_job({"split_seed": value})
        self.assertEqual(worker.split_seed_from_job({"split_seed": 157}), 157)

    def test_sigma_dimension_boundary_matches_actual_gaussian_component(self):
        import torch
        from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

        with self.assertRaisesRegex(ValueError, "at most 7"):
            worker.normalize_model_parameters("PatchCore", {
                "image_size": {"width": 32, "height": 64}, "sigma": 8,
            })
        sigma = worker.validate_patchcore_sigma(7, (64, 32))
        result = AnomalyMapGenerator(sigma=sigma)(
            torch.zeros((1, 1, 4, 4)), image_size=(64, 32),
        )
        self.assertEqual(tuple(result.shape), (1, 1, 64, 32))

    def test_padim_layer_selection_accepts_only_constructible_feature_counts(self):
        from anomalib.models.image.padim.torch_model import PadimModel

        base = {"backbone": "resnet18", "layers": ["layer1"], "pre_trained": False}
        for value in (None, 65):
            with self.subTest(n_features=value), self.assertRaisesRegex(ValueError, "n_features"):
                worker.normalize_model_parameters("PaDiM", {**base, "n_features": value})
        accepted = worker.normalize_model_parameters("PaDiM", {**base, "n_features": 64})
        model = PadimModel(**{key: value for key, value in accepted.items() if key != "image_size"})
        self.assertEqual(model.n_features, model.n_features_original)
        with self.assertRaisesRegex(ValueError, "automatic n_features"):
            worker.normalize_model_parameters("PaDiM", {
                "backbone": "wide_resnet50_2", "layers": ["layer2"],
            })

    def test_teacher_channels_fail_before_model_construction(self):
        constructor = mock.Mock()
        for channels in (1, 383, 385, 65_536):
            with self.subTest(channels=channels), self.assertRaisesRegex(ValueError, "teacher_out_channels"):
                worker.create_model("EfficientAD", {"teacher_out_channels": channels},
                                    constructors={"EfficientAD": constructor})
        constructor.assert_not_called()
        self.assertEqual(worker.normalize_model_parameters("EfficientAD")["teacher_out_channels"], 384)


if __name__ == "__main__":
    unittest.main()
