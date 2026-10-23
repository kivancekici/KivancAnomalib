from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

from worker import anomalib_worker as worker


def write_weights(path: Path, value: bytes = b"\x00\x00\x80?") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = json.dumps({"weight": {
        "dtype": "F32", "shape": [1], "data_offsets": [0, 4],
    }}).encode()
    header += b" " * (-len(header) % 8)
    path.write_bytes(len(header).to_bytes(8, "little") + header + value)


class AssetRegressionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def make_hf_cache(self, cache: Path, revision: str, value=b"\x00\x00\x80?"):
        definition = worker.ASSET_DEFINITIONS["patchcore_backbone"]
        repo = worker._hf_repo_directory(cache, definition["repo_id"])
        (repo / "refs").mkdir(parents=True)
        (repo / "refs/main").write_text(revision)
        weights = repo / "snapshots" / revision / definition["filename"]
        write_weights(weights, value)
        return repo, weights

    def make_replacement(self):
        old = self.root / "managed"
        new = self.root / "staging"
        old.mkdir()
        new.mkdir()
        (old / "weights").write_bytes(b"old")
        (new / "weights").write_bytes(b"new")
        return new, old

    def test_failed_promotion_restores_old_directory(self):
        staging, destination = self.make_replacement()
        original_replace = Path.replace

        def fail_promotion(path, target):
            if path == staging:
                raise PermissionError("rename blocked")
            return original_replace(path, target)

        with mock.patch.object(Path, "replace", fail_promotion):
            with self.assertRaisesRegex(PermissionError, "rename blocked"):
                worker._replace_directory(staging, destination)
        self.assertEqual((destination / "weights").read_bytes(), b"old")
        self.assertEqual((staging / "weights").read_bytes(), b"new")
        self.assertFalse(list(self.root.glob("*.backup")))

    def test_failed_metadata_commit_restores_old_directory(self):
        staging, destination = self.make_replacement()
        with self.assertRaisesRegex(OSError, "lock write failed"):
            worker._replace_directory(
                staging, destination,
                on_commit=mock.Mock(side_effect=OSError("lock write failed")),
            )
        self.assertEqual((destination / "weights").read_bytes(), b"old")
        self.assertEqual((staging / "weights").read_bytes(), b"new")

    def test_successful_commit_replaces_contents_and_removes_backup(self):
        staging, destination = self.make_replacement()
        committed = []
        worker._replace_directory(
            staging, destination,
            on_commit=lambda: committed.append((destination / "weights").read_bytes()),
        )
        self.assertEqual(committed, [b"new"])
        self.assertEqual((destination / "weights").read_bytes(), b"new")
        self.assertFalse(staging.exists())
        self.assertEqual(list(self.root.iterdir()), [destination])

    def test_corrupt_shared_cache_is_rejected_before_managed_replacement(self):
        asset_root = self.root / "assets"
        shared = self.root / "shared"
        _, old = self.make_hf_cache(worker._managed_hf_cache(asset_root), "old")
        _, corrupt = self.make_hf_cache(shared, "new")
        corrupt.write_bytes(b"not-safetensors")
        previous = old.read_bytes()
        with mock.patch.dict(os.environ, {"INSPECTION_SHARED_HF_CACHE": str(shared)}):
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                worker._import_hf_asset(
                    "patchcore_backbone", worker.ASSET_DEFINITIONS["patchcore_backbone"],
                    asset_root,
                )
        self.assertEqual(old.read_bytes(), previous)
        self.assertEqual(worker._hf_cached_file(
            worker._managed_hf_cache(asset_root), worker.ASSET_DEFINITIONS["patchcore_backbone"]
        )[1], "old")

    def test_import_lock_failure_restores_old_weights_and_lock(self):
        asset_root = self.root / "assets"
        shared = self.root / "shared"
        _, old = self.make_hf_cache(worker._managed_hf_cache(asset_root), "old")
        self.make_hf_cache(shared, "new", b"\x00\x00\x00@")
        worker._record_ready_asset("patchcore_backbone", asset_root, path=old,
                                   revision="old", source="imported")
        lock = worker._asset_lock_path(asset_root)
        previous = (old.read_bytes(), lock.read_bytes())
        with mock.patch.dict(os.environ, {"INSPECTION_SHARED_HF_CACHE": str(shared)}), \
                mock.patch.object(worker, "_write_asset_lock", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                worker._import_hf_asset(
                    "patchcore_backbone", worker.ASSET_DEFINITIONS["patchcore_backbone"],
                    asset_root,
                )
        self.assertEqual((old.read_bytes(), lock.read_bytes()), previous)
        self.assertTrue(worker._asset_state("patchcore_backbone", asset_root, {})["ready"])

    def test_network_download_is_validated_in_staging(self):
        asset_root = self.root / "assets"
        _, old = self.make_hf_cache(worker._managed_hf_cache(asset_root), "old")
        previous = old.read_bytes()

        def corrupt_download(**kwargs):
            _, downloaded = self.make_hf_cache(Path(kwargs["cache_dir"]), "new")
            downloaded.write_bytes(b"broken download")
            return str(downloaded)

        fake_hub = types.SimpleNamespace(hf_hub_download=corrupt_download)
        with mock.patch.dict(sys.modules, {"huggingface_hub": fake_hub}), \
                mock.patch.object(worker, "_import_hf_asset", return_value=None), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                worker._download_hf_asset(
                    "patchcore_backbone", worker.ASSET_DEFINITIONS["patchcore_backbone"],
                    asset_root,
                )
        self.assertEqual(old.read_bytes(), previous)

    def test_archive_lock_failure_preserves_previous_teacher_and_lock(self):
        asset_root = self.root / "assets"
        definition = worker.ASSET_DEFINITIONS["efficientad_teacher"]
        destination = asset_root / definition["destination"]
        destination.mkdir(parents=True)
        for filename in definition["required_files"]:
            (destination / filename).write_bytes(b"old teacher")
        worker._record_ready_asset("efficientad_teacher", asset_root,
                                   path=destination, source="downloaded")
        lock = worker._asset_lock_path(asset_root)
        previous_lock = lock.read_bytes()

        def fake_download(_id, _definition, partial):
            partial.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(partial, "w") as archive:
                for filename in definition["required_files"]:
                    archive.writestr(filename, b"new teacher")

        with mock.patch.object(worker, "_download_file", side_effect=fake_download), \
                mock.patch.object(worker, "_write_asset_lock", side_effect=OSError("disk full")), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(OSError, "disk full"):
                worker._download_archive_asset("efficientad_teacher", definition, asset_root)
        self.assertEqual(lock.read_bytes(), previous_lock)
        for filename in definition["required_files"]:
            self.assertEqual((destination / filename).read_bytes(), b"old teacher")

    def test_asset_lock_rejects_second_writer_and_releases_after_exception(self):
        asset_root = self.root / "assets"
        with self.assertRaisesRegex(ValueError, "cancelled"):
            with worker._AssetPreparationLock(asset_root):
                with self.assertRaisesRegex(RuntimeError, "Another model asset"):
                    with worker._AssetPreparationLock(asset_root):
                        self.fail("Second writer acquired lock")
                raise ValueError("cancelled")
        with worker._AssetPreparationLock(asset_root):
            pass


if __name__ == "__main__":
    unittest.main()
