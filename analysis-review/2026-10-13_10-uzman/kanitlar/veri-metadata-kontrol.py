"""Read-only source inspection plus tiny, synthetic metadata-only Folder checks.

No model, training, inference, camera, network, or source mutation is used.
All generated fixtures and evidence stay inside this review directory.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from worker import anomalib_worker as worker

worker.install_offline_network_guard()
from anomalib.data import Folder
import anomalib

EVIDENCE = Path(__file__).resolve().parent
FIXTURES = EVIDENCE / "rol02-sentetik-veri"
FIXTURES.mkdir(exist_ok=True)


def stub(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"metadata-only fixture; never decoded")


def paths(dm: Folder) -> dict:
    return {
        split: sorted(Path(p).relative_to(FIXTURES).as_posix() for p in getattr(dm, f"{split}_data").samples.image_path)
        for split in ("train", "val", "test")
    }


balanced = FIXTURES / "balanced"
for i in range(20):
    stub(balanced / "normal" / f"n{i:02}.png")
for i in range(8):
    stub(balanced / "abnormal" / f"a{i:02}.png")
job = {"dataset_name": "review_fixture", "dataset_root": str(balanced), "normal_dir": "normal", "abnormal_dir": "abnormal"}
params = {"train_batch_size": 1, "eval_batch_size": 1, "num_workers": 0}
runs = []
for _ in range(2):
    dm = worker.create_datamodule(job, params)
    dm.setup()
    runs.append(paths(dm))

seeded = []
for _ in range(2):
    dm = Folder(name="seeded", root=balanced, normal_dir="normal", abnormal_dir="abnormal", num_workers=0, seed=42)
    dm.setup()
    seeded.append(paths(dm))

tiny = FIXTURES / "tiny"
stub(tiny / "normal" / "n.png")
tiny_result = {}
try:
    dm = worker.create_datamodule({"dataset_root": str(tiny), "normal_dir": "normal"}, params)
    dm.setup()
    tiny_result = paths(dm)
except Exception as exc:
    tiny_result = {"error": f"{type(exc).__name__}: {exc}"}

ambiguous = FIXTURES / "ambiguous_masks"
requested = ambiguous / "abnormal" / "b" / "000.png"
stub(requested)
stub(ambiguous / "masks" / "a" / "000_mask.png")
stub(ambiguous / "masks" / "c" / "000_mask.png")
mask = worker.matching_mask({"dataset_root": str(ambiguous), "abnormal_dir": "abnormal", "mask_dir": "masks"}, requested)

extensions = FIXTURES / "extensions"
stub(extensions / "normal" / "n.png")
stub(extensions / "normal" / "extra.webp")
dm = Folder(name="extensions", root=extensions, normal_dir="normal", num_workers=0)
dm._setup()

result = {
    "anomalib_version": anomalib.__version__,
    "method": "Folder setup metadata only; fixture bytes are not images and are never decoded",
    "default_split_runs": runs,
    "default_split_equal": runs[0] == runs[1],
    "within_run_intersections": {a + "_" + b: sorted(set(runs[0][a]) & set(runs[0][b])) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
    "seed42_split_equal": seeded[0] == seeded[1],
    "tiny_normal_only": tiny_result,
    "ambiguous_requested_image": requested.relative_to(FIXTURES).as_posix(),
    "ambiguous_selected_mask": mask.relative_to(FIXTURES).as_posix() if mask else None,
    "worker_extension_count": len(worker.image_files(extensions / "normal")),
    "folder_extension_count": len(dm.train_data.samples),
}
(EVIDENCE / "veri-metadata-kontrol.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
