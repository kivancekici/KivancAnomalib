# Inspection Workbench

This checkout now lives at
`C:\Users\kivanc.ekici\Documents\GitHub\KivancAnomalib`.
See [local setup and migrated model/data locations](docs/LOCAL_SETUP.md) and
the [2026-10-14 migration verification](docs/migration/2026-10-14.md).
The original project filename and executable name are retained. Local data,
model weights, virtual environments and build outputs are excluded from Git;
a fresh clone needs its own Python environment and model/data assets.

A native Qt desktop workbench for:

- collecting normal and anomalous images from a Basler camera;
- converting and saving camera frames with OpenCV;
- training and testing several Anomalib models in an isolated Python process;
- comparing image/pixel metrics, latency, and checkpoint size;
- exporting the selected model to OpenVINO or ONNX;
- promoting an explicit production-model manifest.

## Architecture

The UI and camera pipeline stay native:

```text
Qt Widgets
├── CameraService
│   ├── Basler Pylon device discovery and single-frame grabbing
│   └── OpenCV BGR conversion and image writing
└── AnomalibProcess
    └── QProcess + JSON Lines
        └── worker/anomalib_worker.py
            ├── Folder datamodule
            ├── Engine.fit
            ├── Engine.test
            ├── Engine.predict (training preview and file/camera inference)
            └── Engine.export
```

Training and inference never run on the Qt UI thread. The Python worker emits
one JSON object per line for stage, progress, result, inference, environment,
asset status/progress/result, export, and error events.

## Build on this machine

The qmake project auto-enables the camera pipeline when it finds:

- Qt 6.11 MSVC 2022;
- `PYLON_DEV_DIR` or `C:\Program Files\Basler\pylon\Development`;
- OpenCV under `C:\opencv\build`.

From an x64 Native Tools prompt:

```bat
mkdir build\desktop-debug
cd build\desktop-debug
C:\Qt\6.11.1\msvc2022_64\bin\qmake.exe ..\..\KivancAnomaLibTest.pro -spec win32-msvc CONFIG+=debug
nmake
```

For a debug launch, ensure these directories are on `PATH`, or place the
required DLLs beside the executable:

```text
C:\Qt\6.11.1\msvc2022_64\bin
C:\Program Files\Basler\pylon\Runtime\x64
C:\opencv\build\x64\vc16\bin
```

## Configure Anomalib

Create or select a dedicated Python environment and install the Anomalib
version you intend to use. Then either:

- set `ANOMALIB_PYTHON` to that environment's `python.exe`; or
- choose the executable on the app's **Settings** page.

Use **Verify environment**, then turn off **Use demonstration results**.
Real jobs require Anomalib `>=2.5,<3`; verification reports an explicit error for
older or incompatible environments.

For NVIDIA-accelerated ONNX inference, install the CUDA 13 ONNX Runtime provider
in the same environment:

```powershell
python -m pip install -r requirements-inference.txt
```

The environment check reports the CUDA device, ONNX execution providers, and
OpenVINO devices it discovers. If the CUDA ONNX provider is unavailable, ONNX
models fall back to a compatible OpenVINO or CPU runtime.

## Persistent inference and loading models

The **Inspect** page keeps one model loaded in a dedicated Python process. Use
**Load model…** for an Anomalib `.ckpt`, `.pt`, `.onnx`, OpenVINO `.xml/.bin`,
or an app-generated `.json` model manifest. **Use trained winner** selects the
latest winning checkpoint from the training page. The last valid artifact is
remembered and loaded only when Inspect is opened.

Torch `.pt` exports use Python pickle. The app asks for confirmation before
loading a new file and remembers its SHA-256 fingerprint; a modified file must
be approved again. Lightning `.ckpt` detection/restoration uses weights-only
loading. Repeated predictions reuse the warmed model and report model latency
separately from total image/artifact latency.

For PatchCore checkpoints, **Inspect** can temporarily override sigma without
changing the checkpoint or its remembered settings. New app-generated ONNX
exports retain their normal smoothed outputs and also expose a
`raw_anomaly_map` output, allowing the same override in the workbench. Legacy
or third-party ONNX files and Torch/OpenVINO exports keep their saved,
graph-baked sigma and show the value as read-only.

## Prepare local model assets

Real training, inference, verification, and export jobs are offline by default.
Open **Settings → Local model assets** to check or explicitly prepare:

- PatchCore `wide_resnet50_2` weights;
- PaDiM `resnet18` weights;
- EfficientAD small and medium teacher weights;
- EfficientAD ImageNette data, either from the configured local directory or
  from a managed copy.

**Download missing assets…** first shows every public source, estimated size,
and the managed destination. Compatible weights already present in the user's
Hugging Face cache are copied and verified before a download is attempted. This
explicit action is the only worker job allowed to use the network.

Managed assets and `assets-lock.json` live under the discovered project root at
`.data/model-assets`. Downloads are checksummed where the provider publishes a
hash, staged, and promoted only after validation. Normal model jobs force
Hugging Face offline mode, disable telemetry, and install a socket guard.
Missing or corrupt resources produce a model-specific error directing the user
back to Settings; no dataset, result, or local path is uploaded.

## Configure training and model parameters

On **Train**, expand **Training settings** to configure epochs
(`1–2,000,000,000`), train/evaluation batch sizes, data workers, accelerator,
and device count. These values are remembered locally with Qt application
settings.

Each model card has a **Configure…** dialog with typed controls:

- All models: a checkpoint-bound input width and height, with common resolution
  presets and exact resizing. The default is `256 × 256`; PatchCore and PaDiM
  accept `32–2048` pixels per dimension, while EfficientAD accepts `256–2048`;
- PatchCore: backbone, feature layers, pretrained weights, coreset ratio,
  nearest neighbors, precision, Gaussian-map sigma, patch size, and patch
  stride;
- PaDiM: backbone, feature layers, pretrained weights, and retained features;
- EfficientAD: ImageNette directory, teacher channels, model size, learning
  rate, weight decay, and padding behavior.

EfficientAD requires a training batch size of one. The worker applies that
constraint only to EfficientAD while preserving the configured shared batch
size for other selected models. Every completed result records both the model
parameters and the effective training parameters. Inspect, export, and
production promotion reuse that checkpoint-bound snapshot, so later settings
changes affect only future training runs.

PatchCore defaults to sigma `4`, patch size `3`, and patch stride `1`. Sigma is
an integer from `1–32`. Patch size is an odd integer from `1–31`, and stride is
between `1` and the selected patch size. Patch size and stride are measured in
backbone feature-map positions rather than input-image pixels; both affect the
memory bank and therefore require retraining. Sigma controls Gaussian smoothing
of the pixel anomaly map and is used for validation, testing, and inference.

The worker expects a Folder-style dataset:

```text
dataset/
├── normal/       defect-free training images
├── abnormal/     known anomalous validation/test images
└── masks/        optional binary pixel masks
```

`good`, `anomalous`, `defect`, `bad`, `mask`, and `ground_truth` folder names
are also detected.

Selecting or rescanning a dataset immediately updates the Train page's
inspection panel with the first anomalous image (or the first normal image when
there is no anomalous folder) and its matching mask. Prediction surfaces remain
empty until a real model run emits that model's anomaly map and heatmap. The
threshold control then rebuilds the displayed predicted mask from the normalized
anomaly map.

The **Inspect** page displays the input size stored with the loaded model and
uses it automatically. It does not allow an inference-only size override,
because changing preprocessing dimensions requires retraining for comparable
results. Legacy artifacts without recorded dimensions use `256 × 256`.

## Test a trained model

After a training run completes, open **Inspect**:

1. Select one of the trained models. The workbench uses the exact checkpoint
   recorded for that model's result.
2. Either choose an image with **Browse…** or select a Basler device on
   **Capture** and click **Capture & test**.
3. Adjust the decision threshold if needed and run the test.

Inference runs in the isolated Python worker and returns the anomaly decision,
anomaly score, end-to-end prediction latency, heatmap, and thresholded pixel
mask. Camera frames are saved under `.data/inference-captures`; generated
prediction artifacts are stored under `.data/inference`.

When **Use demonstration results** is enabled, this flow produces synthetic
preview results without importing Anomalib. Turn demonstration mode off to load
the real checkpoint with `Engine.predict`.

## Current model set

- PatchCore
- PaDiM
- EfficientAD

The worker's model map is centralized in
`worker/anomalib_worker.py::create_model`, so adding another Anomalib model
does not require changing the process protocol.

## Verification

The native release, camera and demo-flow entries below are earlier source-project
checks. Current checkout validation and its limits are recorded in the
[migration report](docs/migration/2026-10-14.md).

- Native debug build: passed with MSVC 2022.
- Worker unit tests (37): `python -m unittest discover -s tests -v`.
- Native release build: passed with MSVC 2022.
- Basler discovery: one installed camera detected during the smoke test.
- Basler/OpenCV capture: a real PNG frame was acquired and written.
- Demo train/test/inference/export and inspection-artifact worker flow: passed.
- Visual comparison: see `design-qa.md`.

The **Inspect** page supports one-shot checkpoint inference from an image file
or a freshly acquired Basler frame. The Train page also renders one
representative prediction per completed model. Continuous exported-model
inference remains a separate deployment increment.
