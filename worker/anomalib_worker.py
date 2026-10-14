#!/usr/bin/env python3
"""JSON-lines worker that isolates Anomalib jobs from the Qt UI process."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
import logging
import os
from pathlib import Path
import re
import shutil
import socket
import tarfile
import time
import traceback
from typing import Any
from urllib.request import Request, urlopen
import uuid
import zipfile


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
SUPPORTED_MODELS = ("PatchCore", "PaDiM", "EfficientAD")
SUPPORTED_BACKBONES = ("wide_resnet50_2", "resnet18")
SUPPORTED_LAYERS = ("layer1", "layer2", "layer3", "layer4")
SUPPORTED_ACCELERATORS = ("auto", "cpu", "gpu")
DEFAULT_IMAGE_SIZE = {"width": 256, "height": 256}
MAXIMUM_IMAGE_DIMENSION = 2_048
PATCHCORE_RUNTIME_PARAMETER_KEYS = {"sigma", "patch_size", "patch_stride"}
PATCHCORE_RAW_ANOMALY_MAP_OUTPUT = "raw_anomaly_map"
ONNX_METADATA_MODEL_PARAMETERS = "inspection_workbench.model_parameters"
ONNX_METADATA_RAW_MAP_OUTPUT = "inspection_workbench.raw_anomaly_map_output"
TRAINING_PARAMETER_KEYS = {
    "max_epochs",
    "train_batch_size",
    "eval_batch_size",
    "num_workers",
    "accelerator",
    "devices",
}
MODEL_PARAMETER_DEFAULTS: dict[str, dict[str, Any]] = {
    "PatchCore": {
        "backbone": "wide_resnet50_2",
        "layers": ["layer2", "layer3"],
        "pre_trained": True,
        "coreset_sampling_ratio": 0.1,
        "num_neighbors": 9,
        "precision": "float32",
        "sigma": 4,
        "patch_size": 3,
        "patch_stride": 1,
        "image_size": deepcopy(DEFAULT_IMAGE_SIZE),
    },
    "PaDiM": {
        "backbone": "resnet18",
        "layers": ["layer1", "layer2", "layer3"],
        "pre_trained": True,
        "n_features": None,
        "image_size": deepcopy(DEFAULT_IMAGE_SIZE),
    },
    "EfficientAD": {
        "imagenet_dir": "./datasets/imagenette",
        "teacher_out_channels": 384,
        "model_size": "s",
        "lr": 0.0001,
        "weight_decay": 0.00001,
        "padding": False,
        "pad_maps": True,
        "image_size": deepcopy(DEFAULT_IMAGE_SIZE),
    },
}

ASSET_SCHEMA_VERSION = 1
ASSET_DEFINITIONS: dict[str, dict[str, Any]] = {
    "patchcore_backbone": {
        "display_name": "PatchCore pretrained backbone (wide_resnet50_2)",
        "kind": "huggingface",
        "source": "Hugging Face Hub",
        "source_url": "https://huggingface.co/timm/wide_resnet50_2.racm_in1k",
        "repo_id": "timm/wide_resnet50_2.racm_in1k",
        "provider_id": "timm/wide_resnet50_2.racm_in1k",
        "filename": "model.safetensors",
        "expected_files": ["model.safetensors"],
        "model_dependencies": ["PatchCore"],
        "estimated_bytes": 275_835_296,
    },
    "padim_backbone": {
        "display_name": "PaDiM pretrained backbone (resnet18)",
        "kind": "huggingface",
        "source": "Hugging Face Hub",
        "source_url": "https://huggingface.co/timm/resnet18.a1_in1k",
        "repo_id": "timm/resnet18.a1_in1k",
        "provider_id": "timm/resnet18.a1_in1k",
        "filename": "model.safetensors",
        "expected_files": ["model.safetensors"],
        "model_dependencies": ["PaDiM"],
        "estimated_bytes": 47_000_000,
    },
    "efficientad_teacher": {
        "display_name": "EfficientAD teacher weights (small and medium)",
        "kind": "archive",
        "source": "Anomalib GitHub release",
        "source_url": (
            "https://github.com/open-edge-platform/anomalib/releases/download/"
            "efficientad_pretrained_weights/efficientad_pretrained_weights.zip"
        ),
        "filename": "efficientad_pretrained_weights.zip",
        "provider_id": "open-edge-platform/anomalib:efficientad_pretrained_weights",
        "sha256": "c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
        "destination": "anomalib/pre_trained/efficientad_pretrained_weights",
        "required_files": [
            "pretrained_teacher_small.pth",
            "pretrained_teacher_medium.pth",
        ],
        "expected_files": [
            "pretrained_teacher_small.pth",
            "pretrained_teacher_medium.pth",
        ],
        "model_dependencies": ["EfficientAD"],
        "estimated_bytes": 85_000_000,
    },
    "efficientad_imagenette": {
        "display_name": "EfficientAD ImageNette auxiliary data",
        "kind": "archive",
        "source": "fast.ai S3",
        "source_url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        "filename": "imagenette2.tgz",
        "provider_id": "fast-ai/imagenette2",
        "sha256": "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
        "destination": "datasets/imagenette",
        "expected_files": ["<class>/<image>"],
        "model_dependencies": ["EfficientAD"],
        "estimated_bytes": 1_560_000_000,
    },
}


class OfflineResourceError(RuntimeError):
    """Raised when a real job is missing a required local model asset."""


def require_supported_anomalib_version(version: str) -> None:
    """Require the Anomalib 2.5+ constructor API used by this worker."""
    match = re.match(r"^\s*(\d+)\.(\d+)", str(version))
    if not match:
        raise RuntimeError(
            f"Could not determine the installed Anomalib version from {version!r}. "
            "Install Anomalib 2.5 or newer (but lower than 3.0)."
        )
    major, minor = (int(value) for value in match.groups())
    if major != 2 or minor < 5:
        raise RuntimeError(
            f"Unsupported Anomalib version {version}. "
            "Inspection Workbench requires Anomalib >=2.5,<3."
        )


def _require_int(value: Any, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer.")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


def _require_float(
    value: Any,
    name: str,
    minimum: float,
    maximum: float,
    *,
    include_minimum: bool = True,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number.")
    converted = float(value)
    valid_minimum = converted >= minimum if include_minimum else converted > minimum
    if not valid_minimum or converted > maximum:
        operator = "between" if include_minimum else "greater than"
        detail = (
            f"{operator} {minimum} and {maximum}"
            if include_minimum
            else f"{operator} {minimum} and at most {maximum}"
        )
        raise ValueError(f"{name} must be {detail}.")
    return converted


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false.")
    return value


def normalize_image_size(model_name: str, value: Any) -> dict[str, int]:
    """Validate and normalize the app-owned model input dimensions."""
    if not isinstance(value, dict) or set(value) != {"width", "height"}:
        raise ValueError(
            f"{model_name}.image_size must contain only width and height."
        )
    minimum = 256 if model_name == "EfficientAD" else 32
    return {
        "width": _require_int(
            value["width"],
            f"{model_name}.image_size.width",
            minimum,
            MAXIMUM_IMAGE_DIMENSION,
        ),
        "height": _require_int(
            value["height"],
            f"{model_name}.image_size.height",
            minimum,
            MAXIMUM_IMAGE_DIMENSION,
        ),
    }


def image_size_tuple(parameters: dict[str, Any]) -> tuple[int, int]:
    """Return Anomalib's (height, width) tuple from app parameters."""
    image_size = parameters["image_size"]
    return int(image_size["height"]), int(image_size["width"])


def normalize_training_parameters(
    job: dict[str, Any], model_name: str
) -> dict[str, Any]:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")
    accelerator = job.get("accelerator", "auto")
    if not isinstance(accelerator, str):
        raise ValueError("accelerator must be a string.")
    accelerator = accelerator.strip().lower()
    if accelerator not in SUPPORTED_ACCELERATORS:
        raise ValueError("accelerator must be one of: auto, cpu, gpu.")
    parameters = {
        "max_epochs": _require_int(
            job.get("max_epochs", 5), "max_epochs", 1, 2_000_000_000
        ),
        "train_batch_size": _require_int(
            job.get("train_batch_size", 8), "train_batch_size", 1, 1_024
        ),
        "eval_batch_size": _require_int(
            job.get("eval_batch_size", 8), "eval_batch_size", 1, 1_024
        ),
        "num_workers": _require_int(
            job.get("num_workers", 2), "num_workers", 0, 64
        ),
        "accelerator": accelerator,
        "devices": _require_int(job.get("devices", 1), "devices", 1, 16),
    }
    if model_name == "EfficientAD":
        parameters["train_batch_size"] = 1
    return parameters


def normalize_model_parameters(
    model_name: str, parameters: dict[str, Any] | None = None
) -> dict[str, Any]:
    if model_name not in MODEL_PARAMETER_DEFAULTS:
        raise ValueError(f"Unsupported model: {model_name}")
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        raise ValueError(f"{model_name} parameters must be an object.")

    defaults = MODEL_PARAMETER_DEFAULTS[model_name]
    unknown = sorted(set(parameters) - set(defaults))
    if unknown:
        raise ValueError(
            f"Unsupported {model_name} parameter(s): {', '.join(unknown)}."
        )
    values = deepcopy(defaults)
    values.update(parameters)
    values["image_size"] = normalize_image_size(
        model_name, values["image_size"]
    )

    if model_name in ("PatchCore", "PaDiM"):
        backbone = values["backbone"]
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"{model_name} backbone must be one of: "
                f"{', '.join(SUPPORTED_BACKBONES)}."
            )
        layers = values["layers"]
        if not isinstance(layers, list) or not layers:
            raise ValueError(f"{model_name} must select at least one feature layer.")
        if (
            any(not isinstance(layer, str) for layer in layers)
            or len(set(layers)) != len(layers)
            or any(layer not in SUPPORTED_LAYERS for layer in layers)
        ):
            raise ValueError(
                f"{model_name} layers must be unique values from: "
                f"{', '.join(SUPPORTED_LAYERS)}."
            )
        values["layers"] = list(layers)
        values["pre_trained"] = _require_bool(
            values["pre_trained"], f"{model_name}.pre_trained"
        )

    if model_name == "PatchCore":
        values["coreset_sampling_ratio"] = _require_float(
            values["coreset_sampling_ratio"],
            "PatchCore.coreset_sampling_ratio",
            0.0,
            1.0,
            include_minimum=False,
        )
        values["num_neighbors"] = _require_int(
            values["num_neighbors"], "PatchCore.num_neighbors", 1, 1_024
        )
        if values["precision"] not in ("float32", "float16"):
            raise ValueError("PatchCore.precision must be float32 or float16.")
        values["sigma"] = _require_int(
            values["sigma"], "PatchCore.sigma", 1, 32
        )
        values["patch_size"] = _require_int(
            values["patch_size"], "PatchCore.patch_size", 1, 31
        )
        if values["patch_size"] % 2 == 0:
            raise ValueError("PatchCore.patch_size must be an odd integer.")
        values["patch_stride"] = _require_int(
            values["patch_stride"],
            "PatchCore.patch_stride",
            1,
            values["patch_size"],
        )
    elif model_name == "PaDiM":
        n_features = values["n_features"]
        if n_features is not None:
            values["n_features"] = _require_int(
                n_features, "PaDiM.n_features", 1, 65_536
            )
    else:
        imagenet_dir = values["imagenet_dir"]
        if not isinstance(imagenet_dir, str) or not imagenet_dir.strip():
            raise ValueError("EfficientAD.imagenet_dir must be a directory path.")
        values["imagenet_dir"] = imagenet_dir.strip()
        values["teacher_out_channels"] = _require_int(
            values["teacher_out_channels"],
            "EfficientAD.teacher_out_channels",
            1,
            65_536,
        )
        if values["model_size"] not in ("s", "m"):
            raise ValueError("EfficientAD.model_size must be s or m.")
        values["lr"] = _require_float(
            values["lr"], "EfficientAD.lr", 0.0, 1.0, include_minimum=False
        )
        values["weight_decay"] = _require_float(
            values["weight_decay"], "EfficientAD.weight_decay", 0.0, 1.0
        )
        values["padding"] = _require_bool(
            values["padding"], "EfficientAD.padding"
        )
        values["pad_maps"] = _require_bool(
            values["pad_maps"], "EfficientAD.pad_maps"
        )
    return values


def model_parameters_from_job(
    job: dict[str, Any], model_name: str
) -> dict[str, Any]:
    parameter_map = job.get("model_parameters", {})
    if not isinstance(parameter_map, dict):
        raise ValueError("model_parameters must be an object keyed by model name.")
    unknown_models = sorted(set(parameter_map) - set(SUPPORTED_MODELS))
    if unknown_models:
        raise ValueError(
            f"Unsupported model parameter group(s): {', '.join(unknown_models)}."
        )
    raw_parameters = parameter_map.get(model_name, {})
    return normalize_model_parameters(model_name, raw_parameters)


def training_parameters_from_job(
    job: dict[str, Any], model_name: str
) -> dict[str, Any]:
    raw_parameters = job.get("training_parameters")
    if raw_parameters is None:
        return normalize_training_parameters(job, model_name)
    if not isinstance(raw_parameters, dict):
        raise ValueError("training_parameters must be an object.")
    unknown = sorted(set(raw_parameters) - TRAINING_PARAMETER_KEYS)
    if unknown:
        raise ValueError(
            f"Unsupported training parameter(s): {', '.join(unknown)}."
        )
    return normalize_training_parameters(raw_parameters, model_name)


def parameter_snapshot(model_name: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {model_name: deepcopy(parameters)}


def emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **payload}, ensure_ascii=False), flush=True)


def load_job(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def asset_root_from_job(job: dict[str, Any]) -> Path:
    raw_root = job.get("asset_root")
    if not isinstance(raw_root, str) or not raw_root.strip():
        raise ValueError("asset_root must be an absolute directory path.")
    root = Path(raw_root).expanduser()
    if not root.is_absolute():
        raise ValueError("asset_root must be an absolute directory path.")
    return root.resolve()


def _asset_lock_path(asset_root: Path) -> Path:
    return asset_root / "assets-lock.json"


def _load_asset_lock(asset_root: Path) -> dict[str, Any]:
    path = _asset_lock_path(asset_root)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"schema_version": ASSET_SCHEMA_VERSION, "assets": {}}
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != ASSET_SCHEMA_VERSION
        or not isinstance(value.get("assets"), dict)
    ):
        return {"schema_version": ASSET_SCHEMA_VERSION, "assets": {}}
    return value


def _write_asset_lock(asset_root: Path, lock: dict[str, Any]) -> None:
    asset_root.mkdir(parents=True, exist_ok=True)
    path = _asset_lock_path(asset_root)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(lock, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _supported_model_name(value: Any) -> str | None:
    """Return the canonical supported model name for a loose string value."""
    compact = re.sub(r"[^a-z0-9]", "", str(value).lower())
    aliases = {
        "patchcore": "PatchCore",
        "padim": "PaDiM",
        "efficientad": "EfficientAD",
    }
    for alias, model_name in aliases.items():
        if alias in compact:
            return model_name
    return None


def _detect_model_from_names(names: list[str]) -> str | None:
    """Detect one of the supported models from state or graph symbol names."""
    text = "\n".join(str(name).lower() for name in names)
    matches: set[str] = set()
    if any(token in text for token in ("memory_bank", "nearest_neighbors", "nearest-neighbors")):
        matches.add("PatchCore")
    if any(
        token in text
        for token in (
            "gaussian.mean",
            "gaussian.inv_covariance",
            "inv_covariance",
            "multi_variate_gaussian",
        )
    ):
        matches.add("PaDiM")
    if (
        ("teacher" in text and "student" in text)
        or "student_teacher" in text
        or "autoencoder" in text
    ):
        matches.add("EfficientAD")
    return next(iter(matches)) if len(matches) == 1 else None


def _model_parameters_from_hparams(
    model_name: str,
    hyper_parameters: Any,
    base_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw = hyper_parameters if isinstance(hyper_parameters, dict) else {}
    allowed = MODEL_PARAMETER_DEFAULTS[model_name]
    selected = {key: raw[key] for key in allowed if key in raw}
    if model_name == "EfficientAD":
        if "imagenet_dir" in selected:
            selected["imagenet_dir"] = str(selected["imagenet_dir"])
        model_size = selected.get("model_size")
        model_size = getattr(model_size, "value", model_size)
        if model_size in ("small", "medium"):
            selected["model_size"] = {"small": "s", "medium": "m"}[model_size]
    merged = deepcopy(base_parameters) if isinstance(base_parameters, dict) else {}
    merged.update(selected)
    return normalize_model_parameters(model_name, merged)


def _manifest_sigma_capability(manifest: dict[str, Any]) -> str:
    capabilities = manifest.get("capabilities")
    if not isinstance(capabilities, dict):
        return ""
    sigma_override = capabilities.get("sigma_override")
    if not isinstance(sigma_override, dict) or not sigma_override.get("supported"):
        return ""
    raw_output = sigma_override.get("raw_output")
    return str(raw_output) if isinstance(raw_output, str) else ""


def _resolve_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read model manifest {path}: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ValueError("Model manifest must contain a JSON object.")

    artifact = manifest.get("artifact")
    if isinstance(artifact, dict):
        artifact_path = artifact.get("path")
    else:
        artifact_path = manifest.get("checkpoint") or manifest.get("path")
    if not isinstance(artifact_path, str) or not artifact_path.strip():
        raise ValueError("Model manifest does not contain an artifact path.")

    resolved = Path(artifact_path).expanduser()
    if not resolved.is_absolute():
        resolved = path.parent / resolved
    model_name = _supported_model_name(manifest.get("model"))
    model_parameters: dict[str, Any] = {}
    raw_parameter_map = manifest.get("model_parameters")
    if model_name and isinstance(raw_parameter_map, dict):
        raw_parameters = raw_parameter_map.get(model_name, raw_parameter_map)
        if isinstance(raw_parameters, dict):
            model_parameters = normalize_model_parameters(model_name, raw_parameters)
    return {
        "path": resolved.resolve(),
        "manifest_path": path.resolve(),
        "manifest": manifest,
        "model": model_name,
        "model_parameters": model_parameters,
        "training_parameters": (
            manifest.get("training_parameters")
            if isinstance(manifest.get("training_parameters"), dict)
            else {}
        ),
        "raw_anomaly_map_output": _manifest_sigma_capability(manifest),
    }


def _artifact_format(path: Path) -> str:
    suffix = path.suffix.lower()
    formats = {
        ".ckpt": "checkpoint",
        ".pt": "torch",
        ".onnx": "onnx",
        ".xml": "openvino",
        ".bin": "openvino",
    }
    try:
        return formats[suffix]
    except KeyError as exc:
        raise ValueError(
            "Unsupported model file. Choose .ckpt, .pt, .onnx, .xml, .bin, or a model manifest .json."
        ) from exc


def _canonical_artifact_path(path: Path) -> Path:
    """Treat either OpenVINO companion file as the same model artifact."""
    resolved = path.resolve()
    return resolved.with_suffix(".xml") if resolved.suffix.lower() == ".bin" else resolved


def inspect_model_artifact(path_value: str) -> dict[str, Any]:
    """Validate an artifact and return safe metadata without executing pickle."""
    selected_path = Path(path_value).expanduser()
    if not selected_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {selected_path}")

    manifest_info: dict[str, Any] = {}
    artifact_path = selected_path.resolve()
    if selected_path.suffix.lower() == ".json":
        manifest_info = _resolve_manifest(artifact_path)
        artifact_path = manifest_info["path"]
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Manifest model artifact does not exist: {artifact_path}"
            )
    else:
        companion_manifest = selected_path.with_name(
            f"{selected_path.stem}.manifest.json"
        )
        if companion_manifest.is_file():
            manifest_info = _resolve_manifest(companion_manifest.resolve())
            described_artifact = Path(manifest_info["path"])
            if _canonical_artifact_path(described_artifact) != _canonical_artifact_path(
                artifact_path
            ):
                raise ValueError(
                    f"Companion manifest {companion_manifest} does not describe "
                    f"the selected artifact {selected_path}."
                )
            artifact_path = described_artifact

    artifact_format = _artifact_format(artifact_path)
    if artifact_format == "openvino":
        xml_path = artifact_path.with_suffix(".xml")
        bin_path = artifact_path.with_suffix(".bin")
        if not xml_path.is_file() or not bin_path.is_file():
            raise FileNotFoundError(
                "OpenVINO models require matching .xml and .bin files in the same folder."
            )
        artifact_path = xml_path.resolve()

    fingerprint_paths = [artifact_path]
    if artifact_format == "openvino":
        fingerprint_paths.append(artifact_path.with_suffix(".bin"))
    if len(fingerprint_paths) == 1:
        fingerprint_value = _sha256(fingerprint_paths[0])
    else:
        fingerprint = hashlib.sha256()
        for fingerprint_path in sorted(
            fingerprint_paths, key=lambda item: item.suffix
        ):
            fingerprint.update(fingerprint_path.suffix.lower().encode("ascii"))
            fingerprint.update(_sha256(fingerprint_path).encode("ascii"))
        fingerprint_value = fingerprint.hexdigest()
    if manifest_info:
        artifact_spec = manifest_info["manifest"].get("artifact")
        expected_hash = (
            artifact_spec.get("sha256") if isinstance(artifact_spec, dict) else None
        )
        if expected_hash and str(expected_hash).lower() != fingerprint_value:
            raise ValueError(
                "Model artifact fingerprint does not match its manifest. "
                "The weights may have been modified or corrupted."
            )

    model_name = manifest_info.get("model")
    model_parameters = manifest_info.get("model_parameters", {})
    raw_anomaly_map_output = manifest_info.get("raw_anomaly_map_output", "")
    checkpoint: dict[str, Any] | None = None
    if artifact_format == "checkpoint":
        import torch

        checkpoint = torch.load(
            artifact_path, map_location="cpu", weights_only=True
        )
        if not isinstance(checkpoint, dict) or not isinstance(
            checkpoint.get("state_dict"), dict
        ):
            raise ValueError("Lightning checkpoint does not contain a state_dict.")
        state_names = list(checkpoint["state_dict"].keys())
        detected = _detect_model_from_names(state_names)
        if model_name and detected and model_name != detected:
            raise ValueError(
                f"Manifest identifies {model_name}, but checkpoint weights look like {detected}."
            )
        model_name = model_name or detected
        if model_name:
            model_parameters = _model_parameters_from_hparams(
                model_name,
                checkpoint.get("hyper_parameters"),
                model_parameters,
            )
    elif artifact_format == "onnx":
        import onnx

        onnx_model = onnx.load(str(artifact_path), load_external_data=False)
        graph = onnx_model.graph
        names = [node.name for node in graph.node]
        names += [value.name for value in graph.initializer]
        names += [value.name for value in graph.input]
        names += [value.name for value in graph.output]
        model_name = model_name or _detect_model_from_names(names)
        metadata = {item.key: item.value for item in onnx_model.metadata_props}
        embedded_parameters = metadata.get(ONNX_METADATA_MODEL_PARAMETERS)
        if model_name and embedded_parameters:
            try:
                decoded = json.loads(embedded_parameters)
            except (TypeError, json.JSONDecodeError):
                decoded = {}
            if isinstance(decoded, dict):
                raw_parameters = decoded.get(model_name, decoded)
                if isinstance(raw_parameters, dict):
                    merged_parameters = deepcopy(model_parameters)
                    merged_parameters.update(raw_parameters)
                    model_parameters = normalize_model_parameters(
                        model_name, merged_parameters
                    )
        embedded_raw_output = metadata.get(ONNX_METADATA_RAW_MAP_OUTPUT, "")
        if embedded_raw_output in {value.name for value in graph.output}:
            raw_anomaly_map_output = embedded_raw_output
    elif artifact_format == "openvino" and not model_name:
        try:
            graph_text = artifact_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ValueError(f"Could not inspect OpenVINO graph: {exc}") from exc
        model_name = _detect_model_from_names([graph_text])

    requires_trust = artifact_format == "torch"
    if not model_name and not requires_trust:
        raise ValueError(
            "Could not unambiguously identify this artifact as PatchCore, PaDiM, or EfficientAD. "
            "Select an app-generated manifest alongside the model."
        )
    if model_name and not model_parameters:
        model_parameters = normalize_model_parameters(model_name)
    sigma_override_supported = bool(
        model_name == "PatchCore"
        and (
            artifact_format == "checkpoint"
            or (
                artifact_format == "onnx"
                and raw_anomaly_map_output == PATCHCORE_RAW_ANOMALY_MAP_OUTPUT
            )
        )
    )

    return {
        "selected_path": str(selected_path.resolve()),
        "path": str(artifact_path),
        "manifest_path": (
            str(manifest_info["manifest_path"]) if manifest_info else ""
        ),
        "format": artifact_format,
        "model": model_name or "",
        "model_parameters": (
            parameter_snapshot(model_name, model_parameters) if model_name else {}
        ),
        "training_parameters": manifest_info.get("training_parameters", {}),
        "fingerprint": fingerprint_value,
        "requires_trust": requires_trust,
        "saved_sigma": (
            model_parameters.get("sigma", 4) if model_name == "PatchCore" else None
        ),
        "sigma_override_supported": sigma_override_supported,
        "raw_anomaly_map_output": raw_anomaly_map_output,
    }


def _safetensors_validation_error(path: Path) -> str:
    """Validate the self-describing safetensors layout without loading tensors."""
    try:
        size = path.stat().st_size
        if size < 16:
            return "Cached safetensors file is too small."
        with path.open("rb") as stream:
            header_size = int.from_bytes(stream.read(8), "little")
            if header_size <= 0 or header_size > size - 8:
                return "Cached safetensors header size is invalid."
            header = json.loads(stream.read(header_size).decode("utf-8"))
        if not isinstance(header, dict):
            return "Cached safetensors header is invalid."
        data_size = size - 8 - header_size
        tensors = [
            value
            for key, value in header.items()
            if key != "__metadata__" and isinstance(value, dict)
        ]
        if not tensors:
            return "Cached safetensors file contains no tensors."
        for tensor in tensors:
            offsets = tensor.get("data_offsets")
            if (
                not isinstance(offsets, list)
                or len(offsets) != 2
                or any(not isinstance(value, int) for value in offsets)
                or offsets[0] < 0
                or offsets[0] > offsets[1]
                or offsets[1] > data_size
            ):
                return "Cached safetensors tensor offsets are invalid."
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return f"Could not validate cached safetensors: {exc}"
    return ""


def _hf_repo_directory(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / f"models--{repo_id.replace('/', '--')}"


def _hf_cached_file(
    cache_dir: Path,
    definition: dict[str, Any],
) -> tuple[Path | None, str]:
    repo_dir = _hf_repo_directory(cache_dir, str(definition["repo_id"]))
    revision = ""
    reference = repo_dir / "refs" / "main"
    try:
        revision = reference.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    if revision:
        candidate = repo_dir / "snapshots" / revision / str(definition["filename"])
        if candidate.is_file():
            return candidate, revision
    snapshots = repo_dir / "snapshots"
    if snapshots.is_dir():
        for snapshot in sorted(snapshots.iterdir(), reverse=True):
            candidate = snapshot / str(definition["filename"])
            if candidate.is_file():
                return candidate, snapshot.name
    return None, revision


def _managed_hf_cache(asset_root: Path) -> Path:
    return asset_root / "huggingface" / "hub"


def _shared_hf_cache() -> Path:
    configured = os.environ.get("INSPECTION_SHARED_HF_CACHE", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "huggingface" / "hub"


def _managed_imagenette_path(asset_root: Path) -> Path:
    return asset_root / str(ASSET_DEFINITIONS["efficientad_imagenette"]["destination"])


def _configured_imagenette_path(job: dict[str, Any]) -> Path | None:
    raw_path = job.get("imagenette_dir")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        workspace = Path(str(job.get("workspace_root", "."))).expanduser()
        path = workspace / path
    return path.resolve()


def _contains_images(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return next(
            (
                True
                for item in path.rglob("*")
                if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
            ),
            False,
        )
    except OSError:
        return False


def _teacher_files(asset_root: Path) -> list[Path]:
    definition = ASSET_DEFINITIONS["efficientad_teacher"]
    destination = asset_root / str(definition["destination"])
    return [destination / str(name) for name in definition["required_files"]]


def _asset_state(
    asset_id: str,
    asset_root: Path,
    job: dict[str, Any],
) -> dict[str, Any]:
    definition = ASSET_DEFINITIONS[asset_id]
    lock = _load_asset_lock(asset_root)
    locked = lock["assets"].get(asset_id, {})
    status: dict[str, Any] = {
        "event": "asset_status",
        "asset_id": asset_id,
        "display_name": definition["display_name"],
        "source": definition["source"],
        "source_url": definition["source_url"],
        "estimated_bytes": definition.get("estimated_bytes", 0),
        "state": "missing",
        "ready": False,
        "path": "",
        "revision": "",
        "import_available": False,
        "message": "Missing",
    }

    if definition["kind"] == "huggingface":
        status["path"] = str(
            _hf_repo_directory(
                _managed_hf_cache(asset_root),
                str(definition["repo_id"]),
            ).resolve()
        )
        managed_file, revision = _hf_cached_file(
            _managed_hf_cache(asset_root),
            definition,
        )
        shared_file, _ = _hf_cached_file(_shared_hf_cache(), definition)
        status["import_available"] = bool(shared_file)
        if managed_file is None:
            return status
        status["path"] = str(managed_file.resolve())
        status["revision"] = revision
        validation_error = _safetensors_validation_error(managed_file)
        if validation_error:
            status.update(state="corrupt", message=validation_error)
            return status
        expected_hash = locked.get("sha256") if isinstance(locked, dict) else None
        if expected_hash and _sha256(managed_file) != expected_hash:
            status.update(
                state="corrupt",
                message="Cached file does not match the recorded checksum.",
            )
            return status
        status.update(state="ready", ready=True, message="Ready")
        return status

    if asset_id == "efficientad_teacher":
        files = _teacher_files(asset_root)
        status["path"] = str(files[0].parent.resolve())
        missing = [path.name for path in files if not path.is_file()]
        if missing:
            if files[0].parent.exists():
                status.update(
                    state="corrupt",
                    message=f"Missing teacher file(s): {', '.join(missing)}.",
                )
            return status
        locked_files = locked.get("files", {}) if isinstance(locked, dict) else {}
        for path in files:
            expected = locked_files.get(path.name)
            if path.stat().st_size <= 0 or (expected and _sha256(path) != expected):
                status.update(
                    state="corrupt",
                    message=f"Teacher file is corrupt: {path.name}.",
                )
                return status
        status.update(state="ready", ready=True, message="Ready")
        return status

    configured = _configured_imagenette_path(job)
    if configured is not None and _contains_images(configured):
        status.update(
            state="ready",
            ready=True,
            path=str(configured),
            source="Local directory",
            source_url="",
            message="Ready (configured local directory)",
        )
        return status
    managed = _managed_imagenette_path(asset_root)
    status["path"] = str(managed.resolve())
    if _contains_images(managed):
        status.update(state="ready", ready=True, message="Ready")
    elif managed.exists():
        status.update(
            state="corrupt",
            message="Managed ImageNette directory contains no supported images.",
        )
    return status


def emit_asset_statuses(job: dict[str, Any]) -> list[dict[str, Any]]:
    asset_root = asset_root_from_job(job)
    statuses = [
        _asset_state(asset_id, asset_root, job)
        for asset_id in ASSET_DEFINITIONS
    ]
    for status in statuses:
        event = dict(status)
        event.pop("event", None)
        emit("asset_status", **event)
    return statuses


def _record_ready_asset(
    asset_id: str,
    asset_root: Path,
    *,
    path: Path,
    revision: str = "",
    source: str,
) -> None:
    lock = _load_asset_lock(asset_root)
    entry: dict[str, Any] = {
        "path": str(path.resolve()),
        "source": source,
        "revision": revision,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if path.is_file():
        entry["sha256"] = _sha256(path)
    elif asset_id == "efficientad_teacher":
        entry["files"] = {
            teacher.name: _sha256(teacher)
            for teacher in _teacher_files(asset_root)
        }
    lock["assets"][asset_id] = entry
    _write_asset_lock(asset_root, lock)


def _replace_directory(staging: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.rmtree(destination)
    staging.replace(destination)


def _import_hf_asset(
    asset_id: str,
    definition: dict[str, Any],
    asset_root: Path,
) -> tuple[Path, str] | None:
    source_file, source_revision = _hf_cached_file(
        _shared_hf_cache(),
        definition,
    )
    if source_file is None:
        return None
    source_repo = _hf_repo_directory(
        _shared_hf_cache(),
        str(definition["repo_id"]),
    )
    destination_repo = _hf_repo_directory(
        _managed_hf_cache(asset_root),
        str(definition["repo_id"]),
    )
    temporary = destination_repo.with_name(
        f".{destination_repo.name}.{uuid.uuid4().hex}.import"
    )
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_repo, temporary, symlinks=False)
    _replace_directory(temporary, destination_repo)
    managed_file, managed_revision = _hf_cached_file(
        _managed_hf_cache(asset_root),
        definition,
    )
    if managed_file is None:
        raise RuntimeError(f"Could not import {definition['display_name']}.")
    validation_error = _safetensors_validation_error(managed_file)
    if validation_error:
        raise RuntimeError(
            f"Imported {definition['display_name']} is corrupt: "
            f"{validation_error}"
        )
    if _sha256(managed_file) != _sha256(source_file):
        raise RuntimeError(
            f"Imported {definition['display_name']} failed checksum validation."
        )
    emit(
        "asset_progress",
        asset_id=asset_id,
        stage="imported",
        downloaded_bytes=managed_file.stat().st_size,
        total_bytes=managed_file.stat().st_size,
        message="Imported from the existing user cache.",
    )
    return managed_file, managed_revision or source_revision


def _download_hf_asset(
    asset_id: str,
    definition: dict[str, Any],
    asset_root: Path,
) -> tuple[Path, str, str]:
    imported = _import_hf_asset(asset_id, definition, asset_root)
    if imported is not None:
        return imported[0], imported[1], "imported"
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is required to download pretrained backbones."
        ) from exc
    emit(
        "asset_progress",
        asset_id=asset_id,
        stage="downloading",
        downloaded_bytes=0,
        total_bytes=definition.get("estimated_bytes", 0),
        message=f"Downloading {definition['display_name']}.",
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=str(definition["repo_id"]),
            filename=str(definition["filename"]),
            cache_dir=str(_managed_hf_cache(asset_root)),
        )
    )
    revision = downloaded.parent.name
    emit(
        "asset_progress",
        asset_id=asset_id,
        stage="validating",
        downloaded_bytes=downloaded.stat().st_size,
        total_bytes=downloaded.stat().st_size,
        message="Validating downloaded backbone.",
    )
    validation_error = _safetensors_validation_error(downloaded)
    if validation_error:
        raise RuntimeError(
            f"Downloaded {definition['display_name']} is corrupt: "
            f"{validation_error}"
        )
    return downloaded, revision, "downloaded"


def _download_file(
    asset_id: str,
    definition: dict[str, Any],
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(
        str(definition["source_url"]),
        headers={"User-Agent": "Inspection-Workbench/0.1"},
    )
    downloaded = 0
    digest = hashlib.sha256()
    try:
        with urlopen(request, timeout=30) as response, destination.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0") or "0")
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                downloaded += len(chunk)
                emit(
                    "asset_progress",
                    asset_id=asset_id,
                    stage="downloading",
                    downloaded_bytes=downloaded,
                    total_bytes=total,
                    message=f"Downloading {definition['display_name']}.",
                )
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    expected = str(definition.get("sha256", ""))
    if expected and digest.hexdigest().lower() != expected.lower():
        destination.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum validation failed for {definition['display_name']}."
        )


def _safe_extract_archive(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()

    def validate_member(name: str) -> None:
        candidate = (destination / name).resolve()
        if (
            candidate != resolved_destination
            and resolved_destination not in candidate.parents
        ):
            raise RuntimeError(f"Unsafe archive member path: {name}")

    archive_name = archive.name.lower()
    if archive_name.endswith((".zip", ".zip.part")):
        with zipfile.ZipFile(archive) as compressed:
            for member in compressed.infolist():
                validate_member(member.filename)
            compressed.extractall(destination)
        return

    if archive_name.endswith(
        (".tgz", ".tgz.part", ".tar.gz", ".tar.gz.part", ".tar", ".tar.part")
    ):
        with tarfile.open(archive) as compressed:
            members = compressed.getmembers()
            for member in members:
                validate_member(member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(
                        f"Archive links are not allowed: {member.name}"
                    )
            compressed.extractall(destination, members=members)
        return
    raise RuntimeError(f"Unsupported asset archive format: {archive.name}")


def _find_extracted_directory(staging: Path, required_files: list[str]) -> Path:
    for directory in [staging, *sorted(path for path in staging.rglob("*") if path.is_dir())]:
        if all((directory / name).is_file() for name in required_files):
            return directory
    raise RuntimeError(
        f"Downloaded archive does not contain: {', '.join(required_files)}."
    )


def _find_imagenette_directory(staging: Path) -> Path:
    candidates = [
        staging,
        *sorted(path for path in staging.rglob("*") if path.is_dir()),
    ]
    for candidate in candidates:
        training_split = candidate / "train"
        if training_split.is_dir() and _contains_images(training_split):
            return training_split
    for candidate in candidates:
        if candidate.name.lower() == "train" and _contains_images(candidate):
            return candidate
    for candidate in candidates:
        if _contains_images(candidate):
            return candidate
    raise RuntimeError("Downloaded ImageNette archive contains no supported images.")


def _download_archive_asset(
    asset_id: str,
    definition: dict[str, Any],
    asset_root: Path,
) -> Path:
    downloads = asset_root / "downloads"
    partial = downloads / f"{definition['filename']}.part"
    staging = asset_root / ".staging" / f"{asset_id}-{uuid.uuid4().hex}"
    partial.unlink(missing_ok=True)
    if staging.exists():
        shutil.rmtree(staging)
    try:
        _download_file(asset_id, definition, partial)
        emit(
            "asset_progress",
            asset_id=asset_id,
            stage="extracting",
            downloaded_bytes=partial.stat().st_size,
            total_bytes=partial.stat().st_size,
            message=f"Extracting {definition['display_name']}.",
        )
        _safe_extract_archive(partial, staging)
        if asset_id == "efficientad_teacher":
            extracted = _find_extracted_directory(
                staging,
                [str(name) for name in definition["required_files"]],
            )
        else:
            extracted = _find_imagenette_directory(staging)
        destination = asset_root / str(definition["destination"])
        _replace_directory(extracted, destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def run_assets(job: dict[str, Any]) -> None:
    action = str(job.get("action", "check")).lower()
    if action == "check":
        statuses = emit_asset_statuses(job)
        emit(
            "asset_result",
            action="check",
            completed=True,
            result="completed",
            ready_count=sum(bool(status["ready"]) for status in statuses),
            asset_count=len(statuses),
            message="Local model asset status refreshed.",
        )
        return
    if action != "download":
        raise ValueError("assets action must be check or download.")

    asset_root = asset_root_from_job(job)
    requested = job.get("asset_ids")
    if not isinstance(requested, list) or not requested:
        raise ValueError("asset_ids must contain at least one asset identifier.")
    unknown = sorted(
        str(asset_id)
        for asset_id in requested
        if str(asset_id) not in ASSET_DEFINITIONS
    )
    if unknown:
        raise ValueError(f"Unknown asset identifier(s): {', '.join(unknown)}.")

    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    for raw_id in requested:
        asset_id = str(raw_id)
        definition = ASSET_DEFINITIONS[asset_id]
        current = _asset_state(asset_id, asset_root, job)
        if current["ready"]:
            emit(
                "asset_result",
                asset_id=asset_id,
                completed=True,
                result="completed",
                message=f"{definition['display_name']} is already ready.",
            )
            continue
        try:
            if definition["kind"] == "huggingface":
                path, revision, result = _download_hf_asset(
                    asset_id,
                    definition,
                    asset_root,
                )
                _record_ready_asset(
                    asset_id,
                    asset_root,
                    path=path,
                    revision=revision,
                    source=result,
                )
            else:
                path = _download_archive_asset(
                    asset_id,
                    definition,
                    asset_root,
                )
                _record_ready_asset(
                    asset_id,
                    asset_root,
                    path=path,
                    source="downloaded",
                )
                result = "downloaded"
            status = _asset_state(asset_id, asset_root, job)
            if not status["ready"]:
                raise RuntimeError(str(status["message"]))
            emit(
                "asset_result",
                asset_id=asset_id,
                completed=True,
                result="imported" if result == "imported" else "completed",
                message=f"{definition['display_name']} is ready.",
            )
        except KeyboardInterrupt:
            emit(
                "asset_result",
                asset_id=asset_id,
                completed=False,
                result="cancelled",
                message=f"Preparing {definition['display_name']} was cancelled.",
            )
            raise
        except Exception as exc:
            emit(
                "asset_result",
                asset_id=asset_id,
                completed=False,
                result="failed",
                message=f"{type(exc).__name__}: {exc}",
            )
            raise


def configure_offline_environment(job: dict[str, Any]) -> None:
    asset_root = asset_root_from_job(job)
    os.environ["HF_HOME"] = str(asset_root / "huggingface")
    os.environ["HF_HUB_CACHE"] = str(_managed_hf_cache(asset_root))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    os.environ["DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TORCH_HOME"] = str(asset_root / "torch")


def install_offline_network_guard() -> None:
    def blocked(*_args: Any, **_kwargs: Any) -> Any:
        raise OfflineResourceError(
            "Network access is disabled for model jobs. "
            "Download required public assets explicitly from Settings."
        )

    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]
    socket.socket.connect = blocked  # type: ignore[assignment]
    socket.socket.connect_ex = blocked  # type: ignore[assignment]


def _preflight_model_assets(
    job: dict[str, Any],
    model_name: str,
    parameters: dict[str, Any],
) -> dict[str, Any]:
    asset_root = asset_root_from_job(job)
    normalized = deepcopy(parameters)
    required_ids: list[str] = []
    if normalized.get("pre_trained", False):
        required_ids.append(
            "patchcore_backbone"
            if normalized.get("backbone") == "wide_resnet50_2"
            else "padim_backbone"
        )
    if model_name == "EfficientAD":
        configured = _configured_imagenette_path(
            {
                **job,
                "imagenette_dir": str(normalized.get("imagenet_dir", "")),
            }
        )
        if configured is not None and _contains_images(configured):
            normalized["imagenet_dir"] = str(configured)
        else:
            managed = _managed_imagenette_path(asset_root)
            if _contains_images(managed):
                normalized["imagenet_dir"] = str(managed.resolve())
            else:
                required_ids.append("efficientad_imagenette")

    missing: list[str] = []
    for asset_id in required_ids:
        status = _asset_state(asset_id, asset_root, job)
        if not status["ready"]:
            missing.append(f"{status['display_name']}: {status['message']}")
    if model_name == "EfficientAD":
        size = "small" if normalized.get("model_size") == "s" else "medium"
        teacher = (
            asset_root
            / str(ASSET_DEFINITIONS["efficientad_teacher"]["destination"])
            / f"pretrained_teacher_{size}.pth"
        )
        if not teacher.is_file():
            missing.append(f"EfficientAD {size} teacher file is missing.")
        elif teacher.stat().st_size <= 0:
            missing.append(f"EfficientAD {size} teacher file is corrupt.")
        else:
            locked = _load_asset_lock(asset_root)["assets"].get(
                "efficientad_teacher", {}
            )
            expected = (
                locked.get("files", {}).get(teacher.name)
                if isinstance(locked, dict)
                and isinstance(locked.get("files", {}), dict)
                else None
            )
            if expected and _sha256(teacher) != expected:
                missing.append(
                    f"EfficientAD {size} teacher file failed checksum validation."
                )
    if missing:
        raise OfflineResourceError(
            "Required local model assets are not ready. "
            + " ".join(missing)
            + " Open Settings > Local model assets to prepare them."
        )
    return normalized


def preflight_selected_models(
    job: dict[str, Any],
    model_names: list[str],
) -> dict[str, dict[str, Any]]:
    """Validate all selected models so one job reports every missing asset."""
    normalized: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for model_name in model_names:
        try:
            parameters = model_parameters_from_job(job, model_name)
            normalized[model_name] = _preflight_model_assets(
                job,
                model_name,
                parameters,
            )
        except OfflineResourceError as exc:
            failures.append(f"{model_name}: {exc}")
    if failures:
        raise OfflineResourceError(
            "Offline resource preflight failed for the selected models. "
            + " ".join(failures)
        )
    return normalized


def configure_anomalib_asset_paths(asset_root: Path) -> None:
    try:
        from anomalib.models.image.efficient_ad import lightning_model
    except ImportError:
        return
    local_pretrained = asset_root / "anomalib" / "pre_trained"
    lightning_model.get_pretrained_weights_dir = lambda: local_pretrained


def image_files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda item: str(item).lower(),
    )


def matching_mask(job: dict[str, Any], image_path: Path) -> Path | None:
    dataset_root = Path(job.get("dataset_root", ""))
    mask_dir = job.get("mask_dir")
    if not mask_dir:
        return None
    mask_root = dataset_root / str(mask_dir)
    if not mask_root.is_dir():
        return None

    abnormal_dir = job.get("abnormal_dir")
    try:
        relative = (
            image_path.relative_to(dataset_root / str(abnormal_dir))
            if abnormal_dir
            else Path(image_path.name)
        )
    except ValueError:
        relative = Path(image_path.name)

    relative_parent = relative.parent
    for stem in (relative.stem, f"{relative.stem}_mask"):
        for extension in IMAGE_EXTENSIONS:
            candidate = mask_root / relative_parent / f"{stem}{extension}"
            if candidate.is_file():
                return candidate

    expected_names = {relative.stem.lower(), f"{relative.stem.lower()}_mask"}
    return next(
        (
            candidate
            for candidate in image_files(mask_root)
            if candidate.stem.lower() in expected_names
        ),
        None,
    )


def dataset_inspection(job: dict[str, Any]) -> dict[str, Any]:
    dataset_root_value = job.get("dataset_root")
    if not dataset_root_value:
        return {}
    dataset_root = Path(dataset_root_value)
    sample_dir = job.get("abnormal_dir") or job.get("normal_dir")
    if not sample_dir:
        return {}
    samples = image_files(dataset_root / str(sample_dir))
    if not samples:
        return {}
    image_path = samples[0]
    mask_path = matching_mask(job, image_path)
    return {
        "image_path": str(image_path.resolve()),
        "mask_path": str(mask_path.resolve()) if mask_path else "",
        "sample_index": 1,
        "sample_count": len(samples),
    }


def demo_result(
    model: str,
    job: dict[str, Any] | None = None,
    model_parameters: dict[str, Any] | None = None,
    training_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    values = {
        "PatchCore": (0.978, 0.962, 0.917, 38.2, 1270.0),
        "PaDiM": (0.957, 0.929, 0.881, 22.7, 412.0),
        "EfficientAD": (0.942, 0.909, 0.846, 16.4, 298.0),
    }
    image_auroc, pixel_auroc, image_f1, latency, size = values.get(
        model, (0.900, 0.870, 0.820, 30.0, 500.0)
    )
    result = {
        "model": model,
        "image_auroc": image_auroc,
        "pixel_auroc": pixel_auroc,
        "image_f1": image_f1,
        "latency_ms": latency,
        "model_size_mb": size,
        "checkpoint": f"demo/{model.lower()}/model.ckpt",
        "model_parameters": parameter_snapshot(
            model,
            model_parameters
            if model_parameters is not None
            else normalize_model_parameters(model),
        ),
        "training_parameters": (
            deepcopy(training_parameters)
            if training_parameters is not None
            else normalize_training_parameters(job or {}, model)
        ),
    }
    if job:
        inspection = dataset_inspection(job)
        if inspection:
            result["inspection"] = inspection
    return result


def run_demo(job: dict[str, Any]) -> None:
    mode = job.get("mode", "train")
    if mode == "check":
        emit(
            "environment",
            ok=True,
            demo=True,
            message="Demo worker ready. Configure an Anomalib Python environment for real jobs.",
        )
        return

    if mode == "inference":
        run_demo_inference(job)
        return

    if mode == "export":
        output_root = Path(job["output_root"])
        output_root.mkdir(parents=True, exist_ok=True)
        model = job.get("model", "PatchCore")
        model_parameters = model_parameters_from_job(job, model)
        training_parameters = training_parameters_from_job(job, model)
        export_format = job.get("format", "OPENVINO").upper()
        manifest = output_root / f"{model.lower()}-{export_format.lower()}-demo.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "demo": True,
                    "model": model,
                    "format": export_format,
                    "artifact": {
                        "format": export_format.lower(),
                        "path": job.get("checkpoint", ""),
                    },
                    "source_checkpoint": job.get("checkpoint", ""),
                    "model_parameters": parameter_snapshot(
                        model, model_parameters
                    ),
                    "training_parameters": training_parameters,
                    "metrics": job.get("metrics", {}),
                    "capabilities": _sigma_override_capabilities(
                        model, export_format
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        emit(
            "exported",
            model=model,
            format=export_format,
            path=str(manifest),
            model_parameters=parameter_snapshot(model, model_parameters),
            training_parameters=training_parameters,
        )
        return

    models = job.get("models") or ["PatchCore"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list.")
    emit("hello", demo=True, anomalib_version="demo")
    results: list[dict[str, Any]] = []
    for model in models:
        if not isinstance(model, str):
            raise ValueError("Every selected model name must be a string.")
        model_parameters = model_parameters_from_job(job, model)
        training_parameters = normalize_training_parameters(job, model)
        epochs = training_parameters["max_epochs"]
        emit("stage", model=model, status="training", message=f"Preparing {model}")
        simulation_duration = 0.12 * min(epochs, 8)
        simulation_started = time.perf_counter()
        if epochs <= 1_000:
            progress_epochs = range(1, epochs + 1)
        else:
            # Keep very large demo jobs responsive while still reporting the
            # requested epoch count and a final 100% progress event.
            progress_epochs = sorted(
                {1, *(round(epochs * step / 100) for step in range(1, 101))}
            )
        for epoch in progress_epochs:
            target_time = simulation_started + simulation_duration * epoch / epochs
            remaining = target_time - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            emit(
                "progress",
                model=model,
                epoch=epoch,
                max_epochs=epochs,
                message=f"{model}: epoch {epoch}/{epochs}",
            )
        emit("stage", model=model, status="testing", message=f"Testing {model}")
        time.sleep(0.18)
        result = demo_result(
            model, job, model_parameters, training_parameters
        )
        results.append(result)
        emit("result", **result)

    winner = max(results, key=lambda item: item["image_auroc"])["model"]
    emit("finished", winner=winner, result_count=len(results), demo=True)


def metric_value(metrics: dict[str, Any], names: list[str]) -> float | None:
    lowered = {str(key).lower(): value for key, value in metrics.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value is None:
            continue
        try:
            if hasattr(value, "item"):
                value = value.item()
            return float(value)
        except (TypeError, ValueError):
            pass
    return None


def directory_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file()) / (
        1024 * 1024
    )


def first_prediction(value: Any) -> Any | None:
    if value is None:
        return None
    if isinstance(value, dict) and value.get("anomaly_map") is not None:
        return value
    if getattr(value, "anomaly_map", None) is not None:
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            prediction = first_prediction(item)
            if prediction is not None:
                return prediction
        return None
    if type(value).__name__.lower().endswith("batch"):
        try:
            for item in value:
                prediction = first_prediction(item)
                if prediction is not None:
                    return prediction
        except TypeError:
            pass
    return None


def prediction_field(prediction: Any, name: str) -> Any:
    if isinstance(prediction, dict):
        return prediction.get(name)
    return getattr(prediction, name, None)


def scalar_value(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        import numpy as np

        array = np.asarray(value)
        if not array.size:
            return None
        return float(array.reshape(-1)[0])
    except (TypeError, ValueError):
        return None


def tensor_to_numpy(value: Any):
    import numpy as np

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    while array.ndim > 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 3:
        array = array[0]
    return np.asarray(array, dtype=np.float32)


def normalized_map(value: Any):
    import numpy as np

    array = tensor_to_numpy(value)
    if array is None or array.ndim != 2 or not array.size:
        raise ValueError("Prediction did not contain a two-dimensional anomaly map.")
    finite = np.isfinite(array)
    if not finite.any():
        raise ValueError("Prediction anomaly map did not contain finite values.")
    minimum = float(array[finite].min())
    maximum = float(array[finite].max())
    if maximum <= minimum:
        return np.zeros(array.shape, dtype=np.float32)
    result = (array - minimum) / (maximum - minimum)
    result[~finite] = 0.0
    return np.clip(result, 0.0, 1.0)


def gaussian_blur_anomaly_map(value: Any, sigma: int):
    """Apply Anomalib's Gaussian blur to a tensor/array anomaly map."""
    import numpy as np
    import torch
    from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

    sigma = _require_int(sigma, "sigma_override", 1, 32)
    array = tensor_to_numpy(value)
    if array is None:
        raise ValueError("Raw anomaly map is unavailable for sigma override.")
    tensor = torch.as_tensor(np.asarray(array), dtype=torch.float32)
    while tensor.ndim < 4:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("Raw anomaly map must have at most four dimensions.")
    with torch.inference_mode():
        blurred = AnomalyMapGenerator(sigma=sigma).blur(tensor)
    return blurred.detach().cpu().numpy()


def sigma_override_from_command(command: dict[str, Any]) -> int | None:
    value = command.get("sigma_override")
    if value is None:
        return None
    return _require_int(value, "sigma_override", 1, 32)


def colorize_anomaly_map(anomaly_map):
    import numpy as np

    stops = np.asarray([0.0, 0.20, 0.45, 0.70, 1.0], dtype=np.float32)
    colors = np.asarray(
        [
            (8, 18, 80),
            (20, 90, 220),
            (0, 210, 190),
            (255, 225, 30),
            (220, 20, 30),
        ],
        dtype=np.float32,
    )
    channels = [
        np.interp(anomaly_map, stops, colors[:, channel])
        for channel in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def write_prediction_artifacts(
    image_path: Path,
    prediction: Any,
    artifact_root: Path,
    threshold: float,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    anomaly_map = normalized_map(prediction_field(prediction, "anomaly_map"))
    artifact_root.mkdir(parents=True, exist_ok=True)
    anomaly_path = artifact_root / "anomaly-map.png"
    heatmap_path = artifact_root / "heatmap.png"
    predicted_mask_path = artifact_root / "predicted-mask.png"

    Image.fromarray((anomaly_map * 255.0).astype(np.uint8), mode="L").save(
        anomaly_path
    )
    with Image.open(image_path) as source_file:
        source = source_file.convert("RGB")
    heatmap = Image.fromarray(colorize_anomaly_map(anomaly_map), mode="RGB")
    heatmap = heatmap.resize(source.size, Image.Resampling.BILINEAR)
    Image.blend(source, heatmap, 0.55).save(heatmap_path)

    predicted_mask_array = anomaly_map >= threshold
    mask_rgb = np.zeros((*predicted_mask_array.shape, 3), dtype=np.uint8)
    mask_rgb[predicted_mask_array] = (239, 55, 52)
    mask_image = Image.fromarray(mask_rgb, mode="RGB")
    mask_image.resize(source.size, Image.Resampling.NEAREST).save(
        predicted_mask_path
    )

    score = scalar_value(prediction_field(prediction, "pred_score"))
    if score is None:
        score = float(anomaly_map.max())
    return {
        "image_path": str(image_path.resolve()),
        "anomaly_score": score,
        "label": "anomalous" if score >= threshold else "normal",
        "threshold": threshold,
        "anomaly_map_path": str(anomaly_path.resolve()),
        "heatmap_path": str(heatmap_path.resolve()),
        "predicted_mask_path": str(predicted_mask_path.resolve()),
    }


def create_inspection_artifacts(
    job: dict[str, Any],
    model_name: str,
    engine: Any,
    model: Any,
    checkpoint: str,
    model_root: Path,
) -> dict[str, Any]:
    import numpy as np
    from PIL import Image

    inspection = dataset_inspection(job)
    image_path_value = inspection.get("image_path")
    if not image_path_value:
        return {}
    image_path = Path(image_path_value)

    predictions = engine.predict(
        model=model,
        data_path=str(image_path),
        ckpt_path=checkpoint or None,
        return_predictions=True,
    )
    prediction = first_prediction(predictions)
    if prediction is None:
        raise ValueError("Anomalib did not return an inspection prediction.")
    anomaly_map = normalized_map(prediction_field(prediction, "anomaly_map"))

    artifact_root = model_root / "inspection"
    artifact_root.mkdir(parents=True, exist_ok=True)
    anomaly_path = artifact_root / "anomaly-map.png"
    heatmap_path = artifact_root / "heatmap.png"
    predicted_mask_path = artifact_root / "predicted-mask.png"

    Image.fromarray((anomaly_map * 255.0).astype(np.uint8), mode="L").save(
        anomaly_path
    )
    with Image.open(image_path) as source_file:
        source = source_file.convert("RGB")
    heatmap = Image.fromarray(colorize_anomaly_map(anomaly_map), mode="RGB")
    heatmap = heatmap.resize(source.size, Image.Resampling.BILINEAR)
    Image.blend(source, heatmap, 0.55).save(heatmap_path)

    predicted_mask = prediction_field(prediction, "pred_mask")
    predicted_mask_array = tensor_to_numpy(predicted_mask)
    if predicted_mask_array is None or predicted_mask_array.ndim != 2:
        predicted_mask_array = anomaly_map >= 0.60
    else:
        predicted_mask_array = predicted_mask_array > 0
    mask_rgb = np.zeros((*predicted_mask_array.shape, 3), dtype=np.uint8)
    mask_rgb[predicted_mask_array] = (239, 55, 52)
    mask_image = Image.fromarray(mask_rgb, mode="RGB")
    mask_image.resize(source.size, Image.Resampling.NEAREST).save(
        predicted_mask_path
    )

    inspection.update(
        {
            "model": model_name,
            "anomaly_map_path": str(anomaly_path.resolve()),
            "heatmap_path": str(heatmap_path.resolve()),
            "predicted_mask_path": str(predicted_mask_path.resolve()),
        }
    )
    return inspection


def run_demo_inference(job: dict[str, Any]) -> None:
    import numpy as np
    from PIL import Image, ImageFilter

    model_name = str(job.get("model", "PatchCore"))
    model_parameters = model_parameters_from_job(job, model_name)
    training_parameters = training_parameters_from_job(job, model_name)
    image_path = Path(job["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Test image does not exist: {image_path}")
    threshold = min(1.0, max(0.0, float(job.get("threshold", 0.60))))
    output_root = Path(job["output_root"])

    with Image.open(image_path) as source_file:
        grayscale = np.asarray(source_file.convert("L"), dtype=np.float32) / 255.0
    deviation = np.abs(grayscale - float(grayscale.mean()))
    if deviation.max() > 0:
        deviation /= float(deviation.max())
    score = float(np.quantile(deviation, 0.99)) if deviation.size else 0.0
    sigma_override = sigma_override_from_command(job)
    saved_sigma = model_parameters["sigma"] if model_name == "PatchCore" else None
    if model_name == "PatchCore":
        effective_sigma = sigma_override if sigma_override is not None else saved_sigma
        deviation = np.asarray(
            Image.fromarray((deviation * 255.0).astype(np.uint8), mode="L").filter(
                ImageFilter.GaussianBlur(radius=effective_sigma)
            ),
            dtype=np.float32,
        ) / 255.0
    else:
        effective_sigma = None
    prediction = {
        "anomaly_map": deviation,
        "pred_score": score,
    }
    started = time.perf_counter()
    result = write_prediction_artifacts(
        image_path,
        prediction,
        output_root / model_name.lower(),
        threshold,
    )
    result.update(
        {
            "model": model_name,
            "checkpoint": job.get("checkpoint", ""),
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "demo": True,
            "model_parameters": parameter_snapshot(
                model_name, model_parameters
            ),
            "training_parameters": training_parameters,
            "saved_sigma": saved_sigma,
            "effective_sigma": effective_sigma,
            "sigma_overridden": sigma_override is not None,
            "sigma_override_supported": model_name == "PatchCore",
        }
    )
    emit("inference", **result)


def create_datamodule(
    job: dict[str, Any], training_parameters: dict[str, Any] | None = None
):
    from anomalib.data import Folder

    if training_parameters is None:
        training_parameters = normalize_training_parameters(
            job, str(job.get("model", "PatchCore"))
        )
    kwargs: dict[str, Any] = {
        "name": job.get("dataset_name", "custom_dataset"),
        "root": job["dataset_root"],
        "normal_dir": job.get("normal_dir", "normal"),
        "train_batch_size": training_parameters["train_batch_size"],
        "eval_batch_size": training_parameters["eval_batch_size"],
        "num_workers": training_parameters["num_workers"],
    }
    if job.get("abnormal_dir"):
        kwargs["abnormal_dir"] = job["abnormal_dir"]
    if job.get("mask_dir"):
        kwargs["mask_dir"] = job["mask_dir"]
    return Folder(**kwargs)


def _record_patchcore_hyperparameters(
    model: Any, parameters: dict[str, Any]
) -> None:
    """Persist app-owned PatchCore settings in Lightning checkpoints."""
    hparams = getattr(model, "hparams", None)
    if hparams is None:
        return
    for key in PATCHCORE_RUNTIME_PARAMETER_KEYS:
        try:
            hparams[key] = parameters[key]
        except (AttributeError, KeyError, TypeError):
            return


def configure_model_preprocessor(
    model: Any,
    parameters: dict[str, Any],
    *,
    require_components: bool = True,
) -> Any:
    """Apply and persist the app-owned exact-resize preprocessing shape."""
    configure = getattr(model, "configure_pre_processor", None)
    if not callable(configure):
        if require_components:
            raise RuntimeError(
                "Installed Anomalib model does not expose pre-processor configuration."
            )
        return model
    model.pre_processor = configure(image_size=image_size_tuple(parameters))
    hparams = getattr(model, "hparams", None)
    if hparams is not None:
        try:
            hparams["image_size"] = deepcopy(parameters["image_size"])
        except (AttributeError, KeyError, TypeError) as exc:
            if require_components:
                raise RuntimeError(
                    "Could not persist the configured model image size."
                ) from exc
    return model


def restore_checkpoint_state(model: Any, checkpoint_path: str | Path) -> Any:
    """Safely restore a Lightning state dictionary into a configured model."""
    import torch

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("state_dict"), dict
    ):
        raise ValueError("Lightning checkpoint does not contain a state_dict.")
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(
            "Checkpoint does not exactly match the configured model: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return model


def configure_patchcore_components(
    model: Any,
    parameters: dict[str, Any],
    *,
    require_components: bool = True,
) -> Any:
    """Apply app-owned pooling and anomaly-map settings to PatchCore."""
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        if require_components:
            raise RuntimeError(
                "Installed Anomalib PatchCore does not expose its inner model."
            )
        return model

    import torch
    from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

    patch_size = parameters["patch_size"]
    inner_model.feature_pooler = torch.nn.AvgPool2d(
        kernel_size=patch_size,
        stride=parameters["patch_stride"],
        padding=patch_size // 2,
    )
    generator = AnomalyMapGenerator(sigma=parameters["sigma"])
    reference = next(inner_model.parameters(), None)
    if reference is not None:
        generator = generator.to(device=reference.device, dtype=reference.dtype)
    inner_model.anomaly_map_generator = generator
    _record_patchcore_hyperparameters(model, parameters)
    return model


def configure_patchcore_sigma(model: Any, sigma: int) -> None:
    """Replace only PatchCore's Gaussian map generator on the active device."""
    sigma = _require_int(sigma, "sigma_override", 1, 32)
    inner_model = getattr(model, "model", None)
    if inner_model is None:
        raise RuntimeError("PatchCore does not expose its anomaly-map generator.")

    from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator

    generator = AnomalyMapGenerator(sigma=sigma)
    reference = next(inner_model.parameters(), None)
    if reference is not None:
        generator = generator.to(device=reference.device, dtype=reference.dtype)
    inner_model.anomaly_map_generator = generator


def create_model(
    model_name: str,
    parameters: dict[str, Any] | None = None,
    *,
    constructors: dict[str, Any] | None = None,
):
    custom_constructors = constructors is not None
    if constructors is None:
        from anomalib.models import EfficientAd, Padim, Patchcore

        constructors = {
            "PatchCore": Patchcore,
            "PaDiM": Padim,
            "EfficientAD": EfficientAd,
        }
    if model_name not in constructors:
        raise ValueError(f"Unsupported model: {model_name}")
    normalized = normalize_model_parameters(model_name, parameters)
    constructor = constructors[model_name]
    constructor_parameters = deepcopy(normalized)
    constructor_parameters.pop("image_size")
    if model_name == "PatchCore":
        for key in PATCHCORE_RUNTIME_PARAMETER_KEYS:
            constructor_parameters.pop(key)
    if model_name == "EfficientAD":
        constructor_parameters["model_size"] = {
            "s": "small",
            "m": "medium",
        }[constructor_parameters["model_size"]]
    try:
        signature = inspect.signature(constructor)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
        unsupported = sorted(
            name
            for name in constructor_parameters
            if name not in signature.parameters and not accepts_kwargs
        )
        if unsupported:
            raise RuntimeError(
                f"Installed Anomalib does not support {model_name} parameter(s): "
                f"{', '.join(unsupported)}."
            )
    model = constructor(**constructor_parameters)
    configure_model_preprocessor(
        model,
        normalized,
        require_components=not custom_constructors,
    )
    if model_name == "PatchCore":
        configure_patchcore_components(
            model,
            normalized,
            require_components=not custom_constructors,
        )
    return model


def run_real_train(job: dict[str, Any]) -> None:
    asset_root = asset_root_from_job(job)
    models = job.get("models") or ["PatchCore"]
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list.")
    if any(not isinstance(model_name, str) for model_name in models):
        raise ValueError("Every selected model name must be a string.")
    model_names = [str(model_name) for model_name in models]
    prepared_parameters = preflight_selected_models(job, model_names)

    import anomalib
    from anomalib.engine import Engine

    configure_anomalib_asset_paths(asset_root)
    anomalib_version = getattr(anomalib, "__version__", "unknown")
    require_supported_anomalib_version(anomalib_version)
    emit("hello", demo=False, anomalib_version=anomalib_version)
    output_root = Path(job["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for model_name in model_names:
        model_parameters = prepared_parameters[model_name]
        training_parameters = normalize_training_parameters(job, model_name)
        model_root = output_root / model_name.lower()
        model_root.mkdir(parents=True, exist_ok=True)
        emit("stage", model=model_name, status="training", message=f"Training {model_name}")
        started = time.perf_counter()
        datamodule = create_datamodule(job, training_parameters)
        model = create_model(model_name, model_parameters)
        engine = Engine(
            default_root_dir=str(model_root),
            max_epochs=training_parameters["max_epochs"],
            accelerator=training_parameters["accelerator"],
            devices=training_parameters["devices"],
            enable_progress_bar=False,
        )
        engine.fit(model=model, datamodule=datamodule)
        emit("stage", model=model_name, status="testing", message=f"Testing {model_name}")
        raw_results = engine.test(model=model, datamodule=datamodule)
        metrics = raw_results[0] if isinstance(raw_results, list) and raw_results else {}
        checkpoint = str(engine.best_model_path or "")
        duration = time.perf_counter() - started
        inspection = dataset_inspection(job)
        try:
            generated_inspection = create_inspection_artifacts(
                job, model_name, engine, model, checkpoint, model_root
            )
            if generated_inspection:
                inspection = generated_inspection
        except Exception as exc:
            emit(
                "warning",
                model=model_name,
                message=(
                    f"Metrics are available, but the inspection preview could not "
                    f"be generated: {type(exc).__name__}: {exc}"
                ),
            )
        result = {
            "model": model_name,
            "image_auroc": metric_value(
                metrics, ["image_AUROC", "image_auroc", "test_image_AUROC"]
            ),
            "pixel_auroc": metric_value(
                metrics, ["pixel_AUROC", "pixel_auroc", "test_pixel_AUROC"]
            ),
            "image_f1": metric_value(
                metrics, ["image_F1Score", "image_f1", "test_image_F1Score"]
            ),
            "latency_ms": None,
            "model_size_mb": directory_size_mb(Path(checkpoint)) if checkpoint else 0.0,
            "checkpoint": checkpoint,
            "duration_seconds": round(duration, 2),
            "metrics": {str(key): str(value) for key, value in metrics.items()},
            "inspection": inspection,
            "model_parameters": parameter_snapshot(
                model_name, model_parameters
            ),
            "training_parameters": deepcopy(training_parameters),
        }
        results.append(result)
        emit("result", **result)

    if not results:
        raise RuntimeError("No models were selected.")
    scored = [item for item in results if item.get("image_auroc") is not None]
    winner = (
        max(scored, key=lambda item: item["image_auroc"])["model"]
        if scored
        else results[0]["model"]
    )
    emit("finished", winner=winner, result_count=len(results), demo=False)


def _sigma_override_capabilities(
    model_name: str, export_format: str
) -> dict[str, Any]:
    if model_name == "PatchCore" and export_format.upper() == "ONNX":
        return {
            "sigma_override": {
                "supported": True,
                "raw_output": PATCHCORE_RAW_ANOMALY_MAP_OUTPUT,
                "minimum": 1,
                "maximum": 32,
            }
        }
    return {"sigma_override": {"supported": False}}


def _set_onnx_metadata(model: Any, key: str, value: str) -> None:
    for item in model.metadata_props:
        if item.key == key:
            item.value = value
            return
    item = model.metadata_props.add()
    item.key = key
    item.value = value


def add_patchcore_raw_map_output(
    path: Path, model_parameters: dict[str, Any]
) -> None:
    """Expose the tensor immediately before PatchCore's Gaussian ONNX Conv."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper, shape_inference

    model = onnx.load(str(path), load_external_data=True)
    initializers = {item.name: item for item in model.graph.initializer}
    sigma = model_parameters["sigma"]
    kernel_size = 2 * int(4.0 * sigma + 0.5) + 1
    axis = np.arange(kernel_size, dtype=np.float64) - (kernel_size - 1) / 2.0
    gaussian = np.exp(-(axis**2) / (2.0 * sigma**2))
    gaussian /= gaussian.sum()
    expected = np.outer(gaussian, gaussian).astype(np.float32)

    candidates: list[Any] = []
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2:
            continue
        initializer = initializers.get(node.input[1])
        if initializer is None:
            continue
        weights = numpy_helper.to_array(initializer)
        # FP16 export rounds the small Gaussian coefficients more than the
        # original FP32-specific tolerance allowed.
        rtol, atol = ((5e-3, 5e-5) if weights.dtype == np.float16 else (1e-4, 1e-6))
        if (
            weights.shape == (1, 1, kernel_size, kernel_size)
            and np.allclose(weights[0, 0], expected, rtol=rtol, atol=atol)
        ):
            candidates.append(node)
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely identify PatchCore's Gaussian blur in the ONNX graph."
        )

    blur_input = candidates[0].input[0]
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    padding_node = producers.get(blur_input)
    raw_map_input = (
        padding_node.input[0]
        if padding_node is not None and padding_node.op_type == "Pad"
        else blur_input
    )
    inferred = shape_inference.infer_shapes(model)
    value_infos = {
        item.name: item
        for collection in (
            inferred.graph.input,
            inferred.graph.value_info,
            inferred.graph.output,
        )
        for item in collection
    }
    raw_value = helper.make_tensor_value_info(
        PATCHCORE_RAW_ANOMALY_MAP_OUTPUT,
        onnx.TensorProto.FLOAT,
        [None, 1, None, None],
    )
    source_value = value_infos.get(raw_map_input)
    if source_value is not None:
        raw_value.type.CopyFrom(source_value.type)
    model.graph.node.append(
        helper.make_node(
            "Identity",
            inputs=[raw_map_input],
            outputs=[PATCHCORE_RAW_ANOMALY_MAP_OUTPUT],
            name="InspectionWorkbenchRawAnomalyMap",
        )
    )
    model.graph.output.append(raw_value)
    _set_onnx_metadata(
        model,
        ONNX_METADATA_MODEL_PARAMETERS,
        json.dumps(parameter_snapshot("PatchCore", model_parameters), sort_keys=True),
    )
    _set_onnx_metadata(
        model,
        ONNX_METADATA_RAW_MAP_OUTPUT,
        PATCHCORE_RAW_ANOMALY_MAP_OUTPUT,
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def embed_onnx_model_parameters(
    path: Path, model_name: str, model_parameters: dict[str, Any]
) -> None:
    """Embed app-owned parameters so direct ONNX selection remains self-describing."""
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    _set_onnx_metadata(
        model,
        ONNX_METADATA_MODEL_PARAMETERS,
        json.dumps(parameter_snapshot(model_name, model_parameters), sort_keys=True),
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def package_onnx_as_single_file(path: Path) -> None:
    """Embed manageable external tensors so the delivered ONNX is self-contained."""
    import onnx
    from onnx.external_data_helper import convert_model_from_external_data

    model = onnx.load(str(path), load_external_data=True)
    tensor_bytes = sum(len(tensor.raw_data) for tensor in model.graph.initializer)
    if tensor_bytes >= 1_800_000_000:
        return
    convert_model_from_external_data(model)
    onnx.checker.check_model(model)
    temporary = path.with_name(f"{path.stem}.embedded{path.suffix}")
    onnx.save_model(model, temporary, save_as_external_data=False)
    temporary.replace(path)


def _onnx_export_postprocessor_forward(self: Any, predictions: Any) -> Any:
    """Trace-friendly equivalent of Anomalib's trained post-processor."""
    from anomalib.data import InferenceBatch

    pred_score = (
        (predictions.pred_score - self._image_threshold)
        / (self.image_max - self.image_min)
        + 0.5
    ).clamp(min=0, max=1)
    anomaly_map = (
        (predictions.anomaly_map - self._pixel_threshold)
        / (self.pixel_max - self.pixel_min)
        + 0.5
    ).clamp(min=0, max=1)
    return InferenceBatch(
        pred_score=pred_score,
        pred_label=pred_score > (1.0 - self.image_sensitivity),
        anomaly_map=anomaly_map,
        pred_mask=anomaly_map > (1.0 - self.pixel_sensitivity),
    )


def make_postprocessor_onnx_export_friendly(model: Any) -> None:
    """Remove data-dependent Python branches while retaining trained values."""
    import types

    import torch

    post_processor = model.post_processor
    if post_processor is None:
        return
    image_threshold = post_processor.image_threshold.detach().clone()
    pixel_threshold = post_processor.pixel_threshold.detach().clone()
    required = (
        image_threshold,
        pixel_threshold,
        post_processor.image_min,
        post_processor.image_max,
        post_processor.pixel_min,
        post_processor.pixel_max,
    )
    if not all(bool(torch.isfinite(value).all().item()) for value in required):
        raise ValueError("Trained post-processing values are incomplete; refusing an uncalibrated ONNX export.")
    post_processor._image_threshold.copy_(image_threshold)
    post_processor._pixel_threshold.copy_(pixel_threshold)
    post_processor.forward = types.MethodType(
        _onnx_export_postprocessor_forward,
        post_processor,
    )


def run_real_export(job: dict[str, Any]) -> None:
    asset_root = asset_root_from_job(job)
    model_name = str(job["model"])
    model_parameters = preflight_selected_models(job, [model_name])[model_name]

    import anomalib
    from anomalib.deploy import ExportType
    from anomalib.engine import Engine

    configure_anomalib_asset_paths(asset_root)
    anomalib_version = getattr(anomalib, "__version__", "unknown")
    require_supported_anomalib_version(anomalib_version)
    training_parameters = training_parameters_from_job(job, model_name)
    model = create_model(model_name, model_parameters)
    restore_checkpoint_state(model, job["checkpoint"])
    export_format = job.get("format", "OPENVINO").upper()
    export_type = getattr(ExportType, export_format)
    output_root = Path(job["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    if export_format == "ONNX":
        import torch

        make_postprocessor_onnx_export_friendly(model)
        # Anomalib performs a real sample forward before graph capture to infer
        # output names.  For a large PatchCore memory bank this is the dominant
        # cost, so use the available accelerator for that pass.
        if torch.cuda.is_available():
            model = model.to("cuda")
    engine = Engine(default_root_dir=str(output_root))
    # PatchCore checkpoints can contain a memory bank with hundreds of millions
    # of values.  The legacy TorchScript-based exporter may spend hours tracing
    # nearest-neighbour operations before it writes anything.  The dynamo
    # exporter captures this graph directly.  Disable graph optimization here:
    # walking a 500+ MiB initializer is both expensive and unnecessary for a
    # faithful recovery export.
    onnx_kwargs = (
        {
            "dynamo": True,
            "optimize": False,
            "external_data": True,
            "dynamic_axes": {},
            "opset_version": 18,
        }
        if export_format == "ONNX"
        else None
    )
    exported = engine.export(
        model=model,
        export_type=export_type,
        export_root=output_root,
        input_size=image_size_tuple(model_parameters),
        ckpt_path=None,
        onnx_kwargs=onnx_kwargs,
    )
    if exported is None:
        raise RuntimeError(f"Anomalib did not produce a {export_format} artifact.")
    exported = Path(exported).resolve()
    if model_name == "PatchCore" and export_format == "ONNX":
        add_patchcore_raw_map_output(exported, model_parameters)
    if export_format == "ONNX":
        embed_onnx_model_parameters(exported, model_name, model_parameters)
        package_onnx_as_single_file(exported)
    artifact_hash_value = _sha256(exported)
    if exported.suffix.lower() in {".xml", ".bin"}:
        companion = exported.with_suffix(
            ".bin" if exported.suffix.lower() == ".xml" else ".xml"
        )
        if companion.is_file():
            combined_hash = hashlib.sha256()
            for artifact in sorted((exported, companion), key=lambda item: item.suffix):
                combined_hash.update(artifact.suffix.lower().encode("ascii"))
                combined_hash.update(_sha256(artifact).encode("ascii"))
            artifact_hash_value = combined_hash.hexdigest()
    manifest_path = exported.with_name(f"{exported.stem}.manifest.json")
    manifest = {
        "schema_version": 2,
        "model": model_name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "artifact": {
            "format": export_format.lower(),
            "path": exported.name,
            "sha256": artifact_hash_value,
        },
        "source_checkpoint": str(Path(job["checkpoint"]).resolve()),
        "model_parameters": parameter_snapshot(model_name, model_parameters),
        "training_parameters": training_parameters,
        "metrics": job.get("metrics", {}),
        "capabilities": _sigma_override_capabilities(model_name, export_format),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    emit(
        "exported",
        model=model_name,
        format=export_format,
        path=str(exported),
        manifest_path=str(manifest_path),
        fingerprint=artifact_hash_value,
        model_parameters=parameter_snapshot(model_name, model_parameters),
        training_parameters=training_parameters,
    )


def run_real_inference(job: dict[str, Any]) -> None:
    asset_root = asset_root_from_job(job)
    image_path = Path(job["image_path"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Test image does not exist: {image_path}")
    checkpoint = Path(job["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Model checkpoint does not exist: {checkpoint}")

    model_name = str(job["model"])
    model_parameters = preflight_selected_models(job, [model_name])[model_name]

    import anomalib
    from anomalib.engine import Engine

    configure_anomalib_asset_paths(asset_root)
    anomalib_version = getattr(anomalib, "__version__", "unknown")
    require_supported_anomalib_version(anomalib_version)
    training_parameters = training_parameters_from_job(job, model_name)
    threshold = min(1.0, max(0.0, float(job.get("threshold", 0.60))))
    output_root = Path(job["output_root"])
    model_root = output_root / model_name.lower()
    model_root.mkdir(parents=True, exist_ok=True)

    emit(
        "hello",
        demo=False,
        anomalib_version=anomalib_version,
    )
    emit(
        "stage",
        model=model_name,
        status="inference",
        message=f"Testing {image_path.name} with {model_name}",
    )
    model = create_model(model_name, model_parameters)
    engine = Engine(default_root_dir=str(model_root), enable_progress_bar=False)
    started = time.perf_counter()
    predictions = engine.predict(
        model=model,
        data_path=str(image_path),
        ckpt_path=str(checkpoint),
        return_predictions=True,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    prediction = first_prediction(predictions)
    if prediction is None:
        raise ValueError("Anomalib did not return an image prediction.")

    result = write_prediction_artifacts(
        image_path,
        prediction,
        model_root,
        threshold,
    )
    result.update(
        {
            "model": model_name,
            "checkpoint": str(checkpoint.resolve()),
            "latency_ms": round(latency_ms, 2),
            "demo": False,
            "model_parameters": parameter_snapshot(
                model_name, model_parameters
            ),
            "training_parameters": training_parameters,
        }
    )
    emit("inference", **result)


class _CheckpointInferenceBackend:
    def __init__(self, metadata: dict[str, Any], session_root: Path) -> None:
        import torch
        from anomalib.engine import Engine

        self.torch = torch
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        self.model_name = str(metadata["model"])
        parameter_map = metadata.get("model_parameters", {})
        parameters = parameter_map.get(self.model_name, {})
        self.model_parameters = normalize_model_parameters(
            self.model_name, parameters
        )
        self.saved_sigma = (
            self.model_parameters["sigma"] if self.model_name == "PatchCore" else None
        )
        self.sigma_override_supported = self.model_name == "PatchCore"
        constructor_parameters = deepcopy(self.model_parameters)
        if self.model_name in ("PatchCore", "PaDiM"):
            # All learned backbone weights are restored below. Avoid resolving
            # pretrained assets while loading a self-contained checkpoint.
            constructor_parameters["pre_trained"] = False
        self.model = create_model(self.model_name, constructor_parameters)
        restore_checkpoint_state(self.model, metadata["path"])
        self.input_size = image_size_tuple(self.model_parameters)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
        self.model.eval()
        self.model.to(self.device)
        self.engine = Engine(
            default_root_dir=str(session_root),
            enable_progress_bar=False,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            logger=False,
        )
        self.backend_name = "PyTorch"
        self._warm_up()

    def _warm_up(self) -> None:
        try:
            inner_model = getattr(self.model, "model", None)
            if inner_model is None:
                return
            sample = self.torch.zeros(
                (1, 3, *self.input_size),
                dtype=self.torch.float32,
                device=self.device,
            )
            with self.torch.inference_mode():
                inner_model(sample)
            if self.device.startswith("cuda"):
                self.torch.cuda.synchronize()
        except Exception:
            # Some models require the complete Anomalib input batch. Their
            # first real prediction performs the harmless warm-up instead.
            pass

    def predict(
        self, image_path: Path, sigma_override: int | None = None
    ) -> tuple[Any, float]:
        if self.model_name == "PatchCore":
            configure_patchcore_sigma(
                self.model,
                sigma_override if sigma_override is not None else self.saved_sigma,
            )
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            predictions = self.engine.predict(
                model=self.model,
                data_path=str(image_path),
                ckpt_path=None,
                return_predictions=True,
            )
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        prediction = first_prediction(predictions)
        if prediction is None:
            raise ValueError("Anomalib did not return an image prediction.")
        return prediction, latency_ms


class _TorchExportInferenceBackend:
    def __init__(self, metadata: dict[str, Any]) -> None:
        import torch
        from anomalib.deploy import TorchInferencer

        self.torch = torch
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True
        previous_trust = os.environ.get("TRUST_REMOTE_CODE")
        os.environ["TRUST_REMOTE_CODE"] = "1"
        try:
            self.inferencer = TorchInferencer(
                metadata["path"], device="cuda" if torch.cuda.is_available() else "cpu"
            )
        finally:
            if previous_trust is None:
                os.environ.pop("TRUST_REMOTE_CODE", None)
            else:
                os.environ["TRUST_REMOTE_CODE"] = previous_trust
        candidates = [
            type(self.inferencer.model).__name__,
            type(getattr(self.inferencer.model, "model", None)).__name__,
            self.inferencer.model.__class__.__module__,
        ]
        detected = next(
            (name for value in candidates if (name := _supported_model_name(value))),
            None,
        )
        expected = metadata.get("model") or None
        if expected and detected and expected != detected:
            raise ValueError(
                f"Manifest identifies {expected}, but Torch export contains {detected}."
            )
        self.model_name = expected or detected
        if not self.model_name:
            raise ValueError(
                "Could not identify the Torch export as PatchCore, PaDiM, or EfficientAD."
            )
        self.model_parameters = normalize_model_parameters(
            self.model_name,
            metadata.get("model_parameters", {}).get(self.model_name, {}),
        )
        exported_input_size = getattr(self.inferencer.model, "input_size", None)
        if exported_input_size is not None and len(exported_input_size) == 2:
            self.model_parameters["image_size"] = {
                "width": int(exported_input_size[1]),
                "height": int(exported_input_size[0]),
            }
            self.model_parameters = normalize_model_parameters(
                self.model_name, self.model_parameters
            )
        self.input_size = image_size_tuple(self.model_parameters)
        self.saved_sigma = (
            self.model_parameters["sigma"] if self.model_name == "PatchCore" else None
        )
        self.sigma_override_supported = False
        self.device = str(self.inferencer.device)
        self.device_name = (
            torch.cuda.get_device_name(0) if self.device.startswith("cuda") else "CPU"
        )
        self.backend_name = "PyTorch export"
        try:
            self.inferencer.predict(
                torch.zeros((3, *self.input_size), dtype=torch.float32)
            )
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
        except Exception:
            pass

    def predict(
        self, image_path: Path, sigma_override: int | None = None
    ) -> tuple[Any, float]:
        if sigma_override is not None:
            raise ValueError("Sigma override is not supported by this Torch export.")
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        started = time.perf_counter()
        with self.torch.inference_mode():
            prediction = self.inferencer.predict(image_path)
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize()
        return prediction, (time.perf_counter() - started) * 1000.0


class _OnnxInferenceBackend:
    def __init__(self, metadata: dict[str, Any]) -> None:
        import numpy as np
        import torch

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "ONNX Runtime GPU is not installed. Install onnxruntime-gpu>=1.27,<1.30."
            ) from exc
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
        available = ort.get_available_providers()
        providers: list[Any] = []
        if torch.cuda.is_available() and "CUDAExecutionProvider" in available:
            providers.append(
                (
                    "CUDAExecutionProvider",
                    {
                        "device_id": 0,
                        "cudnn_conv_use_max_workspace": "1",
                        "do_copy_in_default_stream": "1",
                    },
                )
            )
        providers.append("CPUExecutionProvider")
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            metadata["path"], sess_options=options, providers=providers
        )
        self.np = np
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [item.name for item in self.session.get_outputs()]
        active = self.session.get_providers()[0]
        self.device = "cuda:0" if active == "CUDAExecutionProvider" else "cpu"
        self.device_name = (
            torch.cuda.get_device_name(0) if self.device.startswith("cuda") else "CPU"
        )
        self.backend_name = f"ONNX Runtime · {active}"
        self.model_name = str(metadata["model"])
        self.model_parameters = normalize_model_parameters(
            self.model_name,
            metadata.get("model_parameters", {}).get(self.model_name, {}),
        )
        self.input_size = image_size_tuple(self.model_parameters)
        self.saved_sigma = (
            self.model_parameters["sigma"] if self.model_name == "PatchCore" else None
        )
        self.raw_anomaly_map_output = str(
            metadata.get("raw_anomaly_map_output", "")
        )
        self.sigma_override_supported = bool(
            metadata.get("sigma_override_supported")
            and self.raw_anomaly_map_output in self.output_names
        )
        self._run_array(np.zeros((*self.input_size, 3), dtype=np.uint8))

    def _run_array(self, image: Any) -> dict[str, Any]:
        value = self.np.asarray(image, dtype=self.np.float32)
        if value.max(initial=0.0) > 1.0:
            value /= 255.0
        if value.ndim == 3:
            value = value[None, ...]
        if value.shape[-1] == 3:
            value = value.transpose(0, 3, 1, 2)
        outputs = self.session.run(None, {self.input_name: value})
        return dict(zip(self.output_names, outputs, strict=True))

    def predict(
        self, image_path: Path, sigma_override: int | None = None
    ) -> tuple[Any, float]:
        from PIL import Image

        with Image.open(image_path) as source:
            height, width = self.input_size
            resized = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            image = self.np.asarray(resized)
        started = time.perf_counter()
        prediction = self._run_array(image)
        if sigma_override is not None and not self.sigma_override_supported:
            raise ValueError("Sigma override is not supported by this ONNX model.")
        if self.sigma_override_supported:
            sigma = sigma_override if sigma_override is not None else self.saved_sigma
            prediction["anomaly_map"] = gaussian_blur_anomaly_map(
                prediction[self.raw_anomaly_map_output], sigma
            )
        return prediction, (time.perf_counter() - started) * 1000.0


class _OpenVinoOnnxInferenceBackend:
    """Run override-capable ONNX graphs through OpenVINO without losing outputs."""

    def __init__(
        self,
        metadata: dict[str, Any],
        device_cache: dict[str, str],
    ) -> None:
        import numpy as np
        import openvino as ov

        self.np = np
        self.model_name = str(metadata["model"])
        self.model_parameters = normalize_model_parameters(
            self.model_name,
            metadata.get("model_parameters", {}).get(self.model_name, {}),
        )
        self.input_size = image_size_tuple(self.model_parameters)
        self.saved_sigma = self.model_parameters["sigma"]
        self.raw_anomaly_map_output = str(metadata["raw_anomaly_map_output"])
        self.sigma_override_supported = True
        core = ov.Core()
        source_model = core.read_model(metadata["path"])
        device_names = {
            device: str(core.get_property(device, "FULL_DEVICE_NAME"))
            for device in core.available_devices
        }
        cached = device_cache.get(str(metadata["fingerprint"]))
        candidates = [cached] if cached in device_names else []
        candidates += [
            device
            for device in device_names
            if device.startswith("GPU") and device not in candidates
        ]
        if "CPU" in device_names and "CPU" not in candidates:
            candidates.append("CPU")
        failures: list[str] = []
        for device in candidates:
            try:
                self.compiled_model = core.compile_model(source_model, device)
                self.device = device
                self.device_name = device_names[device]
                break
            except Exception as exc:
                failures.append(f"{device}: {exc}")
        else:
            raise RuntimeError(
                "OpenVINO could not compile the override-capable ONNX model. "
                + "; ".join(failures)
            )
        device_cache[str(metadata["fingerprint"])] = self.device
        self.input = self.compiled_model.input(0)
        self.outputs = list(self.compiled_model.outputs)
        self.output_names = []
        for index, output in enumerate(self.outputs):
            names = set(output.get_names())
            preferred = next(
                (
                    name
                    for name in (
                        "pred_score",
                        "pred_label",
                        "anomaly_map",
                        "pred_mask",
                        PATCHCORE_RAW_ANOMALY_MAP_OUTPUT,
                    )
                    if name in names
                ),
                None,
            )
            fallback_name = f"output_{index}"
            if preferred is None and names:
                fallback_name = sorted(names)[0]
            self.output_names.append(preferred or fallback_name)
        if self.raw_anomaly_map_output not in self.output_names:
            raise RuntimeError("OpenVINO did not preserve the raw anomaly-map output.")
        self.backend_name = f"OpenVINO ONNX · {self.device}"
        self._run_array(np.zeros((*self.input_size, 3), dtype=np.uint8))

    def _run_array(self, image: Any) -> dict[str, Any]:
        value = self.np.asarray(image, dtype=self.np.float32)
        if value.max(initial=0.0) > 1.0:
            value /= 255.0
        if value.ndim == 3:
            value = value[None, ...]
        if value.shape[-1] == 3:
            value = value.transpose(0, 3, 1, 2)
        output_values = self.compiled_model({self.input: value})
        return {
            name: self.np.asarray(output_values[output])
            for name, output in zip(self.output_names, self.outputs, strict=True)
        }

    def predict(
        self, image_path: Path, sigma_override: int | None = None
    ) -> tuple[Any, float]:
        from PIL import Image

        with Image.open(image_path) as source:
            height, width = self.input_size
            resized = source.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            image = self.np.asarray(resized)
        started = time.perf_counter()
        prediction = self._run_array(image)
        sigma = sigma_override if sigma_override is not None else self.saved_sigma
        prediction["anomaly_map"] = gaussian_blur_anomaly_map(
            prediction[self.raw_anomaly_map_output], sigma
        )
        return prediction, (time.perf_counter() - started) * 1000.0


class _OpenVINOInferenceBackend:
    def __init__(
        self,
        metadata: dict[str, Any],
        device_cache: dict[str, str],
    ) -> None:
        import numpy as np
        import openvino as ov
        from anomalib.deploy import OpenVINOInferencer

        self.np = np
        self.model_name = str(metadata["model"])
        self.model_parameters = normalize_model_parameters(
            self.model_name,
            metadata.get("model_parameters", {}).get(self.model_name, {}),
        )
        self.input_size = image_size_tuple(self.model_parameters)
        self.saved_sigma = (
            self.model_parameters["sigma"] if self.model_name == "PatchCore" else None
        )
        self.sigma_override_supported = False
        fingerprint = str(metadata["fingerprint"])
        core = ov.Core()
        device_names = {
            device: str(core.get_property(device, "FULL_DEVICE_NAME"))
            for device in core.available_devices
        }
        cached = device_cache.get(fingerprint)
        candidates = [cached] if cached in device_names else []
        candidates += ["CPU"] if "CPU" in device_names and "CPU" not in candidates else []
        candidates += [
            device
            for device, name in device_names.items()
            if device.startswith("GPU")
            and "intel" in name.lower()
            and device not in candidates
        ]
        sample = np.zeros((*self.input_size, 3), dtype=np.uint8)
        successes: list[tuple[float, str, str, Any]] = []
        failures: list[str] = []
        for device in candidates:
            try:
                inferencer = OpenVINOInferencer(metadata["path"], device=device)
                timings = []
                for _ in range(3):
                    started = time.perf_counter()
                    inferencer.predict(sample)
                    timings.append((time.perf_counter() - started) * 1000.0)
                steady = sorted(timings[1:] or timings)[len(timings[1:] or timings) // 2]
                successes.append((steady, device, device_names[device], inferencer))
                if cached == device:
                    break
            except Exception as exc:
                failures.append(f"{device}: {exc}")
        if not successes:
            raise RuntimeError(
                "OpenVINO could not compile the model on a compatible device. "
                + "; ".join(failures)
            )
        _, selected_device, selected_name, self.inferencer = min(
            successes, key=lambda item: item[0]
        )
        device_cache[fingerprint] = selected_device
        self.device = selected_device
        self.device_name = selected_name
        self.backend_name = "OpenVINO"

    def predict(
        self, image_path: Path, sigma_override: int | None = None
    ) -> tuple[Any, float]:
        from PIL import Image

        if sigma_override is not None:
            raise ValueError("Sigma override is not supported by this OpenVINO model.")
        height, width = self.input_size
        with Image.open(image_path) as source:
            image = self.np.asarray(
                source.convert("RGB").resize(
                    (width, height), Image.Resampling.BILINEAR
                )
            )
        started = time.perf_counter()
        prediction = self.inferencer.predict(image)
        return prediction, (time.perf_counter() - started) * 1000.0


class PersistentInferenceSession:
    """Own exactly one loaded inference backend for the lifetime of the worker."""

    def __init__(self, asset_root: Path, session_root: Path) -> None:
        self.asset_root = asset_root
        self.session_root = session_root
        self.backend: Any | None = None
        self.metadata: dict[str, Any] = {}
        self.device_cache: dict[str, str] = {}
        configure_anomalib_asset_paths(asset_root)

    def inspect(self, path: str) -> dict[str, Any]:
        return inspect_model_artifact(path)

    def load(self, command: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        metadata = self.inspect(str(command.get("path", "")))
        if metadata["requires_trust"] and command.get("trusted_sha256") != metadata["fingerprint"]:
            return "trust_required", metadata

        started = time.perf_counter()
        artifact_format = metadata["format"]
        if artifact_format == "checkpoint":
            candidate = _CheckpointInferenceBackend(metadata, self.session_root)
        elif artifact_format == "torch":
            candidate = _TorchExportInferenceBackend(metadata)
            metadata["model"] = candidate.model_name
            metadata["model_parameters"] = parameter_snapshot(
                candidate.model_name, candidate.model_parameters
            )
        elif artifact_format == "onnx":
            try:
                candidate = _OnnxInferenceBackend(metadata)
            except Exception as onnx_error:
                if metadata.get("sigma_override_supported"):
                    candidate = _OpenVinoOnnxInferenceBackend(
                        metadata, self.device_cache
                    )
                else:
                    candidate = _OpenVINOInferenceBackend(
                        metadata, self.device_cache
                    )
                candidate.backend_name += f" · ONNX fallback ({type(onnx_error).__name__})"
        else:
            candidate = _OpenVINOInferenceBackend(metadata, self.device_cache)

        previous = self.backend
        self.backend = candidate
        self.metadata = metadata
        del previous
        self._release_accelerator_cache()
        load_ms = (time.perf_counter() - started) * 1000.0
        return "model_loaded", {
            **metadata,
            "model": candidate.model_name,
            "model_parameters": parameter_snapshot(
                candidate.model_name, candidate.model_parameters
            ),
            "backend": candidate.backend_name,
            "device": candidate.device,
            "device_name": candidate.device_name,
            "load_ms": round(load_ms, 2),
            "saved_sigma": getattr(candidate, "saved_sigma", None),
            "sigma_override_supported": getattr(
                candidate, "sigma_override_supported", False
            ),
        }

    def predict(self, command: dict[str, Any]) -> dict[str, Any]:
        if self.backend is None:
            raise RuntimeError("Load a model before running inference.")
        image_path = Path(str(command.get("image_path", ""))).expanduser()
        if not image_path.is_file():
            raise FileNotFoundError(f"Test image does not exist: {image_path}")
        threshold = min(1.0, max(0.0, float(command.get("threshold", 0.60))))
        sigma_override = sigma_override_from_command(command)
        sigma_override_supported = getattr(
            self.backend, "sigma_override_supported", False
        )
        saved_sigma = getattr(self.backend, "saved_sigma", None)
        if sigma_override is not None and not sigma_override_supported:
            raise ValueError(
                "The loaded model does not support a runtime sigma override."
            )
        output_root = Path(str(command.get("output_root", ""))).expanduser()
        if not str(output_root):
            raise ValueError("output_root is required.")
        total_started = time.perf_counter()
        if sigma_override is None:
            prediction, model_latency_ms = self.backend.predict(image_path)
        else:
            prediction, model_latency_ms = self.backend.predict(
                image_path, sigma_override
            )
        result = write_prediction_artifacts(
            image_path, prediction, output_root, threshold
        )
        total_latency_ms = (time.perf_counter() - total_started) * 1000.0
        result.update(
            {
                "model": self.backend.model_name,
                "artifact_path": self.metadata["path"],
                "artifact_format": self.metadata["format"],
                "fingerprint": self.metadata["fingerprint"],
                "backend": self.backend.backend_name,
                "device": self.backend.device,
                "device_name": self.backend.device_name,
                "latency_ms": round(model_latency_ms, 2),
                "model_latency_ms": round(model_latency_ms, 2),
                "total_latency_ms": round(total_latency_ms, 2),
                "cache_hit": True,
                "model_parameters": parameter_snapshot(
                    self.backend.model_name, self.backend.model_parameters
                ),
                "saved_sigma": saved_sigma,
                "effective_sigma": (
                    sigma_override
                    if sigma_override is not None
                    else saved_sigma
                ),
                "sigma_overridden": sigma_override is not None,
                "sigma_override_supported": sigma_override_supported,
            }
        )
        return result

    def unload(self) -> None:
        self.backend = None
        self.metadata = {}
        self._release_accelerator_cache()

    @staticmethod
    def _release_accelerator_cache() -> None:
        import gc

        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass


def run_inference_service(asset_root_value: str, session_root_value: str) -> int:
    asset_root = Path(asset_root_value).expanduser().resolve()
    session_root = Path(session_root_value).expanduser().resolve()
    session_root.mkdir(parents=True, exist_ok=True)
    environment_job = {"asset_root": str(asset_root)}
    configure_offline_environment(environment_job)
    install_offline_network_guard()
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    logging.getLogger("anomalib").setLevel(logging.ERROR)
    session = PersistentInferenceSession(asset_root, session_root)
    emit("session_ready", pid=os.getpid())
    for raw_line in __import__("sys").stdin:
        request_id = ""
        try:
            command = json.loads(raw_line)
            if not isinstance(command, dict):
                raise ValueError("Session command must be a JSON object.")
            request_id = str(command.get("request_id", ""))
            action = str(command.get("command", ""))
            if action == "inspect_model":
                emit(
                    "model_inspected",
                    request_id=request_id,
                    **session.inspect(str(command.get("path", ""))),
                )
            elif action == "load_model":
                event_name, payload = session.load(command)
                emit(event_name, request_id=request_id, **payload)
            elif action == "predict":
                emit("inference", request_id=request_id, **session.predict(command))
            elif action == "unload":
                session.unload()
                emit("model_unloaded", request_id=request_id)
            elif action == "shutdown":
                session.unload()
                emit("session_stopped", request_id=request_id)
                return 0
            else:
                raise ValueError(f"Unknown inference session command: {action}")
        except Exception as exc:
            emit(
                "error",
                request_id=request_id,
                message=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
    session.unload()
    return 0


def run_check() -> None:
    try:
        import anomalib
        import torch

        anomalib_version = getattr(anomalib, "__version__", "unknown")
        require_supported_anomalib_version(anomalib_version)
        onnx_providers: list[str] = []
        try:
            import onnxruntime as ort

            onnx_providers = list(ort.get_available_providers())
        except ImportError:
            pass
        openvino_devices: list[str] = []
        try:
            import openvino as ov

            core = ov.Core()
            openvino_devices = [
                f"{device}: {core.get_property(device, 'FULL_DEVICE_NAME')}"
                for device in core.available_devices
            ]
        except ImportError:
            pass
        cuda_available = bool(torch.cuda.is_available())
        cuda_device = torch.cuda.get_device_name(0) if cuda_available else ""
        onnx_status = (
            "ONNX CUDA provider ready"
            if "CUDAExecutionProvider" in onnx_providers
            else "ONNX CUDA provider unavailable"
        )
        openvino_status = f"OpenVINO devices: {len(openvino_devices)}"
        emit(
            "environment",
            ok=True,
            demo=False,
            anomalib_version=anomalib_version,
            torch_version=getattr(torch, "__version__", "unknown"),
            cuda_available=cuda_available,
            cuda_device=cuda_device,
            onnx_providers=onnx_providers,
            openvino_devices=openvino_devices,
            message=(
                f"Anomalib environment is ready. CUDA: {cuda_device}. "
                f"{onnx_status}. {openvino_status}."
                if cuda_available
                else "Anomalib environment is ready. CUDA is unavailable; "
                f"inference will use CPU fallbacks. {onnx_status}. {openvino_status}."
            ),
        )
    except Exception as exc:
        emit(
            "environment",
            ok=False,
            demo=False,
            message=f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job")
    mode.add_argument("--serve", action="store_true")
    parser.add_argument("--asset-root")
    parser.add_argument("--session-root")
    args = parser.parse_args()
    if args.serve:
        if not args.asset_root or not args.session_root:
            parser.error("--serve requires --asset-root and --session-root")
        try:
            return run_inference_service(args.asset_root, args.session_root)
        except Exception as exc:
            emit(
                "error",
                request_id="",
                message=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
            )
            return 1
    job = load_job(args.job)
    mode = job.get("mode", "train")

    try:
        if job.get("demo", False):
            run_demo(job)
        elif mode == "assets":
            if str(job.get("action", "check")).lower() != "download":
                configure_offline_environment(job)
                install_offline_network_guard()
            run_assets(job)
        elif mode == "check":
            configure_offline_environment(job)
            install_offline_network_guard()
            run_check()
        elif mode == "export":
            configure_offline_environment(job)
            install_offline_network_guard()
            run_real_export(job)
        elif mode == "inference":
            configure_offline_environment(job)
            install_offline_network_guard()
            run_real_inference(job)
        else:
            configure_offline_environment(job)
            install_offline_network_guard()
            run_real_train(job)
        return 0
    except Exception as exc:
        emit(
            "error",
            message=f"{type(exc).__name__}: {exc}",
            traceback=traceback.format_exc(),
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
