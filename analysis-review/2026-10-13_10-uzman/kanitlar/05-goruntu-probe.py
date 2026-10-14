"""Small synthetic preprocessing probes; no training, models, camera or downloads."""
from pathlib import Path
import importlib.util
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
FIXTURE = OUT / "rol05-sentetik-goruntu"
FIXTURE.mkdir(exist_ok=True)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
spec = importlib.util.spec_from_file_location("review_worker", ROOT / "worker/anomalib_worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
worker.install_offline_network_guard()

import numpy as np
import torch
from PIL import Image
from anomalib.data.utils import read_image
from anomalib.deploy import OpenVINOInferencer
from anomalib.models import Patchcore
from torchvision import tv_tensors

result = {"scope": "CPU synthetic preprocessing only; fake backends; no model weights loaded"}

class CaptureSession:
    def run(self, _, feeds):
        self.value = next(iter(feeds.values())).copy()
        return [np.zeros((1, 32, 32), dtype=np.float32)]

class CaptureCompiled:
    def __call__(self, feeds):
        self.value = next(iter(feeds.values())).copy()
        return {"out": np.zeros((1, 32, 32), dtype=np.float32)}

dark = np.ones((64, 64, 3), dtype=np.uint8)
dark[0, 0] = 0
dark_path = FIXTURE / "dark-uint8-one.png"
Image.fromarray(dark).save(dark_path)
onnx = worker._OnnxInferenceBackend.__new__(worker._OnnxInferenceBackend)
onnx.np = np
onnx.input_name = "input"
onnx.output_names = ["anomaly_map"]
onnx.session = CaptureSession()
onnx.input_size = (32, 32)
onnx.sigma_override_supported = False
onnx.predict(dark_path)
ov_onnx = worker._OpenVinoOnnxInferenceBackend.__new__(worker._OpenVinoOnnxInferenceBackend)
ov_onnx.np = np
ov_onnx.input = "input"
ov_onnx.outputs = ["out"]
ov_onnx.output_names = ["anomaly_map"]
ov_onnx.compiled_model = CaptureCompiled()
ov_onnx._run_array(dark)
legacy = OpenVINOInferencer.pre_process(dark.copy())
torch_pixels = read_image(dark_path, as_tensor=True)
result["dark_uint8_scale"] = {
    "torch_loader_max": float(torch_pixels.max()),
    "onnx_wrapper_actual_predict_input_max": float(onnx.session.value.max()),
    "openvino_onnx_run_array_input_max": float(ov_onnx.compiled_model.value.max()),
    "openvino_legacy_pre_process_input_max": float(legacy.max()),
    "ratio": float(onnx.session.value.max() / torch_pixels.max()),
}
assert result["dark_uint8_scale"]["onnx_wrapper_actual_predict_input_max"] == 1.0
assert abs(result["dark_uint8_scale"]["ratio"] - 255.0) < 0.01

pp = Patchcore.configure_pre_processor(image_size=(32, 48))
result["resize_flags"] = {
    "train_antialias": pp.transform.transforms[0].antialias,
    "export_antialias": pp.export_transform.transforms[0].antialias,
    "size_height_width": list(pp.transform.transforms[0].size),
}

def compare_resize(name, array):
    path = FIXTURE / f"{name}.png"
    Image.fromarray(array).save(path)
    raw = read_image(path, as_tensor=True).unsqueeze(0)
    train = pp.transform(raw)
    exported_torch = pp(raw)
    pil = Image.open(path).convert("RGB").resize((48, 32), Image.Resampling.BILINEAR)
    pil_array = np.asarray(pil).copy()
    external = torch.from_numpy(pil_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    exported_pil = pp(external)
    def stats(a, b):
        delta = (a - b).abs()
        return {"max_abs_normalized_tensor": float(delta.max()), "mean_abs_normalized_tensor": float(delta.mean())}
    return {"train_vs_torch_export": stats(train, exported_torch),
            "train_vs_onnx_openvino_pil": stats(train, exported_pil),
            "torch_export_vs_onnx_openvino_pil": stats(exported_torch, exported_pil)}

rng = np.random.default_rng(42)
result["resize_numeric_differences"] = {
    "random_96x80_to_48x32": compare_resize("random", rng.integers(0, 256, (80, 96, 3), dtype=np.uint8)),
    "checker_96x80_to_48x32": compare_resize("checker", np.repeat((((np.indices((80, 96)).sum(0) % 2) * 255).astype(np.uint8))[..., None], 3, axis=2)),
}

colors = np.asarray([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]], dtype=np.uint8)
color_path = FIXTURE / "rgb-colors.png"
Image.fromarray(colors).save(color_path)
torch_colors = read_image(color_path, as_tensor=True).permute(1, 2, 0).numpy()
onnx._run_array(colors)
result["rgb_color_confirmation"] = {
    "torch_rgb_values": torch_colors.tolist(),
    "onnx_rgb_values": onnx.session.value[0].transpose(1, 2, 0).tolist(),
    "equal": bool(np.array_equal(torch_colors, onnx.session.value[0].transpose(1, 2, 0))),
}

mask = tv_tensors.Mask(torch.tensor(np.indices((80, 96)).sum(0) % 2, dtype=torch.uint8))
image = read_image(FIXTURE / "random.png", as_tensor=True)
_, resized_mask = pp.transform(image, mask)
result["ground_truth_mask_confirmation"] = {
    "resized_shape": list(resized_mask.shape), "unique_values": torch.unique(resized_mask).tolist(),
    "mask_type_preserved": isinstance(resized_mask, tv_tensors.Mask),
}

raw_16 = np.asarray([[0, 1, 255, 256, 1024, 32768, 65535]], dtype=np.uint16)
depth_path = FIXTURE / "grayscale-16bit.tiff"
Image.fromarray(raw_16).save(depth_path)
with Image.open(depth_path) as im:
    result["accepted_16bit_tiff_rgb_conversion"] = {
        "source_mode": im.mode, "source_pixel_values": raw_16.tolist(),
        "convert_rgb_values": np.asarray(im.convert("RGB"))[:, :, 0].tolist(),
        "torch_read_image_values": read_image(depth_path, as_tensor=True)[0].tolist(),
    }

def crop_geometry(source, viewport):
    sw, sh = source
    vw, vh = viewport
    scale = max(vw / sw, vh / sh)
    return {"scaled_size": [sw * scale, sh * scale],
            "fraction_source_visible_x": vw / (sw * scale),
            "fraction_source_visible_y": vh / (sh * scale),
            "center_crop_source_y_range": [(sh - vh / scale) / 2, (sh + vh / scale) / 2]}

result["qt_documented_expand_geometry_math_no_gui"] = {
    "training_input_640x480_into_330x112": crop_geometry((640, 480), (330, 112)),
    "training_heatmap_640x480_into_330x96": crop_geometry((640, 480), (330, 96)),
    "training_threshold_raw_256x256_into_330x74": crop_geometry((256, 256), (330, 74)),
    "inference_640x480_into_230x230": crop_geometry((640, 480), (230, 230)),
    "normalized_y_point": 0.625,
    "same_point_heatmap_viewport_y": (0.625 - 0.5) * 247.5 + 48,
    "same_point_raw_mask_viewport_y": (0.625 - 0.5) * 330 + 37,
}

(OUT / "05-goruntu-probe.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
