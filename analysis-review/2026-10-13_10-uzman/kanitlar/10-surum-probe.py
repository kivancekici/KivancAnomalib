"""Read-only source/resource/test inventory; outputs stay in the review folder."""
from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import re
import sys
import zlib

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_name("10-surum-probe.json")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def describe(path: Path) -> dict:
    payload = path.read_bytes()
    return {
        "path": str(path.relative_to(ROOT)),
        "length": len(payload),
        "sha256": digest(payload),
    }


source = ROOT / "worker/anomalib_worker.py"
source_hash = digest(source.read_bytes())
cached = [describe(source)]
for base in (ROOT / ".data", ROOT / "build"):
    cached.extend(describe(p) for p in sorted(base.rglob("anomalib_worker.py")))
for item in cached:
    item["matches_source"] = item["sha256"] == source_hash

resources = []
for path in sorted((ROOT / "build").rglob("qrc_resources.cpp")):
    text = path.read_text(encoding="utf-8")
    # Inspect only the first resource, which is named explicitly in these files.
    first = re.search(
        r"qt_resource_data\[\] = \{\s*// anomalib_worker\.py\s*(.*?)(?=\s*//|\s*\};)",
        text,
        re.S,
    )
    if not first:
        resources.append({"path": str(path.relative_to(ROOT)), "decoded": False})
        continue
    data = bytes(int(h, 16) for h in re.findall(r"0x([0-9a-fA-F]+)", first[1]))
    stored_size = int.from_bytes(data[:4], "big")
    compressed = data[4 : 4 + stored_size]
    uncompressed_size = int.from_bytes(compressed[:4], "big")
    payload = zlib.decompress(compressed[4:])
    assert len(payload) == uncompressed_size
    assert payload.startswith(b"#!/usr/bin/env python")
    resources.append({
        "path": str(path.relative_to(ROOT)),
        "decoded": True,
        "declared_stored_size": stored_size,
        "decoded_length": len(payload),
        "decoded_worker_sha256": digest(payload),
        "matches_source": digest(payload) == source_hash,
        "proof_scope": "generated qrc C++ only; executable not inspected/launched",
    })

tree = ast.parse((ROOT / "tests/test_worker.py").read_text(encoding="utf-8"))
tests = [
    {"name": node.name, "line": node.lineno}
    for node in ast.walk(tree)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
]
result = {
    "python_executable": sys.executable,
    "python_version": sys.version,
    "source_worker_sha256": source_hash,
    "worker_copies": cached,
    "generated_resources": resources,
    "test_count": len(tests),
    "tests": tests,
    "package_versions": {
        name: metadata.version(name)
        for name in ("pip", "anomalib", "torch", "torchvision", "lightning", "timm", "onnxruntime-gpu")
    },
}
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(result, ensure_ascii=False, indent=2))
