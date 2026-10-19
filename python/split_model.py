"""Split an exported ONNX model into <25 MiB chunks (Cloudflare's per-file limit) plus a manifest.

    python split_model.py build/timestral-28m.onnx ../web/model

Files are named by content hash so they can be cached forever; manifest.json is the only mutable file.
"""
import hashlib
import json
import sys
from pathlib import Path

CHUNK = 24 * 1024 * 1024  # 24 MiB, under the 25 MiB limit

src, out = Path(sys.argv[1]), Path(sys.argv[2])
data = src.with_name(src.name + ".data")
out.mkdir(parents=True, exist_ok=True)
for old in out.glob("ts-*"):
    old.unlink()

blob = data.read_bytes()
tag = hashlib.sha256(blob + src.read_bytes()).hexdigest()[:10]
graph_name = f"ts-{tag}.onnx"
(out / graph_name).write_bytes(src.read_bytes())
parts = []
for i in range(0, len(blob), CHUNK):
    name = f"ts-{tag}.data.{i // CHUNK}"
    (out / name).write_bytes(blob[i:i + CHUNK])
    parts.append({"file": name, "size": min(CHUNK, len(blob) - i)})
manifest = {"version": tag, "graph": graph_name, "dataName": data.name, "parts": parts, "totalBytes": len(blob)}
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print(json.dumps(manifest))
