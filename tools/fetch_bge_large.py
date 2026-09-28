"""Resumable background download of the bge-large ONNX model.

The migration scenes (migrate apply / quality-gate / cutover) cannot run until
model.onnx is present in the fastembed cache. This script is safe to re-run:
huggingface_hub resumes from whatever is already on disk.
"""

import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO = "qdrant/bge-large-en-v1.5-onnx"
CACHE = Path.home() / "AppData" / "Local" / "Temp" / "fastembed_cache"

started = time.perf_counter()
attempt = 0
while attempt < 40:
    attempt += 1
    try:
        path = snapshot_download(
            repo_id=REPO,
            cache_dir=str(CACHE),
            allow_patterns=["model.onnx", "*.json", "*.txt"],
            max_workers=4,
        )
        size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
        print(
            f"DONE attempt={attempt} path={path} "
            f"size={size/1e6:.1f}MB elapsed={(time.perf_counter()-started)/60:.1f}min",
            flush=True,
        )
        sys.exit(0)
    except Exception as error:  # noqa: BLE001 - retry on any transport error
        print(f"attempt {attempt} failed: {type(error).__name__}: {error}", flush=True)
        time.sleep(min(30 * attempt, 300))

print("GAVE UP after repeated failures", flush=True)
sys.exit(1)
