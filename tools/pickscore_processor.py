"""Fetch the exact tokenizer/image processor used by the authors' PickScore inference recipe."""
import hashlib
import json
from pathlib import Path
import time
import requests

REPO="laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
REV="1c2b8495b28150b8a4922ee1c8edee224c284c0c"

def fetch_processor(root):
    root=Path(root)
    root.mkdir(parents=True,exist_ok=True)
    pins=json.loads((Path(__file__).resolve().parents[1]/"configs/processor-sha256.json").read_text())
    for file,sha in pins.items():
        dest=root/file
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest()==sha:
            continue
        for retry in range(3):
            try:
                r=requests.get(f"https://huggingface.co/{REPO}/resolve/{REV}/{file}",timeout=45)
                r.raise_for_status()
                if hashlib.sha256(r.content).hexdigest()!=sha:
                    raise IOError(f"Unexpected processor contents: {file}")
                tmp=dest.with_suffix(".tmp")
                tmp.write_bytes(r.content)
                tmp.replace(dest)
                break
            except requests.RequestException:
                if retry==2:
                    raise
                time.sleep(1+retry)
    return root
