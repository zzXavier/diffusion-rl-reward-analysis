"""Download only the immutable PickScore inference artifacts with resumable weights."""
from pathlib import Path
import argparse
import hashlib
import json
import os

import requests

MODEL = "yuvalkirstain/PickScore_v1"
REVISION = "a4e4367c6dfa7288a00c550414478f865b875800"
FILES = ["config.json","preprocessor_config.json","tokenizer_config.json","tokenizer.json",
         "vocab.json","merges.txt","special_tokens_map.json","model.safetensors"]


def fetch(target):
    target.mkdir(parents=True,exist_ok=True)
    pins=json.loads((Path(__file__).resolve().parents[1]/"configs/model-sha256.json").read_text())
    def verify():
        for file,expected_digest in pins.items():
            with (target/file).open("rb") as handle:
                if hashlib.file_digest(handle,"sha256").hexdigest()!=expected_digest:
                    raise IOError(f"Cached file does not match pinned SHA-256: {file}")
    if all((target/name).is_file() for name in FILES):
        verify()
        return target
    root=f"https://huggingface.co/{MODEL}/resolve/{REVISION}"
    tree=requests.get(f"https://huggingface.co/api/models/{MODEL}/tree/{REVISION}",timeout=45)
    tree.raise_for_status()
    entries={e["path"]:e for e in tree.json()}
    for name in FILES:
        dest=target/name
        expected=entries[name]
        if dest.exists() and dest.stat().st_size==expected["size"]:
            continue
        tmp=target/(name+".part")
        start=tmp.stat().st_size if tmp.exists() else 0
        r=requests.get(root+"/"+name,params={"start":start},headers={"Range":f"bytes={start}-"} if start else {},stream=True,timeout=(15,90))
        r.raise_for_status()
        if start and r.status_code==200:
            start=0
        with tmp.open("ab" if start else "wb") as fh:
            for block in r.iter_content(1024*1024):
                fh.write(block)
        if tmp.stat().st_size!=expected["size"]:
            raise IOError(f"Downloaded size mismatch for {name}")
        if expected.get("lfs"):
            with tmp.open("rb") as fh:
                digest=hashlib.file_digest(fh,"sha256").hexdigest()
            if digest!=expected["lfs"]["oid"]:
                raise IOError(f"SHA mismatch for {name}")
        tmp.replace(dest)
    verify()
    return target


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("target",type=Path)
    fetch(p.parse_args().target)
