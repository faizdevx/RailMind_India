"""Download historical delay data (cc-by-4.0) from Hugging Face pjay-in/indian-railways-delays.

Usage: python scripts/download_data.py [--only observed|derived|all]
Files go to data/raw/ (git-ignored). Existing files are skipped.
"""
import argparse
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = "pjay-in/indian-railways-delays"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw"


def listing():
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main?recursive=true"
    return [f for f in json.load(urllib.request.urlopen(url)) if f["type"] == "file"
            and f["path"].endswith(".parquet")]


def fetch(f):
    dst = OUT / f["path"]
    if dst.exists() and dst.stat().st_size == f.get("size"):
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(f"https://huggingface.co/datasets/{REPO}/resolve/main/{f['path']}", dst)
    return f.get("size", 0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["observed", "derived", "all"], default="all")
    a = ap.parse_args()
    files = listing()
    if a.only == "observed":
        files = [f for f in files if f["path"].startswith("data/")]
    elif a.only == "derived":
        files = [f for f in files if not f["path"].startswith("data/")]
    with ThreadPoolExecutor(8) as ex:
        got = sum(ex.map(fetch, files))
    print(f"{len(files)} files, {got / 1e6:.1f} MB downloaded")
