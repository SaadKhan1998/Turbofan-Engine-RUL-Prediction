"""Download the NASA CMAPSS turbofan dataset into a local folder.

Usage:
    python download_data.py --dataset_dir data/CMAPSS

Only the plain-text train/test/RUL files are extracted (nested zips are handled).
"""
from __future__ import annotations

import argparse
import io
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = [
    "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip",
    "https://data.nasa.gov/download/ff5v-kuh6/application%2Fzip",
]
WANTED = re.compile(r"^(train|test|RUL)_FD00[1-4]\.txt$")


def _extract(zf: zipfile.ZipFile, out: Path, depth: int = 0) -> list[str]:
    found = []
    for info in zf.infolist():
        name = Path(info.filename).name  # basename only: never trust archive paths
        if WANTED.match(name):
            (out / name).write_bytes(zf.read(info))
            found.append(name)
        elif name.lower().endswith(".zip") and depth < 2:
            with zipfile.ZipFile(io.BytesIO(zf.read(info))) as inner:
                found += _extract(inner, out, depth + 1)
    return found


def download(dataset_dir: str | Path) -> Path:
    out = Path(dataset_dir)
    out.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in URLS:
        try:
            print(f"Downloading {url} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "cmapss-downloader"})
            with urllib.request.urlopen(req, timeout=120) as r:
                payload = r.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                found = _extract(zf, out)
            if found:
                print(f"Extracted {len(found)} files to {out.resolve()}: {', '.join(sorted(found))}")
                return out
            errors.append(f"{url}: archive had no CMAPSS txt files")
        except Exception as e:  # try the next mirror
            errors.append(f"{url}: {e}")
    raise RuntimeError("Could not download CMAPSS. Download it manually from the NASA Prognostics "
                       "Data Repository and extract the txt files into the folder.\n" + "\n".join(errors))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset_dir", default="data/CMAPSS")
    try:
        download(p.parse_args().dataset_dir)
    except RuntimeError as e:
        sys.exit(str(e))
