"""Download a batch of UR Fall Detection (URFD) RGB sequences and build a validation
manifest. cam1 = ceiling/overhead (best geometry match to our deployment); cam0 = frontal.
ADL (normal activity) sequences are cam0 only.

    python scripts/download_urfd.py --falls 20 --adl 20
"""
from __future__ import annotations

import argparse
import json
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://fenix.ur.edu.pl/~mkepski/ds/data"
ROOT = Path("data/real/urfd")


def fetch(url: str, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size > 1000:
        return True
    try:
        urllib.request.urlretrieve(url, dst)
        return dst.stat().st_size > 1000
    except Exception as e:
        print(f"  fail {url}: {e}")
        return False


def unzip(z: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(out)
        return True
    except Exception as e:
        print(f"  bad zip {z}: {e}")
        return False


def frames_dir_of(extract_root: Path) -> Path:
    """URFD zips extract to a subfolder of PNGs; find the dir actually holding images."""
    pngs = list(extract_root.rglob("*.png"))
    if pngs:
        return pngs[0].parent
    return extract_root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--falls", type=int, default=20)
    ap.add_argument("--adl", type=int, default=20)
    args = ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    manifest = []

    # falls: cam1 (overhead) primary; fall-01..N
    for i in range(1, args.falls + 1):
        name = f"fall-{i:02d}-cam1-rgb"
        z = ROOT / f"{name}.zip"
        if fetch(f"{BASE}/{name}.zip", z) and unzip(z, ROOT / name):
            fd = frames_dir_of(ROOT / name)
            manifest.append({"id": name, "frames_dir": str(fd), "label": "fall", "split": "overhead"})
            print(f"  ok {name} ({len(list(fd.glob('*.png')))} frames)")

    # ADL (normal): cam0 only
    for i in range(1, args.adl + 1):
        name = f"adl-{i:02d}-cam0-rgb"
        z = ROOT / f"{name}.zip"
        if fetch(f"{BASE}/{name}.zip", z) and unzip(z, ROOT / name):
            fd = frames_dir_of(ROOT / name)
            manifest.append({"id": name, "frames_dir": str(fd), "label": "adl", "split": "frontal"})
            print(f"  ok {name} ({len(list(fd.glob('*.png')))} frames)")

    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    n_fall = sum(m["label"] == "fall" for m in manifest)
    print(f"\nmanifest: {len(manifest)} clips ({n_fall} fall / {len(manifest)-n_fall} adl) "
          f"-> {ROOT/'manifest.json'}")


if __name__ == "__main__":
    main()
