#!/usr/bin/env python3
"""Build the NACC-only training set used for the released model: the tiles of NACC slides
from the full dataset written by build_dataset.py, keeping its train/val split (symlinks)."""
import argparse
import os
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="dataset", help="output folder of build_dataset.py")
ap.add_argument("--out", default="dataset_nacc_only")
a = ap.parse_args()
src, out = Path(a.src).resolve(), Path(a.out)
for split in ("train", "val"):
    for kind, ext in (("images", ".jpg"), ("labels", ".txt")):
        (out / kind / split).mkdir(parents=True, exist_ok=True)
    for img in sorted((src / "images" / split).glob("NACC*.jpg")):
        for kind, p in (("images", img), ("labels", src / "labels" / split / (img.stem + ".txt"))):
            dst = out / kind / split / p.name
            if p.exists() and not dst.exists():
                os.symlink(p, dst)
(out / "data.yaml").write_text(f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\n"
                               "names: ['Amyloid_Plaque']\n")
print(f"written {out}")
