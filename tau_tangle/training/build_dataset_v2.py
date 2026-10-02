#!/usr/bin/env python
"""Materialize the case_train_valid_plan_v2_20260920 patient-level split into a
YOLO dataset directory by symlinking existing images/labels into new
train/valid folders. No relabeling or filtering is applied.
"""
import argparse
import os
import csv
import sys
from pathlib import Path

TANGLES_DIR = Path(os.environ.get("TANGLES_DIR", "."))  # folder holding the original train/ valid/ test/ tile splits
OLD_SPLITS = ["train", "valid", "test"]


def find_source(filename):
    stem = Path(filename).stem
    for split in OLD_SPLITS:
        img = TANGLES_DIR / split / "images" / filename
        lbl = TANGLES_DIR / split / "labels" / f"{stem}.txt"
        if img.exists():
            return img, lbl
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=False)
    dest_dirs = {}
    for split in ["train", "valid"]:
        for kind in ["images", "labels"]:
            d = args.output / split / kind
            d.mkdir(parents=True)
            dest_dirs[(split, kind)] = d

    counts = {"train": 0, "valid": 0}
    missing_image = []
    missing_label = []

    with open(args.manifest, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            split = row["split"]
            if split not in ("train", "valid"):
                continue
            src_img, src_lbl = find_source(filename)
            if src_img is None:
                missing_image.append(filename)
                continue
            (dest_dirs[(split, "images")] / filename).symlink_to(src_img.resolve())
            if src_lbl.exists():
                (dest_dirs[(split, "labels")] / src_lbl.name).symlink_to(src_lbl.resolve())
            else:
                # Background/negative patch: empty label file is the YOLO convention.
                (dest_dirs[(split, "labels")] / f"{Path(filename).stem}.txt").touch()
                missing_label.append(filename)
            counts[split] += 1

    print(f"train images linked: {counts['train']}")
    print(f"valid images linked: {counts['valid']}")
    print(f"images with no source label file (written empty, expected for background patches): {len(missing_label)}")
    if missing_image:
        print(f"ERROR: {len(missing_image)} filenames had no source image found:", file=sys.stderr)
        for m in missing_image[:20]:
            print(f"  {m}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
