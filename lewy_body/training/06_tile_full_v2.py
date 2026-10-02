#!/usr/bin/env python3
"""
Full-dataset (all 14 a-syn slides) tiling, v2 -- rebuilt after auditing
against YOLO_TRAINING_PLAYBOOK.md. Fixes vs. the original 02_tile_and_label.py
run:

  - ROI_to_test excluded from sampling (fixed upstream in
    01_select_annotations.py -- see that file's comment: it has zero Lewy
    body boxes in every NACC slide, so it's an unreviewed/held-out region,
    not a valid negative source).
  - Split is by CASE, not slide file. The original split put
    NACC036233 and NACC054803 blocks in BOTH train and val (leakage).
    Val cases here: NACC075420 (1 case, 101 boxes) + NPBB51 (1 case, 45
    boxes) -- keeps both sources represented in val while leaving the
    majority of data (6 of 8 cases, 1454 of 1600 boxes) in train.
  - Positive tiles capped per slide (300) so the densest slides
    (NACC054803_11: 415 boxes, NPBB0.24_AS_B8: 188 boxes) don't dominate
    via near-duplicate overlapping crops.
  - Negative:positive tile ratio 1.0 (playbook default), down from 2.0.
"""

import glob
import json
import os
import random

import numpy as np
import openslide
from PIL import Image

DATASET_DIR = os.environ.get("LEWY_DATASET_DIR", "dataset")  # output of 01_select_annotations.py
DESC_DIR = os.path.join(DATASET_DIR, "slide_annotations")

OUT_DIR = os.environ.get("LEWY_OUT_DIR", "dataset_v2")
IMAGES_DIR = os.path.join(OUT_DIR, "images")
LABELS_DIR = os.path.join(OUT_DIR, "labels")
REPORT = os.path.join(OUT_DIR, "tiling_report.tsv")

TILE = 640
GRID_STRIDE = 480
MIN_BOX_COVERAGE = 0.30
BG_WHITE_THRESH = 235
BG_MAX_FRACTION = 0.92
NEG_PER_POS_RATIO = 1.0
MIN_NEG_TILES = 15
MAX_NEG_TILES = 400
MAX_POS_TILES_PER_SLIDE = 300

random.seed(0)

# Case-level split: both sources represented in val, no case appears in
# both splits. 6 of 8 cases / 1454 of 1600 boxes stay in train.
VAL_CASES = {"NACC075420", "NPBB51"}


def is_background(tile_rgb):
    gray = np.asarray(tile_rgb.convert("L"))
    return (gray >= BG_WHITE_THRESH).mean() >= BG_MAX_FRACTION


def clip_box(box, ox, oy):
    x1, y1, x2, y2 = box
    x1c, y1c = max(x1, ox), max(y1, oy)
    x2c, y2c = min(x2, ox + TILE), min(y2, oy + TILE)
    if x2c <= x1c or y2c <= y1c:
        return None
    orig_area = max(1.0, (x2 - x1) * (y2 - y1))
    clipped_area = (x2c - x1c) * (y2c - y1c)
    if clipped_area / orig_area < MIN_BOX_COVERAGE:
        return None
    return x1c - ox, y1c - oy, x2c - ox, y2c - oy


def to_yolo_line(box_local):
    x1, y1, x2, y2 = box_local
    cx = (x1 + x2) / 2.0 / TILE
    cy = (y1 + y2) / 2.0 / TILE
    w = (x2 - x1) / TILE
    h = (y2 - y1) / TILE
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_candidate_origins(desc, slide_w, slide_h):
    origins = set()

    for (x1, y1, x2, y2) in desc["positives"]:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ox = int(round(cx - TILE / 2))
        oy = int(round(cy - TILE / 2))
        origins.add((ox, oy))

    for (x1, y1, x2, y2) in desc["roi_rects"]:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(slide_w, int(x2)), min(slide_h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        gx = x1
        while gx < x2:
            gy = y1
            while gy < y2:
                origins.add((gx, gy))
                gy += GRID_STRIDE
            gx += GRID_STRIDE

    for (x1, y1, x2, y2) in desc["negative_tile_rects"]:
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        ox = int(round(cx - TILE / 2))
        oy = int(round(cy - TILE / 2))
        origins.add((ox, oy))

    clamped = set()
    for ox, oy in origins:
        ox = max(0, min(ox, slide_w - TILE))
        oy = max(0, min(oy, slide_h - TILE))
        clamped.add((ox, oy))
    return clamped


def process_slide(desc_path, images_dir, labels_dir):
    desc = json.load(open(desc_path))
    stem = desc["slide_stem"]
    slide = openslide.OpenSlide(desc["slide_path"])
    slide_w, slide_h = slide.dimensions

    origins = build_candidate_origins(desc, slide_w, slide_h)

    positive_tiles = []
    negative_tiles = []

    for (ox, oy) in sorted(origins):
        region = slide.read_region((ox, oy), 0, (TILE, TILE)).convert("RGB")
        region.info.pop("icc_profile", None)

        lines = []
        for box in desc["positives"]:
            clipped = clip_box(box, ox, oy)
            if clipped is not None:
                lines.append(to_yolo_line(clipped))

        if lines:
            positive_tiles.append((ox, oy, lines, region))
        else:
            if is_background(region):
                continue
            negative_tiles.append((ox, oy, region))

    if len(positive_tiles) > MAX_POS_TILES_PER_SLIDE:
        positive_tiles = random.sample(positive_tiles, MAX_POS_TILES_PER_SLIDE)

    n_pos = len(positive_tiles)
    cap = int(max(MIN_NEG_TILES, min(MAX_NEG_TILES, n_pos * NEG_PER_POS_RATIO)))
    if len(negative_tiles) > cap:
        negative_tiles = random.sample(negative_tiles, cap)

    for (ox, oy, lines, region) in positive_tiles:
        name = f"{stem}__x{ox}_y{oy}"
        region.save(os.path.join(images_dir, name + ".png"))
        with open(os.path.join(labels_dir, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    for (ox, oy, region) in negative_tiles:
        name = f"{stem}__x{ox}_y{oy}"
        region.save(os.path.join(images_dir, name + ".png"))
        open(os.path.join(labels_dir, name + ".txt"), "w").close()

    slide.close()
    return {
        "slide_stem": stem,
        "case_id": desc["case_id"],
        "n_candidate_origins": len(origins),
        "n_positive_tiles": len(positive_tiles),
        "n_negative_tiles_written": len(negative_tiles),
    }


def main():
    for split in ("train", "val"):
        os.makedirs(os.path.join(IMAGES_DIR, split), exist_ok=True)
        os.makedirs(os.path.join(LABELS_DIR, split), exist_ok=True)

    desc_paths = sorted(glob.glob(os.path.join(DESC_DIR, "*.json")))
    rows = []
    for desc_path in desc_paths:
        desc = json.load(open(desc_path))
        stem = desc["slide_stem"]
        split = "val" if desc["case_id"] in VAL_CASES else "train"
        print(f"[{split}] {stem} (case {desc['case_id']}) ...")
        stats = process_slide(
            desc_path,
            os.path.join(IMAGES_DIR, split),
            os.path.join(LABELS_DIR, split),
        )
        stats["split"] = split
        rows.append(stats)
        print(
            f"    origins={stats['n_candidate_origins']} "
            f"pos_tiles={stats['n_positive_tiles']} "
            f"neg_tiles={stats['n_negative_tiles_written']}"
        )

    with open(REPORT, "w") as f:
        cols = [
            "slide_stem",
            "case_id",
            "split",
            "n_candidate_origins",
            "n_positive_tiles",
            "n_negative_tiles_written",
        ]
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    total_pos = sum(r["n_positive_tiles"] for r in rows)
    total_neg = sum(r["n_negative_tiles_written"] for r in rows)
    print(f"\nTotal tiles: {total_pos + total_neg} "
          f"(positive={total_pos}, negative={total_neg})")
    print(f"Report: {REPORT}")


if __name__ == "__main__":
    main()
