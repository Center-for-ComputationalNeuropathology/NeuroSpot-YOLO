#!/usr/bin/env python3
"""
STEP3_build_nacc_cv.py - NACC-only dataset for leave-one-case-out CV, built per
YOLO_TRAINING_PLAYBOOK.md (2026-09-22). Separate from STEP3_build_dataset.py;
reuses its calibrated DAB / tissue helpers but none of its split logic.

Scope: the 5 annotated NACC cases (6 slides; NACC036233 has blocks 11 + 14).
Labels come from data/resolved_annotations/ (STEP1 output): MH + BOP unioned,
IoU-deduped, ghost_pTDP-43 kept as positive -- all as agreed with the user.

Audit facts this build is designed around (see playbook steps 1-3):
  - 100% of MH positives fall inside MH "ROI" boxes -> only ROI interiors are
    exhaustively annotated.
  - All 10 ROI_TEST boxes have zero GT boxes but visibly contain many unboxed
    inclusions (NACC928383 especially). ROI_TEST is NOT background: it is
    excluded from every split, and negatives are never mined from it.
  - BOP files (NACC036233_11, NACC054803) have no ROI, so their completeness
    is unverified -> their out-of-ROI positives are TRAIN-ONLY, and background
    near them is only taken via the calibrated DAB-clean test, never from
    "no box".

Tile sources (all 640px at native 40x, ~0.263 um/px; no resize, no stain norm
-- single-domain NACC data, keeps deployment preprocessing trivial):
  roi      : grid over each ROI, stride 512 (20% overlap), tiles fully inside
             the ROI. Positive if any box keeps >=20% of its area, else a
             TRUSTED background tile (DAB allowed -> real hard negatives).
             Used for train AND for validation of the held-out case.
  outroi   : 640 window around positive boxes outside any ROI (BOP), jittered.
             Train only.
  colorneg : random tissue tiles outside ROI/ROI_TEST/positives that pass the
             STEP2a-calibrated DAB-blob test. Train only.
Background budget per slide (train) = 1x that slide's positive tile count
(ROI background first, then colorneg). Positive tiles capped per slide.

Output: data/dataset_nacc_cv/
  pool/{images,labels}/          every tile, written once
  tiles.csv                      per-tile metadata (case, slide, source, n_boxes)
  fold_<CASE>/{train,valid}/...  symlinks; valid = held-out case's ROI tiles
  all/train/...                  every train-eligible tile (final model)
"""

import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import openslide
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from STEP3_build_dataset import (  # noqa: E402  calibrated helpers, reused as-is
    BLOB_MIN_AREA, any_overlap, calibrate_tissue_thresholds, is_tissue_patch,
    max_dab_blob_area, read_patch,
)

RESOLVED_DIR = REPO_ROOT / "data" / "resolved_annotations"
OUTPUT_DIR = REPO_ROOT / "data" / "dataset_nacc_cv"

PATCH = 640
ROI_STRIDE = 512                 # ~20% overlap (playbook step 5)
MIN_BOX_VISIBLE_FRAC = 0.20      # drop a clipped box below this (playbook step 5)
OUTROI_JITTER = 160              # px; random offset of window around an out-of-ROI box
MAX_POS_TILES_PER_SLIDE = 150    # cap so one dense slide can't dominate
BG_PER_POS = 1.0                 # background budget per slide = this x positive tiles
COLORNEG_MAX_ATTEMPTS = 6000
JPEG_QUALITY = 95
RNG_SEED = 0


def case_of(slide_name):
    return slide_name.split("_")[0]


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def tile_labels(x, y, boxes):
    """YOLO lines for boxes clipped to tile; drops boxes with <20% area visible."""
    lines = []
    for b in boxes:
        cx1, cy1 = max(b[0], x), max(b[1], y)
        cx2, cy2 = min(b[2], x + PATCH), min(b[3], y + PATCH)
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        if area((cx1, cy1, cx2, cy2)) < MIN_BOX_VISIBLE_FRAC * area(b):
            continue
        xc, yc = ((cx1 + cx2) / 2 - x) / PATCH, ((cy1 + cy2) / 2 - y) / PATCH
        w, h = (cx2 - cx1) / PATCH, (cy2 - cy1) / PATCH
        lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    return lines


def grid(lo, hi):
    """Tile origins covering [lo, hi) with PATCH windows fully inside; last one edge-aligned."""
    lo, hi = int(lo), int(hi)
    if hi - lo < PATCH:
        return []
    v = list(range(lo, hi - PATCH + 1, ROI_STRIDE))
    if v[-1] != hi - PATCH:
        v.append(hi - PATCH)
    return v


def inside(b, r):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return r[0] <= cx <= r[2] and r[1] <= cy <= r[3]


def roi_tiles(rec, boxes):
    out = []
    for r in rec["roi_boxes"]:
        for y in grid(r[1], r[3]):
            for x in grid(r[0], r[2]):
                out.append((x, y, tile_labels(x, y, boxes)))
    return out


def outroi_tiles(rec, boxes, wsi, rng):
    """Cover every out-of-ROI positive with at least one window containing it."""
    W, H = wsi.dimensions
    todo = [b for b in boxes if not any(inside(b, r) for r in rec["roi_boxes"])]
    rng.shuffle(todo)
    covered, out = set(), []
    for i, b in enumerate(todo):
        if i in covered:
            continue
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x = int(min(max(cx - PATCH / 2 + rng.randint(-OUTROI_JITTER, OUTROI_JITTER), 0), W - PATCH))
        y = int(min(max(cy - PATCH / 2 + rng.randint(-OUTROI_JITTER, OUTROI_JITTER), 0), H - PATCH))
        # a window that pokes into ROI_TEST would carry its unboxed positives
        if any_overlap([x, y, x + PATCH, y + PATCH], rec["roi_test_boxes"]):
            continue
        for j, o in enumerate(todo):
            if o[0] >= x and o[1] >= y and o[2] <= x + PATCH and o[3] <= y + PATCH:
                covered.add(j)
        out.append((x, y, tile_labels(x, y, boxes)))
    return out


def colorneg_tiles(rec, boxes, wsi, rng, target):
    if target <= 0:
        return []
    W, H = wsi.dimensions
    bmax, smin = calibrate_tissue_thresholds(rec, wsi, rng)
    # keep a margin around positives so a partially visible inclusion can't sneak in
    avoid = rec["roi_boxes"] + rec["roi_test_boxes"] + [
        [b[0] - PATCH / 2, b[1] - PATCH / 2, b[2] + PATCH / 2, b[3] + PATCH / 2] for b in boxes]
    out, seen = [], set()
    for _ in range(COLORNEG_MAX_ATTEMPTS):
        if len(out) >= target:
            break
        x = rng.randint(0, (W - PATCH) // PATCH) * PATCH
        y = rng.randint(0, (H - PATCH) // PATCH) * PATCH
        if (x, y) in seen:
            continue
        seen.add((x, y))
        if any_overlap([x, y, x + PATCH, y + PATCH], avoid):
            continue
        patch = read_patch(wsi, x, y, size=PATCH)
        if patch is None or not is_tissue_patch(patch, bmax, smin):
            continue
        if max_dab_blob_area(patch) >= BLOB_MIN_AREA:
            continue
        out.append((x, y, [], patch))
    return out


def save(pool, name, patch, lines):
    Image.fromarray(patch).save(pool / "images" / f"{name}.jpg", quality=JPEG_QUALITY)
    (pool / "labels" / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def link_split(dst, rows):
    for sub in ("images", "labels"):
        (dst / sub).mkdir(parents=True, exist_ok=True)
    for row in rows:
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            src = OUTPUT_DIR / "pool" / sub / f"{row['name']}{ext}"
            (dst / sub / f"{row['name']}{ext}").symlink_to(src)


def write_yaml(d, note, val_rel="valid/images"):
    (d / "dataset.yaml").write_text(
        f"# {note}\npath: {d}\ntrain: train/images\nval: {val_rel}\nnc: 1\nnames:\n  0: TDP-43\n")


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        sys.exit(f"{OUTPUT_DIR} already exists and is not empty -- move it aside first.")
    rng = random.Random(RNG_SEED)
    pool = OUTPUT_DIR / "pool"
    for sub in ("images", "labels"):
        (pool / sub).mkdir(parents=True, exist_ok=True)

    rows = []
    for f in sorted(RESOLVED_DIR.glob("NACC*.json")):
        rec = json.load(open(f))
        slide = Path(rec["slide_name"]).stem
        case = case_of(slide)
        boxes = [p["box"] for p in rec["positive_boxes"]]
        wsi = openslide.OpenSlide(rec["image_path"])

        roi = roi_tiles(rec, boxes)
        roi_pos = [t for t in roi if t[2]]
        roi_bg = [t for t in roi if not t[2]]
        out_pos = outroi_tiles(rec, boxes, wsi, rng)
        pos = roi_pos + out_pos
        if len(pos) > MAX_POS_TILES_PER_SLIDE:
            # cap out-of-ROI (train-only) tiles first; ROI tiles are also the val set
            keep_out = max(0, MAX_POS_TILES_PER_SLIDE - len(roi_pos))
            out_pos = rng.sample(out_pos, keep_out)
            pos = roi_pos + out_pos
        bg_target = int(round(BG_PER_POS * len(pos)))
        cneg = colorneg_tiles(rec, boxes, wsi, rng, bg_target - len(roi_bg))

        def emit(src, x, y, lines, patch=None):
            patch = patch if patch is not None else read_patch(wsi, x, y, size=PATCH)
            name = f"{slide}__{src}__{int(x)}_{int(y)}"
            save(pool, name, patch, lines)
            rows.append(dict(name=name, case=case, slide=slide, source=src,
                             n_boxes=len(lines), x=int(x), y=int(y)))

        for x, y, lines in roi:
            emit("roi", x, y, lines)
        for x, y, lines in out_pos:
            emit("outroi", x, y, lines)
        for x, y, lines, patch in cneg:
            emit("colorneg", x, y, lines, patch)
        print(f"{slide:32s} roi_pos={len(roi_pos):3d} roi_bg={len(roi_bg):3d} "
              f"outroi_pos={len(out_pos):3d} colorneg={len(cneg):3d} (bg target {bg_target})", flush=True)

    with open(OUTPUT_DIR / "tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    cases = sorted({r["case"] for r in rows})
    for held in cases:
        d = OUTPUT_DIR / f"fold_{held}"
        link_split(d / "train", [r for r in rows if r["case"] != held])
        # val = held-out case's ROI tiles only: the only exhaustively annotated GT
        link_split(d / "valid", [r for r in rows if r["case"] == held and r["source"] == "roi"])
        write_yaml(d, f"NACC-only LOCO fold, held-out case {held}; valid = its ROI tiles only")
    d = OUTPUT_DIR / "all"
    link_split(d / "train", rows)
    write_yaml(d, "NACC-only final model: all 5 cases in train; train with val=False, fixed epochs",
               val_rel="train/images")

    print("\nPER-FOLD SUMMARY (tiles / positive tiles / boxes)")
    for held in cases:
        tr = [r for r in rows if r["case"] != held]
        va = [r for r in rows if r["case"] == held and r["source"] == "roi"]
        s = lambda rs: f"{len(rs):4d} / {sum(r['n_boxes'] > 0 for r in rs):4d} / {sum(r['n_boxes'] for r in rs):5d}"
        print(f"  fold_{held:12s} train {s(tr)}   valid {s(va)}")
    print(f"\nWritten to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
