#!/usr/bin/env python3
"""
STEP2d_build_hardneg_dataset.py - Hard-negative round 1 (2026-09-26): turn the
user's review labels (data/hardneg_round1/labels_round1.json, from the review
page https://claude.ai/artifact/LqPRBi63ndihBvAbZVvhun) into background tiles,
and build data/dataset_nacc_cv_hn1 = dataset_nacc_cv + these negatives.

Review result that motivates this: on 25 random unannotated cohort slides only
30% of detections at conf>=0.25 were real inclusions (42% at >=0.5), vs 77%
precision in annotated ROI cortex; 37/40 dense tiles were artifacts (hair,
pale granular clouds, stained tissue edges, folds, debris).

Negatives (640px native, empty labels, same scale as every other train tile):
  hn_tile : each 1024px tile labelled "artifact" -> its 4 corner 640 windows
            (together they cover the whole tile)
  hn_det  : each detection labelled "not" -> a 640 window containing it, placed
            to avoid its unreviewed neighbouring detections (conf >= 0.25),
            which may be real inclusions; dropped if a neighbour >= 0.35
            can't be avoided
Not used: "real"/"mixed" tiles, "real"/"unsure" detections.

The negatives come from cohort patients that are not among the 5 annotated
cases, so they are added to the TRAIN split of every fold (and all/) with no
leakage; validation sets are unchanged, so CV numbers stay comparable to v1.
dataset_nacc_cv itself is left untouched (symlinks only).
"""
import csv
import json
import os
import random
from pathlib import Path

import openslide
from PIL import Image

BASE = Path(__file__).resolve().parents[1]
LABELS = BASE / "data" / "hardneg_round1" / "labels_round1.json"
SRC = BASE / "data" / "dataset_nacc_cv"
OUT = BASE / "data" / "dataset_nacc_cv_hn1"
SLIDE_LIST = ("/sc/arion/projects/comppath_500k/neuroFM_slides/CraryLab_Slides/Test/"
              "Collection_Brain_Bank_BU_ART-AD_Phase2/pathology_slide_paths_419cases.tsv")
MINING_PRED = Path("/sc/arion/projects/tauomics/YOLO_all_models_Neuropathology/NACC_test/"
                   "tdp43_hardneg_round1/test_predictions_positive")
PATCH = 640
COHORT_TILE = 1024
FP_MARGIN = 96         # reviewed FP centre kept at least this far inside the window
NEIGHBOUR_CONF = 0.25  # unreviewed detections at/above this are avoided when placing a window
DROP_CONF = 0.35       # ...and a window is dropped if one this confident can't be avoided
SAME_OBJECT_PX = 30    # a detection this close to the FP is the FP itself
SEED = 0


def tile_origin(tile_stem):
    """Cohort tile names end in _test_<y>_<x> (level-0 px)."""
    y, x = tile_stem.rsplit("_test_", 1)[1].split("_")
    return int(x), int(y)


def load_mining_detections():
    """Every mining-run detection, in slide level-0 coords: slide -> [(cx, cy, conf)]."""
    out = {}
    for f in MINING_PRED.glob("*/*/*.txt"):
        tx, ty = tile_origin(f.stem)
        for line in open(f):
            _, cx, cy, _, _, c = map(float, line.split())
            out.setdefault(f.parent.name, []).append((tx + cx * COHORT_TILE, ty + cy * COHORT_TILE, c))
    return out


def neighbours(slide_dets, cx, cy):
    """Other detections near a reviewed FP (itself excluded) that could be real inclusions."""
    return [(x, y, c) for x, y, c in slide_dets
            if c >= NEIGHBOUR_CONF and abs(x - cx) < PATCH and abs(y - cy) < PATCH
            and (x - cx) ** 2 + (y - cy) ** 2 > SAME_OBJECT_PX ** 2]


def place_fp_window(cx, cy, nbrs, rng):
    """A 640 window containing the reviewed FP (>= FP_MARGIN from the edge) and, if
    possible, none of its unreviewed neighbours -- a neighbour may be a real
    inclusion, which must not be taught as background. Returns None when every
    placement keeps a neighbour scoring >= DROP_CONF."""
    offs = list(range(FP_MARGIN, PATCH - FP_MARGIN + 1, 32))
    cands = [(int(cx - ox), int(cy - oy)) for ox in offs for oy in offs]
    rng.shuffle(cands)

    def worst(w):
        inside = [c for x, y, c in nbrs if w[0] <= x < w[0] + PATCH and w[1] <= y < w[1] + PATCH]
        return max(inside, default=0.0)

    best = min(cands, key=worst)
    return None if worst(best) >= DROP_CONF else best


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists -- move it aside first")
    rng = random.Random(SEED)
    paths = {os.path.basename(r["PATH"]).rsplit(".", 1)[0]: r["PATH"]
             for r in csv.DictReader(open(SLIDE_LIST), delimiter="\t") if r["PATHOLOGY"] == "pTDP-43"}
    labels = json.load(open(LABELS))

    neg_dir = OUT / "hardneg"
    (neg_dir / "images").mkdir(parents=True)
    (neg_dir / "labels").mkdir(parents=True)
    slides = {}
    dets = load_mining_detections()
    n_tile = n_det = n_dropped = 0
    rows = []
    for it in labels:
        use_tile = it["kind"] == "tile" and it["label"] == "artifact"
        use_det = it["kind"] == "det" and it["label"] == "not"
        if not (use_tile or use_det):
            continue
        wsi = slides.setdefault(it["slide"], openslide.OpenSlide(paths[it["slide"]]))
        W, H = wsi.dimensions
        tx, ty = tile_origin(it["tile"])
        if use_tile:
            off = COHORT_TILE - PATCH
            windows = [(tx, ty), (tx + off, ty), (tx, ty + off), (tx + off, ty + off)]
        else:
            x1, y1, x2, y2 = it["box"]
            cx, cy = tx + (x1 + x2) / 2, ty + (y1 + y2) / 2
            win = place_fp_window(cx, cy, neighbours(dets.get(it["slide"], []), cx, cy), rng)
            if win is None:
                n_dropped += 1
                continue
            windows = [win]
        for k, (x, y) in enumerate(windows):
            x, y = min(max(x, 0), W - PATCH), min(max(y, 0), H - PATCH)
            name = f"HN1__{it['naccid']}__{it['id']}_{k}"
            wsi.read_region((x, y), 0, (PATCH, PATCH)).convert("RGB").save(
                neg_dir / "images" / f"{name}.jpg", quality=95)
            (neg_dir / "labels" / f"{name}.txt").write_text("")
            rows.append(dict(name=name, review_id=it["id"], kind=it["kind"], slide=it["slide"], x=x, y=y))
        n_tile += use_tile
        n_det += use_det

    with open(OUT / "hardneg_tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # mirror every fold + all/, adding the negatives to train only
    for d in sorted(p for p in SRC.iterdir() if p.is_dir() and (p.name.startswith("fold_") or p.name == "all")):
        for split in ("train", "valid"):
            if not (d / split).exists():
                continue
            for sub in ("images", "labels"):
                dst = OUT / d.name / split / sub
                dst.mkdir(parents=True)
                for f in (d / split / sub).iterdir():
                    (dst / f.name).symlink_to(f.resolve())
                if split == "train":
                    for f in (neg_dir / sub).iterdir():
                        (dst / f.name).symlink_to(f.resolve())
        yaml = (d / "dataset.yaml").read_text().replace(str(SRC), str(OUT))
        (OUT / d.name / "dataset.yaml").write_text(
            "# + hard-negative round 1 (STEP2d) in train\n" + yaml)
    print(f"{n_tile} artifact tiles -> {4 * n_tile} windows, {n_det} FP detections -> {n_det} windows "
          f"({n_dropped} more dropped: a likely-real neighbour couldn't be avoided); "
          f"{len(rows)} negative tiles added to every fold's train -> {OUT}")


if __name__ == "__main__":
    main()
