#!/usr/bin/env python3
"""
STEP2e_build_hn2_dataset.py - Hard-negative round 2 dataset (2026-09-27):
data/dataset_nacc_cv_hn2 = dataset_nacc_cv
                         + the 148 gross-artifact windows from hn1 (37 tiles x 4)
                         + the round-1 detections the user labelled REAL, as new positives.

Why (blind whole-slide review of v1 vs hn1 on 10 fresh slides,
runs/wsi_precision_eval_round1.txt): hn1 raised precision (16% -> ~50%) but
overcorrected -- it found only ~50% of v1's real inclusions even at conf 0.10.
hn1's 112 negatives around single "not" detections were mostly subtle
look-alikes (faint cytoplasmic staining, neurite bits, debris) close to faint
real inclusions; they are dropped here. The gross artifacts (hair, folds,
stained edges, pale clouds) stay. The 50 reviewed real inclusions add the first
positives from whole cohort slides outside annotated ROIs.

Positive windows (640px native, 2 placements per real inclusion): the reviewed
box stays >= 96px inside the window; unreviewed detections (any saved,
conf >= 0.10), which may be unlabelled real inclusions, are kept out of the
window where possible, and the placement is dropped if one >= 0.25 can't be
avoided.
Reviewed "not" detections inside a window are fine as background; any other
reviewed "real" box inside it is labelled too.

The 10 slides of the blind evaluation (NACC_test/tdp43_eval_round1) are not
used in any way. Folds: extra tiles go to every fold's TRAIN; validation is
unchanged, so CV stays comparable with v1/hn1.
"""
import csv
import json
import os
import random
from pathlib import Path

import openslide

import STEP2d_build_hardneg_dataset as hn1

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "data" / "dataset_nacc_cv"
HN1_NEG = BASE / "data" / "dataset_nacc_cv_hn1" / "hardneg"
OUT = BASE / "data" / "dataset_nacc_cv_hn2"
LABELS = BASE / "data" / "hardneg_round1" / "labels_round1.json"
PATCH = hn1.PATCH
PLACEMENTS_PER_POSITIVE = 2
# stricter than hn1's negatives: in a POSITIVE window every unboxed object is taught as
# background, so avoid any saved detection (>= 0.10) and drop the placement if one
# >= 0.25 can't be avoided
POS_NEIGHBOUR_CONF = 0.10
POS_DROP_CONF = 0.25
MIN_BOX_VISIBLE_FRAC = 0.20
SEED = 1


def yolo_line(box, x, y):
    x1, y1, x2, y2 = box
    c = [max(x1, x), max(y1, y), min(x2, x + PATCH), min(y2, y + PATCH)]
    if c[2] <= c[0] or c[3] <= c[1]:
        return None
    if (c[2] - c[0]) * (c[3] - c[1]) < MIN_BOX_VISIBLE_FRAC * (x2 - x1) * (y2 - y1):
        return None
    return (f"0 {((c[0] + c[2]) / 2 - x) / PATCH:.6f} {((c[1] + c[3]) / 2 - y) / PATCH:.6f} "
            f"{(c[2] - c[0]) / PATCH:.6f} {(c[3] - c[1]) / PATCH:.6f}")


def main():
    if OUT.exists():
        raise SystemExit(f"{OUT} exists -- move it aside first")
    rng = random.Random(SEED)
    paths = {os.path.basename(r["PATH"]).rsplit(".", 1)[0]: r["PATH"]
             for r in csv.DictReader(open(hn1.SLIDE_LIST), delimiter="\t") if r["PATHOLOGY"] == "pTDP-43"}
    labels = [it for it in json.load(open(LABELS)) if it["kind"] == "det"]
    dets = hn1.load_mining_detections()

    # reviewed detections in slide coords
    reviewed = {}
    for it in labels:
        tx, ty = hn1.tile_origin(it["tile"])
        x1, y1, x2, y2 = it["box"]
        reviewed.setdefault(it["slide"], []).append(
            dict(it=it, box=(tx + x1, ty + y1, tx + x2, ty + y2), cx=tx + (x1 + x2) / 2, cy=ty + (y1 + y2) / 2))

    extra = OUT / "extra"
    (extra / "images").mkdir(parents=True)
    (extra / "labels").mkdir(parents=True)
    rows = []

    # 1) gross-artifact negatives, reused from hn1 (tile-kind windows only)
    for f in sorted((HN1_NEG / "images").glob("*.jpg")):
        if "__t0" not in f.name:     # hn1 tile windows are named HN1__<naccid>__t0NN_k
            continue
        (extra / "images" / f.name).symlink_to(f.resolve())
        (extra / "labels" / f"{f.stem}.txt").write_text("")
        rows.append(dict(name=f.stem, source="artifact_tile", review_id=f.stem.split("__")[2].rsplit("_", 1)[0],
                         slide="", x="", y="", n_boxes=0))
    n_neg = len(rows)

    # 2) reviewed-real positives
    hn1.NEIGHBOUR_CONF, hn1.DROP_CONF = POS_NEIGHBOUR_CONF, POS_DROP_CONF
    slides, n_dropped = {}, 0
    for r in [r for s in reviewed.values() for r in s if r["it"]["label"] == "real"]:
        it = r["it"]
        others = reviewed[it["slide"]]
        # unreviewed detections near it: anything not within 30px of a reviewed item
        nbrs = [(x, y, c) for x, y, c in hn1.neighbours(dets.get(it["slide"], []), r["cx"], r["cy"])
                if all((x - o["cx"]) ** 2 + (y - o["cy"]) ** 2 > hn1.SAME_OBJECT_PX ** 2 for o in others)]
        wsi = slides.setdefault(it["slide"], openslide.OpenSlide(paths[it["slide"]]))
        W, H = wsi.dimensions
        used = set()
        for k in range(PLACEMENTS_PER_POSITIVE):
            win = hn1.place_fp_window(r["cx"], r["cy"], nbrs, rng)
            if win is None:
                n_dropped += 1
                break
            x, y = min(max(win[0], 0), W - PATCH), min(max(win[1], 0), H - PATCH)
            if (x, y) in used:
                continue
            used.add((x, y))
            lines = [ln for o in others if o["it"]["label"] == "real"
                     for ln in [yolo_line(o["box"], x, y)] if ln]
            name = f"HN2POS__{it['naccid']}__{it['id']}_{k}"
            wsi.read_region((x, y), 0, (PATCH, PATCH)).convert("RGB").save(
                extra / "images" / f"{name}.jpg", quality=95)
            (extra / "labels" / f"{name}.txt").write_text("\n".join(lines) + "\n")
            rows.append(dict(name=name, source="reviewed_real", review_id=it["id"], slide=it["slide"],
                             x=x, y=y, n_boxes=len(lines)))

    with open(OUT / "extra_tiles.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

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
                    for f in (extra / sub).iterdir():
                        (dst / f.name).symlink_to(f.resolve())
        (OUT / d.name / "dataset.yaml").write_text(
            "# + hn2 extras (STEP2e) in train\n" + (d / "dataset.yaml").read_text().replace(str(SRC), str(OUT)))

    n_pos = len(rows) - n_neg
    print(f"{n_neg} artifact negatives + {n_pos} reviewed-real positive windows "
          f"({sum(r['n_boxes'] for r in rows)} boxes; {n_dropped} placements dropped) -> {OUT}")


if __name__ == "__main__":
    main()
