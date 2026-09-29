#!/usr/bin/env python3
"""
fig_augmentation_examples.py - One annotated TDP-43 training tile shown under
each augmentation used to train the NACC models, at the SAME settings as
04_train/STEP4_train_nacc_cv.py / STEP4_train_nacc_final.py:

  fliplr 0.5 (ultralytics default)   degrees 10     scale 0.5    translate 0.15
  hsv_h 0.03   hsv_s 0.9   hsv_v 0.6  mosaic 1.0     mixup 0.15
  (flipud 0 -> vertical flips are NOT used)

Boxes are transformed with the image, as the trainer does (rotations/scaling
take the axis-aligned box around the moved corners; boxes pushed out of frame
are dropped). Each panel shows one augmentation at a representative extreme of
its range so the effect is visible; in training they are sampled randomly and
combined, and mosaic is applied to every batch except the last 30 epochs.

Output: Training/TDP-43/augmentation_examples.png (+ .pdf)
"""
import csv
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path(__file__).resolve().parents[1]
POOL = BASE / "data" / "dataset_nacc_cv" / "pool"
MAIN = "NACC036233_11_pTDP-43_164617__roi__29406_33030"
PARTNERS = ["NACC782796", "NACC928383", "NACC068046"]   # mosaic / mixup partners: other patients
S = 640
PAD = 114                                              # ultralytics letterbox grey
BOX = "#00e5ff"


def load(name):
    im = cv2.cvtColor(cv2.imread(str(POOL / "images" / f"{name}.jpg")), cv2.COLOR_BGR2RGB)
    boxes = []
    for line in open(POOL / "labels" / f"{name}.txt"):
        p = line.split()
        if len(p) == 5:
            _, xc, yc, w, h = map(float, p)
            boxes.append([(xc - w / 2) * S, (yc - h / 2) * S, (xc + w / 2) * S, (yc + h / 2) * S])
    return im, np.array(boxes).reshape(-1, 4)


def pick_partner(case):
    rows = [r for r in csv.DictReader(open(BASE / "data" / "dataset_nacc_cv" / "tiles.csv"))
            if r["case"] == case and r["source"] == "roi" and 3 <= int(r["n_boxes"]) <= 8]
    return rows[len(rows) // 2]["name"]


def affine(im, boxes, M):
    """Warp image and boxes by a 2x3 affine; boxes -> AABB of moved corners, clipped, tiny ones dropped."""
    out = cv2.warpAffine(im, M, (S, S), borderValue=(PAD, PAD, PAD))
    if not len(boxes):
        return out, boxes
    corners = np.concatenate([boxes[:, [0, 1]], boxes[:, [2, 1]], boxes[:, [0, 3]], boxes[:, [2, 3]]])
    corners = np.hstack([corners, np.ones((len(corners), 1))]) @ M.T
    n = len(boxes)
    xs, ys = corners[:, 0].reshape(4, n), corners[:, 1].reshape(4, n)
    new = np.stack([xs.min(0), ys.min(0), xs.max(0), ys.max(0)], 1)
    area0 = (new[:, 2] - new[:, 0]) * (new[:, 3] - new[:, 1])
    new = np.clip(new, 0, S)
    area1 = (new[:, 2] - new[:, 0]) * (new[:, 3] - new[:, 1])
    return out, new[area1 > 0.1 * area0]


def rotate(im, boxes, deg):
    return affine(im, boxes, cv2.getRotationMatrix2D((S / 2, S / 2), deg, 1.0))


def scale(im, boxes, f):
    return affine(im, boxes, cv2.getRotationMatrix2D((S / 2, S / 2), 0, f))


def translate(im, boxes, fx, fy):
    return affine(im, boxes, np.float32([[1, 0, fx * S], [0, 1, fy * S]]))


def hsv(im, gh, gs, gv):
    """ultralytics RandomHSV with fixed gains (1 +/- hsv_*)."""
    h, s, v = cv2.split(cv2.cvtColor(im, cv2.COLOR_RGB2HSV))
    x = np.arange(256)
    lut_h = ((x * gh) % 180).astype(np.uint8)
    lut_s = np.clip(x * gs, 0, 255).astype(np.uint8)
    lut_v = np.clip(x * gv, 0, 255).astype(np.uint8)
    out = cv2.merge((cv2.LUT(h, lut_h), cv2.LUT(s, lut_s), cv2.LUT(v, lut_v)))
    return cv2.cvtColor(out, cv2.COLOR_HSV2RGB)


def mosaic(tiles):
    """4 tiles on a 2S canvas around a random centre, then cropped back to S (as in ultralytics Mosaic + affine)."""
    canvas = np.full((2 * S, 2 * S, 3), PAD, np.uint8)
    allb = []
    for (im, b), (ox, oy) in zip(tiles, [(0, 0), (S, 0), (0, S), (S, S)]):
        canvas[oy:oy + S, ox:ox + S] = im
        allb.append(b + [ox, oy, ox, oy])
    allb = np.concatenate(allb)
    cx, cy = int(S * 1.0), int(S * 1.0)          # centre crop keeps a quarter of each tile visible
    x0, y0 = cx - S // 2, cy - S // 2
    return affine(canvas, allb, np.float32([[1, 0, -x0], [0, 1, -y0]]))


def mixup(a, b, r=0.5):
    return (a[0] * r + b[0] * (1 - r)).astype(np.uint8), np.concatenate([a[1], b[1]])


def main():
    im, b = load(MAIN)
    partners = [load(pick_partner(c)) for c in PARTNERS]
    panels = [
        ("Original", "annotated ROI tile, 640 px at 40x", (im, b)),
        ("Horizontal flip", "fliplr = 0.5", (im[:, ::-1].copy(), np.c_[S - b[:, 2], b[:, 1], S - b[:, 0], b[:, 3]])),
        ("Rotate +10°", "degrees = 10", rotate(im, b, 10)),
        ("Rotate −10°", "degrees = 10", rotate(im, b, -10)),
        ("Zoom out ×0.5", "scale = 0.5", scale(im, b, 0.5)),
        ("Zoom in ×1.5", "scale = 0.5", scale(im, b, 1.5)),
        ("Translate", "translate = 0.15", translate(im, b, 0.15, -0.15)),
        ("Hue ×0.97 / ×1.03 (subtle)", "hsv_h = 0.03", None),
        ("Saturation ×0.1 / ×1.9", "hsv_s = 0.9", None),
        ("Brightness ×0.4 / ×1.6", "hsv_v = 0.6", None),
        ("Mosaic", "mosaic = 1.0 (4 tiles, 4 patients)", mosaic([(im, b)] + partners)),
        ("MixUp", "mixup = 0.15 (blend with another patient)", mixup((im, b), partners[0])),
    ]
    # split panels: left half one extreme, right half the other
    lo, hi = hsv(im, 1.0, 0.1, 1.0), hsv(im, 1.0, 1.9, 1.0)
    panels[8] = (panels[8][0], panels[8][1], (np.concatenate([lo[:, :S // 2], hi[:, S // 2:]], 1), b))
    lo, hi = hsv(im, 0.97, 1.0, 1.0), hsv(im, 1.03, 1.0, 1.0)
    panels[7] = (panels[7][0], panels[7][1], (np.concatenate([lo[:, :S // 2], hi[:, S // 2:]], 1), b))
    lo, hi = hsv(im, 1.0, 1.0, 0.4), hsv(im, 1.0, 1.0, 1.6)
    panels[9] = (panels[9][0], panels[9][1], (np.concatenate([lo[:, :S // 2], hi[:, S // 2:]], 1), b))

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 4, figsize=(16, 13.2))
    for ax, (title, param, (img, boxes)) in zip(axes.flat, panels):
        ax.imshow(img)
        for x1, y1, x2, y2 in boxes:
            ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=BOX, lw=1.8))
        if title.startswith(("Saturation", "Brightness", "Hue")):
            ax.axvline(S / 2, color="white", lw=1.5, ls="--")
        ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=16)
        ax.text(0, 1.012, param, transform=ax.transAxes, fontsize=10, color="#555", family="DejaVu Sans Mono")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#ccc")
    fig.suptitle("Data augmentation for pTDP-43 inclusion detection (settings used in training)",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=0.995)
    fig.text(0.01, 0.962, f"Example tile: {MAIN.replace('__roi__', '  ROI @ ')}   ·   cyan boxes = annotated "
             "inclusions, transformed with the image\nEach panel shows one augmentation at an extreme of its range "
             "(split panels: left = low end, right = high end); in training they are sampled at random and combined.",
             fontsize=9.5, color="#444", va="top", linespacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = BASE / "augmentation_examples"
    fig.savefig(f"{out}.png", dpi=160)
    fig.savefig(f"{out}.pdf")
    print(f"written {out}.png / .pdf")


if __name__ == "__main__":
    main()
