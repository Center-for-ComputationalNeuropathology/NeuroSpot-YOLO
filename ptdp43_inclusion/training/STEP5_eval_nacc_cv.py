#!/usr/bin/env python3
"""
STEP5_eval_nacc_cv.py - Pooled leave-one-NACC-case-out evaluation (playbook
step 6: with only 5 patients, report the aggregate, not one noisy fold).

For each held-out NACC case, runs that fold's model on the case's ROI tiles
(data/dataset_nacc_cv/fold_<CASE>/valid -- the only exhaustively annotated GT),
matches predictions to GT at IoU>=0.5, then pools every fold's detections into
one AP50 / P / R / F1 curve.

Scored side by side on the IDENTICAL tiles:
  nacc_only  : runs/TDP43_nacc_cv_<CASE>_*/weights/best.pt   (this experiment)
  all_data   : runs/TDP43_cv_<CASE>_*/weights/best.pt        (Sep-16 CV: trained
               with MA24/NPBB cases too, on Reinhard stain-normalised tiles --
               so its inputs get the same normalisation it was trained with)

Usage: python STEP5_eval_nacc_cv.py
Writes runs/nacc_cv_pooled_eval.csv and prints the summary table.
"""
import csv
import glob
import sys
from pathlib import Path

import json
import numpy as np
from PIL import Image
from ultralytics import YOLO

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "03_build_dataset"))
from STEP3_build_dataset import stain_normalize  # noqa: E402

CASES = ["NACC036233", "NACC054803", "NACC068046", "NACC782796", "NACC928383"]
IMGSZ = 960
CONF_FLOOR = 0.01       # keep low-conf preds so the PR curve is complete
IOU_MATCH = 0.5
STAIN_REF = json.load(open(BASE / "data" / "dataset" / "stain_reference.json"))

MODEL_SETS = {
    "nacc_only": ("runs/TDP43_nacc_cv_{case}_*/weights/best.pt", False),
    # + hard-negative round 1 (02_mine_negatives/STEP2d_build_hardneg_dataset.py)
    "nacc_hn1": ("runs/TDP43_nacc_hn1_cv_{case}_*/weights/best.pt", False),
    # gross-artifact negatives only + reviewed cohort positives (STEP2e_build_hn2_dataset.py)
    "nacc_hn2": ("runs/TDP43_nacc_hn2_cv_{case}_*/weights/best.pt", False),
    "all_data": ("runs/TDP43_cv_{case}_*/weights/best.pt", True),
}


def latest(pattern):
    hits = sorted(glob.glob(str(BASE / pattern)))
    return hits[-1] if hits else None


def load_gt(label_path):
    out = []
    for line in open(label_path):
        p = line.split()
        if len(p) == 5:
            _, xc, yc, w, h = map(float, p)
            out.append([(xc - w / 2) * 640, (yc - h / 2) * 640, (xc + w / 2) * 640, (yc + h / 2) * 640])
    return np.array(out).reshape(-1, 4)


def iou(a, B):
    ix1, iy1 = np.maximum(a[0], B[:, 0]), np.maximum(a[1], B[:, 1])
    ix2, iy2 = np.minimum(a[2], B[:, 2]), np.minimum(a[3], B[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - inter
    return inter / np.maximum(ua, 1e-9)


def score_fold(model, valid_dir, stain_norm):
    """Returns (list of (conf, is_tp), n_gt)."""
    dets, n_gt = [], 0
    for img_path in sorted((valid_dir / "images").glob("*.jpg")):
        gt = load_gt(valid_dir / "labels" / f"{img_path.stem}.txt")
        n_gt += len(gt)
        im = np.array(Image.open(img_path).convert("RGB"))
        if stain_norm:
            im = stain_normalize(im, STAIN_REF["mean"], STAIN_REF["std"])
        r = model.predict(im[:, :, ::-1], imgsz=IMGSZ, conf=CONF_FLOOR, verbose=False, half=True)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        used = np.zeros(len(gt), bool)
        for i in np.argsort(-confs):
            tp = False
            if len(gt):
                ious = iou(boxes[i], gt)
                ious[used] = 0
                j = int(ious.argmax())
                if ious[j] >= IOU_MATCH:
                    used[j] = tp = True
            dets.append((float(confs[i]), tp))
    return dets, n_gt


def summarize(dets, n_gt):
    if not dets or n_gt == 0:
        return dict(AP50=float("nan"), P=float("nan"), R=float("nan"), F1=float("nan"), conf=float("nan"))
    dets = sorted(dets, key=lambda d: -d[0])
    tp = np.cumsum([d[1] for d in dets])
    fp = np.cumsum([not d[1] for d in dets])
    rec, prec = tp / n_gt, tp / (tp + fp)
    # all-point interpolated AP (same definition ultralytics uses)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.maximum.accumulate(np.concatenate([[1], prec, [0]])[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    k = int(f1.argmax())
    return dict(AP50=ap, P=float(prec[k]), R=float(rec[k]), F1=float(f1[k]), conf=dets[k][0])


def main():
    rows = []
    for set_name, (pattern, stain_norm) in MODEL_SETS.items():
        pooled, pooled_gt = [], 0
        for case in CASES:
            ckpt = latest(pattern.format(case=case))
            if ckpt is None:
                print(f"[{set_name}] no checkpoint for {case} -- skipped")
                continue
            dets, n_gt = score_fold(YOLO(ckpt), BASE / "data" / "dataset_nacc_cv" / f"fold_{case}" / "valid", stain_norm)
            pooled += dets
            pooled_gt += n_gt
            s = summarize(dets, n_gt)
            rows.append(dict(model_set=set_name, case=case, n_gt=n_gt, checkpoint=ckpt, **s))
        rows.append(dict(model_set=set_name, case="POOLED", n_gt=pooled_gt, checkpoint="",
                         **summarize(pooled, pooled_gt)))

    out = BASE / "runs" / "nacc_cv_pooled_eval.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'model_set':10s} {'case':12s} {'nGT':>4s} {'AP50':>6s} {'P':>6s} {'R':>6s} {'F1':>6s} {'@conf':>6s}")
    for r in rows:
        print(f"{r['model_set']:10s} {r['case']:12s} {r['n_gt']:4d} {r['AP50']:6.3f} {r['P']:6.3f} "
              f"{r['R']:6.3f} {r['F1']:6.3f} {r['conf']:6.3f}")
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
