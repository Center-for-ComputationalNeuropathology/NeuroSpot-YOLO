#!/usr/bin/env python3
"""
Train a YOLO detector for amyloid-beta plaques on the tiled dataset produced
by build_dataset.py.

Notes on hyperparameters chosen for this task:
 - model=yolo11x.pt, imgsz=640 to match this lab's established convention
   for the prior amyloid plaque model (models/amyloid_plaques_best.pt,
   trained yolo11x/imgsz=640 on a separate older dataset). Tiles are
   extracted at 1024x1024 (see build_dataset.py); training at imgsz=640
   downsamples them, which shrinks the smaller plaques (median ~130x126px
   at native tile resolution, p95 ~410x395px) -- a tradeoff accepted here
   for consistency with the existing pipeline rather than optimality.
 - hsv_h raised well above the ultralytics default (0.015 -> 0.10) because
   the dataset mixes two IHC chromogens (bright red Fast-Red on the
   Hippocampus_beta_amyloid "4xxx" slides vs faint brown DAB on the
   NACC/NPBB slides). The model needs to key off plaque shape/texture
   rather than absolute hue to generalize across both stains.
 - close_mosaic keeps the last few epochs mosaic-free so the model sees
   full untiled-composite tiles right before convergence.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset/data.yaml")
    ap.add_argument("--model", default="yolo11x.pt", help="base checkpoint to fine-tune from")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="amyloid_plaque")
    ap.add_argument("--patience", type=int, default=30)
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
        close_mosaic=15,
        hsv_h=0.10,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=180,     # tissue has no canonical orientation
        flipud=0.5,
        fliplr=0.5,
        translate=0.1,
        scale=0.3,
        mosaic=1.0,
        plots=True,
    )


if __name__ == "__main__":
    main()
