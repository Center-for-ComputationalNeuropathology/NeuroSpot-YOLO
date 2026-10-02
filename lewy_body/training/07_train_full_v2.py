#!/usr/bin/env python3
"""Train the Lewy body detector on the full 14-slide dataset, v2 -- case-
level split (no leakage), ROI_to_test excluded, positive tiles capped per
slide, and the yolo11x/no-rotation recipe from the existing well-performing
checkpoint (models/lewy_bodies_best.pt, peak mAP50=0.855). See
06_tile_full_v2.py and YOLO_TRAINING_PLAYBOOK.md for the fixes this
addresses relative to the original run (runs/lewy_body_asyn_v1, mAP50=0.225,
early-stopped at epoch 35).
"""

import os

from ultralytics import YOLO

ROOT = os.environ.get("LEWY_ROOT", ".")  # folder holding dataset_v2/
BASE_MODEL = "yolo11x.pt"  # COCO-pretrained, downloaded by ultralytics
os.chdir(ROOT)


def main():
    model = YOLO(BASE_MODEL)
    model.train(
        data="dataset_v2/lewy_body_v2.yaml",
        imgsz=640,
        epochs=150,
        patience=100,
        batch=16,
        device=0,
        workers=4,  # matches the 4 CPU slots requested from LSF
        project=os.path.join(ROOT, "runs_v2"),
        name="lewy_body_full_v2",
        exist_ok=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=0.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
    )


if __name__ == "__main__":
    main()
