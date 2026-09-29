#!/usr/bin/env python3
"""
STEP4_train_nacc_final.py - Deployment model: NACC-only, trained on all 5
annotated NACC cases (data/dataset_nacc_cv/all), no validation split.

Recipe is identical to the leave-one-case-out CV folds (STEP4_train_nacc_cv.py),
whose pooled held-out AP50 was 0.829 (runs/nacc_cv_pooled_eval.csv) -- that is
this model's expected performance. Epochs fixed at 66 = median peak-mAP50 epoch
across the 5 folds (145/63/66/120/42), since there is no held-out data left to
early-stop on. last.pt is the deployment checkpoint.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = str(Path(__file__).resolve().parents[1])
MODEL_PATH = f"{BASE_DIR}/models/yolo11l.pt"
# env overrides select a dataset variant (e.g. dataset_nacc_cv_hn1 = + hard negatives)
DATASET_YAML = f"{BASE_DIR}/data/{os.environ.get('TDP43_DATASET', 'dataset_nacc_cv')}/all/dataset.yaml"
RUN_TAG = os.environ.get("TDP43_RUN_TAG", "nacc_final")
OUTPUT_DIR = f"{BASE_DIR}/runs"
EPOCHS = 66

if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"TDP43_{RUN_TAG}_{timestamp}"
    logging.basicConfig(filename=os.path.join(OUTPUT_DIR, f"{name}.log"), level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    try:
        logging.info(f"Starting NACC final training ({EPOCHS} epochs, no val) on {DATASET_YAML}")
        model = YOLO(MODEL_PATH)
        # close_mosaic scaled from 30/200 in the CV folds so the final no-mosaic phase
        # is a similar fraction of this shorter run
        results = model.train(
            data=DATASET_YAML, epochs=EPOCHS, val=False, batch=4, imgsz=960, device=0,
            project=OUTPUT_DIR, name=name, exist_ok=True, save_period=-1, plots=True, seed=0,
            warmup_epochs=5, hsv_h=0.03, hsv_s=0.9, hsv_v=0.6, degrees=10, translate=0.15,
            scale=0.5, mosaic=1.0, mixup=0.15, close_mosaic=10, box=10.0,
        )
        last = os.path.join(results.save_dir if results else f"{OUTPUT_DIR}/{name}", "weights", "last.pt")
        logging.info(f"NACC_FINAL_DONE last_pt={last}")
        print(f"NACC_FINAL_DONE last_pt={last}")
    except Exception as e:
        logging.error(f"final training failed: {e}")
        sys.exit(1)
