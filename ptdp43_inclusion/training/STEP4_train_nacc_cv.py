#!/usr/bin/env python3
"""
STEP4_train_nacc_cv.py - Train one NACC-only leave-one-case-out fold on
data/dataset_nacc_cv/fold_<CASE>/ (built by 03_build_dataset/STEP3_build_nacc_cv.py).

Usage: python STEP4_train_nacc_cv.py <HELDOUT_CASE>
Same base model / imgsz / hyperparameters as v5 and the Sep-16 CV folds, so the
only change vs. those runs is the data (NACC-only, ROI-only val, ROI_TEST
excluded) -- comparable per playbook step 7.
"""
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO

BASE_DIR = str(Path(__file__).resolve().parents[1])
MODEL_PATH = f"{BASE_DIR}/models/yolo11l.pt"
OUTPUT_DIR = f"{BASE_DIR}/runs"

if __name__ == '__main__':
    case = sys.argv[1]
    # env overrides select a dataset variant (e.g. dataset_nacc_cv_hn1 = + hard negatives)
    dataset = os.environ.get("TDP43_DATASET", "dataset_nacc_cv")
    tag = os.environ.get("TDP43_RUN_TAG", "nacc_cv")
    dataset_yaml = f"{BASE_DIR}/data/{dataset}/fold_{case}/dataset.yaml"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"TDP43_{tag}_{case}_{timestamp}"
    log_file = os.path.join(OUTPUT_DIR, f"{name}.log")
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")

    def log_and_print(message, level="info"):
        getattr(logging, level)(message)
        print(message, flush=True)

    try:
        log_and_print(f"Starting NACC-only CV fold, held-out case {case}. Data: {dataset_yaml}")
        model = YOLO(MODEL_PATH)
        train_results = model.train(
            data=dataset_yaml,
            epochs=200,
            batch=4,
            imgsz=960,
            device=0,
            project=OUTPUT_DIR,
            name=name,
            exist_ok=True,
            save_period=-1,
            patience=50,
            plots=True,
            seed=0,
            warmup_epochs=5,
            hsv_h=0.03,
            hsv_s=0.9,
            hsv_v=0.6,
            degrees=10,
            translate=0.15,
            scale=0.5,
            mosaic=1.0,
            mixup=0.15,
            close_mosaic=30,
            box=10.0,
        )
        best = os.path.join(train_results.save_dir, "weights", "best.pt")
        v = YOLO(best).val(data=dataset_yaml, imgsz=960, batch=8, device=0, plots=True,
                           project=OUTPUT_DIR, name=f"{name}_val", exist_ok=True)
        log_and_print(f"NACC_CV_RESULT case={case} P={v.box.mp:.4f} R={v.box.mr:.4f} "
                      f"mAP50={v.box.map50:.4f} mAP50-95={v.box.map:.4f} best_pt={best}")
    except Exception as e:
        log_and_print(f"fold {case} failed: {e}", level="error")
        sys.exit(1)
