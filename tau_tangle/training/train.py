#!/usr/bin/env python3
"""Train the tangle detector on the patient-level split built by build_dataset_v2.py.
Settings are those of the released run (run_tau_tangle_v2/args.yaml); augmentation is the
Ultralytics default (no rotation or vertical flip)."""
import argparse

from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="dataset_v2/dataset.yaml")
ap.add_argument("--model", default="yolo11x.pt", help="base checkpoint (COCO-pretrained)")
ap.add_argument("--device", default="0")
ap.add_argument("--project", default="runs")
ap.add_argument("--name", default="tau_tangle_v2")
a = ap.parse_args()

YOLO(a.model).train(data=a.data, epochs=100, batch=16, imgsz=640, device=a.device, project=a.project,
                    name=a.name, save_period=10, patience=15, warmup_epochs=5, seed=0)
