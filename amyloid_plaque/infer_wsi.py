#!/usr/bin/env python3
"""
NeuroSpot-YOLO -- amyloid-beta plaque detection on whole-slide images (WSI).

Runs the trained YOLO11x plaque detector over one or more WSIs and writes, per slide:
  <out>/<slide>_detections.csv   every plaque: x1,y1,x2,y2 (level-0 px), conf
  <out>/<slide>_dsa.json         the same boxes as a DSA / HistomicsTK annotation (--dsa)
and one row per slide to <out>/summary.tsv:
  slide, width, height, mpp, tile_px, n_tiles_scanned, tissue_mm2, n_detections, plaques_per_mm2

How the slide is scanned (matches how the model was trained):
  * Physical scale. Training tiles were 1024 px at ~0.263 um/px (40x), i.e. ~269 um per side,
    resized to 640 px for the network. For a slide at another resolution the crop window is
    resized in pixels so it still covers ~269 um; otherwise plaques appear at the wrong size.
  * Overlap + de-duplication. Tiles overlap by 25 % so a plaque split by one tile edge is whole in
    a neighbour; overlapping boxes of the same plaque are then merged in slide coordinates
    (IoU > 0.3, or a tile-edge box lying >= 50 % inside a complete box); complete boxes win over cut-off ones.
  * Tissue. Tiles that are almost entirely glass are skipped. Density is reported per mm^2 of
    tissue, measured from a tissue mask of the slide thumbnail.

Example:
  python infer_wsi.py --slides /data/slides/*.svs --weights weights/amyloid_plaque_yolo11x.pt \
      --out results/ --dsa
"""
import argparse
import csv
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import openslide
import torch
from PIL import Image
from ultralytics import YOLO

CLASS_NAME = "Amyloid_Plaque"
TRAIN_MPP_UM = 0.263                                # training data resolution (40x)
TRAIN_TILE_PX = 1024                                # training tile size at that resolution
TRAIN_PHYSICAL_UM = TRAIN_TILE_PX * TRAIN_MPP_UM    # ~269 um per tile side
NET_INPUT_PX = 640                                  # training imgsz
OVERLAP_FRAC = 0.25
IOU_DEDUP = 0.3
CONTAIN_DEDUP = 0.5                                 # drop a tile-edge box that lies >= 50 % inside a kept box
EDGE_PX = 3                                         # box within this many network px of a tile border = "edge"
MIN_TISSUE_FRAC = 0.02                              # skip tiles that are ~all glass (as in training)
SLIDE_EXT = (".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".scn", ".vms", ".vmu", ".bif", ".svslide")


def tissue_fraction(rgb):
    """Fraction of non-white pixels in a tile (same rule used to build the training tiles)."""
    return float((rgb.reshape(-1, 3).astype(np.int32).sum(1) < 700).mean())


def tissue_mask(slide):
    """Low-resolution tissue mask: pixels darker than the slide's own glass background
    (IHC sections can be very pale), excluding near-black marks; small or thin pieces
    (dust, glass edges, streaks) are dropped. Returns (mask, downsample)."""
    lvl = slide.level_count - 1
    if max(slide.level_dimensions[lvl]) > 6000:      # slide without a small pyramid level
        thumb = np.array(slide.get_thumbnail((4000, 4000)).convert("RGB"))
        ds = slide.dimensions[0] / thumb.shape[1]
    else:
        thumb = np.array(slide.read_region((0, 0), lvl, slide.level_dimensions[lvl]).convert("RGB"))
        ds = slide.level_downsamples[lvl]
    gray = cv2.GaussianBlur(cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY), (5, 5), 0)
    bg = np.percentile(gray, 90)
    m = ((gray < bg - 8) & (gray > 60)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    keep = np.zeros(m.shape, bool)
    for i in range(1, n):
        w, h, area = st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT], st[i, cv2.CC_STAT_AREA]
        thin = area / (w * h) < 0.2 or max(w, h) / max(min(w, h), 1) > 8
        if area > 0.005 * m.size and not thin:
            keep |= lab == i
    return keep, ds


def dedup(boxes, thr=IOU_DEDUP, contain=CONTAIN_DEDUP):
    """Merge duplicate detections from overlapping tiles, in slide coordinates.

    boxes: N x 6 array (x1, y1, x2, y2, conf, edge), edge = 1 if the box touched the border of
    the tile it came from (the object may be cut off there and seen whole in a neighbouring tile).
    Boxes away from tile borders are taken first, most confident first, so a cut-off fragment never
    replaces the complete box. A box is dropped if it overlaps an already-kept box with IoU > thr;
    an edge box is also dropped if at least `contain` of its area lies inside a kept box.
    Returns N x 5 (x1, y1, x2, y2, conf)."""
    if not len(boxes):
        return boxes[:, :5]
    b = boxes[np.lexsort((-boxes[:, 4], boxes[:, 5]))]
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    edge = b[:, 5] > 0
    keep = []
    alive = np.ones(len(b), bool)
    for i in range(len(b)):
        if not alive[i]:
            continue
        keep.append(i)
        xx1 = np.maximum(b[i, 0], b[:, 0])
        yy1 = np.maximum(b[i, 1], b[:, 1])
        xx2 = np.minimum(b[i, 2], b[:, 2])
        yy2 = np.minimum(b[i, 3], b[:, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area - inter + 1e-9)
        inside = inter / (area + 1e-9)
        alive &= ~((iou > thr) | (edge & (inside >= contain)))
        alive[i] = False
    return b[keep, :5]


def slide_mpp(slide, override):
    if override:
        return override, "--mpp"
    for k in ("openslide.mpp-x", "aperio.MPP"):
        if k in slide.properties:
            return float(slide.properties[k]), k
    print(f"  WARNING: no resolution metadata; assuming {TRAIN_MPP_UM} um/px (40x). Use --mpp if wrong.")
    return TRAIN_MPP_UM, "assumed"


def run_slide(path, model, args, device):
    slide = openslide.OpenSlide(path)
    W, H = slide.dimensions
    mpp, mpp_src = slide_mpp(slide, args.mpp)
    tile = int(round(TRAIN_PHYSICAL_UM / mpp))
    stride = int(tile * (1 - OVERLAP_FRAC))
    mask, ds = tissue_mask(slide)
    tissue_mm2 = mask.sum() * (ds * mpp / 1000) ** 2

    xs = list(range(0, max(1, W - tile + 1), stride))
    ys = list(range(0, max(1, H - tile + 1), stride))
    if xs[-1] + tile < W:
        xs.append(max(0, W - tile))
    if ys[-1] + tile < H:
        ys.append(max(0, H - tile))
    # only read tiles that touch tissue in the thumbnail mask
    pos = []
    for y in ys:
        for x in xs:
            sub = mask[int(y / ds):int((y + tile) / ds) + 1, int(x / ds):int((x + tile) / ds) + 1]
            if sub.any():
                pos.append((x, y))
    print(f"  {W}x{H} px, {mpp:.4f} um/px ({mpp_src}) -> {tile} px tiles, stride {stride}; "
          f"{len(pos)} tissue tiles, {tissue_mm2:.1f} mm^2 tissue", flush=True)

    raw, scanned, t0 = [], 0, time.time()
    scale = tile / NET_INPUT_PX
    for i in range(0, len(pos), args.batch):
        chunk = pos[i:i + args.batch]
        imgs, kept = [], []
        for (x, y) in chunk:
            im = slide.read_region((x, y), 0, (tile, tile)).convert("RGB")
            if tile != NET_INPUT_PX:
                im = im.resize((NET_INPUT_PX, NET_INPUT_PX), Image.BILINEAR)
            a = np.array(im)
            if tissue_fraction(a) >= MIN_TISSUE_FRAC:
                imgs.append(a[..., ::-1])          # ultralytics expects BGR numpy arrays
                kept.append((x, y))
        scanned += len(imgs)
        if imgs:
            res = model.predict(imgs, imgsz=NET_INPUT_PX, conf=args.conf, device=device, verbose=False,
                                half=device != "cpu")
            for (x, y), r in zip(kept, res):
                for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    edge = float(min(b[0], b[1]) < EDGE_PX or max(b[2], b[3]) > NET_INPUT_PX - EDGE_PX)
                    raw.append([x + b[0] * scale, y + b[1] * scale, x + b[2] * scale, y + b[3] * scale, float(c), edge])
        if (i // args.batch) % 20 == 0:
            print(f"    {min(i + args.batch, len(pos))}/{len(pos)} tiles, {len(raw)} raw boxes, "
                  f"{time.time() - t0:.0f}s", flush=True)
        if device != "cpu" and (i // args.batch) % 25 == 0:
            torch.cuda.empty_cache()

    det = dedup(np.array(raw).reshape(-1, 6))
    # keep detections whose centre lies on tissue
    if len(det):
        cy = np.clip(((det[:, 1] + det[:, 3]) / 2 / ds).astype(int), 0, mask.shape[0] - 1)
        cx = np.clip(((det[:, 0] + det[:, 2]) / 2 / ds).astype(int), 0, mask.shape[1] - 1)
        det = det[mask[cy, cx]]
    print(f"  {len(raw)} raw -> {len(det)} plaques after de-duplication ({time.time() - t0:.0f}s)", flush=True)
    return dict(W=W, H=H, mpp=mpp, tile=tile, n_tiles=scanned, tissue_mm2=tissue_mm2, det=det)


def write_dsa(det, path, conf):
    elements = [{
        "type": "rectangle", "center": [float((x1 + x2) / 2), float((y1 + y2) / 2), 0],
        "width": float(x2 - x1), "height": float(y2 - y1), "rotation": 0,
        "group": CLASS_NAME, "label": {"value": CLASS_NAME},
        "lineColor": "rgb(0, 255, 0)", "fillColor": "rgba(0, 255, 0, 0)",
        "user": {"confidence": round(float(c), 4)},
    } for x1, y1, x2, y2, c in det]
    json.dump({"name": "Amyloid_Plaque_detections",
               "description": f"NeuroSpot-YOLO amyloid plaque model, conf >= {conf}",
               "elements": elements}, open(path, "w"))


def collect(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(str(q) for q in Path(p).rglob("*") if q.suffix.lower() in SLIDE_EXT)
        else:
            out += sorted(glob.glob(p))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slides", nargs="+", required=True, help="slide files, globs, or folders")
    ap.add_argument("--weights", default=str(Path(__file__).parent / "weights" / "amyloid_plaque_yolo11x.pt"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (default 0.25)")
    ap.add_argument("--mpp", type=float, default=None, help="override slide resolution, um/px")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None, help="e.g. 0, cpu (default: GPU if available)")
    ap.add_argument("--dsa", action="store_true", help="also write a DSA/HistomicsTK annotation JSON")
    args = ap.parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    slides = collect(args.slides)
    if not slides:
        raise SystemExit("no slides found")
    os.makedirs(args.out, exist_ok=True)
    model = YOLO(args.weights)
    summary = os.path.join(args.out, "summary.tsv")
    new = not os.path.exists(summary)
    with open(summary, "a", newline="") as fs:
        w = csv.writer(fs, delimiter="\t")
        if new:
            w.writerow(["slide", "width", "height", "mpp", "tile_px", "n_tiles_scanned", "tissue_mm2",
                        "n_detections", "plaques_per_mm2", "conf"])
        for k, path in enumerate(slides, 1):
            stem = Path(path).stem
            print(f"[{k}/{len(slides)}] {stem}", flush=True)
            try:
                r = run_slide(path, model, args, device)
            except Exception as e:  # keep going on a bad slide
                print(f"  ERROR: {e}", flush=True)
                continue
            np.savetxt(os.path.join(args.out, f"{stem}_detections.csv"), r["det"], delimiter=",",
                       header="x1,y1,x2,y2,conf", comments="", fmt=["%.1f"] * 4 + ["%.4f"])
            if args.dsa:
                write_dsa(r["det"], os.path.join(args.out, f"{stem}_dsa.json"), args.conf)
            dens = len(r["det"]) / r["tissue_mm2"] if r["tissue_mm2"] > 0 else float("nan")
            w.writerow([stem, r["W"], r["H"], round(r["mpp"], 5), r["tile"], r["n_tiles"],
                        round(r["tissue_mm2"], 2), len(r["det"]), round(dens, 3), args.conf])
            fs.flush()


if __name__ == "__main__":
    main()
