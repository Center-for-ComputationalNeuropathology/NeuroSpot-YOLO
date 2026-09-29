#!/usr/bin/env python3
"""
STEP3_build_dataset.py - Assemble the final single-class YOLO dataset from the
resolved annotations (STEP1) using the calibrated negative-mining method
(STEP2a/STEP2b).

POSITIVES (single merged class, id 0 "TDP-43"):
  1. Grid-tile every confirmed "roi" region (native-resolution PATCH_SIZE tiles
     at ROI_GRID_STRIDE) -- exhaustively-reviewed rectangles, so any tile
     overlapping >=1 positive box becomes a labeled image with ALL overlapping
     boxes as ground truth. Overlapping stride gives multiple translated views
     of each inclusion (free augmentation).
  2. Any positive box NOT captured by an ROI tile (2026-protocol files with no
     roi, or boxes outside the old roi) gets ZOOM-NORMALISED cluster-and-crop:
     nearby boxes are grouped, a crop window is sized to a few multiples of the
     cluster's own extent (clamped), then resized to OUT_SIZE. This makes the
     inclusion occupy a consistent, learnable fraction of the frame regardless
     of how sparse the slide is -- addressing the "50px inclusion is only 5% of
     a downscaled 1024px frame" problem that held back earlier runs.

NEGATIVES -- the tiers validated by human review in STEP2b, at training scale:
  Tier 0 negative_control  - hard-negative distractor structures explicitly
                              rejected by a pathologist. Always trusted.
  Tier 1 roi_clean         - grid tiles inside confirmed ROI with zero box
                              overlap, DAB-blob-checked. Trusted.
  Tier 2 color_threshold   - tissue elsewhere on the slide passing the
                              calibrated DAB-blob test. The METHOD is trusted
                              (human-reviewed 2026-09-05), applied at scale.
  Excluded: likely_missed_positive (data/candidate_negatives/) -- human QC
  called this a MIXED set ("mostly diffuse background, some may be real
  inclusions"), so folding it in as negatives would inject label noise. Needs a
  patch-by-patch QC pass before it can be used.

TEST split: grid-tiled from "roi_test" regions, EMPTY labels (intentional blind
holdout by the annotator; used for visual inspection in STEP5, not scored mAP).

-----------------------------------------------------------------------------
v3 changes (2026-09-10) -- honest held-out-slide performance was ~0.44 mAP50 in
v2, and diagnosis pointed at object scale + cross-slide stain variation + a
1-slide validation set that was too noisy. This build:

  1. SMALLER PATCHES.  PATCH_SIZE 1024 -> 640 (== training imgsz, so ROI/neg
     tiles are saved with NO downscale). A ~57px inclusion goes from ~5% to
     ~9% of the frame just by dropping the old 1024->640 shrink.

  2. ZOOM-NORMALISED CLUSTER CROPS.  Cluster-crop windows are now sized from
     the cluster's own pixel extent (CLUSTER_ZOOM x, clamped to
     [CLUSTER_MIN_CROP, CLUSTER_MAX_CROP]) then resized to OUT_SIZE. Isolated
     inclusions on sparse slides get zoomed in so they're ~15% of the frame
     instead of vanishing. YOLO labels are normalised 0-1 so the resize does
     not change them.

  3. STAIN NORMALISATION (Reinhard, tissue-masked).  Every saved patch is
     colour-matched to a dataset-wide reference (mean/std in LAB over
     tissue-only pixels), removing the biggest source of slide-to-slide
     variation. Reference stats are written to stain_reference.json so STEP5
     inference can apply the identical transform. Before/after montages are
     dumped to _stain_norm_samples/ for a visual sanity check.

  4. MULTI-SLIDE HELD-OUT VALIDATION.  Instead of accumulating slides until
     15% of positive-box volume (which landed on a single slide in v2), hold
     out N_VALID_SLIDES_PER_FAMILY whole slide(s) from EACH stain family
     (MA24 / NACC / NPBB), chosen by a fixed RNG. Gives a less noisy,
     cross-family generalisation estimate.

v2 changes retained: whole-slide (never patch-level) train/valid split;
per-slide data-driven tissue brightness/saturation thresholds.
"""

import json
import os
import random
from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image
from skimage.color import rgb2hed

REPO_ROOT = Path(__file__).resolve().parents[1]
RESOLVED_DIR = Path(os.environ.get("TDP43_RESOLVED_DIR", str(REPO_ROOT / "data" / "resolved_annotations")))
# Overridable so a rebuild (e.g. after re-resolving annotations, v2 2026-09-16)
# can be built into a fresh directory instead of overwriting data/dataset/
# in place while other jobs still hold symlinks into the live one.
OUTPUT_DIR = Path(os.environ.get("TDP43_DATASET_OUTPUT_DIR", str(REPO_ROOT / "data" / "dataset")))

# --- patch geometry (v3) ---
OUT_SIZE = 640                  # what every image is saved at == training imgsz
PATCH_SIZE = 640               # native-resolution window for ROI tiles / negatives / test
ROI_GRID_STRIDE = 320          # 50% overlap -> multiple translated views per inclusion

# --- zoom-normalised cluster crops (v3) ---
CLUSTER_MAX_DIST = 400         # px; group positive-box centres this close into one crop
CLUSTER_ZOOM = 3.5            # crop window = cluster pixel-extent * this ...
CLUSTER_MIN_CROP = 384        # ... clamped to at least this (tight zoom on lone inclusions)
CLUSTER_MAX_CROP = 896        # ... and at most this (keep some context, limit downscale)

NEG_CONTROL_CLUSTER_DIST = 400
JPEG_QUALITY = 95

# --- multi-slide held-out validation ---
# v4 (2026-09-19): replaced the old "N_VALID_SLIDES_PER_FAMILY" per-family
# heuristic with an exact 80/20-by-positive-patch-count slide split (see
# assign_slide_splits). TRAIN_FRACTION is that split's target.
TRAIN_FRACTION = 0.80

# --- stain normalisation (v3) ---
APPLY_STAIN_NORM = True
STAIN_REF_SAMPLES_PER_SLIDE = 15
STAIN_REF_CROP_SIZE = 256
STAIN_TISSUE_GRAY_MAX = 220    # pixels darker than this count as tissue for stats
STAIN_MIN_TISSUE_FRAC = 0.05  # below this, normalise on the whole patch instead

# --- calibrated in STEP2a against this dataset's own labeled pixels ---
DAB_PIXEL_THRESHOLD = 0.0432
MORPH_CLOSE_KERNEL = 5
BLOB_MIN_AREA = 30

# v4 (2026-09-19): was 0.20. Direct sampling showed real on-tissue candidates on
# several slides (MA24-160, NPBB400) have a median of only ~8% genuinely dark
# pixels -- 0.20 was rejecting ~99% of real tissue as "not tissue" on those
# slides (confirmed via Otsu tissue-fraction on a thumbnail: ~35% of the whole
# slide is tissue by a coarse estimate, but only ~1% of samples passed the old
# 0.20 bar). 0.08 recovers the large majority of real tissue while still
# rejecting true-blank patches (which measure exactly 0.000). Verified this
# does NOT relax the actual "is it brown" rule at all -- BLOB_MIN_AREA is
# unchanged; a sample of newly-recovered negatives was visually inspected and
# looked like genuine pale/sparse tissue, not artifacts.
TISSUE_THRESHOLD = 0.08

# --- fallback tissue thresholds, used only when a slide has no positive
# boxes to calibrate from (see calibrate_tissue_thresholds) ---
DEFAULT_BRIGHTNESS_MAX = 248
DEFAULT_SATURATION_MIN = 8
TISSUE_CALIBRATION_CROP_SIZE = 400
TISSUE_CALIBRATION_MAX_BOXES = 150

# --- full-scale negative mining budget (vs. the small QC sample in STEP2b) ---
COLOR_THRESHOLD_MAX_ATTEMPTS_PER_SLIDE = 6000
# v4 (2026-09-19): was 150. Raised so slides with abundant qualifying tissue
# (e.g. NACC068046 hit 150 in just 2824/6000 attempts) can supply more toward
# the global train-set 50/50 positive:negative balance, rather than being
# artificially capped while negative-poor slides can't keep up.
COLOR_THRESHOLD_TARGET_PER_SLIDE = 300

RNG_SEED = 0


# ============================================================================
# geometry / colour helpers
# ============================================================================

def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or ax1 >= bx2 or ay2 <= by1 or ay1 >= by2)


def any_overlap(box, box_list):
    return any(boxes_overlap(box, b) for b in box_list)


def clamp(v, lo, hi):
    return max(lo, min(v, hi))


def dab_channel(patch_rgb_uint8):
    rgb_float = patch_rgb_uint8.astype(np.float64) / 255.0
    return rgb2hed(rgb_float)[:, :, 2]


def max_dab_blob_area(patch_rgb_uint8):
    d = dab_channel(patch_rgb_uint8)
    mask = (d > DAB_PIXEL_THRESHOLD).astype(np.uint8)
    kernel = np.ones((MORPH_CLOSE_KERNEL, MORPH_CLOSE_KERNEL), np.uint8)
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if n_labels <= 1:
        return 0
    return int(stats[1:, cv2.CC_STAT_AREA].max())


def is_tissue_patch(patch_rgb_uint8, brightness_max=DEFAULT_BRIGHTNESS_MAX,
                     saturation_min=DEFAULT_SATURATION_MIN, tissue_threshold=TISSUE_THRESHOLD):
    # brightness_max/saturation_min are calibrated per slide by
    # calibrate_tissue_thresholds(); the global guesses (248/8) were shown to
    # reject real tissue on paler batches and remain only as a fallback.
    gray = cv2.cvtColor(patch_rgb_uint8, cv2.COLOR_RGB2GRAY)
    if np.mean(gray) > brightness_max:
        return False
    if np.std(gray) < 5:
        return False
    hsv = cv2.cvtColor(patch_rgb_uint8, cv2.COLOR_RGB2HSV)
    if np.mean(hsv[:, :, 1]) < saturation_min:
        return False
    if np.sum(gray < 220) / gray.size >= tissue_threshold:
        return True
    return np.sum(hsv[:, :, 1] > 15) / hsv.shape[0] / hsv.shape[1] >= tissue_threshold


def calibrate_tissue_thresholds(record, wsi, rng):
    """Data-driven tissue brightness/saturation cutoffs for this slide, sampled
    from the neighbourhoods of its own confirmed positive boxes (guaranteed
    real tissue). Same percentile philosophy as the STEP2a DAB calibration.
    Falls back to the global defaults if the slide has no positive boxes.
    """
    boxes = [p["box"] for p in record["positive_boxes"]]
    if not boxes:
        return DEFAULT_BRIGHTNESS_MAX, DEFAULT_SATURATION_MIN

    sample_size = min(len(boxes), TISSUE_CALIBRATION_MAX_BOXES)
    sampled = rng.sample(boxes, sample_size)

    gray_vals, sat_vals = [], []
    half = TISSUE_CALIBRATION_CROP_SIZE / 2
    for x1, y1, x2, y2 in sampled:
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        patch = read_patch(wsi, cx - half, cy - half, size=TISSUE_CALIBRATION_CROP_SIZE)
        if patch is None:
            continue
        gray_vals.append(cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY).ravel())
        sat_vals.append(cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)[:, :, 1].ravel())

    if not gray_vals:
        return DEFAULT_BRIGHTNESS_MAX, DEFAULT_SATURATION_MIN

    gray_all = np.concatenate(gray_vals)
    sat_all = np.concatenate(sat_vals)

    # Only ever relax (never tighten) vs. the global default.
    brightness_max = max(DEFAULT_BRIGHTNESS_MAX, float(np.percentile(gray_all, 99.5)))
    saturation_min = min(DEFAULT_SATURATION_MIN, float(np.percentile(sat_all, 0.5)))
    brightness_max = min(brightness_max, 254)
    saturation_min = max(saturation_min, 1)
    return brightness_max, saturation_min


def cluster_points(points, max_dist):
    clusters = []
    for p in points:
        joined = False
        for c in clusters:
            cx = sum(q[0] for q in c["points"]) / len(c["points"])
            cy = sum(q[1] for q in c["points"]) / len(c["points"])
            if ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5 <= max_dist:
                c["points"].append(p)
                joined = True
                break
        if not joined:
            clusters.append({"points": [p]})
    return clusters


def read_patch(wsi, x, y, size=PATCH_SIZE):
    try:
        return np.array(wsi.read_region((int(x), int(y)), 0, (int(size), int(size))).convert("RGB"))
    except Exception:
        return None


def convert_to_yolo(box, patch_x, patch_y, patch_size):
    """Box -> normalised (xc, yc, w, h) within a patch_size window at (patch_x,
    patch_y). Normalised coords are invariant to any later resize of the patch."""
    x1, y1, x2, y2 = box
    rel = [max(x1, patch_x) - patch_x, max(y1, patch_y) - patch_y,
           min(x2, patch_x + patch_size) - patch_x, min(y2, patch_y + patch_size) - patch_y]
    xc = (rel[0] + rel[2]) / 2 / patch_size
    yc = (rel[1] + rel[3]) / 2 / patch_size
    w = (rel[2] - rel[0]) / patch_size
    h = (rel[3] - rel[1]) / patch_size
    return xc, yc, w, h


def clip_patch_origin(x, y, img_w, img_h, size):
    x = max(0, min(x, img_w - size))
    y = max(0, min(y, img_h - size))
    return x, y


# ============================================================================
# stain normalisation (Reinhard, tissue-masked)
# ============================================================================

def _tissue_lab_stats(rgb_uint8):
    """LAB per-channel (mean, std) over tissue-only pixels of one patch."""
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    mask = gray < STAIN_TISSUE_GRAY_MAX
    if mask.sum() < STAIN_MIN_TISSUE_FRAC * mask.size:
        flat = lab.reshape(-1, 3)
    else:
        flat = lab[mask]
    return flat.mean(axis=0), flat.std(axis=0)


def compute_stain_reference(records, rng):
    """Dataset-wide reference = average of per-patch tissue LAB mean/std over a
    sample of crops spanning every slide."""
    means, stds = [], []
    for record in records:
        ip = record["image_path"]
        if not os.path.exists(ip):
            continue
        wsi = openslide.OpenSlide(ip)
        w, h = wsi.dimensions
        got = 0
        tries = 0
        while got < STAIN_REF_SAMPLES_PER_SLIDE and tries < STAIN_REF_SAMPLES_PER_SLIDE * 20:
            tries += 1
            x = rng.randint(0, max(0, w - STAIN_REF_CROP_SIZE))
            y = rng.randint(0, max(0, h - STAIN_REF_CROP_SIZE))
            patch = read_patch(wsi, x, y, size=STAIN_REF_CROP_SIZE)
            if patch is None:
                continue
            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            if (gray < STAIN_TISSUE_GRAY_MAX).mean() < 0.30:
                continue  # mostly background, skip
            m, s = _tissue_lab_stats(patch)
            means.append(m)
            stds.append(s)
            got += 1
        wsi.close()
    ref_mean = np.mean(means, axis=0)
    ref_std = np.mean(stds, axis=0)
    return ref_mean.tolist(), ref_std.tolist()


def stain_normalize(rgb_uint8, ref_mean, ref_std):
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float32)
    src_mean, src_std = _tissue_lab_stats(rgb_uint8)
    src_std = np.where(src_std < 1e-6, 1.0, src_std)
    out = (lab - src_mean) / src_std * np.asarray(ref_std, np.float32) + np.asarray(ref_mean, np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


# ============================================================================
# positive extraction
# ============================================================================

def extract_roi_positive_tiles(record):
    """Grid-tile confirmed ROI regions; keep tiles overlapping >=1 positive box.
    Returns (tiles, covered, positive_boxes) where each tile is
    (x, y, window_size, labels) at native resolution (window_size == PATCH_SIZE)."""
    positive_boxes = [p["box"] for p in record["positive_boxes"]]
    tiles = []
    covered = set()
    for rx1, ry1, rx2, ry2 in record["roi_boxes"]:
        y = ry1
        while y + PATCH_SIZE <= ry2:
            x = rx1
            while x + PATCH_SIZE <= rx2:
                patch_box = [x, y, x + PATCH_SIZE, y + PATCH_SIZE]
                overlapping = [i for i, b in enumerate(positive_boxes) if boxes_overlap(patch_box, b)]
                if overlapping:
                    tiles.append((x, y, PATCH_SIZE, [positive_boxes[i] for i in overlapping]))
                    covered.update(overlapping)
                x += ROI_GRID_STRIDE
            y += ROI_GRID_STRIDE
    return tiles, covered, positive_boxes


def extract_clustered_positive_tiles(record, wsi, covered, positive_boxes):
    """Zoom-normalised cluster-and-crop for positive boxes not covered by an ROI
    tile. Each tile is (x, y, window_size, labels); window_size varies and the
    patch is resized to OUT_SIZE at save time."""
    img_w, img_h = wsi.dimensions
    idx_uncovered = [i for i in range(len(positive_boxes)) if i not in covered]
    if not idx_uncovered:
        return []
    centres = {i: ((positive_boxes[i][0] + positive_boxes[i][2]) / 2,
                   (positive_boxes[i][1] + positive_boxes[i][3]) / 2) for i in idx_uncovered}
    clusters = cluster_points(list(centres.values()), CLUSTER_MAX_DIST)

    tiles = []
    for c in clusters:
        pts = set(c["points"])
        members = [i for i in idx_uncovered if centres[i] in pts]
        mb = [positive_boxes[i] for i in members]
        minx = min(b[0] for b in mb); maxx = max(b[2] for b in mb)
        miny = min(b[1] for b in mb); maxy = max(b[3] for b in mb)
        span = max(maxx - minx, maxy - miny)
        win = int(clamp(span * CLUSTER_ZOOM, CLUSTER_MIN_CROP, CLUSTER_MAX_CROP))
        win = int(min(win, img_w, img_h))
        cx = (minx + maxx) / 2
        cy = (miny + maxy) / 2
        x, y = clip_patch_origin(cx - win / 2, cy - win / 2, img_w, img_h, win)
        patch_box = [x, y, x + win, y + win]
        labels = [b for b in positive_boxes if boxes_overlap(patch_box, b)]
        if labels:
            tiles.append((int(x), int(y), win, labels))
    return tiles


# ============================================================================
# negative extraction (same tiers as STEP2b, run at training scale)
# ============================================================================

def mine_negative_control_tiles(record, wsi, rng):
    nc_boxes = record["negative_control_boxes"]
    if not nc_boxes:
        return []
    positives = [p["box"] for p in record["positive_boxes"]]
    centers = [(((b["box"][0] + b["box"][2]) / 2, (b["box"][1] + b["box"][3]) / 2)) for b in nc_boxes]
    clusters = cluster_points(centers, NEG_CONTROL_CLUSTER_DIST)
    tiles = []
    for c in clusters:
        cx = sum(p[0] for p in c["points"]) / len(c["points"])
        cy = sum(p[1] for p in c["points"]) / len(c["points"])
        x, y = cx - PATCH_SIZE / 2, cy - PATCH_SIZE / 2
        patch_box = [x, y, x + PATCH_SIZE, y + PATCH_SIZE]
        if any_overlap(patch_box, positives):
            continue
        patch = read_patch(wsi, x, y)
        if patch is None or max_dab_blob_area(patch) >= BLOB_MIN_AREA:
            continue
        tiles.append((x, y, patch))
    return tiles


def mine_roi_clean_tiles(record, wsi):
    roi_boxes = record["roi_boxes"]
    if not roi_boxes:
        return []
    positives = [p["box"] for p in record["positive_boxes"]]
    tiles = []
    for rx1, ry1, rx2, ry2 in roi_boxes:
        y = ry1
        while y + PATCH_SIZE <= ry2:
            x = rx1
            while x + PATCH_SIZE <= rx2:
                patch_box = [x, y, x + PATCH_SIZE, y + PATCH_SIZE]
                if not any_overlap(patch_box, positives):
                    patch = read_patch(wsi, x, y)
                    if patch is not None and max_dab_blob_area(patch) < BLOB_MIN_AREA:
                        tiles.append((x, y, patch))
                x += ROI_GRID_STRIDE
            y += ROI_GRID_STRIDE
    return tiles


def mine_color_threshold_tiles(record, wsi, rng, target_count, brightness_max, saturation_min):
    img_w, img_h = wsi.dimensions
    positives = [p["box"] for p in record["positive_boxes"]]
    roi_boxes = record["roi_boxes"]
    tiles = []
    attempts = 0
    seen = set()
    while attempts < COLOR_THRESHOLD_MAX_ATTEMPTS_PER_SLIDE and len(tiles) < target_count:
        attempts += 1
        gx = rng.randint(0, max(0, (img_w - PATCH_SIZE) // PATCH_SIZE))
        gy = rng.randint(0, max(0, (img_h - PATCH_SIZE) // PATCH_SIZE))
        x, y = gx * PATCH_SIZE, gy * PATCH_SIZE
        if (x, y) in seen:
            continue
        seen.add((x, y))
        box = [x, y, x + PATCH_SIZE, y + PATCH_SIZE]
        if any_overlap(box, roi_boxes) or any_overlap(box, positives):
            continue
        patch = read_patch(wsi, x, y)
        if patch is None or not is_tissue_patch(patch, brightness_max, saturation_min):
            continue
        if max_dab_blob_area(patch) >= BLOB_MIN_AREA:
            continue
        tiles.append((x, y, patch))
    return tiles, attempts


# ============================================================================
# test split (roi_test, unlabeled by design)
# ============================================================================

def extract_test_tiles(record, wsi):
    tiles = []
    for rx1, ry1, rx2, ry2 in record["roi_test_boxes"]:
        y = ry1
        while y + PATCH_SIZE <= ry2:
            x = rx1
            while x + PATCH_SIZE <= rx2:
                patch = read_patch(wsi, x, y)
                if patch is not None:
                    tiles.append((x, y, patch))
                x += ROI_GRID_STRIDE
            y += ROI_GRID_STRIDE
    return tiles


# ============================================================================
# I/O
# ============================================================================

def prepare_dirs(base):
    for split in ("train", "valid"):  # v4 (2026-09-19): dropped "test", see write_dataset_yaml
        os.makedirs(base / split / "images", exist_ok=True)
        os.makedirs(base / split / "labels", exist_ok=True)
    os.makedirs(base / "_stain_norm_samples", exist_ok=True)


def finalize_patch(patch_arr, ref_mean, ref_std):
    """Resize to OUT_SIZE (if needed) then stain-normalise."""
    h = patch_arr.shape[0]
    if h != OUT_SIZE or patch_arr.shape[1] != OUT_SIZE:
        interp = cv2.INTER_AREA if h > OUT_SIZE else cv2.INTER_CUBIC
        patch_arr = cv2.resize(patch_arr, (OUT_SIZE, OUT_SIZE), interpolation=interp)
    if APPLY_STAIN_NORM and ref_mean is not None:
        patch_arr = stain_normalize(patch_arr, ref_mean, ref_std)
    return patch_arr


def save_example(base, split, name, patch_arr, labels):
    img_path = base / split / "images" / f"{name}.jpg"
    lbl_path = base / split / "labels" / f"{name}.txt"
    Image.fromarray(patch_arr).save(img_path, "JPEG", quality=JPEG_QUALITY, optimize=True)
    with open(lbl_path, "w") as f:
        for xc, yc, w, h in labels:
            f.write(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")


def save_stain_sample(base, name, before_rgb, after_rgb):
    b = cv2.resize(before_rgb, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_AREA)
    montage = np.concatenate([b, after_rgb], axis=1)
    Image.fromarray(montage).save(base / "_stain_norm_samples" / f"{name}.jpg",
                                  "JPEG", quality=90)


def write_dataset_yaml(base):
    # v4 (2026-09-19): dropped the "test" split. roi_test tiles are
    # deliberately unlabeled (the annotator left that region blind by
    # design), so they can't be folded into train/valid as verified
    # positives or negatives -- they were only ever used for unscored visual
    # inspection. Given the 8 slides are now split cleanly 80/20 into
    # train/valid only, that third bucket added complexity without adding
    # anything usable.
    yaml_path = base / "dataset.yaml"
    yaml_path.write_text(
        f"# TDP-43 Dataset YAML (single merged class)\n"
        f"path: {base}\n"
        f"train: train/images\n"
        f"val: valid/images\n\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: TDP-43\n"
    )
    return yaml_path


# ============================================================================
# main
# ============================================================================

def slide_family(name):
    for p in ("MA24", "NACC", "NPBB"):
        if name.startswith(p):
            return p
    return "other"


class _DimsOnly:
    """Duck-types the one WSI attribute extract_clustered_positive_tiles reads
    (.dimensions) for count-only planning, so slide-count planning doesn't need
    to open any actual WSI files. Safe because clip_patch_origin only nudges a
    tile's (x, y) to stay in-bounds -- it never changes whether a cluster with
    real member boxes produces a tile, so the count is dimension-independent
    as long as the fake bound is larger than any real slide."""
    dimensions = (2**31, 2**31)


def assign_slide_splits(records, rng):
    """v4 (2026-09-19): exact slide-level 80/20 split by POSITIVE PATCH COUNT
    (not slide count, not a per-family heuristic). Brute-forces every subset
    of the (small, n=12) slide list and keeps whichever one lands closest to
    20% of total positive patches in valid -- exact search is cheap at this
    scale and gives a much tighter ratio than a greedy/random pick would.
    Whole-slide split only -- a slide is never split across train and valid.
    Uses the real extract_roi_positive_tiles/extract_clustered_positive_tiles
    functions (not a re-derived estimate) so the planned counts exactly match
    what main() will actually build.

    v5 (2026-09-20): added a constraint -- no institution (NACC/NPBB/other)
    may have MORE than half its own positive patches land in valid. Without
    this, the pure closest-to-80/20 search can (and did, in the v4 build)
    assign an institution's patches mostly to valid while the overall ratio
    still looks like a fine 80/20 -- v4's valid set ended up disproportionately
    loaded with one hard subtype as a side effect of an unlucky combination,
    not this specific imbalance, but the constraint is a real accident this
    guards against for future rebuilds regardless.

    `rng` is accepted for signature compatibility but unused -- the search is
    exhaustive and deterministic."""
    import itertools

    slide_pos_counts = {}
    for r in records:
        roi_tiles, covered, positive_boxes = extract_roi_positive_tiles(r)
        cluster_tiles = extract_clustered_positive_tiles(r, _DimsOnly(), covered, positive_boxes)
        slide_pos_counts[r["slide_name"]] = len(roi_tiles) + len(cluster_tiles)

    names = list(slide_pos_counts)
    total = sum(slide_pos_counts.values())
    target_frac = 1 - TRAIN_FRACTION

    by_inst_total = {}
    for name in names:
        inst = slide_family(name)
        by_inst_total[inst] = by_inst_total.get(inst, 0) + slide_pos_counts[name]

    best = None
    for k in range(1, len(names)):
        for combo in itertools.combinations(names, k):
            valid_sum = sum(slide_pos_counts[s] for s in combo)
            diff = abs(valid_sum / total - target_frac)

            by_inst_valid = {}
            for s in combo:
                inst = slide_family(s)
                by_inst_valid[inst] = by_inst_valid.get(inst, 0) + slide_pos_counts[s]
            if any(by_inst_valid.get(inst, 0) / tot > 0.5 for inst, tot in by_inst_total.items() if tot > 0):
                continue  # would put an institution's majority in valid -- skip

            if best is None or diff < best[0]:
                best = (diff, combo)
    valid_slides = set(best[1])

    return {name: ("valid" if name in valid_slides else "train") for name in names}


def main():
    prepare_dirs(OUTPUT_DIR)
    rng = random.Random(RNG_SEED)

    resolved_files = sorted(RESOLVED_DIR.glob("*.json"))
    print(f"Found {len(resolved_files)} resolved annotation files\n")
    records = [json.load(open(rf)) for rf in resolved_files]

    slide_split = assign_slide_splits(records, rng)
    print("Slide-level train/valid assignment (whole slide held out, cross-family):")
    for name, split in sorted(slide_split.items()):
        print(f"  {name:45s} [{slide_family(name):5s}] -> {split}")
    n_valid_slides = sum(1 for s in slide_split.values() if s == "valid")
    print(f"  ({n_valid_slides} of {len(slide_split)} slides held out for validation)\n")

    ref_mean = ref_std = None
    if APPLY_STAIN_NORM:
        print("Computing dataset-wide stain reference (Reinhard, tissue-masked)...")
        ref_mean, ref_std = compute_stain_reference(records, random.Random(RNG_SEED + 1))
        (OUTPUT_DIR / "stain_reference.json").write_text(json.dumps(
            {"space": "LAB", "channels": ["L", "a", "b"],
             "mean": ref_mean, "std": ref_std,
             "tissue_gray_max": STAIN_TISSUE_GRAY_MAX}, indent=2))
        print(f"  ref LAB mean = {[round(v, 2) for v in ref_mean]}")
        print(f"  ref LAB std  = {[round(v, 2) for v in ref_std]}")
        print(f"  written to {OUTPUT_DIR / 'stain_reference.json'}\n")

    per_slide_summary = []
    stain_samples_saved = 0

    for rf, record in zip(resolved_files, records):
        image_path = record["image_path"]
        if not os.path.exists(image_path):
            print(f"[skip] WSI not found: {image_path}")
            continue

        wsi = openslide.OpenSlide(image_path)
        slide_stem = rf.stem
        split = slide_split[record["slide_name"]]
        print(f"=== {record['slide_name']} (-> {split}) ===")

        # --- positives ---
        roi_tiles, covered, all_pos = extract_roi_positive_tiles(record)
        cluster_tiles = extract_clustered_positive_tiles(record, wsi, covered, all_pos)

        n_pos = 0
        for x, y, win, labels in roi_tiles + cluster_tiles:
            raw = read_patch(wsi, x, y, size=win)
            if raw is None:
                continue
            patch = finalize_patch(raw, ref_mean, ref_std)
            yolo_labels = [convert_to_yolo(b, x, y, win) for b in labels]
            name = f"{slide_stem}_{int(y)}_{int(x)}_w{win}"
            save_example(OUTPUT_DIR, split, name, patch, yolo_labels)
            if APPLY_STAIN_NORM and stain_samples_saved < 24 and n_pos == 0:
                save_stain_sample(OUTPUT_DIR, name, raw, patch)
                stain_samples_saved += 1
            n_pos += 1

        # --- negatives (validated tiers, at training scale) ---
        brightness_max, saturation_min = calibrate_tissue_thresholds(record, wsi, rng)
        n_neg = 0
        neg_iter = (list(mine_negative_control_tiles(record, wsi, rng)) +
                    list(mine_roi_clean_tiles(record, wsi)))
        color_tiles, attempts = mine_color_threshold_tiles(
            record, wsi, rng, COLOR_THRESHOLD_TARGET_PER_SLIDE, brightness_max, saturation_min)
        neg_iter += list(color_tiles)
        for x, y, raw in neg_iter:
            patch = finalize_patch(raw, ref_mean, ref_std)
            save_example(OUTPUT_DIR, split, f"{slide_stem}_{int(y)}_{int(x)}_neg", patch, [])
            n_neg += 1

        # v4 (2026-09-19): dropped test-tile extraction -- see write_dataset_yaml
        # for why. roi_test_boxes are simply unused now (train/valid only).

        print(f"  positives: {n_pos} ({len(roi_tiles)} ROI-grid, {len(cluster_tiles)} zoom-cluster)")
        print(f"  negatives: {n_neg} (color-threshold attempts {attempts}, "
              f"cal brightness_max={brightness_max:.1f} saturation_min={saturation_min:.1f})")

        per_slide_summary.append({
            "slide": record["slide_name"], "split": split,
            "positives": n_pos, "negatives": n_neg,
        })
        wsi.close()

    # v4 (2026-09-19): enforce global 50/50 positive:negative balance in
    # train, pooled across all train slides (not per-slide -- some slides
    # structurally can't supply many negatives no matter what, see
    # TISSUE_THRESHOLD/COLOR_THRESHOLD_TARGET_PER_SLIDE history above). Mining
    # is done per-slide above with no advance guarantee of which direction the
    # real yield lands, so this reconciles it after the fact: if negatives
    # came up short, keep everything (never discard real positive annotations
    # to force an exact ratio); if they came up long, randomly trim the
    # excess negatives down to match, so train ends at 50/50 or as close as
    # achievable without ever throwing away a positive.
    train_lab_dir = OUTPUT_DIR / "train" / "labels"
    train_img_dir = OUTPUT_DIR / "train" / "images"
    train_labels = list(train_lab_dir.glob("*.txt"))
    train_pos = [p for p in train_labels if p.stat().st_size > 0]
    train_neg = [p for p in train_labels if p.stat().st_size == 0]
    n_trim = max(0, len(train_neg) - len(train_pos))
    if n_trim:
        rebalance_rng = random.Random(RNG_SEED + 2)
        to_remove = rebalance_rng.sample(train_neg, n_trim)
        for lab_path in to_remove:
            lab_path.unlink()
            img_path = train_img_dir / f"{lab_path.stem}.jpg"
            if img_path.exists():
                img_path.unlink()
        print(f"\nRebalanced train: trimmed {n_trim} excess negatives "
              f"({len(train_neg)} -> {len(train_neg) - n_trim}) to match "
              f"{len(train_pos)} positives (50/50 target).")
    elif len(train_neg) < len(train_pos):
        print(f"\nTrain negative supply ({len(train_neg)}) fell short of positives "
              f"({len(train_pos)}) even after maximizing mining -- kept all "
              f"available negatives rather than discard real positive annotations.")

    yaml_path = write_dataset_yaml(OUTPUT_DIR)

    print("\n" + "=" * 104)
    print("PER-SLIDE SUMMARY")
    print("=" * 104)
    print(f"{'slide':40s} {'split':>6s} {'positives':>10s} {'negatives':>10s}")
    for s in per_slide_summary:
        print(f"{s['slide']:40s} {s['split']:>6s} {s['positives']:10d} {s['negatives']:10d}")
    # recompute pos/neg from disk (not the per-slide tally above) since the
    # train-rebalancing step may have trimmed negatives after that tally was taken
    valid_lab_dir = OUTPUT_DIR / "valid" / "labels"
    valid_labels = list(valid_lab_dir.glob("*.txt")) if valid_lab_dir.exists() else []
    v_pos = sum(1 for p in valid_labels if p.stat().st_size > 0)
    v_neg = sum(1 for p in valid_labels if p.stat().st_size == 0)
    train_labels_final = list(train_lab_dir.glob("*.txt"))
    t_pos = sum(1 for p in train_labels_final if p.stat().st_size > 0)
    t_neg = sum(1 for p in train_labels_final if p.stat().st_size == 0)
    tp, tn = t_pos + v_pos, t_neg + v_neg

    print("\n" + "=" * 104)
    print("DATASET COMPLETE")
    print("=" * 104)
    print(f"Positive (labeled) images:   {tp}   (train {t_pos}, valid {v_pos})")
    print(f"Negative (empty-label) images: {tn}   (train {t_neg}, valid {v_neg})")
    print(f"Train pos:neg ratio: {t_pos}:{t_neg} ({100*t_neg/max(1,t_pos+t_neg):.1f}% negative)")
    total = tp + tn
    if total:
        print(f"Negative fraction: {tn / total:.1%}")
    print(f"Patch geometry: OUT_SIZE={OUT_SIZE}, ROI/neg window={PATCH_SIZE}, "
          f"cluster window [{CLUSTER_MIN_CROP}..{CLUSTER_MAX_CROP}] x{CLUSTER_ZOOM}")
    print(f"Stain normalisation: {'ON' if APPLY_STAIN_NORM else 'OFF'}")
    print(f"Dataset written to: {OUTPUT_DIR}")
    print(f"dataset.yaml: {yaml_path}")
    if APPLY_STAIN_NORM:
        print(f"Stain before/after montages: {OUTPUT_DIR / '_stain_norm_samples'} "
              f"({stain_samples_saved} saved) -- eyeball these before trusting the build")
    print("\nExcluded: data/candidate_negatives/likely_missed_positive/ "
          "(MIXED per human QC -- needs patch-by-patch review before use)")


if __name__ == "__main__":
    main()
