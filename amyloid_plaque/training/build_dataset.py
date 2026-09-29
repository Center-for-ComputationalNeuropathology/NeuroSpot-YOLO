#!/usr/bin/env python3
"""
Build a YOLO-format tiled detection dataset for amyloid-beta plaques from
DSA/HistomicsTK JSON annotations referenced in amyloid-beta_manifest.tsv.

See README.md in this directory for the full design rationale. Summary:
 - Elements are deduped by annotation doc `_id` across all Json_path rows
   for a given image (the same doc can appear standalone AND embedded in a
   multi-doc *.annotations.json list).
 - Only plaque-group elements become the single "Amyloid_Plaque" class.
   Blood vessel / CAA / cluster / misc groups are dropped.
 - Tiles are only extracted from inside ROI / area_of_interest boxes (the
   regions that were exhaustively annotated). For slides with no ROI box
   at all, a padded bounding box around that doc's own annotations is used
   instead (conservative fallback).
 - Any tile overlapping an ROI_to_test box is skipped entirely (no ground
   truth exists there).
 - Negative Patch boxes are extracted directly as confirmed-empty tiles.
 - Slide-level train/val split, stratified by stain type (red Fast-Red vs
   brown DAB, inferred from source folder), to avoid tile leakage and to
   keep both stain domains represented in both splits.
 - For the "noisy" no-ROI slides (confirmed, by spot check, to have real
   plaques missing from the JSON even inside dense-pathology regions): a
   background tile is only trusted if it is also verified chromogen-free by
   color (no contiguous red/brown blob anywhere in the tile), since "no box"
   alone isn't reliable ground truth there. This only works for brown-DAB
   slides (red Fast-Red slides also color vessel-associated amyloid, which
   isn't the target class, so color alone can't confirm true negatives) --
   all 9 noisy slides happen to be brown-DAB, so this covers them.
 - Positive tiles are capped per slide (MAX_POS_TILES_PER_IMAGE, randomly
   subsampled) so a handful of extremely dense slides (1000+ raw boxes)
   can't dominate the training positive-tile pool via near-duplicate
   overlapping crops.
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict

import numpy as np
import openslide
from PIL import Image
from scipy import ndimage
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

Image.MAX_IMAGE_PIXELS = None

PLAQUE_GROUPS = {
    "Amyloid_plaques_BTO:0002774",
    "Diffuse Plaque",
    "Cored Plaque",
    "AMYLOID_Cored",
    "AMYLOID_Diffuse",
    "AMYLOID_Compact",
}
ROI_GROUPS = {"ROI", "area_of_interest"}
ROI_TEST_GROUPS = {"ROI_to_test"}
NEG_PATCH_GROUPS = {"Negative Patch"}
CLASS_NAMES = ["Amyloid_Plaque"]

TILE = 1024
OVERLAP_FRAC = 0.20
STRIDE = int(TILE * (1 - OVERLAP_FRAC))
MIN_BOX_KEEP_FRAC = 0.20   # keep a box clipped at a tile edge only if this much of its area survives
MIN_TISSUE_FRAC = 0.02     # skip near-blank (all background) tiles
NEG_TILE_TO_POS_TILE_RATIO = 1.0  # cap background-only tiles per image at this multiple of positive tiles
MAX_POS_TILES_PER_IMAGE = 150      # cap on positive tiles from any single slide (avoids a few dense slides dominating)
NO_ROI_PAD_PX = 300        # padding around annotations bbox for docs that have no ROI box
VAL_FRACTION = 0.20
SEED = 42

# color-based background verification for noisy brown-DAB slides (calibrated
# against known plaque-interior pixels vs. confirmed-clean background pixels
# from the ROI-restricted slides; validated by visual spot check).
CHROMOGEN_RB_THRESH = 15
CHROMOGEN_MIN_BLOB_AREA = 150
CHROMOGEN_CANDIDATE_BUDGET_MULT = 15  # candidates to screen per background tile wanted


def load_docs(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return [d] if isinstance(d, dict) else d


def elem_box(e):
    cx, cy = e["center"][0], e["center"][1]
    w, h = e["width"], e["height"]
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def stain_type(image_path):
    return "red_fastred" if "Hippocampus_beta_amyloid" in image_path else "brown_dab"


def collect_image_annotations(image_path, json_paths):
    """Dedup docs by _id across all json_paths for this image, then split
    elements into plaque boxes / ROI regions / exclude regions / negative
    patches for each doc, applying the no-ROI fallback per-doc."""
    docs_by_id = {}
    for jp in json_paths:
        for doc in load_docs(jp):
            docs_by_id[doc["_id"]] = doc

    plaque_boxes = []
    roi_polys = []
    exclude_polys = []
    neg_patch_boxes = []
    has_real_roi = False

    for doc in docs_by_id.values():
        els = doc.get("annotation", {}).get("elements", [])
        doc_plaques = [e for e in els if e.get("group") in PLAQUE_GROUPS and e.get("type") == "rectangle"]
        doc_roi = [elem_box(e) for e in els if e.get("group") in ROI_GROUPS and e.get("type") == "rectangle"]
        doc_exclude = [elem_box(e) for e in els if e.get("group") in ROI_TEST_GROUPS and e.get("type") == "rectangle"]
        doc_neg = [elem_box(e) for e in els if e.get("group") in NEG_PATCH_GROUPS and e.get("type") == "rectangle"]

        plaque_boxes.extend(elem_box(e) for e in doc_plaques)
        exclude_polys.extend(shapely_box(*b) for b in doc_exclude)
        neg_patch_boxes.extend(doc_neg)

        if doc_roi:
            has_real_roi = True
            roi_polys.extend(shapely_box(*b) for b in doc_roi)
        elif doc_plaques:
            xs0 = [elem_box(e)[0] for e in doc_plaques]
            ys0 = [elem_box(e)[1] for e in doc_plaques]
            xs1 = [elem_box(e)[2] for e in doc_plaques]
            ys1 = [elem_box(e)[3] for e in doc_plaques]
            pad = NO_ROI_PAD_PX
            roi_polys.append(shapely_box(min(xs0) - pad, min(ys0) - pad, max(xs1) + pad, max(ys1) + pad))

    roi_union = unary_union(roi_polys) if roi_polys else None
    exclude_union = unary_union(exclude_polys) if exclude_polys else None

    # dedupe plaque boxes (multiple docs could in principle carry the exact same box)
    plaque_boxes = sorted(set(tuple(round(v, 1) for v in b) for b in plaque_boxes))

    # "noisy" = every positive region for this image came from the no-ROI bbox
    # fallback -- confirmed (via spot check) to have real plaques missing from
    # the JSON even inside that bbox. These images must never be used to
    # sample background/negative tiles (an empty label there isn't
    # trustworthy) and must never land in the validation split.
    noisy = (roi_union is not None) and (not has_real_roi)

    return plaque_boxes, roi_union, exclude_union, neg_patch_boxes, noisy


def tissue_fraction(rgb_arr):
    flat = rgb_arr.reshape(-1, 3).astype(np.int32).sum(axis=1)
    return float((flat < 700).mean())


def chromogen_blob_count(rgb_arr, rb_thresh=CHROMOGEN_RB_THRESH, min_area=CHROMOGEN_MIN_BLOB_AREA):
    """Count contiguous red/brown-chromogen blobs (DAB or Fast-Red) in a
    tile, ignoring isolated speckle noise. Zero means no visible stain
    anywhere -- i.e. a color-verified true negative, regardless of whether
    the annotator got around to labeling that region."""
    r = rgb_arr[:, :, 0].astype(np.int32)
    b = rgb_arr[:, :, 2].astype(np.int32)
    mask = (r - b) > rb_thresh
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    lbl, n = ndimage.label(mask)
    if n == 0:
        return 0
    sizes = ndimage.sum(mask, lbl, range(1, n + 1))
    return int((sizes >= min_area).sum())


def clip_box_to_tile(b, x0, y0, tile):
    bx0, by0, bx1, by1 = b
    orig_area = max(bx1 - bx0, 0) * max(by1 - by0, 0)
    if orig_area <= 0:
        return None
    cx0, cy0, cx1, cy1 = max(bx0, x0), max(by0, y0), min(bx1, x0 + tile), min(by1, y0 + tile)
    if cx1 <= cx0 or cy1 <= cy0:
        return None
    clipped_area = (cx1 - cx0) * (cy1 - cy0)
    if clipped_area / orig_area < MIN_BOX_KEEP_FRAC:
        return None
    # normalized yolo coords, local to tile
    lx0, ly0, lx1, ly1 = cx0 - x0, cy0 - y0, cx1 - x0, cy1 - y0
    xc = (lx0 + lx1) / 2 / tile
    yc = (ly0 + ly1) / 2 / tile
    w = (lx1 - lx0) / tile
    h = (ly1 - ly0) / tile
    return xc, yc, w, h


def gen_tile_positions(region_poly, tile):
    """Yield (x0, y0) integer tile origins fully contained in region_poly,
    scanning per constituent rectangle-ish piece of the (possibly multi-part)
    region to keep the grid search cheap."""
    geoms = region_poly.geoms if region_poly.geom_type == "MultiPolygon" else [region_poly]
    seen = set()
    for g in geoms:
        minx, miny, maxx, maxy = g.bounds
        x = int(minx)
        while x + tile <= maxx + 1:
            y = int(miny)
            while y + tile <= maxy + 1:
                key = (x, y)
                if key not in seen:
                    seen.add(key)
                    tb = shapely_box(x, y, x + tile, y + tile)
                    if region_poly.covers(tb):
                        yield x, y
                y += STRIDE
            x += STRIDE


def slide_reader(image_path):
    return openslide.OpenSlide(image_path)


def read_tile_rgb(slide, x0, y0, w, h):
    region = slide.read_region((x0, y0), 0, (w, h)).convert("RGB")
    return np.array(region)


def safe_name(image_path):
    base = os.path.splitext(os.path.basename(image_path))[0]
    h = hashlib.md5(image_path.encode()).hexdigest()[:6]
    return f"{base}_{h}"


def process_image(image_path, annotations, split, out_dir, rng, stats):
    slide_tag = safe_name(image_path)
    stain = stain_type(image_path)
    plaque_boxes, roi_union, exclude_union, neg_patches, noisy = annotations
    if noisy:
        # confirmed (spot-checked) incomplete annotation in this annotator
        # batch: real plaques are missing even within the fallback bbox, so
        # an empty label here can't be trusted as a true negative.
        neg_patches = []

    stats["images"] += 1
    stats["plaque_boxes_total"] += len(plaque_boxes)

    if roi_union is None and not neg_patches:
        stats["images_no_region"] += 1
        return

    slide = slide_reader(image_path)
    W, H = slide.dimensions

    img_out = os.path.join(out_dir, "images", split)
    lbl_out = os.path.join(out_dir, "labels", split)
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    def boxes_for_tile(x0, y0, tile):
        yolo_lines = []
        for b in plaque_boxes:
            if b[2] <= x0 or b[0] >= x0 + tile or b[3] <= y0 or b[1] >= y0 + tile:
                continue
            clipped = clip_box_to_tile(b, x0, y0, tile)
            if clipped:
                xc, yc, w, h = clipped
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        return yolo_lines

    pos_tiles = 0
    neg_tiles = []       # candidate empty tile origins, subsampled later
    pos_candidates = []  # (x0, y0, lines) -- subsampled before the (expensive) pixel reads

    if roi_union is not None:
        for x0, y0 in gen_tile_positions(roi_union, TILE):
            tb = shapely_box(x0, y0, x0 + TILE, y0 + TILE)
            if exclude_union is not None and tb.intersects(exclude_union):
                continue
            lines = boxes_for_tile(x0, y0, TILE)
            if lines:
                pos_candidates.append((x0, y0, lines))
            else:
                neg_tiles.append((x0, y0))

    # no single slide (esp. the noisy ones with very dense raw box counts,
    # e.g. 1000+) should dominate the positive-tile pool via heavily
    # overlapping near-duplicate crops -- cap and randomly subsample.
    if len(pos_candidates) > MAX_POS_TILES_PER_IMAGE:
        pos_candidates = rng.sample(pos_candidates, MAX_POS_TILES_PER_IMAGE)

    for x0, y0, lines in pos_candidates:
        arr = read_tile_rgb(slide, x0, y0, TILE, TILE)
        if tissue_fraction(arr) < MIN_TISSUE_FRAC:
            continue
        name = f"{slide_tag}_pos_x{x0}_y{y0}"
        Image.fromarray(arr).save(os.path.join(img_out, name + ".jpg"), quality=95)
        with open(os.path.join(lbl_out, name + ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")
        pos_tiles += 1

    kept_neg = 0
    color_verified_neg = 0
    if not noisy:
        rng.shuffle(neg_tiles)
        max_neg = int(max(pos_tiles, 1) * NEG_TILE_TO_POS_TILE_RATIO) if pos_tiles > 0 else min(len(neg_tiles), 20)
        for x0, y0 in neg_tiles:
            if kept_neg >= max_neg:
                break
            arr = read_tile_rgb(slide, x0, y0, TILE, TILE)
            if tissue_fraction(arr) < MIN_TISSUE_FRAC:
                continue
            name = f"{slide_tag}_bg_x{x0}_y{y0}"
            Image.fromarray(arr).save(os.path.join(img_out, name + ".jpg"), quality=95)
            open(os.path.join(lbl_out, name + ".txt"), "w").close()
            kept_neg += 1
    elif stain == "brown_dab":
        # "no box" isn't trustworthy here, but "no chromogen signal at all"
        # is -- a missed plaque would still be visibly stained.
        rng.shuffle(neg_tiles)
        max_neg = int(max(pos_tiles, 1) * NEG_TILE_TO_POS_TILE_RATIO) if pos_tiles > 0 else min(len(neg_tiles), 20)
        budget = max_neg * CHROMOGEN_CANDIDATE_BUDGET_MULT
        for x0, y0 in neg_tiles[:budget]:
            if kept_neg >= max_neg:
                break
            arr = read_tile_rgb(slide, x0, y0, TILE, TILE)
            if tissue_fraction(arr) < MIN_TISSUE_FRAC:
                continue
            if chromogen_blob_count(arr) > 0:
                continue
            name = f"{slide_tag}_bgcolor_x{x0}_y{y0}"
            Image.fromarray(arr).save(os.path.join(img_out, name + ".jpg"), quality=95)
            open(os.path.join(lbl_out, name + ".txt"), "w").close()
            kept_neg += 1
            color_verified_neg += 1

    neg_patch_tiles = 0
    for (bx0, by0, bx1, by1) in neg_patches:
        x0, y0 = int(bx0), int(by0)
        tw, th = int(bx1 - bx0), int(by1 - by0)
        if tw < 32 or th < 32:
            continue
        x0c, y0c = max(0, x0), max(0, y0)
        tw = min(tw, W - x0c)
        th = min(th, H - y0c)
        arr = read_tile_rgb(slide, x0c, y0c, tw, th)
        if tissue_fraction(arr) < MIN_TISSUE_FRAC:
            continue
        name = f"{slide_tag}_negpatch_x{x0c}_y{y0c}"
        Image.fromarray(arr).save(os.path.join(img_out, name + ".jpg"), quality=95)
        open(os.path.join(lbl_out, name + ".txt"), "w").close()
        neg_patch_tiles += 1

    stats["tiles_pos"] += pos_tiles
    stats["tiles_bg"] += kept_neg
    stats["tiles_negpatch"] += neg_patch_tiles
    stats["per_image"].append({
        "image": image_path, "slide_tag": slide_tag, "split": split, "stain": stain, "noisy": noisy,
        "n_plaque_boxes": len(plaque_boxes), "tiles_pos": pos_tiles,
        "tiles_bg": kept_neg, "tiles_bg_color_verified": color_verified_neg, "tiles_negpatch": neg_patch_tiles,
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="amyloid-beta_manifest.tsv")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N images (for dry runs)")
    args = ap.parse_args()

    rng = random.Random(SEED)

    rows = list(csv.DictReader(open(args.manifest), delimiter="\t"))
    by_image = defaultdict(list)
    for r in rows:
        ip = r["Image_path"]
        if not ip or ip == "NA" or not os.path.exists(ip):
            continue
        by_image[ip].append(r["Json_path"])

    images = sorted(by_image.keys())
    if args.limit:
        images = images[:args.limit]

    print("Loading annotations for all images...", flush=True)
    annotations_by_image = {img: collect_image_annotations(img, by_image[img]) for img in images}
    noisy_images = {img for img, a in annotations_by_image.items() if a[4]}
    if noisy_images:
        print(f"{len(noisy_images)} image(s) flagged noisy (no-ROI fallback, incomplete GT) "
              f"-> positive-tiles-only, forced into train split:")
        for img in sorted(noisy_images):
            print(f"  {img}")

    by_stain = defaultdict(list)
    for img in images:
        by_stain[stain_type(img)].append(img)

    split_of = {}
    for stain, imgs in by_stain.items():
        clean_imgs = [i for i in imgs if i not in noisy_images]
        rng.shuffle(clean_imgs)
        n_val = max(1, round(len(clean_imgs) * VAL_FRACTION)) if clean_imgs else 0
        val_set = set(clean_imgs[:n_val])
        for img in imgs:
            split_of[img] = "val" if img in val_set else "train"

    stats = {"images": 0, "images_no_region": 0, "plaque_boxes_total": 0,
             "tiles_pos": 0, "tiles_bg": 0, "tiles_negpatch": 0, "per_image": []}

    for i, img in enumerate(images):
        split = split_of[img]
        print(f"[{i+1}/{len(images)}] {split:5s} {stain_type(img):11s} {img}", flush=True)
        process_image(img, annotations_by_image[img], split, args.out, rng, stats)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "data.yaml"), "w") as f:
        f.write(f"path: {os.path.abspath(args.out)}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write(f"nc: {len(CLASS_NAMES)}\n")
        f.write(f"names: {CLASS_NAMES}\n")

    with open(os.path.join(args.out, "build_report.tsv"), "w", newline="") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=["image", "slide_tag", "split", "stain", "noisy",
                                                            "n_plaque_boxes", "tiles_pos", "tiles_bg",
                                                            "tiles_bg_color_verified", "tiles_negpatch"])
        w.writeheader()
        for row in stats["per_image"]:
            w.writerow(row)

    print("\n=== Summary ===")
    print(f"Images processed: {stats['images']} (no usable region: {stats['images_no_region']})")
    print(f"Total plaque boxes (deduped): {stats['plaque_boxes_total']}")
    print(f"Tiles written: pos={stats['tiles_pos']} bg={stats['tiles_bg']} negpatch={stats['tiles_negpatch']}")


if __name__ == "__main__":
    main()
