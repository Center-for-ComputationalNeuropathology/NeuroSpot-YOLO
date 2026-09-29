#!/usr/bin/env python3
"""
STEP1_resolve_annotations.py - Consolidate multi-annotator, multi-protocol
annotations per slide into one clean label set, driven by the pTDP-43 manifest:
  data/pTDP43_manifest_all_slides.tsv

v2 (2026-09-16): this manifest is a "prefer 2026" cut of the original
comppath_500k manifest -- for NPBB400's 3 blocks, where an annotator (MH)
re-annotated the same slide in 2026 with a revised vocabulary, only the 2026
file is kept and the superseded 2025 file for that same slide+annotator is
dropped (an independent annotator's file for the same slide, e.g. BOP, with no
2026 re-annotation of its own, is left untouched). The other 9 slides have no
2026 file at all and are byte-identical to the original manifest's selection.
Net effect: 2,572 -> 2,463 resolved positive instances (-4.2%), confined
entirely to NPBB400. See data/pTDP43_manifest_all_slides.tsv for the exact
row-level diff against the original manifest at
/sc/arion/projects/comppath_500k/neuroFM_slides/CraryLab_Slides/Cell_Atlas/manifests/pTDP-43/pTDP-43_manifest.tsv

Design decisions (agreed with user, 2026-09-05):
- SINGLE merged class: every real positive box -- any morphology subtype
  (compact/ring/flame/lentiform/short_neurite), the old generic
  "pTDP-43_MONDO:0700038", AND "ghost_pTDP-43_MONDO:0700038" (uncertain calls)
  -- collapses to one class, "TDP-43" (id 0).
- Multiple annotation files for the same slide are UNIONED, not replaced.
  Different annotators/years catching different inclusions is treated as
  legitimate augmentation of the positive set.
- Byte-identical duplicate files (same content, different filename/date) are
  deduped by content hash BEFORE unioning, so a resave doesn't fake a second
  independent annotator's agreement.
- Near-duplicate positive boxes from different sources (same physical
  inclusion marked twice) are merged via IoU clustering so one object isn't
  double-counted as two ground-truth boxes.
- Explicit "roi" / "roi_test" boxes are trusted-negative regions: the
  annotator exhaustively reviewed that rectangle, so anywhere inside it with
  no positive box is a safe negative.
- Explicit "Negative Control" boxes are hard-negative distractor markers --
  small structures a pathologist looked at and explicitly rejected as
  non-TDP-43. These are a stronger negative signal than plain background.
- Elements with BOTH label and group empty are "ambiguous" and are reported,
  not auto-classified into any bucket.
"""

import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# Default manifest is now the "prefer 2026" cut (see module docstring). Still
# overridable via env vars, e.g. to resolve against the original upstream
# manifest for comparison, without touching the canonical resolved_annotations/.
MANIFEST_PATH = os.environ.get(
    "TDP43_MANIFEST_PATH",
    str(REPO_ROOT / "data" / "pTDP43_manifest_all_slides.tsv"),
)
OUTPUT_DIR = os.environ.get("TDP43_RESOLVED_OUTPUT_DIR", str(REPO_ROOT / "data" / "resolved_annotations"))

# IoU threshold above which two boxes (from different source files) are
# considered the same physical inclusion and merged into one.
POSITIVE_IOU_DEDUP_THRESHOLD = 0.3

# A rectangle with empty label AND group is only trusted as an implicit ROI
# if it's at least this large in BOTH dimensions (real ROI boxes in this
# dataset run ~1100-2300px; this stays well above that so we don't guess).
IMPLICIT_ROI_MIN_SIDE = 100000  # effectively "never auto-trust" -- see report

# Positive box area (px^2) below which an element is treated as a degenerate
# mis-click rather than a real annotation. Real boxes in this dataset have a
# 5th-percentile area of ~616px^2 (~24x24px); this stays far below that.
MIN_POSITIVE_BOX_AREA = 25


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_elements(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return d.get("annotation", {}).get("elements", [])


def element_to_box(e):
    cx, cy = e["center"][0], e["center"][1]
    w, h = e["width"], e["height"]
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


# v5 (2026-09-20): optional subtype exclusion, e.g. EXCLUDED_SUBTYPES=short_neurite
# to drop that morphology from training (user call: 26.6% recall vs ~85% for
# every other subtype, and at this project stage that tradeoff isn't worth
# chasing). Only catches subtypes recorded via Asees/BOP's detailed vocabulary
# (label or group containing the substring) -- NACC's generic/ghost labels
# never recorded a subtype at all, so short_neurite instances hiding inside
# those can't be identified or excluded this way. Comma-separated substrings,
# matched against label OR group (case-insensitive).
EXCLUDED_SUBTYPES = [s.strip().lower() for s in os.environ.get("TDP43_EXCLUDED_SUBTYPES", "").split(",") if s.strip()]


def classify_element(e):
    label = (e.get("label", {}).get("value", "") or "").strip().lower()
    group = str(e.get("group", "") or "").strip().lower()

    if label == "roi_test" or group == "roi_test":
        return "roi_test"
    if label == "roi" or group == "roi":
        return "roi"
    if label == "negative control" or group == "negative control":
        return "negative_control"
    if label == "" and group == "":
        return "ambiguous"
    if any(s in label or s in group for s in EXCLUDED_SUBTYPES):
        return "excluded_subtype"
    return "positive"  # every remaining label/group combo observed is a real inclusion mark


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / (a_area + b_area - inter)


def dedup_boxes_iou(boxes_with_source, threshold):
    """Greedy IoU clustering. boxes_with_source: list of (box, source_filename)."""
    kept = []  # list of {"box": box, "sources": [source, ...]}
    for box, source in boxes_with_source:
        merged = False
        for entry in kept:
            if iou(box, entry["box"]) >= threshold:
                entry["sources"].append(source)
                merged = True
                break
        if not merged:
            kept.append({"box": box, "sources": [source]})
    return kept


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    slides = defaultdict(list)  # image_path -> [json_path, ...]
    for row in rows:
        slides[row["Image_path"]].append(row["Json_path"])

    print(f"Manifest: {len(rows)} rows, {len(slides)} unique slides\n")

    summary_rows = []
    ambiguous_report = []

    for image_path, json_paths in sorted(slides.items()):
        slide_name = os.path.basename(image_path)

        # --- dedupe byte-identical annotation files ---
        seen_hashes = {}
        unique_json_paths = []
        for jp in json_paths:
            h = sha256_of_file(jp)
            if h in seen_hashes:
                print(f"[{slide_name}] skipping byte-identical duplicate: "
                      f"{os.path.basename(jp)} == {os.path.basename(seen_hashes[h])}")
                continue
            seen_hashes[h] = jp
            unique_json_paths.append(jp)

        roi_boxes, roi_test_boxes = [], []
        neg_control_boxes = []
        positive_raw = []  # (box, source_filename)
        ambiguous_raw = []
        degenerate_boxes = []

        for jp in unique_json_paths:
            source = os.path.basename(jp)
            for e in load_elements(jp):
                if e.get("type") != "rectangle":
                    continue
                try:
                    box = element_to_box(e)
                except KeyError:
                    continue
                kind = classify_element(e)
                if kind == "roi":
                    roi_boxes.append(box)
                elif kind == "roi_test":
                    roi_test_boxes.append(box)
                elif kind == "negative_control":
                    neg_control_boxes.append({"box": box, "source": source})
                elif kind == "ambiguous":
                    ambiguous_raw.append({"box": box, "source": source})
                elif kind == "excluded_subtype":
                    continue
                else:
                    area = (box[2] - box[0]) * (box[3] - box[1])
                    if area < MIN_POSITIVE_BOX_AREA:
                        degenerate_boxes.append({"box": box, "source": source, "area": area})
                    else:
                        positive_raw.append((box, source))

        positives = dedup_boxes_iou(positive_raw, POSITIVE_IOU_DEDUP_THRESHOLD)
        n_positive_raw = len(positive_raw)
        n_positive_deduped = len(positives)
        n_multi_source = sum(1 for p in positives if len(set(p["sources"])) > 1)

        has_trusted_negative_region = len(roi_boxes) > 0

        for amb in ambiguous_raw:
            w = amb["box"][2] - amb["box"][0]
            h = amb["box"][3] - amb["box"][1]
            ambiguous_report.append({
                "slide": slide_name, "source": amb["source"],
                "width": round(w), "height": round(h),
            })

        record = {
            "slide_name": slide_name,
            "image_path": image_path,
            "source_json_files": unique_json_paths,
            "roi_boxes": roi_boxes,
            "roi_test_boxes": roi_test_boxes,
            "positive_boxes": [{"box": p["box"], "sources": p["sources"]} for p in positives],
            "negative_control_boxes": neg_control_boxes,
            "ambiguous_elements": ambiguous_raw,
            "degenerate_boxes_dropped": degenerate_boxes,
            "has_trusted_negative_region": has_trusted_negative_region,
        }

        out_path = os.path.join(OUTPUT_DIR, f"{Path(slide_name).stem}.json")
        with open(out_path, "w") as f:
            json.dump(record, f, indent=2)

        summary_rows.append({
            "slide": slide_name,
            "n_files": len(unique_json_paths),
            "n_dup_files_skipped": len(json_paths) - len(unique_json_paths),
            "n_roi": len(roi_boxes),
            "n_roi_test": len(roi_test_boxes),
            "n_positive_raw": n_positive_raw,
            "n_positive_deduped": n_positive_deduped,
            "n_merged_from_multiple_sources": n_multi_source,
            "n_negative_control": len(neg_control_boxes),
            "n_ambiguous": len(ambiguous_raw),
            "trusted_negative_region": has_trusted_negative_region,
        })

    # --- print summary table ---
    print("\n" + "=" * 145)
    hdr = (f"{'slide':45s} {'files':>5s} {'dupSkip':>7s} {'roi':>4s} {'roi_test':>8s} "
           f"{'pos_raw':>7s} {'pos_dedup':>9s} {'merged':>6s} {'neg_ctrl':>8s} {'ambig':>5s} {'trusted_neg':>11s}")
    print(hdr)
    print("-" * 145)
    tot = defaultdict(int)
    for r in summary_rows:
        print(f"{r['slide']:45s} {r['n_files']:5d} {r['n_dup_files_skipped']:7d} "
              f"{r['n_roi']:4d} {r['n_roi_test']:8d} {r['n_positive_raw']:7d} "
              f"{r['n_positive_deduped']:9d} {r['n_merged_from_multiple_sources']:6d} "
              f"{r['n_negative_control']:8d} {r['n_ambiguous']:5d} {str(r['trusted_negative_region']):>11s}")
        for k in ("n_positive_raw", "n_positive_deduped", "n_merged_from_multiple_sources",
                  "n_negative_control", "n_ambiguous"):
            tot[k] += r[k]
    print("-" * 145)
    print(f"{'TOTAL':45s} {'':5s} {'':7s} {'':4s} {'':8s} {tot['n_positive_raw']:7d} "
          f"{tot['n_positive_deduped']:9d} {tot['n_merged_from_multiple_sources']:6d} "
          f"{tot['n_negative_control']:8d} {tot['n_ambiguous']:5d}")
    n_no_trusted_neg = sum(1 for r in summary_rows if not r["trusted_negative_region"])
    print(f"\nSlides with NO trusted negative (roi) region: {n_no_trusted_neg} / {len(summary_rows)}")

    if ambiguous_report:
        print("\n" + "=" * 80)
        print("AMBIGUOUS ELEMENTS (empty label AND group -- not auto-classified):")
        for a in ambiguous_report:
            print(f"  {a['slide']:40s} src={a['source']:35s} size={a['width']}x{a['height']}")
        print("These are excluded from both ROI and positive sets. Review manually if needed.")

    print(f"\nPer-slide resolved annotations written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
