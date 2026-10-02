#!/usr/bin/env python3
"""
Curate the a-syn Lewy body annotations into one clean, per-slide descriptor.

For every a-syn stained slide in the lewy_body manifest:
  - pick the single most complete annotation JSON (Girder DSA exports keep a
    "trash" doc plus the real one, and some slides have several export
    snapshots over time -- we keep the superset).
  - pull out Lewy body boxes (several historical group names all mean the
    same thing) as axis-aligned boxes in level-0 pixel coordinates, since
    the rectangles are frequently rotated by a degree or two.
  - pull out the pathologist-drawn ROI / ROI_to_test / Negative_Tiles boxes
    when present.
  - for slides that were never given an explicit ROI (the "Asees" multi-class
    exports and the simple "Lewy Bodies"-only exports), synthesize a
    pseudo-ROI: a padded square window around every positive annotation.
    This is a deliberate approximation -- see the manifest's `roi_source`
    column, which is "real" or "pseudo" per slide.

Output:
  dataset/slide_annotations/<slide_stem>.json   (one descriptor per slide)
  dataset/slide_manifest.tsv                    (one row per slide, summary)
"""

import json
import math
import os
import re

MANIFEST = os.environ.get("LEWY_MANIFEST", "lewy_body_manifest.tsv")  # TSV: TARGET, Image_path, Json_path

OUT_DIR = os.environ.get("LEWY_DATASET_DIR", "dataset")

ANN_OUT_DIR = os.path.join(OUT_DIR, "slide_annotations")
SUMMARY_OUT = os.path.join(OUT_DIR, "slide_manifest.tsv")

# Groups that mean "this box is a Lewy body" (case/whitespace normalized).
POSITIVE_GROUPS = {
    "lewy_body_bto:0000754",
    "lewy body bto:0000754",
    "lewy bodies",
    "a-syn_lewy_body",
    "a-syn_cortical_lewy_body",
    "a-syn_sn-type_lewy_body",
}

# Groups that are real a-syn pathology but NOT a Lewy body -- excluded from
# the positive class and NOT treated as background either (we skip tiles
# that are dominated by these rather than mislabel them as negative).
IGNORE_GROUPS = {
    "pale bodies",
    "a-syn_lewy_neurite",
    "a-syn_dystrophic_neurite",
    "a-syn_glial_cytoplasmic",
    "a-syn_glial_nuclear",
    "a-syn_neuronal_nuclear_round",
    "a-syn_neuronal_nuclear_ring",
}

ROI_GROUPS = {"roi"}
# "ROI_to_test" is a reserved/held-out region: across every NACC slide it
# contains zero Lewy body boxes even though positives exist elsewhere on the
# same slide, so it was never exhaustively annotated for this class. It is
# NOT a valid source of confirmed-negative tiles -- treat it as unreviewed
# and skip it entirely (don't sample tiles from it either way).
ROI_TO_TEST_GROUPS = {"roi_to_test"}
NEG_TILE_GROUPS = {"negative_tiles"}

# Half-width (px, level-0) of the synthesized pseudo-ROI window drawn around
# each positive annotation on slides with no real ROI. ~660um across at
# 0.26 um/px -- several visual fields, matches how these were annotated.
PSEUDO_ROI_HALF = 1250


def norm_group(g):
    return re.sub(r"\s+", " ", (g or "").strip()).lower()


def rect_to_aabb(el):
    """Axis-aligned bounding box (x1,y1,x2,y2) for a possibly-rotated
    Girder/HistomicsUI rectangle element, in level-0 pixel coords."""
    cx, cy = el["center"][0], el["center"][1]
    w, h = el["width"], el["height"]
    theta = el.get("rotation", 0) or 0

    hw, hh = w / 2.0, h / 2.0
    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    xs, ys = [], []
    for dx, dy in corners:
        rx = dx * cos_t - dy * sin_t
        ry = dx * sin_t + dy * cos_t
        xs.append(cx + rx)
        ys.append(cy + ry)

    return min(xs), min(ys), max(xs), max(ys)


def load_best_doc(json_path):
    """Return the annotation dict {name, elements} with the most elements,
    handling both the flat single-doc format and the Girder list-of-docs
    format (which usually includes an empty 'trash' doc)."""
    data = json.load(open(json_path))
    if isinstance(data, dict):
        return data.get("annotation", {})
    best = {}
    for item in data:
        ann = item.get("annotation", {})
        if len(ann.get("elements", [])) > len(best.get("elements", [])):
            best = ann
    return best


def pick_annotation_file(json_paths):
    """Among the candidate annotation JSONs for one slide, keep the one
    with the most total elements (a strict superset in every case we
    inspected: the '6765ce4*' hash exports supersede the older '671933*'
    exports and the flat *_2024.json renames)."""
    best_path, best_doc, best_n = None, None, -1
    for p in json_paths:
        doc = load_best_doc(p)
        n = len(doc.get("elements", []))
        if n > best_n:
            best_path, best_doc, best_n = p, doc, n
    return best_path, best_doc


def slide_stem(slide_path):
    return os.path.splitext(os.path.basename(slide_path))[0]


def extract_case_id(slide_path, stem):
    """Patient/case identifier, used to keep all of a patient's slides
    (different blocks/sections) in the same train/val split. Prefer the
    parent directory name (e.g. .../NPBB47/MA18-77_AS_B25_Duplicate.svs ->
    "NPBB47", which the filename alone doesn't reveal); fall back to a
    leading NACC/NPBB token parsed from the stem."""
    parent = os.path.basename(os.path.dirname(slide_path))
    if re.match(r"^(NACC\d+|NPBB[\d.]+)$", parent, re.IGNORECASE):
        return parent
    m = re.match(r"^(NACC\d+|NPBB[\d.]+)", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    return stem


def main():
    os.makedirs(ANN_OUT_DIR, exist_ok=True)

    # slide_path -> list of json_paths, restricted to a-syn stain slides
    by_slide = {}
    with open(MANIFEST) as f:
        next(f)
        for line in f:
            target, image_path, json_path = line.rstrip("\n").split("\t")
            if image_path == "NA":
                continue
            fname = os.path.basename(json_path)
            is_asyn = bool(re.search(r"a-syn|_AS_", fname, re.IGNORECASE))
            if not is_asyn:
                continue
            by_slide.setdefault(image_path, []).append(json_path)

    summary_rows = []

    for slide_path in sorted(by_slide):
        json_path, doc = pick_annotation_file(by_slide[slide_path])
        elements = doc.get("elements", [])

        positives, roi_rects, neg_tile_rects, ignored_n = [], [], [], 0
        n_roi_to_test = 0

        for el in elements:
            if el.get("type") != "rectangle":
                continue
            g = norm_group(el.get("group"))
            aabb = rect_to_aabb(el)

            if g in POSITIVE_GROUPS:
                positives.append(aabb)
            elif g in ROI_GROUPS:
                roi_rects.append(aabb)
            elif g in ROI_TO_TEST_GROUPS:
                n_roi_to_test += 1  # tracked only -- never sampled from
            elif g in NEG_TILE_GROUPS:
                neg_tile_rects.append(aabb)
            elif g in IGNORE_GROUPS:
                ignored_n += 1
            # anything else (unlabeled/other) is silently skipped

        roi_source = "real"
        if not roi_rects:
            roi_source = "pseudo"
            for (x1, y1, x2, y2) in positives:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                roi_rects.append(
                    (
                        cx - PSEUDO_ROI_HALF,
                        cy - PSEUDO_ROI_HALF,
                        cx + PSEUDO_ROI_HALF,
                        cy + PSEUDO_ROI_HALF,
                    )
                )

        stem = slide_stem(slide_path)
        source = "NACC" if stem.upper().startswith("NACC") else "NPBB"
        case_id = extract_case_id(slide_path, stem)

        descriptor = {
            "slide_path": slide_path,
            "slide_stem": stem,
            "case_id": case_id,
            "source": source,
            "annotation_json": json_path,
            "roi_source": roi_source,
            "positives": positives,
            "roi_rects": roi_rects,
            "negative_tile_rects": neg_tile_rects,
            "n_ignored_elements": ignored_n,
            "n_roi_to_test_elements": n_roi_to_test,
        }

        out_path = os.path.join(ANN_OUT_DIR, f"{stem}.json")
        with open(out_path, "w") as f:
            json.dump(descriptor, f, indent=2)

        summary_rows.append(
            (
                stem,
                case_id,
                source,
                roi_source,
                len(positives),
                len(roi_rects),
                len(neg_tile_rects),
                ignored_n,
                n_roi_to_test,
                slide_path,
                json_path,
            )
        )

    with open(SUMMARY_OUT, "w") as f:
        f.write(
            "slide_stem\tcase_id\tsource\troi_source\tn_positive_boxes\t"
            "n_roi_rects\tn_negative_tiles\tn_ignored_elements\t"
            "n_roi_to_test_elements\tslide_path\tannotation_json\n"
        )
        for row in summary_rows:
            f.write("\t".join(str(x) for x in row) + "\n")

    print(f"Wrote {len(summary_rows)} slide descriptors to {ANN_OUT_DIR}")
    print(f"Summary: {SUMMARY_OUT}")
    total_pos = sum(r[4] for r in summary_rows)
    print(f"Total Lewy body boxes: {total_pos}")
    n_real = sum(1 for r in summary_rows if r[3] == "real")
    n_pseudo = sum(1 for r in summary_rows if r[3] == "pseudo")
    print(f"Slides with real ROI: {n_real} | pseudo ROI: {n_pseudo}")
    n_cases = len(set(r[1] for r in summary_rows))
    print(f"Unique cases: {n_cases}")


if __name__ == "__main__":
    main()
