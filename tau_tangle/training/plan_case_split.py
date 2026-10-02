"""Plan whole-patient splits from the audited inventory; never touch source data.

Patient independence was confirmed by the user on 2026-09-20. Object counts
remain unresolved until mask annotation semantics are confirmed. Consequently
this plan balances case, image, positive and background counts only.
"""
import argparse
import csv
import itertools
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-valid-only", action="store_true",
                        help="85%% train / 15%% validation; external test slides excluded")
    parser.add_argument("--coordinate-audit", type=Path,
                        help="Y-X exact-overlap annotation audit directory; otherwise keep original provisional coordinates")
    args = parser.parse_args()
    cases = sorted(
        csv.DictReader((args.audit / "candidate_case_summary.csv").open()),
        key=lambda row: row["candidate_case_id"],
    )
    patches = list(csv.DictReader((args.audit / "filename_manifest.csv").open()))
    coordinate_sources = set()
    if args.coordinate_audit:
        coordinate_summary = json.loads((args.coordinate_audit / "summary.json").read_text())
        assert coordinate_summary["coordinate_order"] == "yx"
        coordinate_sources = {
            r["source_id"] for r in csv.DictReader(
                (args.coordinate_audit / "aligned_patch_annotation_audit.csv").open())
            if r["exactly_equal_overlap_pixels"] == "True"
        }
    assert len(cases) == 18, "Reconfirm patient identity if the inventory changes"
    assert len(patches) == 8443
    ids = [r["candidate_case_id"] for r in cases]
    expected_ids = {
        "BU-P12691", "BU-P12699", "BU-P12700", "BU-P12701", "BU-P12702",
        "MS-P13648", "MS-P13654", "PART41997", "PART42018", "PART42020",
        "PART42021", "PART42022", "PART42023", "PART42024", "PART42025",
        "PART42026", "PART42027", "PART42029",
    }
    assert set(ids) == expected_ids
    values = np.array([
        [1, int(r["number_images"]), int(r["number_positive_images"]),
         int(r["number_background_images"])] for r in cases
    ], dtype=np.int64)
    totals = values.sum(axis=0)
    weights = np.array([0.25, 1.0, 1.0, 1.0])
    # Enumerate all 2–4 patient holdout subsets. All remaining patients train.
    subsets = [s for size in range(2, 5)
               for s in itertools.combinations(range(len(cases)), size)]
    rng = np.random.default_rng(42)
    rng.shuffle(subsets)  # Fixed tie order; the objective remains exhaustive.
    bits = np.array([sum(1 << i for i in s) for s in subsets], dtype=np.int64)
    sums = np.array([values[list(s)].sum(axis=0) for s in subsets])
    fractions = sums / totals
    holdout_losses = (((fractions - 0.15) / 0.15) ** 2 * weights).sum(axis=1)
    best = None
    evaluated = 0
    if args.train_valid_only:
        train_values = totals - sums
        eligible = ((sums[:, 2:] > 0).all(axis=1)
                    & (train_values[:, 2:] > 0).all(axis=1))
        losses = ((((train_values / totals - 0.85) / 0.85) ** 2) * weights).sum(axis=1)
        losses += holdout_losses
        losses[~eligible] = np.inf
        vi = int(np.argmin(losses))
        best = (float(losses[vi]), vi, None)
        evaluated = int(eligible.sum())
    for i, mask in enumerate(bits) if not args.train_valid_only else []:
        candidates = np.flatnonzero(((bits & mask) == 0) & (bits > mask))
        if not len(candidates):
            continue
        train_values = totals - sums[i] - sums[candidates]
        valid = (sums[i, 2:] > 0).all() & (sums[candidates, 2:] > 0).all(axis=1)
        valid &= (train_values[:, 2:] > 0).all(axis=1)
        candidates = candidates[valid]
        train_values = train_values[valid]
        if not len(candidates):
            continue
        losses = ((((train_values / totals - 0.7) / 0.7) ** 2) * weights).sum(axis=1)
        losses += holdout_losses[i] + holdout_losses[candidates]
        k = int(np.argmin(losses))
        evaluated += len(candidates)
        candidate = (float(losses[k]), i, int(candidates[k]))
        if best is None or candidate[0] < best[0] - 1e-12:
            best = candidate
    assert best is not None
    _, vi, ti = best
    if not args.train_valid_only and rng.integers(2):
        vi, ti = ti, vi  # Deterministic symmetric holdout assignment.
    assignments = {case: "train" for case in ids}
    split_indices = [("valid", vi)] if args.train_valid_only else [("valid", vi), ("test", ti)]
    split_names = ["train", "valid"] if args.train_valid_only else ["train", "valid", "test"]
    for split, index in split_indices:
        for i in subsets[index]:
            assignments[ids[i]] = split
    case_rows = []
    for r in cases:
        case_rows.append(dict(
            case_id=r["candidate_case_id"], split=assignments[r["candidate_case_id"]],
            number_images=int(r["number_images"]),
            number_positive_images=int(r["number_positive_images"]),
            number_background_images=int(r["number_background_images"]),
            number_objects="", raw_mask_components=int(r["raw_components"]),
            identity_status="user_confirmed_independent_patient",
            object_count_status="pending_annotation_semantics",
        ))
    for r in patches:
        r["split"] = assignments[r["case_id"]]
        r["parse_status"] = "patient_identity_confirmed_by_user_2026-09-20"
        r["slide_roi_status"] = (
            "not_encoded_in_PART_filename" if r["case_id"].startswith("PART")
            else "explicit_filename_identifiers"
        )
        if args.coordinate_audit:
            raw1, raw2 = r["x"], r["y"]
            r["raw_coordinate_1"], r["raw_coordinate_2"] = raw1, raw2
            if r["source_id"] in coordinate_sources:
                r["x"], r["y"] = raw2, raw1
                r["coordinate_status"] = "YX_order_and_unit_pixel_scale_supported_by_exact_overlap_within_source"
            else:
                r["x"], r["y"] = "", ""
                r["coordinate_status"] = "axis_order_or_pixel_scale_unverified_use_raw_fields"
    checks = {}
    for key in ["filename", "case_id", "slide_id", "roi_id", "source_id", "pixel_sha256"]:
        sets = {s: {r[key] for r in patches if r["split"] == s and r[key]}
                for s in split_names}
        for a, b in itertools.combinations(sets, 2):
            count = len(sets[a] & sets[b])
            assert count == 0, (key, a, b, count)
            checks[f"{key}:{a}/{b}"] = count
    summaries = []
    for split in split_names:
        selected = [r for r in patches if r["split"] == split]
        positive = sum(r["has_mask"] == "True" for r in selected)
        background = sum(r["positive_or_background"] == "background" for r in selected)
        assert positive + background == len(selected)
        summaries.append(dict(
            split=split, number_cases=len({r["case_id"] for r in selected}),
            number_images=len(selected), number_positive_images=positive,
            number_background_images=background, number_objects="",
            number_explicit_slides=len({r["slide_id"] for r in selected} - {""}),
            number_explicit_rois=len({r["roi_id"] for r in selected} - {""}),
            PART_sources_without_slide_roi_metadata=len({r["case_id"] for r in selected
                                                       if r["case_id"].startswith("PART")}),
            positive_to_background_ratio=positive / background,
        ))
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "proposed_case_split_manifest.csv", case_rows)
    write_csv(args.output / "confirmed_patient_patch_manifest.csv", patches)
    write_csv(args.output / "proposed_split_summary.csv", summaries)
    metadata = dict(
        status="proposed_only_pending_annotation_semantics",
        seed=42, algorithm="exhaustive_disjoint_2_to_4_patient_holdout_subsets",
        independent_patient_identity="explicitly confirmed by user on 2026-09-20",
        target_fractions=dict(train=0.7, valid=0.15, test=0.15),
        objective="sum over splits and metrics: weight*((observed_fraction-target)/target)**2",
        metric_weights=dict(cases=0.25, images=1, positives=1, backgrounds=1),
        object_counts="not inferred from raw components; unresolved",
        valid_test_symmetric_partitions_evaluated=evaluated,
        score=best[0], overlap_checks=checks,
        PART_slide_roi_counts="unknown, not fabricated from patient IDs",
        versions=dict(numpy=np.__version__),
    )
    if args.train_valid_only:
        metadata.update(
            algorithm="exhaustive_2_to_4_patient_validation_subsets",
            target_fractions=dict(train=0.85, valid=0.15),
            test_cohort="Separate user-held slides, excluded from this dataset and model selection",
            validation_partitions_evaluated=evaluated,
        )
        del metadata["valid_test_symmetric_partitions_evaluated"]
    if args.coordinate_audit:
        metadata["coordinate_audit"] = str(args.coordinate_audit.resolve())
        metadata["coordinate_policy"] = "Y-X on sources supported by exact image overlap; preserve raw coordinates and leave x/y blank elsewhere"
    (args.output / "plan_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(dict(output=str(args.output.resolve()), summary=summaries,
                          metadata=metadata), indent=2))


if __name__ == "__main__":
    main()
