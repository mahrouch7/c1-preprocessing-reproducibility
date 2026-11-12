"""Aggregate completed outer-fold predictions without selecting a winning seed."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
from .data import LABELS, load_source, resolve_input
from .engine import job_definitions, verify_receipt
from .io import digest, read_json, require, sha256, write_csv, write_json
from .metrics import score

INDEX_SHA = "4297577e85d1e95f7175a7af21cfcd7b91a29fbb39de6d20205ccbfacafac9ea"
METRICS = ("macro_f1", "balanced_accuracy", "mcc", "healthy_false_alarm_rate", "missed_fault_rate", "brier", "log_loss")


def aggregate(output, destination, *, allow_partial=False):
    output, destination = Path(output).resolve(), Path(destination).resolve()
    require(not destination.exists(), "Analysis destination must be new; never overwrite an earlier interpretation")
    manifest = read_json(output / "run_manifest.json")
    config, profile = manifest["config"], manifest["profile"]
    rows, _ = load_source(config, verify_rows=False)
    expected = list(job_definitions(config, profile))
    missing, groups, records = [], defaultdict(list), []
    for job in expected:
        directory = output / "jobs" / job["training_condition"] / f"fold{job['fold']:02d}" / f"seed{job['seed']}"
        if not (directory / "completed.json").exists():
            missing.append(job)
            continue
        identity_hash = digest(read_json(directory / "identity.json"))
        receipt = verify_receipt(directory, identity_hash)
        require(receipt["fit_procedures"] == 2, "Unexpected fit accounting")
        for report in read_json(directory / "prediction_catalogue.json"):
            condition = report["condition"]
            # One clean reference; no duplicate clean-trained/noise-matched row.
            exposure = "shared_clean" if condition == "clean" else (
                "clean_trained" if job["training_condition"] == "clean" else "noise_matched")
            key = (job["seed"], exposure, condition)
            path = directory / "predictions" / report["prediction_file"]
            require(sha256(path) == report["prediction_sha256"], "Prediction digest changed")
            with np.load(path, allow_pickle=False) as data:
                ids, p = data["row_ids"].copy(), data["probabilities"].copy()
                require(set(ids.tolist()) == {r["Id"] for r in rows if r["fold"] == job["fold"]}, "Outer test identities changed")
                require(np.array_equal(data["truth"], [LABELS.index(rows[int(i)]["fault"]) for i in ids]), "Saved truth labels changed")
                require(np.array_equal(data["predicted"], p.argmax(axis=1)), "Saved probability decision changed")
                groups[key].append((ids, p))
    require(not missing or allow_partial, f"{len(missing)} jobs missing; use --allow-partial only for progress inspection")
    destination.mkdir(parents=True)
    (destination / "oof").mkdir()
    pooled = {}
    for (seed, exposure, condition), chunks in sorted(groups.items()):
        ids = np.concatenate([x[0] for x in chunks])
        p = np.concatenate([x[1] for x in chunks])
        order = np.argsort(ids)
        ids, p = ids[order], p[order]
        require(len(set(ids.tolist())) == len(ids), "An acquisition is repeated across outer predictions")
        complete = np.array_equal(ids, np.arange(len(rows)))
        if not complete:
            continue  # Never mislabel one-fold F1 as the full-cohort estimand.
        result = score(rows, ids, p)
        name = f"seed{seed}_{exposure}_{condition}"
        with (destination / "oof" / (name + ".npz")).open("xb") as stream:
            np.savez_compressed(stream, row_ids=ids, probabilities=p,
                                truth=np.array([LABELS.index(r["fault"]) for r in rows]))
        write_json(destination / "oof" / (name + ".json"), result)
        record = {"seed": seed, "exposure": exposure, "condition": condition,
                  **{key: result[key] for key in METRICS}}
        records.append(record)
        pooled[(seed, exposure, condition)] = result
    if records:
        write_csv(destination / "metrics_per_seed.csv", records)
    summary, by_case = [], defaultdict(list)
    for item in records:
        by_case[(item["exposure"], item["condition"])].append(item)
    for (exposure, condition), items in sorted(by_case.items()):
        row = {"exposure": exposure, "condition": condition, "initializations": len(items)}
        for metric in METRICS:
            values = [item[metric] for item in items]
            row[metric + "_mean"] = float(np.mean(values))
            row[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        summary.append(row)
    if summary:
        write_csv(destination / "metrics_across_initializations.csv", summary)
    comparisons = baseline_comparisons(config, rows, pooled)
    if comparisons:
        write_csv(destination / "paired_descriptive_differences_vs_E32.csv", comparisons)
    status = {"status": "PARTIAL_NOT_PUBLICATION_READY" if missing else "ALL_EXPECTED_JOBS_COMPLETE",
              "missing_jobs": missing, "completed_jobs": len(expected) - len(missing),
              "expected_jobs": len(expected), "complete_oof_rows": len(records),
              "estimand": "pooled family/class-standardized; metrics are fractions, differences are percentage points",
              "seed_policy": "all declared initializations reported separately; no best-seed selection or probability ensemble",
              "sd_scope": "initialization variability, not a family-bootstrap confidence interval",
              "inference": "descriptive only; no new p-values or superiority claims; extension of the 728 E32 contrasts is NOT automatic",
              "run_manifest_sha256": sha256(output / "run_manifest.json")}
    write_json(destination / "analysis_receipt.json", status)
    print(f"[ANALYSIS] {status['status']}; destination={destination}", flush=True)
    return status


def baseline_comparisons(config, rows, pooled):
    """All 24 pre-existing unfiltered core pipelines, no test-based winner choice."""
    if not pooled:
        return []
    root = resolve_input(config, "baseline_root") / "final_analysis"
    index = root / "pooled_prediction_index.json"
    require(sha256(index) == INDEX_SHA, "Pinned E32 prediction index changed")
    indexed = read_json(index)
    require(isinstance(indexed, dict) and bool(indexed), "Expected the pinned case-id keyed E32 index mapping")
    require(all(isinstance(entry, dict) and {"case", "path", "sha256", "eligible_for_inference"} <= set(entry)
                for entry in indexed.values()), "Invalid E32 index entry schema")
    entries = [{**entry, "case_id": case_id} for case_id, entry in indexed.items()]
    candidates = []
    for entry in entries:
        case = entry["case"]
        if (case["study"] == "robustness" and case["preprocessor"] == "none"
                and case["representation"] in ("RMS6", "RAW12", "MULTI18")
                and not case["anchor_only"] and not case["density_balanced"]):
            candidates.append(entry)
    result, scores = [], {}
    for (seed, exposure, condition), inception in sorted(pooled.items()):
        protocol = "matched" if exposure == "noise_matched" else "deployment"
        selected = [e for e in candidates if e["case"]["condition"] == condition and e["case"]["protocol"] == protocol]
        require(len(selected) == 24, "Expected 24 fixed E32 core comparators")
        for entry in selected:
            path = (root / entry["path"]).resolve()
            require(root in path.parents, "Unsafe baseline prediction path")
            if entry["sha256"] not in scores:
                require(sha256(path) == entry["sha256"], "Baseline prediction bytes changed")
                with np.load(path, allow_pickle=False) as data:
                    ids, p = data["row_ids"], data["probabilities"]
                    require(np.array_equal(ids, np.arange(len(rows))), "Baseline acquisition order changed")
                    scores[entry["sha256"]] = score(rows, ids, p)
            case, baseline = entry["case"], scores[entry["sha256"]]
            result.append({"seed": seed, "exposure": exposure, "condition": condition,
                "baseline_case_id": entry["case_id"],
                "baseline_classifier": case["classifier"], "baseline_representation": case["representation"],
                "baseline_normalizer": case["normalizer"],
                "inception_macro_f1": inception["macro_f1"], "baseline_macro_f1": baseline["macro_f1"],
                "difference_macro_f1_pp": 100 * (inception["macro_f1"] - baseline["macro_f1"]),
                "inception_balanced_accuracy": inception["balanced_accuracy"], "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                "difference_balanced_accuracy_pp": 100 * (inception["balanced_accuracy"] - baseline["balanced_accuracy"]),
                "inception_mcc": inception["mcc"], "baseline_mcc": baseline["mcc"],
                "difference_mcc": inception["mcc"] - baseline["mcc"],
                "E32_inferential_eligibility": entry["eligible_for_inference"],
                "status": "DESCRIPTIVE_PAIRED_DIFFERENCE_NOT_A_NEW_SIGNIFICANCE_TEST"})
    return result
