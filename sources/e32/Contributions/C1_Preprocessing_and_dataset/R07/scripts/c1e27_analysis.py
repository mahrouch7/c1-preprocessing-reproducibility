"""Complete-OOF R03 inference and figures, adapted to current acquisition IDs."""
from __future__ import annotations

import csv
from html import escape
import io
import shutil
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import beta, spearmanr
import c1e25 as base
import c1e27_spec as spec
import c1e27_signal as signal


def write_csv(path, rows):
    if not rows:
        raise ValueError("Refuse an empty scientific table")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def confusion(truth, predicted):
    truth, predicted = np.asarray(truth, dtype=int), np.asarray(predicted, dtype=int)
    if truth.shape != predicted.shape or np.any(truth < 0) or np.any(truth >= 12) or np.any(predicted < 0) or np.any(predicted >= 12):
        raise ValueError("Invalid encoded labels")
    return np.bincount(12 * truth + predicted, minlength=144).reshape(12, 12)


def score_matrix(matrix):
    m = np.asarray(matrix, dtype=float)
    diagonal = np.diagonal(m, axis1=-2, axis2=-1)
    actual, guessed = m.sum(axis=-1), m.sum(axis=-2)
    n = actual.sum(axis=-1)
    f1 = np.divide(2 * diagonal, actual + guessed, out=np.zeros_like(diagonal), where=(actual + guessed) > 0)
    recalls = np.divide(diagonal, actual, out=np.zeros_like(diagonal), where=actual > 0)
    trace = diagonal.sum(axis=-1)
    denominator = np.sqrt(np.maximum(0, (n * n - np.square(actual).sum(axis=-1)) *
                                      (n * n - np.square(guessed).sum(axis=-1))))
    mcc = np.divide(trace * n - (actual * guessed).sum(axis=-1), denominator,
                    out=np.zeros_like(n), where=denominator > 0)
    return {"accuracy": trace / n, "macro_f1": f1.mean(axis=-1),
            "weighted_f1": (f1 * actual).sum(axis=-1) / n, "balanced_accuracy": recalls.mean(axis=-1), "mcc": mcc}


def cluster_matrices(truth, predicted, groups):
    labels, index = np.unique(groups, return_inverse=True)
    result = np.zeros((len(labels), 144), dtype=float)
    np.add.at(result, (index, np.asarray(truth) * 12 + np.asarray(predicted)), 1)
    return result


def cluster_interval(matrices, repetitions, seed):
    rng = np.random.Generator(np.random.PCG64(seed))
    g = len(matrices)
    values = []
    for start in range(0, repetitions, 250):
        weights = rng.multinomial(g, np.full(g, 1 / g), size=min(250, repetitions - start))
        values.extend(score_matrix((weights @ matrices).reshape(-1, 12, 12))["macro_f1"].tolist())
    low, high = np.quantile(values, [.025, .975], method="linear")
    return {"lower_95": float(low), "upper_95": float(high)}


def paired_inference(left, right, groups, bootstrap_reps=10000, swaps=50000, seed=42):
    if not np.array_equal(left["truth"], right["truth"]) or not np.array_equal(left["row_ids"], right["row_ids"]):
        raise ValueError("Paired OOF identities or truths differ")
    a = cluster_matrices(left["truth"], left["predicted"], groups)
    b = cluster_matrices(right["truth"], right["predicted"], groups)
    observed = float(score_matrix(a.sum(axis=0).reshape(12, 12))["macro_f1"] -
                     score_matrix(b.sum(axis=0).reshape(12, 12))["macro_f1"])
    rng = np.random.Generator(np.random.PCG64(seed))
    g, differences, exceedances = len(a), [], 0
    if np.array_equal(a, b):
        return {"difference": observed, "lower_95": 0.0, "upper_95": 0.0, "permutation_p": 1.0,
                "p_mc_lower_99": 1.0, "p_mc_upper_99": 1.0, "groups": g, "swaps": swaps}
    for start in range(0, bootstrap_reps, 250):
        weights = rng.multinomial(g, np.full(g, 1 / g), size=min(250, bootstrap_reps - start))
        sa = score_matrix((weights @ a).reshape(-1, 12, 12))["macro_f1"]
        sb = score_matrix((weights @ b).reshape(-1, 12, 12))["macro_f1"]
        differences.extend((sa - sb).tolist())
    total = a.sum(axis=0) + b.sum(axis=0)
    for start in range(0, swaps, 500):
        weights = rng.integers(0, 2, size=(min(500, swaps - start), g))
        pa = b.sum(axis=0) + weights @ (a - b)
        null = score_matrix(pa.reshape(-1, 12, 12))["macro_f1"] - score_matrix((total - pa).reshape(-1, 12, 12))["macro_f1"]
        exceedances += int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-14))
    low, high = np.quantile(differences, [.025, .975], method="linear")
    pl = 0.0 if exceedances == 0 else float(beta.ppf(.005, exceedances, swaps - exceedances + 1))
    pu = 1.0 if exceedances == swaps else float(beta.ppf(.995, exceedances + 1, swaps - exceedances))
    return {"difference": observed, "lower_95": float(low), "upper_95": float(high),
            "permutation_p": (1 + exceedances) / (swaps + 1), "p_mc_lower_99": (1 + swaps * pl) / (swaps + 1),
            "p_mc_upper_99": (1 + swaps * pu) / (swaps + 1), "groups": g, "swaps": swaps}


def holm(values):
    values = np.asarray(values, dtype=float)
    if np.any(values < 0) or np.any(values > 1) or not np.isfinite(values).all():
        raise ValueError("Invalid multiplicity input")
    order = np.argsort(values, kind="stable")
    result = np.empty_like(values)
    result[order] = np.minimum(1, np.maximum.accumulate((len(values) - np.arange(len(values))) * values[order]))
    return result


def robust_case(model, condition, representation, prep="none"):
    return {"study": "robustness", "protocol": "deployment", "condition": condition,
            "preprocessor": prep, "representation": representation, "classifier": model,
            "normalizer": spec.default_normalizer(model)}


def comparison_plan():
    result = []
    def add(family, left, right):
        result.append({"family": family, "left": left, "right": right,
                       "comparison_id": spec.digest({"family": family, "left": left, "right": right})})
    for model in spec.MODELS:
        for noise in spec.NOISES:
            c = noise + "__10"
            for left, right, family in (("MULTI18", "RAW12", "early_vs_RAW"), ("MULTI18", "RMS6", "early_vs_RMS"),
                    ("LATE_SIMPLE", "RAW12", "late_simple_vs_RAW"), ("LATE_SIMPLE", "RMS6", "late_simple_vs_RMS")):
                add(family, robust_case(model, c, left), robust_case(model, c, right))
            for prep in spec.PREPROCESSORS[1:]:
                add("denoiser_vs_none", robust_case(model, c, "MULTI18", prep), robust_case(model, c, "MULTI18"))
        if model in spec.EXTRA_MODELS:
            for c in spec.EXTRA_CONDITIONS[1:]:
                for extended, original in (("RAW_PLUS", "RAW12"), ("MULTI_PLUS", "MULTI18")):
                    left = {**robust_case(model, c, extended), "study": "raw_plus"}
                    add("RAW_PLUS_vs_RAW" if extended == "RAW_PLUS" else "MULTI_PLUS_vs_MULTI18", left, robust_case(model, c, original))
        for representation in base.REPRESENTATIONS:
            case = {**robust_case(model, "clean", representation), "study": "normalization", "protocol": "normalization"}
            for normalizer in spec.NORMALIZERS[1:]:
                add("normalizer_vs_none", {**case, "normalizer": normalizer}, {**case, "normalizer": "none"})
    if len(result) != 488 or len({r["comparison_id"] for r in result}) != 488:
        raise ValueError("Frozen inferential family count changed")
    return sorted(result, key=lambda r: r["comparison_id"])


def expected_cases():
    cases = {}
    from c1e27_worker import prediction_case
    for unit in spec.learning_plan():
        for job in spec.jobs(unit["preprocessor"], unit["fold"], unit["classifier"]):
            for protocol, condition in spec.evaluation_cases(job):
                case = prediction_case(job, protocol, condition)
                cases[spec.digest(case)] = case
        for protocol in ("deployment", "matched"):
            for condition in spec.CONDITIONS:
                for name in ("LATE_SIMPLE", "LATE_ORACLE_SNR"):
                    case = {**robust_case(unit["classifier"], condition, name, unit["preprocessor"]), "protocol": protocol}
                    cases[spec.digest(case)] = case
    return cases


def read_pooled(output, entry):
    with np.load(output / entry["path"], allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def aggregate(assignment, transport, prepared, execution, output):
    from run_c1e25_pipeline import returned_archive
    rows = base.validate_metadata(base.read_json(prepared / "metadata.json"))
    output.mkdir(exist_ok=False)
    (output / "oof").mkdir()
    groups, fits, reused = {}, 0, 0
    for ordinal, unit in enumerate(assignment["units"]):
        archive = returned_archive(unit, assignment, transport)
        expected_ids = np.array([r["Id"] for r in rows if r["fold"] == unit["identity"]["fold"]])
        with zipfile.ZipFile(archive) as bundle:
            records = __import__("json").loads(bundle.read("fit_records.json"))
            if records["fits_completed"] != unit["expected_fits"]:
                raise ValueError("Returned fit count changed")
            fits += records["fits_completed"]
            reused += records["reused_fits"]
            entries = __import__("json").loads(bundle.read("prediction_catalogue.json"))
            for entry in entries:
                case = entry["case"]
                if any(case[k] != unit["identity"][k] for k in ("preprocessor", "classifier")):
                    raise ValueError("OOF case differs from unit identity")
                with np.load(io.BytesIO(bundle.read(entry["path"])), allow_pickle=False) as data:
                    piece = {k: data[k] for k in data.files}
                if not np.array_equal(piece["row_ids"], expected_ids):
                    raise ValueError("Fold/OOF identities changed")
                key = spec.digest(case)
                group = groups.setdefault(key, {"case": case, "pieces": [], "eligible_for_inference": True})
                group["eligible_for_inference"] &= entry["eligible_for_inference"] is True
                group["pieces"].append(piece)
        if ordinal % 10 == 0:
            print(f"[AGGREGATION] {ordinal}/{len(assignment['units'])} verified learning returns", flush=True)
    expected = expected_cases()
    if fits != 16240 or reused != 240 or set(groups) != set(expected):
        raise ValueError("Incomplete scientific budget or prediction coverage")
    pooled, summary, classes = {}, [], []
    truth = np.array([base.LABELS.index(r["fault"]) for r in rows])
    for key, group in sorted(groups.items()):
        if len(group["pieces"]) != 5:
            raise ValueError("Pooled case does not have five distinct folds")
        data = {k: np.concatenate([p[k] for p in group["pieces"]]) for k in group["pieces"][0]}
        order = np.argsort(data["row_ids"])
        data = {k: v[order] for k, v in data.items()}
        if not np.array_equal(data["row_ids"], np.arange(len(rows))) or not np.array_equal(data["truth"], truth):
            raise ValueError("Pooled acquisition/label identity mismatch")
        if not np.isfinite(data["probabilities"]).all() or not np.allclose(data["probabilities"].sum(axis=1), 1, atol=1e-10, rtol=0):
            raise ValueError("Pooled probability contract failed")
        m = confusion(truth, data["predicted"])
        path = "oof/" + key + ".npz"
        with (output / path).open("xb") as stream:
            np.savez_compressed(stream, **data, confusion=m)
        pooled[key] = {"case": group["case"], "path": path, "sha256": base.sha(output / path),
                       "eligible_for_inference": group["eligible_for_inference"]}
        metrics = {name: float(value) for name, value in score_matrix(m).items()}
        abc = np.isin(truth, [base.LABELS.index("abc"), base.LABELS.index("abcg")])
        summary.append({**group["case"], **metrics, "abc_abcg_errors": int(np.count_nonzero(data["predicted"][abc] != truth[abc])),
                        "predicted_class_count": int(np.count_nonzero(m.sum(axis=0))),
                        "max_prediction_fraction": float(m.sum(axis=0).max() / len(truth)),
                        "eligible_for_inference": group["eligible_for_inference"]})
        for label, name in enumerate(base.LABELS):
            tp, support, predicted_count = m[label, label], m[label].sum(), m[:, label].sum()
            classes.append({**group["case"], "label": name, "support": int(support), "eligible_for_inference": group["eligible_for_inference"],
                "precision": float(tp / predicted_count) if predicted_count else 0.0,
                "recall": float(tp / support) if support else 0.0,
                "f1": float(2 * tp / (support + predicted_count)) if support + predicted_count else 0.0})
    for model in spec.MODELS:
        for representation in base.REPRESENTATIONS:
            original = pooled[spec.digest(robust_case(model, "clean", representation))]
            case = {**original["case"], "study": "normalization", "protocol": "normalization"}
            pooled[spec.digest(case)] = {**original, "case": case, "alias_of": spec.digest(original["case"])}
            metric_row = next(r for r in summary if all(r[k] == original["case"][k] for k in original["case"]))
            summary.append({**metric_row, **case})
    write_csv(output / "summary_metrics.csv", summary)
    with (prepared / "base/reused_e25_summary.csv").open(encoding="utf-8", newline="") as stream:
        prior = list(csv.DictReader(stream))
    if len(prior) != 72:
        raise ValueError("E25 retained-result matrix changed")
    for row in prior:
        condition = "clean" if row["condition"] == "clean" else "awgn__" + row["condition"][4:]
        case = robust_case(row["classifier"], condition, row["representation"], row["preprocessor"])
        current = [r for r in summary if all(r[k] == v for k, v in case.items())]
        if len(current) != 1 or abs(current[0]["macro_f1"] - float(row["macro_f1"])) > 1e-12:
            raise ValueError("Reused E25 result changed; do not silently replace the prior result")
    shutil.copyfile(prepared / "base/reused_e25_primary_endpoint.json", output / "prespecified_primary_from_E25.json")
    write_csv(output / "per_class_metrics.csv", classes)
    base.write_json(output / "pooled_catalogue.json", pooled)
    run_statistics(pooled, rows, output)
    tpi_summary(execution, output, summary)
    descriptor_shift(execution, output)
    draw_curves(summary, output)
    draw_normalization(summary, output)
    files = {p.relative_to(output).as_posix(): base.sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    exclusions = sum(not e["eligible_for_inference"] for e in pooled.values())
    base.write_json(output / "analysis_receipt.json", {"status": "COMPLETE_WITH_NUMERICAL_EXCLUSIONS" if exclusions else "COMPLETE_C1_E27_NOT_REVIEWER_CLOSURE",
        "nonconverged_pooled_cases_excluded": exclusions, "reviewer_items_closed": 0,
        "assignment_sha256": assignment["assignment_sha256"], "scope_sha256": spec.digest(spec.scope()),
        "metadata_sha256": base.sha(prepared / "metadata.json"), "labels": base.LABELS,
        "statistics_plan": spec.STATISTICS,
        "new_fits": fits, "reused_fits": reused, "pooled_cases": len(pooled), "files": files,
        "scientific_limits": spec.LIMITS, "oracle_is_deployable": False, "speedup_measured": None})


def run_statistics(pooled, rows, output):
    target = output / "statistics"
    target.mkdir()
    group_ids = np.array([r["acquisition_id"] for r in rows])
    intervals, comparisons = [], []
    for key, entry in sorted(pooled.items()):
        c = entry["case"]
        interval_required = (c["study"] == "normalization" or
            (c["study"] == "raw_plus" and c["protocol"] == "deployment") or
            (c["study"] == "robustness" and c["protocol"] == "deployment" and c["condition"].endswith("__10")))
        if not interval_required:
            continue
        if entry["eligible_for_inference"]:
            data = read_pooled(output, entry)
            seed = int(spec.digest({"master_seed": 42, "identity": key})[:16], 16)
            interval = cluster_interval(cluster_matrices(data["truth"], data["predicted"], group_ids), 10000, seed)
            status = "EVALUATED"
        else:
            interval, status = {"lower_95": None, "upper_95": None}, "EXCLUDED_NONCONVERGED_FOLD"
        intervals.append({**c, **interval, "status": status})
        if len(intervals) % 50 == 0:
            print(f"[INFERENCE] cluster intervals {len(intervals)}/1200", flush=True)
    for ordinal, comparison in enumerate(comparison_plan()):
        a, b = (pooled[spec.digest(comparison[k])] for k in ("left", "right"))
        if a["eligible_for_inference"] and b["eligible_for_inference"]:
            values = paired_inference(read_pooled(output, a), read_pooled(output, b), group_ids,
                                      seed=int(spec.digest({"master_seed": 42, "identity": comparison["comparison_id"]})[:16], 16))
            status = "EVALUATED"
        else:
            values = {"difference": None, "lower_95": None, "upper_95": None, "permutation_p": None,
                      "p_mc_lower_99": None, "p_mc_upper_99": None, "groups": len(set(group_ids)), "swaps": 0}
            status = "EXCLUDED_NONCONVERGED_COMPONENT"
        row = {"comparison_id": comparison["comparison_id"], "family": comparison["family"],
               "left": spec.digest(comparison["left"]), "right": spec.digest(comparison["right"]), **values, "status": status}
        comparisons.append(row)
        base.write_json(target / (comparison["comparison_id"] + ".json"), {**comparison, "result": values})
        if ordinal % 10 == 0:
            print(f"[INFERENCE] paired comparisons {ordinal + 1}/488", flush=True)
    for family in sorted({row["family"] for row in comparisons}):
        selected = [row for row in comparisons if row["family"] == family]
        for field, name in (("permutation_p", "holm_p"), ("p_mc_lower_99", "holm_mc_lower"), ("p_mc_upper_99", "holm_mc_upper")):
            for row, value in zip(selected, holm([r[field] if r[field] is not None else 1.0 for r in selected])):
                row[name] = float(value) if row["status"] == "EVALUATED" else None
        for row in selected:
            row["mc_decision_uncertain"] = row["status"] == "EVALUATED" and row["holm_mc_lower"] <= .05 <= row["holm_mc_upper"]
            row["interpretation"] = "exploratory_extension_prior_partial_results_seen"
    if len(intervals) != 1200:
        raise ValueError("Confidence-interval coverage changed")
    write_csv(output / "macro_f1_intervals.csv", intervals)
    write_csv(output / "paired_comparisons.csv", comparisons)


def tpi_summary(execution, output, metrics):
    summaries = []
    for condition in spec.CONDITIONS:
        for prep in spec.PREPROCESSORS:
            path = execution / "verified_diagnostics" / (condition + "_" + prep + ".npz")
            with np.load(path, allow_pickle=False) as bundle:
                values = bundle["tpi"]
            if values.shape != (858, 6, 3, 9):
                raise ValueError("TPI diagnostic coverage changed")
            for channel in range(6):
                for operator, name in enumerate(signal.TPI_OPERATORS):
                    ratio = values[:, channel, operator, 3]
                    summaries.append({"condition": condition, "preprocessor": prep, "channel": base.CHANNELS[channel],
                        "operator": name, "median_tpi_noisy": float(np.median(ratio)),
                        "q25_tpi_noisy": float(np.quantile(ratio, .25)), "q75_tpi_noisy": float(np.quantile(ratio, .75)),
                        "count_tpi_gt1": int(np.count_nonzero(ratio > 1)),
                        "near_zero_clean": int(values[:, channel, operator, 7].sum()),
                        "near_zero_noisy": int(values[:, channel, operator, 8].sum()),
                        "median_tpi_clean": float(np.median(values[:, channel, operator, 4])),
                        "median_noise_inflation": float(np.median(values[:, channel, operator, 5])),
                        "median_clean_hf_error": float(np.median(values[:, channel, operator, 6]))})
    write_csv(output / "tpi_channel_summary.csv", summaries)
    association = []
    for noise in spec.NOISES:
        for operator in signal.TPI_OPERATORS:
            xs, ys = [], []
            for c in spec.CONDITIONS:
                if not c.startswith(noise + "__"):
                    continue
                for prep in spec.PREPROCESSORS:
                    scores = [r["macro_f1"] for r in metrics if r["study"] == "robustness" and
                        r["protocol"] == "deployment" and r["condition"] == c and r["preprocessor"] == prep and
                        r["representation"] == "MULTI18" and r["eligible_for_inference"]]
                    if not scores:
                        continue
                    xs.append(float(np.median([r["median_tpi_noisy"] for r in summaries if
                        r["condition"] == c and r["preprocessor"] == prep and r["operator"] == operator])))
                    ys.append(float(np.median(scores)))
            defined = len(xs) >= 3 and np.ptp(xs) > 0 and np.ptp(ys) > 0
            rho, p = spearmanr(xs, ys) if defined else (None, None)
            association.append({"noise": noise, "operator": operator, "spearman_rho": float(rho) if defined else None,
                "descriptive_p_not_independent_evidence": float(p) if defined else None,
                "condition_points": len(xs), "status": "EXPLORATORY_NOT_CAUSAL" if defined else "UNDEFINED_CONSTANT_INPUT"})
    write_csv(output / "tpi_exploratory_association.csv", association)
    draw_figure20(summaries, metrics, output)


def descriptor_shift(execution, output):
    clean = np.load(execution / "verified_features/clean_none.npy", allow_pickle=False)
    rows = []
    scale = np.std(clean, axis=0, ddof=1)
    for condition in spec.EXTRA_CONDITIONS:
        values = np.load(execution / "verified_features" / (condition + "_none.npy"), allow_pickle=False)
        difference = values - clean
        for column, name in enumerate(spec.FEATURES):
            rows.append({"condition": condition, "feature": name,
                "median_paired_change": float(np.median(difference[:, column])),
                "median_absolute_paired_change": float(np.median(np.abs(difference[:, column]))),
                "median_abs_change_over_clean_std": float(np.median(np.abs(difference[:, column])) / scale[column]) if scale[column] > 1e-12 else None,
                "status": "DESCRIPTIVE_NOT_FEATURE_SELECTION" if scale[column] > 1e-12 else "UNDEFINED_NEAR_CONSTANT_CLEAN_FEATURE"})
    write_csv(output / "descriptor_shift.csv", rows)


def draw_figure20(summaries, metrics, output):
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1300" height="1000" viewBox="0 0 1300 1000">',
        '<rect width="1300" height="1000" fill="white"/><g font-family="Arial,sans-serif" fill="#172b3a">',
        '<text x="70" y="35" font-size="23">TPI diagnostics and held-out denoising effect</text>']
    palette = ("#0173B2", "#D55E00", "#029E73", "#CC78BC", "#DE8F05")
    for panel, noise in enumerate(("lowfreq", "mixed_gain_additive")):
        x0 = 80 + panel * 640
        parts.append(f'<text x="{x0}" y="75" font-size="22">{noise}</text>')
        bars = []
        for p, prep in enumerate(spec.PREPROCESSORS):
            for o, operator in enumerate(signal.TPI_OPERATORS):
                ratio = np.median([r["median_tpi_noisy"] for r in summaries if r["condition"] == noise + "__10" and
                                   r["preprocessor"] == prep and r["operator"] == operator])
                bars.append((p, o, float(np.log10(max(ratio, 1e-12)))))
        lo, hi = min(-1, min(b[2] for b in bars)), max(1, max(b[2] for b in bars))
        for p, o, value in bars:
            y0 = 130 + 260 * hi / (hi - lo)
            y = 130 + 260 * (hi - value) / (hi - lo)
            parts.append(f'<rect x="{x0 + p * 100 + o * 25}" y="{min(y,y0):.2f}" width="22" height="{abs(y-y0):.2f}" fill="{COLORS[o]}"/>')
        for p, prep in enumerate(spec.PREPROCESSORS):
            parts.append(f'<text x="{x0 + p * 100}" y="430" font-size="12">{prep}</text>')
        parts.append(f'<text x="{x0}" y="110" font-size="16">10 dB; log10 median channel TPI; range [{lo:.2g}, {hi:.2g}]</text>')
        scatter = []
        for prep_index, prep in enumerate(spec.PREPROCESSORS):
            for snr in base.SNR_LEVELS:
                condition = f"{noise}__{snr}"
                ratios = [r["median_tpi_noisy"] for r in summaries if r["condition"] == condition and
                          r["preprocessor"] == prep and r["operator"] == signal.TPI_OPERATORS[0]]
                effects = []
                for model in spec.MODELS:
                    left = [r for r in metrics if all(r[k] == v for k, v in robust_case(model, condition, "MULTI18", prep).items())]
                    right = [r for r in metrics if all(r[k] == v for k, v in robust_case(model, condition, "MULTI18").items())]
                    if len(left) != 1 or len(right) != 1:
                        raise ValueError("Figure 20 requires paired complete metrics")
                    if left[0]["eligible_for_inference"] and right[0]["eligible_for_inference"]:
                        effects.append(left[0]["macro_f1"] - right[0]["macro_f1"])
                if effects:
                    scatter.append((float(np.log10(max(np.median(ratios), 1e-12))), float(np.median(effects)), prep_index))
        if scatter:
            xmin, xmax = min(v[0] for v in scatter), max(v[0] for v in scatter)
            ymin, ymax = min(-.01, min(v[1] for v in scatter)), max(.01, max(v[1] for v in scatter))
            if xmax - xmin < 1e-9:
                xmax = xmin + 1
            parts.append(f'<rect x="{x0}" y="570" width="510" height="280" fill="none" stroke="#999999"/>')
            for x, y, p in scatter:
                xx, yy = x0 + 510 * (x - xmin) / (xmax - xmin), 850 - 280 * (y - ymin) / (ymax - ymin)
                parts.append(f'<circle cx="{xx:.2f}" cy="{yy:.2f}" r="5" fill="{palette[p]}" opacity="0.8"/>')
            parts.append(f'<text x="{x0}" y="890" font-size="15">log10 DWT D3 TPI [{xmin:.2g}, {xmax:.2g}]</text>')
            parts.append(f'<text x="{x0}" y="550" font-size="15">Macro-F1 filtered minus none [{ymin:.2g}, {ymax:.2g}]</text>')
    for o, operator in enumerate(signal.TPI_OPERATORS):
        parts.append(f'<text x="{70 + o * 420}" y="475" fill="{COLORS[o]}" font-size="14">{operator}</text>')
    for p, prep in enumerate(spec.PREPROCESSORS):
        parts.append(f'<text x="{70 + p * 240}" y="935" fill="{palette[p]}" font-size="15">{prep}</text>')
    parts.append('<text x="70" y="978" font-size="14">Channel ratios and model effects summarized separately. Descriptive association is not causal evidence or a denoiser selection rule.</text></g></svg>')
    with (directory / "figure20_tpi_denoising_effect.svg").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(parts))


def draw_normalization(metrics, output):
    for representation in base.REPRESENTATIONS:
        parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="800" viewBox="0 0 1200 800">',
            '<rect width="1200" height="800" fill="white"/><g font-family="Arial,sans-serif" fill="#172b3a">',
            f'<text x="70" y="40" font-size="24">Clean fold-local normalization: {representation}</text>']
        for c, normalizer in enumerate(spec.NORMALIZERS):
            parts.append(f'<text x="{250 + c * 145}" y="105" font-size="13">{normalizer}</text>')
        for r, model in enumerate(spec.MODELS):
            y = 140 + r * 65
            parts.append(f'<text x="40" y="{y + 31}" font-size="17">{model}</text>')
            for c, normalizer in enumerate(spec.NORMALIZERS):
                selected = [row for row in metrics if row["study"] == "normalization" and row["classifier"] == model and
                            row["representation"] == representation and row["normalizer"] == normalizer]
                if len(selected) != 1:
                    raise ValueError("Normalization figure coverage incomplete")
                row = selected[0]
                shade = int(245 - 100 * row["macro_f1"])
                color = f"rgb({shade},{min(255,shade + 20)},245)" if row["eligible_for_inference"] else "#dddddd"
                x = 240 + c * 145
                label = f"{row['macro_f1']:.3f}" if row["eligible_for_inference"] else "EXCLUDED"
                parts.append(f'<rect x="{x}" y="{y}" width="135" height="55" fill="{color}"/><text x="{x + 30}" y="{y + 32}" font-size="17">{label}</text>')
        parts.append('<text x="70" y="750" font-size="16">Pooled acquisition-grouped Macro-F1. Transformations are fitted inside training folds only. No hyperparameter search.</text></g></svg>')
        with (output / "figures" / f"normalization_{representation}.svg").open("x", encoding="utf-8") as stream:
            stream.write("\n".join(parts))


COLORS = ("#56B4E9", "#D55E00", "#0173B2", "#DE8F05", "#029E73", "#CC78BC", "#CA9161", "#666666")


def draw_curves(metrics, output):
    """Dependency-free vector figures; five noise facets and classifier colors."""
    directory = output / "figures"
    directory.mkdir(exist_ok=True)
    for protocol in ("deployment", "matched"):
        for prep in spec.PREPROCESSORS:
            for representation in (*base.REPRESENTATIONS, "LATE_SIMPLE", "LATE_ORACLE_SNR"):
                parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1200" viewBox="0 0 1400 1200">',
                    '<rect width="1400" height="1200" fill="white"/>',
                    '<g font-family="Arial,sans-serif" fill="#172b3a">',
                    f'<text x="70" y="36" font-size="24">Macro-F1: {escape(protocol)} / {escape(prep)} / {escape(representation)}</text>']
                for panel, noise in enumerate(spec.NOISES):
                    x0, y0, width, height = 75 + (panel % 2) * 690, 90 + (panel // 2) * 360, 580, 270
                    parts.append(f'<text x="{x0}" y="{y0 - 16}" font-size="20">{escape(noise)}</text>')
                    for tick in (0, .25, .5, .75, 1):
                        y = y0 + height * (1 - tick)
                        parts.append(f'<path d="M{x0},{y}h{width}" stroke="#dddddd"/><text x="{x0 - 40}" y="{y + 5}" font-size="15">{tick:g}</text>')
                    for level in base.SNR_LEVELS:
                        x = x0 + width * (level - 10) / 110
                        parts.append(f'<text x="{x - 10}" y="{y0 + height + 24}" font-size="15">{level}</text>')
                    for model, color in zip(spec.MODELS, COLORS):
                        selected = [r for r in metrics if r["study"] == "robustness" and r["protocol"] == protocol and
                            r["preprocessor"] == prep and r["representation"] == representation and r["classifier"] == model and r["condition"].startswith(noise + "__") and r["eligible_for_inference"]]
                        selected.sort(key=lambda r: spec.condition(r["condition"])[1])
                        points = [(x0 + width * (spec.condition(r["condition"])[1] - 10) / 110,
                                   y0 + height * (1 - r["macro_f1"])) for r in selected]
                        parts.append(f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2.3"/>')
                        parts.extend(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.7" fill="{color}"/>' for x, y in points)
                    parts.append(f'<text x="{x0 + 190}" y="{y0 + height + 50}" font-size="16">Additive SNR parameter (dB)</text>')
                for i, (model, color) in enumerate(zip(spec.MODELS, COLORS)):
                    y = 870 + i * 28
                    parts.append(f'<path d="M790,{y}h38" stroke="{color}" stroke-width="3"/><text x="840" y="{y + 5}" font-size="18">{escape(model)}</text>')
                parts.append('<text x="75" y="1180" font-size="15">120 dB is not clean. Composite SNR is additive-component SNR. Nonconverged cases are excluded. Oracle fusion is not deployable.</text></g></svg>')
                with (directory / f"macro_f1_{protocol}_{prep}_{representation}.svg").open("x", encoding="utf-8") as stream:
                    stream.write("\n".join(parts))
