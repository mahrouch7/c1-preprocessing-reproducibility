"""C1-E25 saved-waveform science; imported without fitting or starting a service."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import pickle
import time
import warnings

import numpy as np

EXPERIMENT = "C1-E25"
PROTOCOL = "C1-E25-awgn-grid-savgol41-v2"
SNR_LEVELS = (10, 20, 40, 60, 120)
CONDITIONS = ("clean",) + tuple("awgn" + str(value) for value in SNR_LEVELS)
SNR_ATOL_DB = 1e-7  # Float64 subtraction at 120 dB needs a numerical allowance.
CHANNELS = [kind + "B7B5_Raw_" + phase for kind in ("I", "V") for phase in "ABC"]
FEATURES = [c.replace("_Raw_", "_RMS_") for c in CHANNELS] + [c + s for c in CHANNELS for s in ("_mean", "_std")]
REPRESENTATIONS = {"RMS6": list(range(6)), "RAW12": list(range(6, 18)), "MULTI18": list(range(18))}
LABELS = ["noFault", "ab", "ac", "bc", "abc", "ag", "bg", "cg", "abg", "acg", "bcg", "abcg"]
MODELS = ("logistic_regression", "random_forest")
VERSIONS = {"numpy": "2.3.5", "scipy": "1.18.1", "pandas": "2.3.3", "scikit-learn": "1.9.1"}
MODEL_CONTRACTS = {
    "logistic_regression": {"C": 1.0, "l1_ratio": 0.0, "solver": "lbfgs", "tol": 1e-4,
        "max_iter": 10000, "random_state": 42, "class_weight": None},
    "random_forest": {"n_estimators": 200, "criterion": "gini", "max_depth": None,
        "max_features": "sqrt", "bootstrap": True, "class_weight": None, "random_state": 42, "n_jobs": 1},
}
LIMITS = {"initial_detection_requires_clearing": False,
          "true_b7_dijkstra_distance_is_fault_type_predictor": False,
          "e16_voltage_failures_preserved": 14, "abc_abcg_ambiguity_preserved": True,
          "model_validity_limits_unresolved": True, "clean_pilot_accuracy_percent": 93.473193,
          "clean_pilot_scope": "grouped_interpolation_only"}


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(canonical(value))


def validate_metadata(rows):
    if not isinstance(rows, list) or not rows or len(rows) % 2:
        raise ValueError("Paired acquisition rows required")
    windows, acquisitions = set(), set()
    for i in range(0, len(rows), 2):
        healthy, fault = rows[i:i + 2]
        acq = healthy["acquisition_id"]
        if acq in acquisitions or fault["acquisition_id"] != acq:
            raise ValueError("Duplicate or broken acquisition pair")
        acquisitions.add(acq)
        if (healthy["fault"] != "noFault" or fault["fault"] not in LABELS[1:] or
                healthy["group_fault_label"] != fault["fault"] or
                fault["group_fault_label"] != fault["fault"] or
                healthy["fold"] != fault["fold"]):
            raise ValueError("Label/group/fold pairing changed")
        for row, index, suffix in ((healthy, i, "prefault"), (fault, i + 1, "fault")):
            if row["Id"] != index or row["window_id"] != acq + "__" + suffix:
                raise ValueError("Window identity or order changed")
            if row["window_id"] in windows or row["fold"] not in range(1, 6):
                raise ValueError("Duplicate window or invalid fold")
            windows.add(row["window_id"])
    for fold in range(1, 6):
        if {r["fault"] for r in rows if r["fold"] == fold} != set(LABELS):
            raise ValueError("A test fold lacks declared classes")
        if {r["fault"] for r in rows if r["fold"] != fold} != set(LABELS):
            raise ValueError("A training fold lacks declared classes")
    return rows


def feature_vector(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or raw.shape[0] < 2 or not np.isfinite(raw).all():
        raise ValueError("Finite time-by-six RAW window required")
    rms = np.sqrt(np.mean(raw ** 2, axis=0) + 1e-12)
    moments = np.column_stack((raw.mean(axis=0), raw.std(axis=0, ddof=1))).reshape(-1)
    return np.concatenate((rms, moments))


def snr_for_condition(condition):
    if condition not in CONDITIONS:
        raise ValueError("Unreviewed corruption condition")
    return None if condition == "clean" else float(condition[4:])


def awgn(raw, acquisition_id, window_id, snr_db):
    if snr_db not in SNR_LEVELS:
        raise ValueError("SNR is outside the approved first-pass grid")
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or not np.isfinite(raw).all():
        raise ValueError("Finite six-channel waveform required")
    result = raw.copy()
    for i, channel in enumerate(CHANNELS):
        power = float(np.mean(raw[:, i] ** 2))
        if not np.isfinite(power) or power <= 0:
            raise ValueError("SNR is undefined for a zero/nonfinite channel")
        identity = {"experiment_id": EXPERIMENT, "acquisition_id": acquisition_id,
                    "protocol": PROTOCOL, "window_id": window_id, "channel": channel,
                    "noise": "awgn", "replicate_seed": 42}
        seed = int.from_bytes(hashlib.sha256(canonical(identity)).digest()[:16], "big")
        rng = np.random.Generator(np.random.PCG64(seed))
        noise = rng.standard_normal(len(raw))
        noise *= np.sqrt(power * 10.0 ** (-float(snr_db) / 10.0) / np.mean(noise ** 2))
        result[:, i] += noise
    snr = 10 * np.log10(np.mean(raw ** 2, axis=0) / np.mean((result - raw) ** 2, axis=0))
    if not np.allclose(snr, snr_db, rtol=0, atol=SNR_ATOL_DB):
        raise ValueError("AWGN exact per-window/channel SNR contract failed")
    return result


def preprocess(raw, name):
    if name == "none":
        return np.asarray(raw, dtype=np.float64)
    if name != "savgol":
        raise ValueError("Unreviewed preprocessing")
    from scipy.signal import savgol_filter
    return savgol_filter(raw, window_length=41, polyorder=3, axis=0, mode="mirror")


def validate_configuration(config, expected_fits):
    if (config.get("protocol") != PROTOCOL or config.get("experiment_id") != EXPERIMENT or
            config.get("scientific_limits") != LIMITS or config.get("expected_fits") != expected_fits or
            config.get("sampling_rate_hz") != 40000 or config.get("samples_per_window") != 8000 or
            config.get("feature_columns") != FEATURES or config.get("labels") != LABELS or
            config.get("row_count") != 858 or config.get("preprocessor") not in ("none", "savgol") or
            config.get("evaluation_conditions") != list(CONDITIONS)):
        raise ValueError("Unreviewed scientific configuration")
    if config.get("stage") == "A":
        if expected_fits != 0 or config.get("condition") not in CONDITIONS:
            raise ValueError("Stage A requires a declared zero-fit condition")
    elif config.get("stage") == "B":
        if (expected_fits != 6 or config.get("fold") not in range(1, 6) or
                config.get("model_contracts") != MODEL_CONTRACTS):
            raise ValueError("Stage B is exactly one fold and six fits")
    else:
        raise ValueError("Unknown stage")


def verified_inputs(config, root):
    result = {}
    for name, pin in config["inputs"].items():
        relative = Path(pin["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe scientific input path")
        path = Path(root) / relative
        if path.is_symlink() or sha(path) != pin["sha256"]:
            raise ValueError("Scientific input hash mismatch: " + name)
        result[name] = path
    return result


def stage_a(config, inputs, output):
    rows = validate_metadata(read_json(inputs["metadata"]))
    clean = np.load(inputs["clean"], mmap_mode="r", allow_pickle=False)
    source = clean if config["condition"] == "clean" else np.load(inputs["noisy"], mmap_mode="r", allow_pickle=False)
    shape = (config["row_count"], config["samples_per_window"], 6)
    if clean.shape != shape or source.shape != shape or clean.dtype != np.float64 or source.dtype != np.float64:
        raise ValueError("Frozen waveform shape or precision mismatch")
    if len(rows) != shape[0]:
        raise ValueError("Waveform/metadata row mismatch")
    features = np.empty((len(rows), len(FEATURES)), dtype=np.float64)
    errors, snrs = [], []
    for i, row in enumerate(rows):
        original, observed = np.asarray(clean[i]), np.asarray(source[i])
        if not np.isfinite(original).all() or not np.isfinite(observed).all():
            raise ValueError("Nonfinite waveform")
        processed = preprocess(observed, config["preprocessor"])
        features[i] = feature_vector(processed)
        power = np.mean(original ** 2, axis=0)
        if np.any(power <= 0):
            raise ValueError("Undefined relative distortion for zero channel")
        errors.append(np.sqrt(np.mean((processed - original) ** 2, axis=0) / power))
        if config["condition"] != "clean":
            snr = 10 * np.log10(power / np.mean((observed - original) ** 2, axis=0))
            if not np.allclose(snr, snr_for_condition(config["condition"]), rtol=0, atol=SNR_ATOL_DB):
                raise ValueError("Input corruption SNR differs from the declared level")
            snrs.append(snr)
        if i % 100 == 0:
            print(f"[FEATURES] {config['condition']}/{config['preprocessor']} {i}/{len(rows)}", flush=True)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite features")
    np.save(output / "features.npy", features, allow_pickle=False)
    diagnostics = {"condition": config["condition"], "preprocessor": config["preprocessor"],
                   "rows": len(rows), "feature_columns": FEATURES,
                   "clean_waveforms_sha256": config["inputs"]["clean"]["sha256"],
                   "source_waveforms_sha256": config["inputs"]["clean" if config["condition"] == "clean" else "noisy"]["sha256"],
                   "mean_relative_l2_error_by_channel": np.mean(errors, axis=0).tolist(),
                   "maximum_relative_l2_error_by_channel": np.max(errors, axis=0).tolist(),
                   "minimum_input_snr_db": float(np.min(snrs)) if snrs else None,
                   "maximum_input_snr_db": float(np.max(snrs)) if snrs else None,
                   "input_snr_tolerance_db": SNR_ATOL_DB,
                   "rms_epsilon": 1e-12, "std_ddof": 1, "fits_completed": 0}
    write_json(output / "feature_diagnostics.json", diagnostics)
    return 0


def build_estimator(name):
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        # In the pinned sklearn 1.9 line, l1_ratio=0 is L2 without deprecated penalty.
        return Pipeline([("scale", StandardScaler()),
                         ("classifier", LogisticRegression(**MODEL_CONTRACTS[name]))])
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(**MODEL_CONTRACTS[name])
    raise ValueError("Unreviewed classifier")


def checked_prediction(model, x):
    predicted = np.asarray(model.predict(x), dtype=str)
    classes = list(model.classes_)
    if len(classes) != len(LABELS) or set(classes) != set(LABELS):
        raise ValueError("Model class identity mismatch")
    probabilities = model.predict_proba(x)[:, [classes.index(label) for label in LABELS]]
    if (predicted.shape != (len(x),) or not set(predicted).issubset(LABELS) or
            probabilities.shape != (len(x), len(LABELS)) or not np.isfinite(probabilities).all() or
            np.any(probabilities < 0) or np.any(probabilities > 1) or
            not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError("Invalid predictions or probabilities")
    return predicted, probabilities


def save_estimator(model, path):
    # Only future explicitly authorized analysis of these trusted, hash-pinned
    # artifacts may load pickle. Never load models from untrusted sources.
    with Path(path).open("xb") as stream:
        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)


def stage_b(config, inputs, output):
    import pandas as pd
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    rows = validate_metadata(read_json(inputs["metadata"]))
    arrays = {condition: np.load(inputs[condition + "_features"], allow_pickle=False) for condition in CONDITIONS}
    for array in arrays.values():
        if array.shape != (len(rows), 18) or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("Invalid frozen feature table")
    fold = config["fold"]
    test = np.array([r["fold"] == fold for r in rows])
    train = ~test
    if {r["acquisition_id"] for r, flag in zip(rows, test) if flag} & {r["acquisition_id"] for r, flag in zip(rows, train) if flag}:
        raise ValueError("Train/test acquisition overlap")
    y = np.array([r["fault"] for r in rows])
    records = []
    with threadpool_limits(limits=1):
        for representation, columns in REPRESENTATIONS.items():
            x_train = arrays["clean"][train][:, columns]
            for name in MODELS:
                model = build_estimator(name)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    model.fit(x_train, y[train])
                if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                    raise RuntimeError("Convergence failure; no automatic refit")
                fit_seconds = time.perf_counter() - started
                model_path = output / f"model_{representation}_{name}.pickle"
                save_estimator(model, model_path)
                for condition in CONDITIONS:
                    predicted, probabilities = checked_prediction(model, arrays[condition][test][:, columns])
                    selected = [r for r, flag in zip(rows, test) if flag]
                    table = pd.DataFrame([{k: r[k] for k in ("Id", "window_id", "acquisition_id")} for r in selected])
                    table["y_true"], table["y_pred"] = y[test], predicted
                    for j, label in enumerate(LABELS):
                        table["p_" + label] = probabilities[:, j]
                    table.to_csv(output / f"predictions_{representation}_{name}_{condition}.csv", index=False,
                                 lineterminator="\n", float_format="%.17g")
                records.append({"representation": representation, "classifier": name,
                    "fold": fold, "preprocessor": config["preprocessor"], "fit_seconds": fit_seconds,
                    "model_contract": MODEL_CONTRACTS[name],
                    "feature_columns": [FEATURES[i] for i in columns],
                    "saved_model": {"path": model_path.name, "sha256": sha(model_path)},
                    "training_window_ids_sha256": hashlib.sha256(canonical(
                        [r["window_id"] for r, flag in zip(rows, train) if flag])).hexdigest(),
                    "evaluation_conditions": list(CONDITIONS),
                    "train_windows": int(train.sum()), "test_windows": int(test.sum()),
                    "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured]})
                print(f"[FIT COMPLETE] {config['preprocessor']} fold={fold} {representation}/{name} ({len(records)}/6)", flush=True)
    if len(records) != 6:
        raise ValueError("Exact six-fit unit budget violated")
    write_json(output / "fit_records.json", {"fits_completed": 6, "records": records})
    return 6


def execute(config_path, output, unit_id, role, assignment_sha256, configuration_sha256, expected_fits):
    config_path, output = Path(config_path), Path(output)
    if sha(config_path) != configuration_sha256:
        raise ValueError("Configuration digest mismatch")
    config = read_json(config_path)
    validate_configuration(config, expected_fits)
    versions = {name: importlib.metadata.version(name) for name in VERSIONS}
    if versions != VERSIONS:
        raise ValueError("Scientific dependency versions differ from the frozen environment")
    inputs = verified_inputs(config, config_path.parent)
    fits = (stage_a if config["stage"] == "A" else stage_b)(config, inputs, output)
    if fits != expected_fits:
        raise ValueError("Executed fit count differs from unit budget")
    write_json(output / "runtime.json", {"host": platform.node(), "python": platform.python_version(),
        "versions": versions, "stage": config["stage"], "fits_completed": fits, "simulations": 0})
    artifacts = [{"path": p.name, "bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(output.iterdir()) if p.is_file()]
    write_json(output / "scientific_receipt.json", {"status": "COMPLETED", "unit_id": unit_id,
        "role": role, "assignment_sha256": assignment_sha256, "configuration_sha256": configuration_sha256,
        "expected_fits": expected_fits, "fits_completed": fits, "simulations": 0, "artifacts": artifacts})


def macro_f1_from_confusion(cm):
    cm = np.asarray(cm, dtype=float)
    diagonal = np.diagonal(cm, axis1=-2, axis2=-1)
    denominator = cm.sum(axis=-1) + cm.sum(axis=-2)
    return np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal), where=denominator > 0).mean(axis=-1)


def paired_cluster_interval(rows, predicted_a, predicted_b, resamples=10000, seed=42):
    if not isinstance(resamples, int) or isinstance(resamples, bool) or resamples < 1:
        raise ValueError("A positive bootstrap resample count is required")
    labels = {name: i for i, name in enumerate(LABELS)}
    acquisitions = list(dict.fromkeys(r["acquisition_id"] for r in rows))
    group_index = {name: i for i, name in enumerate(acquisitions)}
    matrices = []
    for predicted in (predicted_a, predicted_b):
        if len(predicted) != len(rows):
            raise ValueError("Unaligned bootstrap predictions")
        per_group = np.zeros((len(acquisitions), len(LABELS), len(LABELS)))
        for row, label in zip(rows, predicted):
            per_group[group_index[row["acquisition_id"]], labels[row["fault"]], labels[str(label)]] += 1
        matrices.append(per_group.reshape(len(acquisitions), -1))
    rng = np.random.Generator(np.random.PCG64(seed))
    differences = []
    for offset in range(0, resamples, 250):
        weights = rng.multinomial(len(acquisitions), np.full(len(acquisitions), 1 / len(acquisitions)),
                                  size=min(250, resamples - offset)).astype(float)
        a, b = [(weights @ matrix).reshape(-1, len(LABELS), len(LABELS)) for matrix in matrices]
        differences.extend((macro_f1_from_confusion(b) - macro_f1_from_confusion(a)).tolist())
    low, high = np.quantile(differences, [.025, .975], method="linear")
    return {"lower_95": float(low), "upper_95": float(high), "resamples": resamples,
            "seed": seed, "cluster": "acquisition", "interval": "percentile"}
