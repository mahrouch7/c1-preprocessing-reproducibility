"""Frozen R03-completion units; scientific work occurs only in execute()."""
from __future__ import annotations

import gzip
import csv
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import pickle
import platform
import shutil
import sys
import time
import warnings
import zipfile

import numpy as np
import c1e25 as base
import c1e26 as extension
import c1e27_spec as spec


def activate_wavelet(wheel, root):
    """Load the approved hash-pinned wheel inside this task, not the worker venv."""
    if base.sha(wheel) != spec.WHEEL_SHA256:
        raise ValueError("Unapproved PyWavelets wheel")
    target = root / "_scientific_dependencies"
    target.mkdir(exist_ok=False)
    with zipfile.ZipFile(wheel) as bundle:
        total = 0
        for member in bundle.infolist():
            relative = PurePosixPath(member.filename)
            total += member.file_size
            if (relative.is_absolute() or ".." in relative.parts or "\\" in member.filename or
                    total > 100 * 1024 * 1024 or ((member.external_attr >> 16) & 0o170000) == 0o120000):
                raise ValueError("Unsafe scientific dependency wheel")
            if member.is_dir():
                continue
            destination = target.joinpath(*relative.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output)
    sys.path.insert(0, str(target))
    importlib.invalidate_caches()
    if importlib.metadata.version("PyWavelets") != spec.VERSIONS["PyWavelets"]:
        raise ValueError("Vendored PyWavelets version mismatch")


def save_npy(path, values):
    with Path(path).open("xb") as stream:
        np.save(stream, values, allow_pickle=False)


def stage_a(config, inputs, output):
    # Activate the task-local wheel before importing SciPy signal helpers, which
    # can discover an optional global PyWavelets distribution during import.
    import c1e27_signal as signal
    rows = base.validate_metadata(base.read_json(inputs["metadata"]))
    clean = np.load(inputs["clean"], mmap_mode="r", allow_pickle=False)
    if clean.shape != (len(rows), spec.SAMPLES, 6) or clean.dtype != np.float64:
        raise ValueError("Waveform capsule shape/precision mismatch")
    tables = {p: np.empty((len(rows), 36), dtype=np.float64) for p in spec.PREPROCESSORS}
    tpi = {p: np.empty((len(rows), 6, 3, 9), dtype=np.float64) for p in spec.PREPROCESSORS}
    distortions = {p: np.empty((len(rows), 6), dtype=np.float64) for p in spec.PREPROCESSORS}
    snrs = []
    started = time.perf_counter()
    for index, row in enumerate(rows):
        original = np.asarray(clean[index])
        observed, noise_report = signal.corrupt(original, row, config["condition"])
        snrs.append(noise_report)
        ec, en = signal.energies(original), signal.energies(observed)
        power = np.mean(original ** 2, axis=0)
        if np.any(power <= 0):
            raise ValueError("Undefined clean-relative channel distortion")
        for prep in spec.PREPROCESSORS:
            processed = signal.preprocess(observed, prep)
            tables[prep][index] = signal.features(processed)
            tpi[prep][index] = signal.tpi(original, observed, processed, ec, en)
            distortions[prep][index] = np.sqrt(np.mean((processed - original) ** 2, axis=0) / power)
        if index % 50 == 0:
            print(f"[PREPROCESSING] {config['condition']} {index}/{len(rows)} windows; fits=0", flush=True)
    for prep in spec.PREPROCESSORS:
        save_npy(output / f"features_{prep}.npy", tables[prep])
        with (output / f"diagnostics_{prep}.npz").open("xb") as stream:
            np.savez_compressed(stream, tpi=tpi[prep], relative_l2=distortions[prep])
    base.write_json(output / "preprocessing_receipt.json", {"condition": config["condition"],
        "rows": len(rows), "feature_columns": spec.FEATURES, "tpi_operators": list(signal.TPI_OPERATORS),
        "tpi_fields": list(signal.TPI_FIELDS), "noise_diagnostics": snrs,
        "elapsed_seconds": time.perf_counter() - started, "fits_completed": 0})
    return 0


def make_estimator(model, normalizer):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                       PowerTransformer, QuantileTransformer)
    built = base.build_estimator(model) if model in base.MODELS else extension.build_estimator(model)
    classifier = built.named_steps["classifier"] if hasattr(built, "named_steps") else built
    scalers = {"none": None, "standard": StandardScaler(), "minmax_0_1": MinMaxScaler(feature_range=(0, 1)),
        "robust": RobustScaler(), "power_yeo_johnson": PowerTransformer(method="yeo-johnson"),
        "quantile_normal": QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=42)}
    if normalizer not in scalers:
        raise ValueError("Unreviewed normalizer")
    return classifier if normalizer == "none" else Pipeline([("scale", scalers[normalizer]), ("classifier", classifier)])


def prediction_case(job, protocol, condition):
    return {"study": job["study"], "protocol": protocol, "condition": condition,
        "preprocessor": job["preprocessor"], "representation": job["representation"],
        "classifier": job["classifier"], "normalizer": job["normalizer"]}


def save_prediction(output, case, row_ids, truth, predicted, probabilities, eligible=True):
    row_ids, truth, predicted = (np.asarray(v, dtype=np.int64) for v in (row_ids, truth, predicted))
    p = np.asarray(probabilities, dtype=np.float64)
    if (len(set(row_ids.tolist())) != len(row_ids) or truth.shape != row_ids.shape or predicted.shape != truth.shape
            or np.any(truth < 0) or np.any(truth >= 12) or np.any(predicted < 0) or np.any(predicted >= 12)
            or p.shape != (len(truth), 12) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1)
            or not np.allclose(p.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError("OOF prediction identity/probability contract failed")
    name = "prediction_" + spec.digest(case) + ".npz"
    with (output / name).open("xb") as stream:
        np.savez_compressed(stream, row_ids=row_ids, truth=truth, predicted=predicted, probabilities=p)
    return {"case": case, "path": name, "sha256": base.sha(output / name),
            "eligible_for_inference": bool(eligible)}


def validated_cache(job, inputs, rows, train, clean):
    representation = job["representation"]
    records = base.read_json(inputs["cache_index"])
    record = records[representation]
    training_sha = spec.digest([row["window_id"] for row, selected in zip(rows, train) if selected])
    if (any(record.get(k) != job[k] for k in ("fold", "preprocessor", "classifier", "representation")) or
            record["training_window_ids_sha256"] != training_sha or
            record["model_contract"] != spec.MODEL_CONTRACTS[job["classifier"]] or
            record["feature_columns"] != [spec.FEATURES[i] for i in spec.REPRESENTATIONS[representation]]):
        raise ValueError("Historical model training identity/configuration mismatch")
    previous = np.load(inputs["reuse_base_features"], allow_pickle=False)
    if not np.array_equal(previous, clean[:, :18]):
        raise ValueError("Current clean features differ from cached model training features")
    path = inputs["cache_" + representation]
    if base.sha(path) != record["saved_model"]["sha256"]:
        raise ValueError("Historical model file mismatch")
    # Only the completed E25/E26 archives validated by the coordinator can supply
    # these immutable pickle files. Arbitrary uploaded pickle inputs are forbidden.
    with path.open("rb") as stream:
        return pickle.load(stream), record


def reused_prediction(job, condition, record, inputs, rows, test):
    legacy = "clean" if condition == "clean" else condition.replace("__", "")
    if legacy not in base.CONDITIONS:
        return None
    pin = record["prediction_artifacts"][legacy]
    path = inputs[f"cachepred_{job['representation']}_{legacy}"]
    if base.sha(path) != pin["sha256"]:
        raise ValueError("Reused prediction artifact changed")
    with path.open(encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream))
    selected = [r for r, v in zip(rows, test) if v]
    if len(table) != len(selected) or any(int(t["Id"]) != r["Id"] or t["window_id"] != r["window_id"] or
            t["acquisition_id"] != r["acquisition_id"] or t["y_true"] != r["fault"] for t, r in zip(table, selected)):
        raise ValueError("Historical OOF predictions are not aligned with the current fold")
    return np.array([t["y_pred"] for t in table]), np.array([[float(t["p_" + label]) for label in base.LABELS] for t in table])


def stage_b(config, inputs, output, estimator_factory=make_estimator):
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    rows = base.validate_metadata(base.read_json(inputs["metadata"]))
    arrays = {c: np.load(inputs[c], allow_pickle=False) for c in spec.CONDITIONS}
    if any(a.shape != (len(rows), 36) or a.dtype != np.float64 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Feature schema, dimensions or precision mismatch")
    test = np.array([row["fold"] == config["fold"] for row in rows])
    train = ~test
    if {r["acquisition_id"] for r, v in zip(rows, train) if v} & {r["acquisition_id"] for r, v in zip(rows, test) if v}:
        raise ValueError("Cross-fold acquisition leakage")
    labels = {name: i for i, name in enumerate(base.LABELS)}
    truth = np.array([labels[r["fault"]] for r in rows])
    row_ids = np.array([r["Id"] for r in rows])[test]
    y_train = np.array([r["fault"] for r in rows])[train]
    records, catalogue, fusion_inputs = [], [], {}
    fits = reused = 0
    all_jobs = spec.jobs(config["preprocessor"], config["fold"], config["classifier"])
    with threadpool_limits(limits=1):
        for ordinal, job in enumerate(all_jobs):
            columns = spec.REPRESENTATIONS[job["representation"]]
            if job["reuse_E25_E26"]:
                model, inherited = validated_cache(job, inputs, rows, train, arrays["clean"])
                reused += 1
                model_path = output / ("model_" + job["job_id"] + ".pickle")
                shutil.copyfile(inputs["cache_" + job["representation"]], model_path)
                elapsed, captured = None, []
            else:
                model = estimator_factory(job["classifier"], job["normalizer"])
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    model.fit(arrays[job["training_condition"]][train][:, columns], y_train)
                elapsed = time.perf_counter() - started
                fits += 1
                warning_records = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
                if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                    base.write_json(output / ("nonconvergence_" + job["job_id"] + ".json"), {"job": job, "fits_attempted": fits,
                        "warnings": warning_records, "automatic_refit": False})
                    print(f"[NUMERICAL EXCLUSION] {job['job_id']} nonconverged; retain evidence, no refit", flush=True)
                model_path = output / ("model_" + job["job_id"] + ".pickle.gz")
                with model_path.open("xb") as raw_stream:
                    with gzip.GzipFile(fileobj=raw_stream, mode="wb", compresslevel=3, mtime=0) as stream:
                        pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
                inherited = None
            eligible = not any(issubclass(w.category, ConvergenceWarning) for w in captured)
            record = {"job": job, "fit_seconds": elapsed, "reused": job["reuse_E25_E26"],
                "eligible_for_inference": eligible, "train_windows": int(train.sum()), "test_windows": int(test.sum()),
                "source_fit_record": inherited, "model_contract": spec.MODEL_CONTRACTS[job["classifier"]],
                "training_window_ids_sha256": spec.digest([r["window_id"] for r, v in zip(rows, train) if v]),
                "feature_columns": [spec.FEATURES[i] for i in columns],
                "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured],
                "saved_model": {"path": model_path.name, "sha256": base.sha(model_path)}}
            base.write_json(output / ("fit_" + job["job_id"] + ".json"), record)
            records.append(record)
            for protocol, condition in spec.evaluation_cases(job):
                cached = reused_prediction(job, condition, inherited, inputs, rows, test) if job["reuse_E25_E26"] else None
                predicted, probability = cached if cached is not None else base.checked_prediction(model, arrays[condition][test][:, columns])
                encoded = np.array([labels[p] for p in predicted])
                case = prediction_case(job, protocol, condition)
                catalogue.append(save_prediction(output, case, row_ids, truth[test], encoded, probability, eligible))
                if job["study"] == "robustness" and job["representation"] in ("RMS6", "RAW12"):
                    fusion_inputs.setdefault((protocol, condition), {})[job["representation"]] = (probability, eligible)
            print(f"[FIT PROGRESS] {config['preprocessor']} fold={config['fold']} {config['classifier']} "
                  f"jobs={ordinal + 1}/{len(all_jobs)} new_fits={fits}/{config['expected_fits']} reused={reused}", flush=True)
        for (protocol, condition), components in fusion_inputs.items():
            if set(components) != {"RAW12", "RMS6"}:
                raise ValueError("Missing aligned late-fusion component")
            _, snr = spec.condition(condition)
            for name, rms_weight in (("LATE_SIMPLE", 0.5), ("LATE_ORACLE_SNR", 0.7 if snr is not None and snr <= 20 else 0.3)):
                p = rms_weight * components["RMS6"][0] + (1 - rms_weight) * components["RAW12"][0]
                eligible = components["RMS6"][1] and components["RAW12"][1]
                case = {"study": "robustness", "protocol": protocol, "condition": condition,
                        "preprocessor": config["preprocessor"], "representation": name,
                        "classifier": config["classifier"], "normalizer": spec.default_normalizer(config["classifier"])}
                catalogue.append(save_prediction(output, case, row_ids, truth[test], np.argmax(p, axis=1), p, eligible))
    if fits != config["expected_fits"] or reused != sum(j["reuse_E25_E26"] for j in all_jobs):
        raise ValueError("Learning budget changed")
    base.write_json(output / "prediction_catalogue.json", catalogue)
    base.write_json(output / "fit_records.json", {"fits_completed": fits, "reused_fits": reused,
        "nonconverged_fit_attempts": sum(not r["eligible_for_inference"] for r in records), "records": records})
    return fits


def execute(args):
    configuration, output = Path(args.configuration), Path(args.output)
    if base.sha(configuration) != args.configuration_sha256:
        raise ValueError("Scientific configuration digest mismatch")
    config = base.read_json(configuration)
    spec.validate_configuration(config, args.expected_fits)
    inputs = base.verified_inputs(config, configuration.parent)
    if len(base.validate_metadata(base.read_json(inputs["metadata"]))) != 858:
        raise ValueError("The frozen production campaign requires 429 acquisitions / 858 windows")
    if shutil.disk_usage(configuration.parent).free < 5 * 1024**3:
        raise RuntimeError("Less than 5 GiB free on this worker drive; preserve evidence and stop without fitting")
    if config["stage"] == "A":
        activate_wavelet(inputs["pywavelets_wheel"], configuration.parent)
    needed = spec.VERSIONS if config["stage"] == "A" else base.VERSIONS
    versions = {name: importlib.metadata.version(name) for name in needed}
    if versions != needed:
        raise ValueError("Scientific dependency version mismatch")
    fits = (stage_a if config["stage"] == "A" else stage_b)(config, inputs, output)
    base.write_json(output / "runtime.json", {"host": platform.node(), "python": platform.python_version(),
        "versions": versions, "fits_completed": fits, "simulations": 0, "stage": config["stage"]})
    artifacts = [{"path": p.name, "bytes": p.stat().st_size, "sha256": base.sha(p)}
                 for p in sorted(output.iterdir()) if p.is_file()]
    base.write_json(output / "scientific_receipt.json", {"status": "COMPLETED", "unit_id": args.unit_id,
        "role": args.role, "assignment_sha256": args.assignment_sha256,
        "configuration_sha256": args.configuration_sha256, "expected_fits": args.expected_fits,
        "fits_completed": fits, "simulations": 0, "artifacts": artifacts})
