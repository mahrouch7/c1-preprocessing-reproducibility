"""Read one retained C1-E32 model; diagnose its original probability contract.

No fit, score, data repair, task submission, or campaign import. The failed unit
is read-only. Only a fresh report outside that unit may be written. This is not
a scientific completion receipt or permission to retry the campaign.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import pickle
import platform
import sys
import traceback
import warnings

import numpy as np

UNIT = "a709bff00cb7e27c14e41815072dfc4d5be09092e629a62c278f4737fe43621e"
JOB = "30c30d7cb6ed2f4e63320ec50b0dfc54f725ddb0a7f93db20e2d67495e454aed"
ASSIGNMENT_FILE = "c5ab7307d4b58521c29ed5fea56064102f282ec4fb675cb5add73b20fe15c949"
ASSIGNMENT = "3b44f5570c0a7d76456fe5a7d6141c999fa70439a67889dcf5806f861ec62627"
CONFIG = "91cdc32f892120e3d4e1f1211bfa17336d48b788b9ff7b88b2fcc95214ee5973"
ARCHIVE = "1a19daa6a37f9e005ec5c9b05ca5f064bbaabbcbc44d262f7da3896b48f3862a"
LABELS = ["noFault", "ab", "ac", "bc", "abc", "ag", "bg", "cg", "abg", "acg", "bcg", "abcg"]
CONDITIONS = ["clean", "awgn__20", "mixed_gain_additive__10"]
INPUT_HASHES = {
    "metadata": "d231f5d52c6ca1f6affb625a035805a85e186e2e8e1bcdd38386e6e50593f5e5",
    "clean": "9d2a959b863e8da8c3e0a2ed58183c7832305746f4603c645540ff9f66c1a2c6",
    "awgn__20": "7d5454e399ee0a9a6c3b664948bef424a8236c95e078bd7eff87872ac0d684b4",
    "mixed_gain_additive__10": "254101a2ba453bb9e6d8f5a5616e6117f9374508b2dd250fc795e44a9970711b",
}
VERSIONS = {"numpy": "2.3.5", "scipy": "1.18.1", "pandas": "2.3.3", "scikit-learn": "1.9.1"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def confined_file(root, relative):
    """Reject traversal, Windows drive/ADS syntax, symlinks and junction escape."""
    root = Path(root).resolve(strict=True)
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and relative not in ("", ".") and
            not any(v in relative for v in ("\\", ":")) and ".." not in rel.parts,
            "Unsafe artifact-relative path")
    candidate = root.joinpath(*rel.parts)
    current = candidate
    while current != root:
        require(not current.is_symlink() and not
                (hasattr(current, "is_junction") and current.is_junction()),
                "Linked artifact path is forbidden")
        current = current.parent
    resolved = candidate.resolve(strict=True)
    require(resolved.is_relative_to(root) and resolved.is_file(), "Artifact escaped unit or is not a file")
    return resolved


class ModelReader(pickle.Unpickler):
    """Small allowlist for this locally created sklearn/NumPy model, not arbitrary pickle."""
    ALLOWED = {
        ("sklearn.pipeline", "Pipeline"),
        ("sklearn.preprocessing._data", "PowerTransformer"),
        ("sklearn.preprocessing._data", "StandardScaler"),
        ("sklearn.naive_bayes", "GaussianNB"),
        ("numpy", "ndarray"), ("numpy", "dtype"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "scalar"),
        ("numpy._core.numeric", "_frombuffer"),
    }

    def find_class(self, module, name):
        if (module, name) not in self.ALLOWED:
            raise pickle.UnpicklingError("Unexpected model global: " + module + "." + name)
        return super().find_class(module, name)


def load_local_model(path):
    # No fallback to unrestricted loading and no importing the task's Python code.
    with gzip.open(path, "rb") as stream:
        data = stream.read(32 * 1024 * 1024 + 1)
    require(len(data) <= 32 * 1024 * 1024, "Saved model exceeds diagnostic size limit")
    model = ModelReader(io.BytesIO(data)).load()
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PowerTransformer
    from sklearn.naive_bayes import GaussianNB
    require(type(model) is Pipeline and [v[0] for v in model.steps] == ["scale", "classifier"],
            "Unexpected saved pipeline structure")
    require(type(model.named_steps["scale"]) is PowerTransformer and
            model.named_steps["scale"].method == "yeo-johnson" and
            type(model.named_steps["classifier"]) is GaussianNB,
            "Unexpected saved scaler/classifier")
    return model


def array_summary(values):
    a = np.asarray(values)
    finite = np.isfinite(a)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "nonfinite": int((~finite).sum()),
            "minimum_finite": float(a[finite].min()) if finite.any() else None,
            "maximum_finite": float(a[finite].max()) if finite.any() else None}


def contract(predicted, probabilities, classes, row_ids):
    """Mirror every clause of c1e25.checked_prediction without repairing values."""
    pred = np.asarray(predicted, dtype=str)
    p = np.asarray(probabilities)
    ids = np.asarray(row_ids)
    n = len(ids)
    result = {"violations": [], "prediction_shape": list(pred.shape),
              "native_labels": sorted(set(pred.reshape(-1))),
              "native_classes": [str(v) for v in classes], "raw_probability_shape": list(p.shape)}
    bad = result["violations"]
    if len(classes) != 12 or set(classes) != set(LABELS):
        bad.append("model_class_identity")
        return result
    if pred.shape != (n,):
        bad.append("prediction_shape")
    if not set(pred.reshape(-1)).issubset(LABELS):
        bad.append("prediction_labels")
    if p.shape != (n, 12):
        bad.append("probability_shape")
        return result
    p = p[:, [list(classes).index(label) for label in LABELS]]
    result["probabilities"] = array_summary(p)
    finite = np.isfinite(p)
    invalid = (~finite).any(axis=1)
    if not finite.all():
        bad.append("nonfinite_probabilities")
    outside = ((p < 0) | (p > 1)).any(axis=1)
    if outside.any():
        bad.append("probabilities_outside_0_1")
    sums = p.sum(axis=1)
    close = np.isclose(sums, 1, rtol=0, atol=1e-10)
    if not close.all():
        bad.append("row_sum_atol_1e-10")
    errors = np.abs(sums - 1)
    result["row_sum_absolute_error"] = array_summary(errors)
    result["rows_outside_original_tolerance"] = int((~close).sum())
    result["negative_values"] = int((p < 0).sum())
    result["above_one_values"] = int((p > 1).sum())
    worst = np.argsort(np.where(np.isfinite(errors), errors, np.inf))[::-1][:8]
    result["worst_rows"] = [{"row_id": int(ids[i]),
                             "sum": float(sums[i]) if np.isfinite(sums[i]) else None,
                             "nonfinite": int((~finite[i]).sum())} for i in worst]
    result["bad_row_ids_first_16"] = ids[invalid | outside | ~close][:16].tolist()
    return result


def inspect_condition(model, x, row_ids):
    result = {"input": array_summary(x)}
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        # Match original method order: native predict, then predict_proba.
        try:
            predicted = model.predict(x)
            probabilities = model.predict_proba(x)
            result.update(contract(predicted, probabilities, list(model.classes_), row_ids))
        except Exception as error:
            result.update(violations=["prediction_exception"], error=repr(error), traceback=traceback.format_exc())
        try:
            transformed = model.named_steps["scale"].transform(x)
            result["transformed_input"] = array_summary(transformed)
            # Inspect only; do not use these logs to alter predictions or scores.
            classifier = model.named_steps["classifier"]
            result["joint_log_likelihood"] = array_summary(classifier._joint_log_likelihood(transformed))
        except Exception as error:
            result["transform_or_log_inspection_error"] = repr(error)
    result["warnings"] = [{"category": w.category.__name__, "message": str(w.message)} for w in captured]
    return result


def diagnose(root):
    root = Path(root).resolve(strict=True)
    require(root.name == UNIT, "This diagnostic is restricted to the identified failed unit")
    artifacts = {}
    for name, relative, expected in [
        ("assignment", "assignment.json", ASSIGNMENT_FILE),
        ("archive", "task.zip", ARCHIVE),
        ("configuration", "execution/task/configuration.json", CONFIG),
    ]:
        path = confined_file(root, relative)
        actual = sha(path)
        require(actual == expected, name + " SHA256 mismatch")
        artifacts[name] = {"path": relative, "sha256": actual, "bytes": path.stat().st_size}
    assignment = json.loads(confined_file(root, "assignment.json").read_text(encoding="utf-8"))
    require(assignment["assignment_sha256"] == ASSIGNMENT, "Assignment identity differs")
    units = [u for u in assignment["units"] if u["unit_id"] == UNIT]
    require(len(units) == 1 and units[0]["configuration_sha256"] == CONFIG and
            units[0]["task_archive_sha256"] == ARCHIVE and units[0]["role"] == "PC2",
            "Failed-unit assignment differs")
    task = root / "execution" / "task"
    config = json.loads(confined_file(root, "execution/task/configuration.json").read_text(encoding="utf-8"))
    require(all(config[k] == v for k, v in {
        "experiment_id": "C1-E32", "stage": "B", "classifier": "naive_bayes",
        "preprocessor": "none", "fold": 1, "expected_fits": 151}.items()), "Configuration identity differs")
    paths = {}
    for name, expected in INPUT_HASHES.items():
        entry = config["inputs"][name]
        require(entry["sha256"] == expected, "Frozen input identity differs: " + name)
        path = confined_file(task, entry["path"])
        require(sha(path) == expected, "Input SHA256 mismatch: " + name)
        artifacts[name] = {"path": entry["path"], "sha256": expected, "bytes": path.stat().st_size}
        paths[name] = path
    rows = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    require(len(rows) == 1839 and len({r["window_id"] for r in rows}) == 1839, "Metadata identities differ")
    test = np.array([r["fold"] == 1 for r in rows])
    train_families = {r["family_id"] for r in rows if r["fold"] != 1}
    require(not train_families & {r["family_id"] for r in rows if r["fold"] == 1}, "Cross-fold family overlap")
    ids = np.arange(len(rows))[test]
    model_path = confined_file(root, "execution/output/model_" + JOB + ".pickle.gz")
    model_sha = sha(model_path)
    artifacts["retained_model"] = {"path": str(model_path.relative_to(root)), "sha256": model_sha,
                                   "bytes": model_path.stat().st_size,
                                   "original_model_hash_receipt": "absent: failed unit did not finalize fit_records"}
    model = load_local_model(model_path)
    scaler, classifier = model.named_steps["scale"], model.named_steps["classifier"]
    details = {"yeo_johnson_lambdas": array_summary(scaler.lambdas_),
               "gaussian_variances": array_summary(classifier.var_),
               "gaussian_means": array_summary(classifier.theta_),
               "gaussian_epsilon": float(classifier.epsilon_),
               "var_smoothing": float(classifier.var_smoothing)}
    results = []
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        for condition in CONDITIONS:
            a = np.load(paths[condition], allow_pickle=False)
            require(a.shape == (1839, 36) and a.dtype == np.float64 and np.isfinite(a).all(),
                    "Feature input dimensions/precision/values differ: " + condition)
            outcome = inspect_condition(model, a[test, 6:18], ids)
            outcome["condition"] = condition
            results.append(outcome)
    require(sha(model_path) == model_sha, "Retained model bytes changed during diagnostic")
    first = next((v["condition"] for v in results if v["violations"]), None)
    return {"status": "ORIGINAL_CONTRACT_REJECTION_REPRODUCED" if first else "NOT_REPRODUCED",
            "unit_id": UNIT, "job_id": JOB, "ordinal": 97, "artifacts": artifacts,
            "test_windows": len(ids), "feature_columns": list(range(6, 18)),
            "model_details": details, "conditions": results,
            "first_rejected_condition_in_original_order": first,
            "matched_clean": "same saved model and rows as deployment/clean; not recomputed",
            "interpretation_limit": "A diagnostic reproduction is not an approved fix or a completed scientific unit."}


def write_once(path, report):
    data = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with Path(path).open("xb") as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unit-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--confirm-saved-model-read", action="store_true", required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    root = args.unit_root.resolve(strict=True)
    require(not output.exists() and not output.is_relative_to(root), "Report must be fresh and outside failed unit")
    require(output.parent.is_dir(), "Report parent directory must already exist")
    report = {"experiment_id": "C1-E32", "diagnostic_only": True,
              "recorded_utc": datetime.now(timezone.utc).isoformat(), "host": platform.node(),
              "python": platform.python_version(), "fits": 0, "simulations": 0, "submissions": 0,
              "scientific_artifact_writes": 0, "script_sha256": sha(Path(__file__))}
    try:
        report["versions"] = {k: importlib.metadata.version(k) for k in VERSIONS}
        require(report["versions"] == VERSIONS, "Scientific versions changed; do not install or refit automatically")
        report.update(diagnose(root))
        exit_code = 0
    except Exception as error:
        report.update(status="DIAGNOSTIC_STOPPED", error=repr(error), traceback=traceback.format_exc())
        exit_code = 2
    write_once(output, report)
    print("[DIAGNOSTIC ONLY; FITS=0; NO RETRY] " + report["status"])
    for condition in report.get("conditions", []):
        print(condition["condition"] + ": " + (", ".join(condition["violations"]) or "PASS") +
              "; max_row_sum_error=" + str(condition.get("row_sum_absolute_error", {}).get("maximum_finite")))
    print("report=" + str(output))
    print("report_sha256=" + sha(output))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
