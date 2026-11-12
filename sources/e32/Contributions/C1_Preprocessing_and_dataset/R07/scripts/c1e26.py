"""C1-E26: six additional classifiers on immutable C1-E25 features; no fit on import."""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path
import platform
import time
import warnings

import numpy as np
import c1e25 as base

EXPERIMENT = "C1-E26"
PROTOCOL = "C1-E26-six-classifier-awgn-extension-v1"
MODEL_CONTRACTS = {
    "naive_bayes": {"var_smoothing": 1e-9, "priors": None},
    "knn": {"n_neighbors": 5, "weights": "uniform", "algorithm": "auto", "leaf_size": 30,
            "p": 2, "metric": "minkowski", "n_jobs": 1},
    "decision_tree": {"criterion": "gini", "splitter": "best", "max_depth": None,
                      "min_samples_split": 2, "min_samples_leaf": 1, "random_state": 42,
                      "class_weight": None},
    "svm": {"C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True,
            "tol": 1e-3, "max_iter": -1, "random_state": 42, "class_weight": None},
    "gradient_boosting": {"loss": "log_loss", "learning_rate": 0.1, "n_estimators": 100,
                          "subsample": 1.0, "max_depth": 3, "random_state": 42},
    "mlp": {"hidden_layer_sizes": [64, 32], "activation": "relu", "solver": "adam",
            "alpha": 1e-4, "learning_rate_init": 1e-3, "max_iter": 10000,
            "tol": 1e-4, "early_stopping": False, "n_iter_no_change": 10, "random_state": 42},
}
MODELS = tuple(MODEL_CONTRACTS)
SCALED = ("knn", "svm", "mlp")


def build_estimator(name):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    factories = dict(naive_bayes=GaussianNB, knn=KNeighborsClassifier,
                     decision_tree=DecisionTreeClassifier, svm=SVC,
                     gradient_boosting=GradientBoostingClassifier, mlp=MLPClassifier)
    parameters = dict(MODEL_CONTRACTS[name])
    if name == "mlp":
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
    estimator = factories[name](**parameters)
    return Pipeline([("scale", StandardScaler()), ("classifier", estimator)]) if name in SCALED else estimator


def validate_configuration(config, expected_fits):
    if (config.get("experiment_id") != EXPERIMENT or config.get("protocol") != PROTOCOL or
            expected_fits != 18 or config.get("expected_fits") != 18 or
            config.get("preprocessor") not in ("none", "savgol") or
            config.get("fold") not in range(1, 6) or config.get("row_count") != 858 or
            config.get("model_contracts") != MODEL_CONTRACTS or
            config.get("scaled_classifiers") != list(SCALED) or
            config.get("conditions") != list(base.CONDITIONS) or
            config.get("feature_columns") != base.FEATURES or
            config.get("scientific_limits") != base.LIMITS):
        raise ValueError("Unreviewed C1-E26 model, fold, feature or fit budget")
    if set(config.get("inputs", {})) != {"metadata"} | {c + "_features" for c in base.CONDITIONS}:
        raise ValueError("Incomplete C1-E26 input conditions")


def fit_unit(config, inputs, output):
    import pandas as pd
    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits
    rows = base.validate_metadata(base.read_json(inputs["metadata"]))
    arrays = {c: np.load(inputs[c + "_features"], allow_pickle=False) for c in base.CONDITIONS}
    if any(a.shape != (len(rows), 18) or a.dtype != np.float64 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError("Frozen feature table schema mismatch")
    test = np.array([row["fold"] == config["fold"] for row in rows])
    train = ~test
    if {r["acquisition_id"] for r, flag in zip(rows, train) if flag} & {r["acquisition_id"] for r, flag in zip(rows, test) if flag}:
        raise ValueError("Train/test acquisition overlap")
    truth = np.array([row["fault"] for row in rows])
    selected = [row for row, flag in zip(rows, test) if flag]
    records = []
    with threadpool_limits(limits=1):
        for representation, columns in base.REPRESENTATIONS.items():
            for name in MODELS:
                estimator = build_estimator(name)
                started = time.perf_counter()
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    estimator.fit(arrays["clean"][train][:, columns], truth[train])
                elapsed = time.perf_counter() - started
                if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                    raise RuntimeError("Convergence failure; no automatic refit")
                model_path = output / f"model_{representation}_{name}.pickle"
                base.save_estimator(estimator, model_path)
                for condition in base.CONDITIONS:
                    predicted, probabilities = base.checked_prediction(estimator, arrays[condition][test][:, columns])
                    table = pd.DataFrame([{key: row[key] for key in ("Id", "window_id", "acquisition_id")} for row in selected])
                    table["y_true"], table["y_pred"] = truth[test], predicted
                    for index, label in enumerate(base.LABELS):
                        table["p_" + label] = probabilities[:, index]
                    table.to_csv(output / f"predictions_{representation}_{name}_{condition}.csv",
                                 index=False, lineterminator="\n", float_format="%.17g")
                records.append({"classifier": name, "representation": representation,
                    "preprocessor": config["preprocessor"], "fold": config["fold"],
                    "fit_seconds": elapsed, "model_contract": MODEL_CONTRACTS[name],
                    "scaled_training_fold_only": name in SCALED,
                    "train_windows": int(train.sum()), "test_windows": int(test.sum()),
                    "feature_columns": [base.FEATURES[i] for i in columns],
                    "training_window_ids_sha256": hashlib.sha256(base.canonical(
                        [r["window_id"] for r, flag in zip(rows, train) if flag])).hexdigest(),
                    "saved_model": {"path": model_path.name, "sha256": base.sha(model_path)},
                    "warnings": [{"category": w.category.__name__, "message": str(w.message)} for w in captured]})
                print(f"[FIT COMPLETE] {config['preprocessor']} fold={config['fold']} {representation}/{name} ({len(records)}/18)", flush=True)
    if len(records) != 18:
        raise ValueError("Exact eighteen-fit budget violated")
    base.write_json(output / "fit_records.json", {"fits_completed": 18, "records": records})
    return records


def execute(args):
    config_path, output = Path(args.configuration), Path(args.output)
    if base.sha(config_path) != args.configuration_sha256:
        raise ValueError("Configuration digest mismatch")
    config = base.read_json(config_path)
    validate_configuration(config, args.expected_fits)
    versions = {name: importlib.metadata.version(name) for name in base.VERSIONS}
    if versions != base.VERSIONS:
        raise ValueError("Scientific package versions differ")
    records = fit_unit(config, base.verified_inputs(config, config_path.parent), output)
    base.write_json(output / "runtime.json", {"host": platform.node(), "python": platform.python_version(),
        "versions": versions, "fits_completed": len(records), "simulations": 0})
    artifacts = [{"path": path.name, "bytes": path.stat().st_size, "sha256": base.sha(path)}
                 for path in sorted(output.iterdir()) if path.is_file()]
    base.write_json(output / "scientific_receipt.json", {"status": "COMPLETED", "unit_id": args.unit_id,
        "role": args.role, "assignment_sha256": args.assignment_sha256,
        "configuration_sha256": args.configuration_sha256, "expected_fits": 18,
        "fits_completed": len(records), "simulations": 0, "artifacts": artifacts})
