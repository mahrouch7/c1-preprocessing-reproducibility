"""Bounded clean baseline: Dummy prior and R03-style standardized logistic model."""
import time
import warnings
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from r07.features.profiles import LABELS, PROFILES

def fit_predict(frame, manifest, modality, model_name, fold, config):
    columns = PROFILES[modality]
    test = manifest["fold"].to_numpy() == fold
    train = ~test
    if set(frame.loc[test, "acquisition_id"]) & set(frame.loc[train, "acquisition_id"]):
        raise ValueError("Train/test event overlap")
    x = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite input is not imputed silently")
    y = frame["fault"].to_numpy(dtype=str)
    if model_name == "dummy_prior":
        model = DummyClassifier(strategy="prior", random_state=config["seed"])
    elif model_name == "logistic_regression":
        model = Pipeline([("scale", StandardScaler()),
                          ("classifier", LogisticRegression(max_iter=config["logistic_max_iter"],
                                                           random_state=config["seed"]))])
    else:
        raise ValueError("Unreviewed model: " + model_name)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        model.fit(x[train], y[train])
    convergence = [str(w.message) for w in captured if issubclass(w.category, ConvergenceWarning)]
    if convergence:
        raise RuntimeError("Model did not converge: " + " | ".join(convergence))
    classes = list(model.classes_)
    if set(classes) != set(LABELS):
        raise ValueError("Training classes do not match the declared twelve labels")
    predicted = model.predict(x[test]).astype(str)
    probabilities = model.predict_proba(x[test])[:, [classes.index(label) for label in LABELS]]
    if (not np.isfinite(probabilities).all() or (probabilities < -1e-12).any() or
        (probabilities > 1+1e-12).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-10)):
        raise ValueError("Invalid class probabilities")
    output = frame.loc[test, ["Id", "window_id", "acquisition_id", "fault"]].copy()
    output = output.rename(columns={"fault": "y_true"})
    output["y_pred"] = predicted
    for i, label in enumerate(LABELS):
        output["p_" + label] = probabilities[:, i]
    return output, dict(fit_seconds=time.perf_counter()-started,
                        warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in captured],
                        feature_columns=columns, training_windows=int(train.sum()),
                        test_windows=int(test.sum()), fold=int(fold), converged=True)
