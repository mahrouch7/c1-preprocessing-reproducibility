"""Persist scientific artifacts per engine task; no custom scheduling or retries."""
from pathlib import Path
import traceback
import uuid
import numpy as np
from r07.core.artifacts import child, digest, read_json, sha256, utc_now, write_csv, write_json
from r07.data.views import read_csv
from r07.evaluation.metrics import score
from r07.features.profiles import LABELS
from r07.learning.models import fit_predict

def unit_name(modality, model, fold):
    return f"{modality}__{model}__fold{fold:02d}"

def verify_unit(prepared, modality, model, fold, pending_receipt=None):
    name = unit_name(modality, model, fold)
    root = Path(prepared["output"]) / "tasks" / name
    complete = root / "complete.json"
    if pending_receipt is None and not complete.exists():
        return None
    receipt = read_json(complete) if pending_receipt is None else pending_receipt
    expected_id = digest(dict(run_id=prepared["run_id"], task=name))
    if receipt.get("status") != "VALIDATED" or receipt.get("identity") != expected_id:
        raise ValueError("Changed/stale completed task identity")
    required = {"predictions.csv", "metrics.json", "confusion.csv"}
    if set(receipt["artifacts"]) != required:
        raise ValueError("Missing or unexpected task artifacts")
    attempt = child(root, receipt["attempt"])
    for file, value in receipt["artifacts"].items():
        if sha256(child(attempt, file)) != value:
            raise ValueError("Completed scientific artifact changed")
    predictions = read_csv(attempt / "predictions.csv")
    expected = prepared["frame"].loc[prepared["folds"]["fold"].to_numpy() == fold,
                                      ["Id", "window_id", "acquisition_id", "fault"]].copy()
    expected = expected.rename(columns={"fault": "y_true"})
    if predictions[list(expected)].to_dict("records") != expected.to_dict("records"):
        raise ValueError("Saved predictions differ from this fold/dataset")
    p = predictions[["p_" + label for label in LABELS]].to_numpy(dtype=float)
    if not np.isfinite(p).all() or (p < -1e-12).any() or (p > 1+1e-12).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-10):
        raise ValueError("Saved probabilities are invalid")
    # Hash equality is not sufficient: check the metric formula and row identities.
    metrics, confusion = score(predictions["y_true"], predictions["y_pred"])
    saved = read_json(attempt / "metrics.json")
    if any(not np.isclose(saved["metrics"][key], value, rtol=0, atol=1e-12) for key, value in metrics.items()):
        raise ValueError("Saved metrics do not match predictions")
    if read_csv(attempt / "confusion.csv").to_dict("records") != confusion.to_dict("records"):
        raise ValueError("Saved confusion counts do not match predictions")
    return dict(name=name, predictions=predictions, metrics=metrics,
                fit=saved["fit"], receipt_sha256=sha256(complete) if pending_receipt is None else digest(receipt),
                receipt_path=complete.relative_to(prepared["output"]).as_posix())

def execute_unit(prepared, modality, model, fold):
    previous = verify_unit(prepared, modality, model, fold)
    if previous is not None:
        print("[REUSE VALIDATED] " + previous["name"], flush=True)
        return previous
    name = unit_name(modality, model, fold)
    root = Path(prepared["output"]) / "tasks" / name
    attempts = root / "attempts"
    for status_path in attempts.glob("*/status.json"):
        if read_json(status_path)["status"] == "FAILED":
            raise RuntimeError("A previous explicit failure needs review: " + str(status_path))
    attempt = attempts / uuid.uuid4().hex
    attempt.mkdir(parents=True)
    write_json(attempt / "status.json", dict(status="RUNNING", started_utc=utc_now(), task=name))
    try:
        predictions, fit = fit_predict(prepared["frame"], prepared["folds"], modality, model, fold, prepared["config"])
        metrics, confusion = score(predictions["y_true"], predictions["y_pred"])
        write_csv(attempt / "predictions.csv", predictions)
        write_csv(attempt / "confusion.csv", confusion)
        write_json(attempt / "metrics.json", dict(metrics=metrics, fit=fit))
        write_json(attempt / "status.json", dict(status="COMPLETED_UNPUBLISHED", finished_utc=utc_now(), task=name))
        receipt = dict(status="VALIDATED",
                       identity=digest(dict(run_id=prepared["run_id"], task=name)),
                       attempt=attempt.relative_to(root).as_posix(),
                       artifacts={n: sha256(attempt / n) for n in ("predictions.csv", "metrics.json", "confusion.csv")})
        verify_unit(prepared, modality, model, fold, pending_receipt=receipt)
        # Controller lock excludes concurrent publication. Old attempts are retained.
        write_json(root / "complete.json", receipt)
        result = verify_unit(prepared, modality, model, fold)
        print("[FOLD COMPLETE] " + name, flush=True)
        return result
    except BaseException as exc:
        write_json(attempt / "status.json",
                   dict(status="INTERRUPTED" if isinstance(exc, (KeyboardInterrupt, SystemExit)) else "FAILED",
                        stopped_utc=utc_now(), error=str(exc), traceback=traceback.format_exc()))
        raise
