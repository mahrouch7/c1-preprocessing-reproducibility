"""The E32 pooled family/class-standardized estimand, not mean fold F1."""
from __future__ import annotations

from collections import Counter
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef
from .data import LABELS
from .io import require


def family_class_weights(rows, ids):
    cells = Counter((rows[int(i)]["family_id"], rows[int(i)]["fault"]) for i in ids)
    return np.asarray([1.0 / cells[(rows[int(i)]["family_id"], rows[int(i)]["fault"])] for i in ids])


def score(rows, ids, probabilities):
    ids = np.asarray(ids, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    require(p.shape == (len(ids), len(LABELS)) and len(set(ids.tolist())) == len(ids), "Bad prediction identity/shape")
    require(np.isfinite(p).all() and np.all((p >= 0) & (p <= 1)) and np.allclose(p.sum(axis=1), 1, atol=1e-6, rtol=0), "Invalid probability values")
    y = np.asarray([LABELS.index(rows[int(i)]["fault"]) for i in ids], dtype=np.int64)
    require(set(y.tolist()) == set(range(len(LABELS))), "Evaluation lacks a class")
    predicted = p.argmax(axis=1)
    weights = family_class_weights(rows, ids)
    cm = confusion_matrix(y, predicted, labels=np.arange(len(LABELS)), sample_weight=weights)
    recall = np.diag(cm) / cm.sum(axis=1)
    result = {"macro_f1": float(f1_score(y, predicted, labels=np.arange(len(LABELS)), average="macro", sample_weight=weights, zero_division=0)),
              "balanced_accuracy": float(recall.mean()),
              "mcc": float(matthews_corrcoef(y, predicted, sample_weight=weights)),
              "accuracy": float(np.trace(cm) / cm.sum()),
              "brier": float(np.average(np.sum((p - np.eye(len(LABELS))[y]) ** 2, axis=1), weights=weights)),
              "log_loss": float(np.average(-np.log(np.maximum(p[np.arange(len(y)), y], 1e-15)), weights=weights)),
              "healthy_false_alarm_rate": float(cm[0, 1:].sum() / cm[0].sum()),
              "missed_fault_rate": float(cm[1:, 0].sum() / cm[1:].sum()),
              "class_recall": dict(zip(LABELS, recall.tolist())), "weighted_confusion": cm.tolist(),
              "rows": len(ids), "families": len({rows[int(i)]["family_id"] for i in ids})}
    require(abs(result["accuracy"] - result["balanced_accuracy"]) < 1e-12, "Primary estimand lost equal class support")
    return result
