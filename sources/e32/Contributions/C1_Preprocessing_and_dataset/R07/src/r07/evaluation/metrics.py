"""Class-balanced metrics, explicit labels and confusion-count invariants."""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.metrics import f1_score, matthews_corrcoef
from r07.features.profiles import LABELS

def score(y_true, y_pred, labels=LABELS):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape or not len(y_true):
        raise ValueError("Aligned non-empty prediction vectors required")
    if not set(y_true).issubset(labels) or not set(y_pred).issubset(labels):
        raise ValueError("Unknown prediction label")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if int(cm.sum()) != len(y_true) or any(int(cm[i].sum()) != int(np.sum(y_true == label)) for i, label in enumerate(labels)):
        raise ValueError("Confusion matrix row sums are invalid")
    metrics = dict(n_windows=len(y_true), accuracy=float(accuracy_score(y_true, y_pred)),
                   balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
                   macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
                   weighted_f1=float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
                   mcc=float(matthews_corrcoef(y_true, y_pred)))
    confusion = pd.DataFrame(cm, columns=labels)
    confusion.insert(0, "true_label", labels)
    return metrics, confusion
