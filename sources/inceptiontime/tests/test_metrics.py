from copy import deepcopy
import numpy as np
import pytest
from c1inception.data import LABELS
from c1inception.metrics import family_class_weights, score


def test_perfect_predictions(synthetic_rows):
    ids = list(range(len(synthetic_rows)))
    y = [LABELS.index(r["fault"]) for r in synthetic_rows]
    result = score(synthetic_rows, ids, np.eye(12)[y])
    assert result["macro_f1"] == result["balanced_accuracy"] == result["mcc"] == 1
    assert result["healthy_false_alarm_rate"] == result["missed_fault_rate"] == 0
    assert result["brier"] == result["log_loss"] == 0


def test_majority_prediction_does_not_appear_accurate(synthetic_rows):
    p = np.zeros((len(synthetic_rows), 12))
    p[:, 0] = 1
    result = score(synthetic_rows, range(len(synthetic_rows)), p)
    assert result["balanced_accuracy"] == pytest.approx(1 / 12)
    assert result["macro_f1"] == pytest.approx(2 / (13 * 12))
    assert result["missed_fault_rate"] == 1


def test_family_density_invariance(synthetic_rows):
    rows = deepcopy(synthetic_rows)
    extra = []
    for _ in range(5):
        for source in rows[:12]:
            extra.append({**source, "Id": len(rows) + len(extra)})
    expanded = rows + extra
    def probabilities(items):
        p = np.zeros((len(items), 12))
        for j, r in enumerate(items):
            label = LABELS.index(r["fault"])
            predicted = label if r["family_id"] != "F000" else (label + 1) % 12
            p[j, predicted] = 1
        return p
    first = score(rows, range(len(rows)), probabilities(rows))
    second = score(expanded, range(len(expanded)), probabilities(expanded))
    assert first["macro_f1"] == pytest.approx(second["macro_f1"], abs=1e-14)
    assert first["balanced_accuracy"] == pytest.approx(second["balanced_accuracy"], abs=1e-14)


@pytest.mark.parametrize("bad", [np.nan, -0.1, 1.1])
def test_bad_probabilities_rejected(synthetic_rows, bad):
    p = np.full((len(synthetic_rows), 12), 1 / 12)
    p[0, 0] = bad
    with pytest.raises(ValueError):
        score(synthetic_rows, range(len(synthetic_rows)), p)
