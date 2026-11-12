from copy import deepcopy
import numpy as np
import pytest
from c1resnet.data import fit_scaler, make_split, validate_rows
from c1resnet.engine import WaveformDataset


def test_valid_synthetic_metadata(synthetic_rows):
    validate_rows(synthetic_rows, production=False)


@pytest.mark.parametrize("key", ["window_id", "acquisition_id", "raw_sha256", "clean_window_numeric_sha256"])
def test_duplicate_identity_rejected(synthetic_rows, key):
    rows = deepcopy(synthetic_rows)
    rows[1][key] = rows[0][key]
    with pytest.raises(ValueError, match="Duplicate"):
        validate_rows(rows, production=False)


def test_family_crossing_outer_folds_rejected(synthetic_rows):
    rows = deepcopy(synthetic_rows)
    rows[1]["fold"] = 2
    with pytest.raises(ValueError, match="Family crosses"):
        validate_rows(rows, production=False)


def test_reused_load_point_rejected(synthetic_rows):
    rows = deepcopy(synthetic_rows)
    rows[12]["factors"] = rows[0]["factors"]
    with pytest.raises(ValueError, match="load regime"):
        validate_rows(rows, production=False)


@pytest.mark.parametrize("fold", [1, 2, 3, 4, 5])
def test_nested_split_family_disjoint(synthetic_rows, fold):
    split = make_split(synthetic_rows, fold)
    assert split == make_split(synthetic_rows, fold)
    a, b, c = (set(split[key]) for key in ("inner_train", "validation", "test"))
    assert not (a & b or a & c or b & c)
    assert a | b == set(split["outer_train"])
    assert a | b | c == set(range(len(synthetic_rows)))
    assert not set(split["validation_families"]) & set(split["test_families"])


def test_scaler_cannot_see_held_out_values(synthetic_values):
    original = fit_scaler(synthetic_values, list(range(30)))
    modified = synthetic_values.copy()
    modified[30:] = 1e10
    assert fit_scaler(modified, list(range(30))) == original


def test_scaler_matches_direct_training_moments(synthetic_values):
    ids = list(range(31))
    scaler = fit_scaler(synthetic_values, ids, chunk_size=7)
    x = synthetic_values[ids].astype(np.float64)
    np.testing.assert_allclose(scaler["mean"], x.mean(axis=(0, 2)), atol=1e-14)
    np.testing.assert_allclose(scaler["std"], x.std(axis=(0, 2)), rtol=1e-14)


def test_normalization_preserves_between_window_amplitude(synthetic_rows):
    x = np.arange(1, 1 + 128 * 6, dtype=np.float32).reshape(1, 6, 128)
    values = np.concatenate([x, 2 * x])
    scaler = fit_scaler(values, [0])
    dataset = WaveformDataset(values, synthetic_rows, [0, 1], scaler)
    assert not np.allclose(dataset[0][0].numpy(), dataset[1][0].numpy())
    np.testing.assert_array_equal(values[0], x[0])


def test_duplicate_scaler_rows_rejected(synthetic_values):
    with pytest.raises(ValueError):
        fit_scaler(synthetic_values, [0, 0])
