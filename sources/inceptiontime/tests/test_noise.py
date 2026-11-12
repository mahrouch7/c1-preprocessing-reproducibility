import hashlib
import numpy as np
import pytest
from c1inception.data import ROOT
from c1inception.io import read_json
from c1inception.noise import CONDITIONS, corrupt


def example():
    time = np.arange(8000, dtype=np.float64) / 40000
    raw = np.column_stack([(i + 1) * (2 + np.sin(2 * np.pi * (60 + i) * time)
                            + .1 * np.cos(2 * np.pi * 700 * time)) for i in range(6)])
    return raw, {"acquisition_id": "SYNTHETIC_PARITY_EVENT", "window_id": "SYNTHETIC_PARITY_WINDOW"}


@pytest.mark.parametrize("condition", CONDITIONS)
def test_bitwise_match_to_original_E32_noise(condition):
    raw, row = example()
    result = corrupt(raw, row, condition)
    actual = hashlib.sha256(result.astype("<f8", order="C").tobytes()).hexdigest()
    assert actual == read_json(ROOT / "provenance/noise_regression.json")["float64_sha256"][condition]


@pytest.mark.parametrize("mechanism", ["awgn", "lowfreq", "impulsive"])
def test_additive_snr_per_channel_and_shared_direction(mechanism):
    raw, row = example()
    low = corrupt(raw, row, mechanism + "__10") - raw
    high = corrupt(raw, row, mechanism + "__20") - raw
    observed = 10 * np.log10(np.mean(raw ** 2, axis=0) / np.mean(low ** 2, axis=0))
    np.testing.assert_allclose(observed, 10, atol=1e-7, rtol=0)
    np.testing.assert_allclose(low, high * np.sqrt(10), atol=1e-12, rtol=1e-12)


def test_corruption_does_not_mutate_source():
    raw, row = example()
    before = raw.copy()
    corrupt(raw, row, "mixed_gain_additive__10")
    np.testing.assert_array_equal(raw, before)


def test_different_acquisitions_receive_different_noise():
    raw, row = example()
    other = {**row, "acquisition_id": "another_acquisition"}
    assert not np.array_equal(corrupt(raw, row, "awgn__20"), corrupt(raw, other, "awgn__20"))


def test_gain_mismatch_persists_at_high_additive_snr():
    raw, row = example()
    value = corrupt(raw, row, "gain_plus_awgn__120")
    assert np.sqrt(np.mean((value - raw) ** 2) / np.mean(raw ** 2)) > 1e-3


def test_unknown_noise_is_not_silently_accepted():
    raw, row = example()
    with pytest.raises(ValueError):
        corrupt(raw, row, "unknown__10")
