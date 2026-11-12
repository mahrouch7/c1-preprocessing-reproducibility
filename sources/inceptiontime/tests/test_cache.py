import numpy as np
import pytest
from c1inception.data import condition_cache


def test_small_cache_roundtrip_and_tamper_rejection(tmp_path, config, synthetic_rows, synthetic_values):
    config["samples"] = 128
    clean = synthetic_values.transpose(0, 2, 1).astype(np.float64)
    values, first = condition_cache(config, synthetic_rows, clean, "clean", tmp_path)
    np.testing.assert_array_equal(values, synthetic_values)
    _, second = condition_cache(config, synthetic_rows, clean, "clean", tmp_path)
    assert first == second
    del values
    path = next(tmp_path.rglob("clean.npy"))
    # On Windows another mmap may own the file, but appending is sufficient to change its digest.
    with path.open("ab") as stream:
        stream.write(b"CHANGED")
    with pytest.raises(ValueError, match="Corrupt cached"):
        condition_cache(config, synthetic_rows, clean, "clean", tmp_path)
