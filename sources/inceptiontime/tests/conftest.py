from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from c1inception.data import LABELS, load_config


@pytest.fixture
def config():
    return deepcopy(load_config())


@pytest.fixture
def synthetic_rows():
    rows = []
    for family in range(10):
        for label in LABELS:
            i = len(rows)
            h = hashlib.sha256(str(i).encode()).hexdigest()
            rows.append({"Id": i, "window_id": f"window{i}", "acquisition_id": f"acquisition{i}",
                         "family_id": f"F{family:03d}", "source_group_id": f"F{family:03d}",
                         "fold": family % 5 + 1, "fault": label, "raw_sha256": h,
                         "clean_window_numeric_sha256": h, "factors": [1 + family / 100, 1, 1]})
    return rows


@pytest.fixture
def tiny_config(config):
    config["architecture"]["filters_per_branch"] = 2
    config["architecture"]["bottleneck_channels"] = 4
    config["architecture"]["module_dilations"] = [1, 1, 1, 1, 1, 1]
    config["architecture"]["kernels"] = [7, 5, 3]
    config["training"].update(max_epochs=2, patience=2, batch_size=16, evaluation_batch_size=24)
    return config


@pytest.fixture
def synthetic_values(synthetic_rows):
    rng = np.random.Generator(np.random.PCG64(77))
    return rng.standard_normal((len(synthetic_rows), 6, 128)).astype(np.float32)
