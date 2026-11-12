"""R03 normalization names retained explicitly; no truth metadata as predictors."""
import numpy as np

RAW_CHANNELS = [kind + "B7B5_Raw_" + phase for kind in ("I", "V") for phase in "ABC"]
RMS = [name.replace("_Raw_", "_RMS_") for name in RAW_CHANNELS]
RAW = [name + suffix for name in RAW_CHANNELS for suffix in ("_mean", "_std")]
PROFILES = {"RMS": RMS, "RAW": RAW, "MULTI": RMS + RAW}
LABELS = ["noFault", "ab", "ac", "bc", "abc", "ag", "bg", "cg",
          "abg", "acg", "bcg", "abcg"]

def window_features(raw):
    """Time rows x six ABC channels; separate windows, sample std ddof=1."""
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or len(raw) < 2 or not np.isfinite(raw).all():
        raise ValueError("A finite six-channel RAW window is required")
    result = {}
    for i, name in enumerate(RAW_CHANNELS):
        result[name + "_mean"] = float(raw[:, i].mean())
        result[name + "_std"] = float(raw[:, i].std(ddof=1))
        result[RMS[i]] = float(np.sqrt(np.mean(raw[:, i] ** 2) + 1e-12))
    return result
