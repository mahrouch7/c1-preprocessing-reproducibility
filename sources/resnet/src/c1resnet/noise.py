"""Exact E32 repeat-0 corruption semantics, ported from the pinned E25/E27 code.

Operate in float64; convert to float32 only after constructing the entire
corrupted waveform. No denoising, resampling or new stochastic augmentation.
"""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.signal import butter, sosfiltfilt
from .io import canonical, require

CHANNELS = [kind + "B7B5_Raw_" + phase for kind in ("I", "V") for phase in "ABC"]
LEVELS = (10, 20, 40, 60, 120)
MECHANISMS = ("awgn", "lowfreq", "impulsive", "gain_plus_awgn", "mixed_gain_additive")
CONDITIONS = ("clean",) + tuple(f"{m}__{s}" for m in MECHANISMS for s in LEVELS)
FS = 40000
SNR_ATOL_DB = 1e-7


def unit_rms(values):
    norm = np.sqrt(np.mean(np.asarray(values) ** 2))
    require(np.isfinite(norm) and norm > 0, "Undefined component RMS")
    return values / norm


def drift(rng, n):
    sos = butter(4, 5.0, btype="lowpass", fs=FS, output="sos")
    return unit_rms(sosfiltfilt(sos, rng.standard_normal(n), padtype="odd", padlen=15))


def impulses(rng, n):
    starts = rng.random(n) < (100.0 / FS)
    if not starts.any():
        starts[int(rng.integers(n))] = True
    signed = starts * rng.choice([-1.0, 1.0], size=n)
    return unit_rms(np.convolve(signed, np.ones(4), mode="full")[:n])


def corrupt(raw, row, condition):
    require(condition in CONDITIONS, "Unknown E32 corruption condition")
    raw = np.asarray(raw, dtype=np.float64)
    require(raw.ndim == 2 and raw.shape[1] == 6 and len(raw) >= 32 and np.isfinite(raw).all(),
            "Finite time-by-six input required")
    if condition == "clean":
        return raw.copy()
    mechanism, level = condition.rsplit("__", 1)
    level = int(level)
    window_id = row["window_id"] + "__C1E32_repeat0"
    result = raw.copy()
    for i, channel in enumerate(CHANNELS):
        x = raw[:, i]
        if mechanism == "awgn":
            identity = {"experiment_id": "C1-E25", "acquisition_id": row["acquisition_id"],
                        "protocol": "C1-E25-awgn-grid-savgol41-v2", "window_id": window_id,
                        "channel": channel, "noise": "awgn", "replicate_seed": 42}
        else:
            identity = {"protocol": "C1-E27-R03-completion-v1", "noise": mechanism,
                        "acquisition_id": row["acquisition_id"], "window_id": window_id,
                        "channel": channel, "seed": 42}
        seed = int.from_bytes(hashlib.sha256(canonical(identity)).digest()[:16], "big")
        rng = np.random.Generator(np.random.PCG64(seed))
        if mechanism == "awgn":
            power = float(np.mean(x ** 2))
            require(np.isfinite(power) and power > 0, "Undefined clean power")
            error = rng.standard_normal(len(x))
            error *= np.sqrt(power * 10.0 ** (-float(level) / 10.0) / np.mean(error ** 2))
            reference = x
        else:
            gain = rng.normal(0.0, 0.05) if mechanism in ("gain_plus_awgn", "mixed_gain_additive") else 0.0
            reference = x * (1.0 + gain)
            if mechanism == "lowfreq":
                direction = drift(rng, len(x))
            elif mechanism == "impulsive":
                direction = impulses(rng, len(x))
            elif mechanism == "gain_plus_awgn":
                direction = unit_rms(rng.standard_normal(len(x)))
            else:
                direction = unit_rms(unit_rms(rng.standard_normal(len(x))) + drift(rng, len(x)) + impulses(rng, len(x)))
            amplitude = np.sqrt(np.mean(reference ** 2)) * 10.0 ** (-level / 20)
            require(amplitude > 0, "Undefined gain-perturbed power")
            error = direction * amplitude
        result[:, i] = reference + error
        observed = 10 * np.log10(np.mean(reference ** 2) / np.mean((result[:, i] - reference) ** 2))
        require(abs(observed - level) <= SNR_ATOL_DB, "Exact additive SNR contract failed")
    require(np.isfinite(result).all(), "Nonfinite corrupted waveform")
    return result


def core_features(raw):
    """Only for comparison against saved E32 features; never ResNet predictors."""
    rms = np.sqrt(np.mean(raw ** 2, axis=0) + 1e-12)
    moments = np.column_stack((raw.mean(axis=0), raw.std(axis=0, ddof=1))).reshape(-1)
    return np.concatenate((rms, moments))
