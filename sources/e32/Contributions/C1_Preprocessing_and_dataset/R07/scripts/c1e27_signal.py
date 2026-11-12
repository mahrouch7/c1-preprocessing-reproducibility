"""R03 signal/descriptor adapters with explicit 40-kHz semantics; no simulation."""
from __future__ import annotations

import hashlib
import numpy as np
from scipy.signal import butter, medfilt, sosfiltfilt
from scipy.stats import skew, kurtosis
import c1e25 as base
import c1e27_spec as spec


def finite_window(raw):
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 6 or len(raw) < 32 or not np.isfinite(raw).all():
        raise ValueError("A finite time-by-six-channel window of at least 32 samples is required")
    return raw


def unit_rms(values):
    norm = np.sqrt(np.mean(np.asarray(values) ** 2))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Undefined corruption-component RMS")
    return values / norm


def random_stream(row, channel, noise):
    identity = {"protocol": spec.PROTOCOL, "noise": noise, "acquisition_id": row["acquisition_id"],
                "window_id": row["window_id"], "channel": base.CHANNELS[channel], "seed": 42}
    seed = int.from_bytes(hashlib.sha256(base.canonical(identity)).digest()[:16], "big")
    return np.random.Generator(np.random.PCG64(seed))


def drift(rng, n):
    sos = butter(4, 5.0, btype="lowpass", fs=spec.FS, output="sos")
    return unit_rms(sosfiltfilt(sos, rng.standard_normal(n), padtype="odd", padlen=15))


def impulses(rng, n):
    starts = rng.random(n) < (100.0 / spec.FS)
    if not starts.any():
        starts[int(rng.integers(n))] = True
    signed = starts * rng.choice([-1.0, 1.0], size=n)
    pulse = np.convolve(signed, np.ones(4), mode="full")[:n]
    return unit_rms(pulse)


def corrupt(raw, row, condition):
    raw = finite_window(raw)
    noise, snr = spec.condition(condition)
    if noise == "clean":
        return raw.copy(), {"additive_snr_db": None, "total_error_snr_db": None}
    if noise == "awgn":
        result = base.awgn(raw, row["acquisition_id"], row["window_id"], snr)
        return result, {"additive_snr_db": [float(snr)] * 6, "total_error_snr_db": [float(snr)] * 6}
    result, additive_snr, total_snr = raw.copy(), [], []
    for channel in range(6):
        x = raw[:, channel]
        rng = random_stream(row, channel, noise)
        gain = rng.normal(0.0, 0.05) if noise in ("gain_plus_awgn", "mixed_gain_additive") else 0.0
        reference = x * (1.0 + gain)
        if noise == "lowfreq":
            direction = drift(rng, len(x))
        elif noise == "impulsive":
            direction = impulses(rng, len(x))
        elif noise == "gain_plus_awgn":
            direction = unit_rms(rng.standard_normal(len(x)))
        else:
            direction = unit_rms(unit_rms(rng.standard_normal(len(x))) + drift(rng, len(x)) + impulses(rng, len(x)))
        amplitude = np.sqrt(np.mean(reference ** 2)) * 10.0 ** (-snr / 20)
        if amplitude <= 0:
            raise ValueError("SNR is undefined for a zero-power channel")
        error = direction * amplitude
        result[:, channel] = reference + error
        observed = 10 * np.log10(np.mean(reference ** 2) / np.mean((result[:, channel] - reference) ** 2))
        if abs(observed - snr) > base.SNR_ATOL_DB:
            raise ValueError("Per-channel additive SNR contract failed")
        additive_snr.append(float(observed))
        total_snr.append(float(10 * np.log10(np.mean(x ** 2) / np.mean((result[:, channel] - x) ** 2))))
    return result, {"additive_snr_db": additive_snr, "total_error_snr_db": total_snr}


def wavelet_module():
    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError("PyWavelets 1.8.0 is required; no unfiltered wavelet fallback is allowed") from exc
    return pywt


def preprocess(raw, name):
    raw = finite_window(raw)
    if name in ("none", "savgol"):
        return base.preprocess(raw, name)
    if name == "median":
        return medfilt(raw, kernel_size=(17, 1))
    if name not in ("wavelet_soft", "wavelet_hard"):
        raise ValueError("Unreviewed filter")
    pywt = wavelet_module()
    coefficients = pywt.wavedec(raw, "db4", mode="symmetric", level=5, axis=0)
    finest = coefficients[-1]
    centered = finest - np.median(finest, axis=0)
    sigma = np.median(np.abs(centered), axis=0) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(raw)))
    mode = "soft" if name == "wavelet_soft" else "hard"
    details = [pywt.threshold(c, threshold, mode=mode) for c in coefficients[1:]]
    result = pywt.waverec([coefficients[0]] + details, "db4", mode="symmetric", axis=0)[:len(raw)]
    if result.shape != raw.shape or not np.isfinite(result).all():
        raise ValueError("Wavelet reconstruction contract failed")
    return result


def features(raw):
    raw = finite_window(raw)
    values = [*base.feature_vector(raw)]
    for channel in range(6):
        x = raw[:, channel]
        rms, std = float(np.sqrt(np.mean(x ** 2))), float(np.std(x, ddof=1))
        third, fourth = (0.0, 3.0) if std <= 1e-12 else (
            float(skew(x, bias=False, nan_policy="raise")),
            float(kurtosis(x, fisher=False, bias=False, nan_policy="raise")))
        values.extend((third, fourth, float(np.max(np.abs(x)) / rms) if rms > 1e-12 else 0.0))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (36,) or not np.isfinite(result).all():
        raise ValueError("Nonfinite or misaligned RAW+ descriptors")
    return result


TPI_OPERATORS = ("DWT_D3_2500_5000Hz", "DWT_D1_10000_20000Hz", "FFT_2500_5000Hz")
TPI_FIELDS = ("energy_clean", "energy_noisy", "energy_processed", "tpi_noisy", "tpi_clean",
              "noise_inflation", "hf_error_clean_normalized", "clean_near_zero", "noisy_near_zero")


def energies(raw):
    raw = finite_window(raw)
    pywt = wavelet_module()
    coefficients = pywt.wavedec(raw, "db4", level=3, mode="periodization", axis=0)
    d3, d1 = (np.sum(c ** 2, axis=0) for c in (coefficients[1], coefficients[-1]))
    fourier = np.fft.rfft(raw, axis=0)
    frequencies = np.fft.rfftfreq(len(raw), d=1 / spec.FS)
    weights = np.full(len(frequencies), 2.0)
    weights[0] = 1.0
    if len(raw) % 2 == 0:
        weights[-1] = 1.0
    band = (frequencies >= 2500) & (frequencies <= 5000)
    fft_energy = np.sum(weights[band, None] * np.abs(fourier[band]) ** 2, axis=0) / len(raw)
    return np.column_stack((d3, d1, fft_energy))


def tpi(clean, noisy, processed, clean_energy=None, noisy_energy=None):
    ec = energies(clean) if clean_energy is None else clean_energy
    en = energies(noisy) if noisy_energy is None else noisy_energy
    ep, ee = energies(processed), energies(processed - clean)
    dc, dn = np.maximum(ec, 1e-12), np.maximum(en, 1e-12)
    result = np.stack((ec, en, ep, ep / dn, ep / dc, en / dc, ee / dc, ec <= 1e-12, en <= 1e-12), axis=-1)
    if result.shape != (6, 3, 9) or not np.isfinite(result).all():
        raise ValueError("Invalid TPI diagnostic tensor")
    return result
