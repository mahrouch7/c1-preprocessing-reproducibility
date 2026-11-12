"""Pinned E32 ingestion, grouped partitions, and immutable waveform caches."""
from __future__ import annotations

from collections import Counter
import hashlib
import math
import os
from pathlib import Path
import shutil
import uuid
import numpy as np

from .io import digest, exclusive_lock, read_json, require, sha256, write_json
from .noise import CHANNELS, CONDITIONS, corrupt, core_features

ROOT = Path(__file__).resolve().parents[2]
LABELS = ["noFault", "ab", "ac", "bc", "abc", "ag", "bg", "cg", "abg", "acg", "bcg", "abcg"]
FEATURE_MANIFEST_SHA = "5e487b09df920f060f5039274790a3f0fdad609b4f054fbd48ca56ca8448bed2"


def load_config(path=None):
    config = read_json(path or ROOT / "configs/experiment.json")
    require(config["experiment_id"] == "C1-E43" and config["schema_version"] == 1, "Wrong experiment/schema")
    require(config["labels"] == LABELS and config["channels"] == CHANNELS, "Class/channel ordering changed")
    require(config["samples"] == 8000 and config["sampling_rate_hz"] == 40000, "Unsupported physical rate/window")
    require(config["expected_rows"] == 1839 and config["expected_families"] == 39, "Wrong cohort declaration")
    require(config["profiles"]["paper"] == list(CONDITIONS), "Paper condition grid differs from E32")
    for profile in config["profiles"].values():
        require(bool(profile) and len(set(profile)) == len(profile) and set(profile) <= set(CONDITIONS), "Bad profile")
        require(profile[0] == "clean", "Clean reference is required and shared")
    t = config["training"]
    require(t["outer_folds"] == [1, 2, 3, 4, 5] and len(set(t["seeds"])) == len(t["seeds"]) > 0, "Bad folds/seeds")
    require(0 < t["validation_family_fraction"] < 0.5 and t["max_epochs"] >= 1 and t["patience"] >= 1, "Bad stopping rule")
    require(t["optimizer"] == "AdamW" and t["class_weighting"] == "none" and t["refit_full_outer_training"], "Unreviewed training policy")
    require(t["normalization"] == "per_channel_mean_std_from_training_rows_and_time", "Unreviewed scaling")
    require(t["num_workers"] == 0 and t["deterministic_algorithms"], "Reference runner requires deterministic single-process loading")
    require(t["batch_size"] >= 2 and t["evaluation_batch_size"] >= 1, "Invalid batch sizes")
    return config


def resolve_input(config, name):
    path = Path(config[name])
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def validate_rows(rows, *, production=True):
    require(isinstance(rows, list) and len(rows) > 0, "Metadata list required")
    require([r["Id"] for r in rows] == list(range(len(rows))), "Row identities/order changed")
    for key in ("window_id", "acquisition_id", "raw_sha256", "clean_window_numeric_sha256"):
        require(len({r[key] for r in rows}) == len(rows), "Duplicate or missing identity: " + key)
    families = {}
    load_points = {}
    for row in rows:
        require(row["fault"] in LABELS and row["fold"] in range(1, 6), "Invalid label/fold")
        family = row["family_id"]
        require(row["source_group_id"] == family, "Source/family drift")
        require(family not in families or families[family] == row["fold"], "Family crosses outer folds")
        families[family] = row["fold"]
        point = tuple(round(float(x), 12) for x in row["factors"])
        require(point not in load_points or load_points[point] == family, "Exact load regime crosses families")
        load_points[point] = family
    for fold in range(1, 6):
        for held_out in (False, True):
            require({r["fault"] for r in rows if (r["fold"] == fold) == held_out} == set(LABELS), "A partition lacks a class")
    if production:
        require(len(rows) == 1839 and len(families) == 39, "Wrong E32 cohort")
        require(Counter(r["fault"] for r in rows) == {"noFault": 156, **{x: 153 for x in LABELS[1:]}}, "Class counts changed")
        names = sorted(families)
        order = np.random.Generator(np.random.PCG64(20261026)).permutation(len(names))
        expected = {names[int(i)]: 1 + j % 5 for j, i in enumerate(order)}
        require(families == expected, "Original family fold assignment changed")
    return rows


def load_source(config, *, verify_rows=True):
    root = resolve_input(config, "source_root")
    for relative, expected in config["source_pins"].items():
        path = (root / relative).resolve()
        require(root in path.parents and path.is_file() and not path.is_symlink(), "Invalid source input: " + relative)
        require(sha256(path) == expected, "Pinned source bytes changed: " + relative)
    rows = validate_rows(read_json(root / "metadata.json"))
    scope = read_json(root / "scope.json")
    require(scope["dataset_view_sha256"] == config["dataset_view_sha256"], "Dataset view drift")
    clean = np.load(root / "base/clean.npy", mmap_mode="r", allow_pickle=False)
    require(clean.shape == (1839, 8000, 6) and clean.dtype == np.dtype("float64"), "Wrong clean waveform schema")
    if verify_rows:
        for row in rows:
            x = np.asarray(clean[row["Id"]], dtype="<f8", order="C")
            require(np.isfinite(x).all() and np.all(np.mean(x ** 2, axis=0) > 0), "Nonfinite/zero-power source")
            require(hashlib.sha256(x.tobytes()).hexdigest() == row["clean_window_numeric_sha256"], "Per-window numeric identity changed")
    return rows, clean


def make_split(rows, fold, fraction=0.2, seed=20261029):
    """The inner family split is fixed before model seeds/outcomes are examined."""
    require(fold in range(1, 6), "Invalid outer fold")
    outer_train = [r["Id"] for r in rows if r["fold"] != fold]
    test = [r["Id"] for r in rows if r["fold"] == fold]
    names = sorted({rows[i]["family_id"] for i in outer_train})
    order = np.random.Generator(np.random.PCG64(seed + fold)).permutation(len(names))
    count = max(1, math.ceil(len(names) * fraction))
    validation_families = {names[int(i)] for i in order[:count]}
    inner_train = [i for i in outer_train if rows[i]["family_id"] not in validation_families]
    validation = [i for i in outer_train if rows[i]["family_id"] in validation_families]
    sets = [{rows[i]["family_id"] for i in ids} for ids in (inner_train, validation, test)]
    require(all(not sets[a] & sets[b] for a, b in ((0, 1), (0, 2), (1, 2))), "Inner/outer family leakage")
    for ids in (inner_train, validation, outer_train, test):
        require({rows[i]["fault"] for i in ids} == set(LABELS), "A nested split lacks a class")
    return {"fold": fold, "inner_train": inner_train, "validation": validation,
            "outer_train": outer_train, "test": test,
            "inner_train_families": sorted(sets[0]), "validation_families": sorted(sets[1]),
            "test_families": sorted(sets[2])}


def verify_feature_parity(config, rows, clean):
    """Data-integrity audit, not model selection: compare the saved E32 18 features."""
    root = resolve_input(config, "baseline_root")
    require(sha256(root / "metadata.json") == config["source_pins"]["metadata.json"], "Baseline row alignment changed")
    manifest_path = root / "verified_features/feature_manifest.json"
    require(sha256(manifest_path) == FEATURE_MANIFEST_SHA, "Saved E32 feature manifest changed")
    manifest = read_json(manifest_path)
    selected = [0, 156, 1000]
    reports = []
    for condition in CONDITIONS:
        name = condition + "_none.npy"
        path = root / "verified_features" / name
        require(sha256(path) == manifest[name], "Saved feature bytes changed: " + name)
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        require(features.shape == (1839, 36), "Saved feature schema changed")
        errors = []
        for i in selected:
            actual = core_features(corrupt(clean[i], rows[i], condition))
            expected = features[i, :18]
            require(np.allclose(actual, expected, rtol=1e-12, atol=1e-10), "E32 noise/feature parity failure: " + condition)
            errors.append(float(np.max(np.abs(actual - expected))))
        reports.append({"condition": condition, "row_ids": selected, "max_absolute_feature_error": max(errors)})
    return reports


def cache_key(config):
    return digest({"source_pins": config["source_pins"], "noise_code": sha256(Path(__file__).with_name("noise.py")),
                   "layout": "NCT_float32_after_float64_corruption_v1"})[:20]


def condition_cache(config, rows, clean, condition, cache_root):
    require(condition in CONDITIONS, "Unknown cache condition")
    directory = Path(cache_root) / cache_key(config)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (condition + ".npy")
    receipt = directory / (condition + ".json")
    if receipt.exists():
        record = read_json(receipt)
        require(record["condition"] == condition and record["cache_key"] == cache_key(config), "Cache identity mismatch")
        require(path.is_file() and sha256(path) == record["sha256"], "Corrupt cached waveform")
    else:
        require(not path.exists(), "Unreceipted waveform exists; inspect it rather than overwrite")
        needed = len(rows) * 6 * config["samples"] * 4 + 1024 * 1024 * 1024
        require(shutil.disk_usage(directory).free >= needed, "Insufficient free cache space")
        with exclusive_lock(directory / (condition + ".lock")):
            partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
            values = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32,
                                             shape=(len(rows), 6, config["samples"]))
            for row in rows:
                values[row["Id"]] = corrupt(clean[row["Id"]], row, condition).T.astype(np.float32)
                if row["Id"] % 200 == 0:
                    print(f"[CACHE] {condition}: {row['Id']}/{len(rows)}; model fits=0", flush=True)
            values.flush()
            del values
            os.replace(partial, path)
            write_json(receipt, {"condition": condition, "cache_key": cache_key(config), "sha256": sha256(path),
                                 "shape": [len(rows), 6, config["samples"]], "dtype": "float32"})
    data = np.load(path, mmap_mode="r", allow_pickle=False)
    require(data.shape == (len(rows), 6, config["samples"]) and data.dtype == np.float32, "Invalid cache schema")
    return data, read_json(receipt)


def fit_scaler(values, ids, *, chunk_size=32):
    """Pooled channel mean/std, fitted ONLY on the explicitly supplied row IDs."""
    ids = np.asarray(ids, dtype=np.int64)
    require(len(ids) > 0 and len(set(ids.tolist())) == len(ids), "Invalid scaler fit rows")
    count, mean, m2 = 0, np.zeros(6), np.zeros(6)
    for start in range(0, len(ids), chunk_size):
        block = np.asarray(values[ids[start:start + chunk_size]], dtype=np.float64)
        require(block.ndim == 3 and block.shape[1] == 6 and np.isfinite(block).all(), "Invalid scaler input")
        n = block.shape[0] * block.shape[2]
        block_mean = block.mean(axis=(0, 2))
        block_m2 = np.sum((block - block_mean[None, :, None]) ** 2, axis=(0, 2))
        delta = block_mean - mean
        mean += delta * n / (count + n)
        m2 += block_m2 + delta ** 2 * count * n / (count + n)
        count += n
    std = np.sqrt(m2 / count)
    require(np.isfinite(std).all() and np.all(std > 1e-12), "Degenerate training-channel scale")
    return {"mean": mean.tolist(), "std": std.tolist(), "fit_row_ids": ids.tolist(),
            "fit_row_ids_sha256": digest(ids.tolist()), "sample_count_per_channel": count, "ddof": 0}
