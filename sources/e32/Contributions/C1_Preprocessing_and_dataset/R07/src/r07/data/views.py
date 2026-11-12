"""Self-contained E19 reader. No imports from the frozen acquisition generator."""
from dataclasses import dataclass
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from r07.core.artifacts import child, digest, read_json, sha256
from r07.features.profiles import LABELS, PROFILES, RAW_CHANNELS, RMS

LINES = [("B7_B8", 50), ("B8_B9", 100), ("B9_B6", 100),
         ("B6_B4", 50), ("B4_B5", 50), ("B5_B7", 100)]
PROFILE_ID = "C1E19_case9_365MW_BE1p25us_40k_v1"

@dataclass
class Dataset:
    frame: pd.DataFrame
    adapter: dict
    manifest: dict
    identity: dict
    bundle: Path

def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        fields = next(csv.reader(stream))
    if len(set(fields)) != len(fields):
        raise ValueError("Duplicate CSV column names")
    return pd.read_csv(path, keep_default_na=False, float_precision="round_trip")

def expected_cases(view):
    step = {"reference_10km": 10, "dense_1km": 1}[view]
    return [(f"G1_{line}_p{position:03d}_{fault}", line, position, fault)
            for line, length in LINES for position in range(step, length, step)
            for fault in LABELS[1:]]

def validate_scenarios(frame, links):
    if len(frame) != len(links) * 2:
        raise ValueError("Expected paired healthy/fault windows")
    required = {"Id", "startIndex", "endIndex", "startTime", "endTime", "fault",
                "source_event_group", "group_fault_label", "acquisition_id", "window_id"}
    if not required.issubset(frame):
        raise ValueError("Missing scenario identities or window bounds")
    for field in ("Id", "startIndex", "endIndex", "source_event_group"):
        values = pd.to_numeric(frame[field], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.array_equal(values, np.round(values)):
            raise ValueError("Non-integer scenario field: " + field)
    expected_ids = np.arange(len(frame))
    if not np.array_equal(frame["Id"].to_numpy(), expected_ids):
        raise ValueError("Missing, reordered or duplicated scenario IDs")
    if frame["window_id"].duplicated().any():
        raise ValueError("Duplicate global window identity")
    for i, link in enumerate(links):
        pair = frame.iloc[2*i:2*i+2]
        acq = link["acquisition_id"]
        expected_windows = [acq + "__prefault", acq + "__fault"]
        if (pair["acquisition_id"].tolist() != [acq, acq] or
            pair["window_id"].tolist() != expected_windows or
            pair["fault"].tolist() != ["noFault", link["fault"]] or
            pair["group_fault_label"].tolist() != [link["fault"]] * 2 or
            pair["source_event_group"].tolist() != [i, i]):
            raise ValueError("Label, order or source-event pairing drift")
        if pair["startIndex"].tolist() != [16000*i, 16000*i+8000] or pair["endIndex"].tolist() != [16000*i+7999, 16000*i+15999]:
            raise ValueError("Inclusive window indices differ from paper contract")
        if not np.allclose(pair["startTime"], [i*.4, i*.4+.2], rtol=0, atol=1e-10) or not np.allclose(pair["endTime"], [i*.4+.2, (i+1)*.4], rtol=0, atol=1e-10):
            raise ValueError("Window timing differs from the 40 kHz contract")

def load_view(bundle, registry_path):
    bundle = Path(bundle).resolve()
    registry = read_json(registry_path)
    view = registry["view"]
    pointer = read_json(child(bundle, registry["pointer"]))
    root = child(bundle, pointer["relative_directory"])
    manifest_path = root / "view_manifest.json"
    manifest_sha = sha256(manifest_path)
    if manifest_sha != pointer["manifest_sha256"] or manifest_sha != registry["expected_manifest_sha256"]:
        raise ValueError("Unpinned or altered published dataset manifest")
    report = read_json(manifest_path)
    expected = expected_cases(view)
    if (report.get("status") != "COMPLETE" or report.get("view") != view or
        report.get("profile_id") != PROFILE_ID or report.get("partitioned_primary_dataset") is not True or
        report.get("plan_sha256") != registry["expected_plan_sha256"] or
        report.get("recording_rate_hz") != 40000 or
        report.get("acquisitions") != registry["expected_acquisitions"] or
        report.get("paired_windows") != registry["expected_windows"] or
        len(expected) != registry["expected_acquisitions"]):
        raise ValueError("Dataset status, scope, rate or matrix differs")
    artifacts = report["artifacts"]
    names = [a["file"] for a in artifacts]
    needed = {"adapter_config.json", "merged_data/scenarios.csv",
              "merged_data/window_features.csv", "merged_data/acquisitions.csv"}
    if len(set(names)) != len(names) or not needed.issubset(names):
        raise ValueError("Missing/duplicate declared table artifact")
    for item in artifacts:
        if sha256(child(root, item["file"])) != item["sha256"]:
            raise ValueError("Published table changed: " + item["file"])
    adapter = read_json(root / "adapter_config.json")
    if (adapter["sampling_rate_hz"] != 40000 or adapter["rows_per_acquisition"] != 16000 or
        adapter["native_rows_per_acquisition"] != 18001 or adapter["view"] != view or
        adapter["prefault"] != "[0,.2)" or adapter["fault"] != "[.2,.4)"):
        raise ValueError("Wrong sample rate or window/export semantics")
    declared = adapter["feature_profiles"]["normalization_entrypoint"]
    if any(declared[key] != columns for key, columns in PROFILES.items()):
        raise ValueError("Feature names/order differ from the corrected profile")
    if set(PROFILES["MULTI"]) & set(adapter["excluded_from_predictors"]):
        raise ValueError("Predictors contain forbidden truth metadata")
    links = report["case_links"]
    if [(r["acquisition_id"], r["line"], r["position_km"], r["fault"]) for r in links] != expected:
        raise ValueError("Unexpected acquisition matrix or order")
    for i, link in enumerate(links):
        if (link["view_ordinal"], link["first_row"], link["last_row"]) != (i, i*16000, (i+1)*16000-1):
            raise ValueError("Acquisition boundary drift")
        receipt_path = child(bundle, link["case_receipt_relative_path"])
        if sha256(receipt_path) != link["case_receipt_sha256"]:
            raise ValueError("Source receipt changed")
        receipt = read_json(receipt_path)
        if (receipt["acquisition_id"] != link["acquisition_id"] or
            receipt["plan_sha256"] != report["plan_sha256"] or
            receipt["case"]["fault_label"] != link["fault"] or
            receipt["case"]["position_km"] != link["position_km"] or
            receipt["case"]["line_id"] != link["line"] or
            any(receipt[key] != link[key] for key in ("raw_file_sha256", "raw_numeric_sha256", "time_numeric_sha256")) or
            receipt["complete_recording_rows"] != 18001 or receipt["recording_period_s"] != 25e-6):
            raise ValueError("Receipt belongs to another acquisition/configuration")
    scenarios = read_csv(root / "merged_data/scenarios.csv")
    validate_scenarios(scenarios, links)
    acquisitions = read_csv(root / "merged_data/acquisitions.csv")
    if acquisitions.to_dict("records") != links:
        raise ValueError("Acquisition table does not match the manifest")
    features = read_csv(root / "merged_data/window_features.csv")
    if features["window_id"].tolist() != scenarios["window_id"].tolist():
        raise ValueError("Missing, duplicated or misaligned feature windows")
    if len(features.columns) != 325 or not set(PROFILES["MULTI"]).issubset(features):
        raise ValueError("Wrong feature table schema")
    selected = features[PROFILES["MULTI"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(selected.to_numpy()).all() or (selected[RMS].to_numpy() < 0).any():
        raise ValueError("Non-finite/invalid features; no silent imputation")
    for channel, rms_name in zip(RAW_CHANNELS, RMS):
        mean, std = selected[channel + "_mean"], selected[channel + "_std"]
        if (std < 0).any() or not np.allclose(selected[rms_name] ** 2,
                mean ** 2 + std ** 2 * 7999 / 8000 + 1e-12, rtol=1e-10, atol=1e-8):
            raise ValueError("RMS/mean/sample-std contract mismatch")
    frame = pd.concat([scenarios.reset_index(drop=True), selected.reset_index(drop=True)], axis=1)
    identity = dict(view=view, manifest_sha256=manifest_sha, plan_sha256=report["plan_sha256"],
                    acquisitions=len(links), windows=len(frame), sampling_rate_hz=40000,
                    tables=artifacts, feature_profile=PROFILES,
                    source_receipts_verified=True, bulk_raw_rehashed=False)
    return Dataset(frame, adapter, report, identity, bundle)

def read_acquisition(dataset, ordinal):
    """Optional saved-only numeric MAT boundary, not used by the clean table pilot."""
    import hashlib
    from scipy.io import loadmat
    link = dataset.manifest["case_links"][ordinal]
    receipt = read_json(child(dataset.bundle, link["case_receipt_relative_path"]))
    path = child(dataset.bundle, receipt["raw_relative_path"])
    if sha256(path) != receipt["raw_file_sha256"]:
        raise ValueError("Numeric MAT file hash changed")
    saved = loadmat(path, variable_names=["time", "raw"])
    t, raw = saved["time"], saved["raw"]
    if t.dtype != np.float64 or raw.dtype != np.float64:
        raise ValueError("Lossless native float64 required")
    t = t.reshape(-1)
    if raw.shape != (18001, 108) or t.shape != (18001,) or not np.isfinite(raw).all() or not np.isfinite(t).all():
        raise ValueError("Incomplete/non-finite acquisition")
    if not np.all(np.diff(t) > 0) or not np.allclose(t, np.arange(18001)*25e-6, rtol=0, atol=1e-11):
        raise ValueError("Native recording clock changed")
    for value, name in ((t, "time_numeric_sha256"), (raw, "raw_numeric_sha256")):
        if hashlib.sha256(np.asarray(value, dtype="<f8", order="C").tobytes()).hexdigest() != receipt[name]:
            raise ValueError("Native numeric values changed")
    indices = [dataset.adapter["raw_columns"].index(n) for n in RAW_CHANNELS]
    return t, raw[:, indices]
