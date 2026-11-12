"""R03 stratified event grouping rewritten around stable acquisition identities."""
import numpy as np
import pandas as pd
from r07.core.artifacts import digest

def make_folds(frame, n_splits=5, seed=42):
    if isinstance(n_splits, bool) or not isinstance(n_splits, int) or n_splits < 2:
        raise ValueError("At least two integer folds required")
    if frame["window_id"].duplicated().any():
        raise ValueError("Duplicate window")
    groups = []
    for acquisition, pair in frame.groupby("acquisition_id", sort=False):
        if len(pair) != 2 or pair["fault"].tolist()[0] != "noFault" or pair["fault"].tolist()[1] == "noFault":
            raise ValueError("Healthy/fault pairing changed")
        if pair["group_fault_label"].nunique() != 1 or pair["group_fault_label"].iloc[0] != pair["fault"].iloc[1]:
            raise ValueError("Group stratum label changed")
        groups.append((acquisition, pair["fault"].iloc[1]))
    table = pd.DataFrame(groups, columns=["acquisition_id", "fault"])
    if (table["fault"].value_counts() < n_splits).any():
        raise ValueError("Insufficient groups per fault class")
    rng = np.random.default_rng(seed)
    assignments = {}
    # Preserve view-local acquisition order before seeded shuffling, as in R03.
    for offset, (_, group) in enumerate(table.groupby("fault", sort=True)):
        ids = group["acquisition_id"].to_numpy().copy()
        rng.shuffle(ids)
        for position, acq in enumerate(ids):
            assignments[acq] = (position + offset) % n_splits + 1
    manifest = frame[["Id", "window_id", "acquisition_id", "fault", "group_fault_label"]].copy()
    manifest["fold"] = manifest["acquisition_id"].map(assignments).astype(int)
    manifest["seed"] = seed
    manifest["n_splits"] = n_splits
    validate_folds(frame, manifest, n_splits)
    return manifest

def validate_folds(frame, manifest, n_splits):
    keys = ["Id", "window_id", "acquisition_id", "fault", "group_fault_label"]
    if manifest[keys].to_dict("records") != frame[keys].to_dict("records"):
        raise ValueError("Frozen folds differ from this dataset")
    folds = manifest["fold"].to_numpy()
    if not np.isfinite(folds).all() or not np.array_equal(folds, folds.astype(int)):
        raise ValueError("Invalid fold numbers")
    if sorted(set(folds)) != list(range(1, n_splits + 1)):
        raise ValueError("Missing or unexpected folds")
    if not (manifest.groupby("acquisition_id")["fold"].nunique() == 1).all():
        raise ValueError("One source event leaks across folds")
    for fold in range(1, n_splits + 1):
        train = frame.loc[folds != fold]
        test = frame.loc[folds == fold]
        if set(train["acquisition_id"]) & set(test["acquisition_id"]):
            raise ValueError("Train/test acquisition leakage")
        if set(train["fault"]) != set(test["fault"]):
            raise ValueError("Missing class in a fold")
    return digest(manifest.to_dict("records"))
