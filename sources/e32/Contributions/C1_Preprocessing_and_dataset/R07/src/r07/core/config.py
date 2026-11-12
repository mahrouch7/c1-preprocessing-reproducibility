"""Explicit initial pilot budget, source identities and source provenance checks."""
from pathlib import Path
from r07.core.artifacts import child, digest, read_json, sha256

def package_root():
    return Path(__file__).resolve().parents[3]

def load_config(path):
    value = read_json(path)
    expected = dict(experiment_id="C1-E20", schema_version=1, condition="measured_clean",
                    n_splits=5, seed=42, modalities=["RMS", "RAW", "MULTI"],
                    models=["dummy_prior", "logistic_regression"],
                    scaling="standard_training_fold_only", logistic_max_iter=10000,
                    feature_profile="normalization_entrypoint", sampling_rate_hz=40000,
                    allow_missing_values=False, publication_qualified=False, view="reference_10km")
    if any(value.get(k) != v for k, v in expected.items()):
        raise ValueError("Unsupported/unreviewed initial pilot configuration")
    if set(value) != set(expected):
        raise ValueError("Unknown pilot settings are not silently ignored")
    return value

def source_identity(root):
    root = Path(root)
    sources = {}
    # Presentation/engine changes do not invalidate already verified model fits.
    for domain in ("core", "data", "features", "learning", "evaluation"):
        for path in sorted((root / "src/r07" / domain).glob("*.py")):
            sources[path.relative_to(root).as_posix()] = sha256(path)
    for relative in ("pyproject.toml",):
        sources[relative] = sha256(root / relative)
    for relative in ("src/r07/workflows/preparation.py", "src/r07/workflows/units.py"):
        sources[relative] = sha256(root / relative)
    lineage = read_json(root / "provenance/lineage.json")
    scientific_lineage = {"sources": lineage["sources"],
                          "destinations": [d for d in lineage["destinations"] if d["path"] in sources]}
    sources["provenance/lineage.json:scientific_subset"] = digest(scientific_lineage)
    return {"files": sources, "sha256": digest(sources)}

def verify_lineage(root, repo):
    lineage = read_json(Path(root) / "provenance/lineage.json")
    for source in lineage["sources"]:
        if sha256(child(repo, source["path"])) != source["sha256"]:
            raise ValueError("Historical provenance source changed: " + source["path"])
    for destination in lineage["destinations"]:
        if sha256(child(root, destination["path"])) != destination["sha256"]:
            raise ValueError("R07 implementation changed without updated provenance: " + destination["path"])
    return lineage
