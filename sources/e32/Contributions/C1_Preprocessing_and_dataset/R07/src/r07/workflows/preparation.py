"""Read-only preparation is separate from orchestration and model fitting."""
from pathlib import Path
from r07.core.artifacts import digest, publish_bytes, read_json, sha256, write_csv, write_json
from r07.core.config import load_config, source_identity, verify_lineage
from r07.data.views import load_view
from r07.learning.folds import make_folds, validate_folds

def prepare(bundle, registry_path, config_path, root, repo, workspace, environment_sha):
    config = load_config(config_path)
    verify_lineage(root, repo)
    dataset = load_view(bundle, registry_path)
    if dataset.identity["view"] != config["view"]:
        raise ValueError("Experiment and dataset views differ")
    folds = make_folds(dataset.frame, config["n_splits"], config["seed"])
    fold_sha = validate_folds(dataset.frame, folds, config["n_splits"])
    science = source_identity(root)
    identity = dict(experiment=config, dataset=dataset.identity, folds_sha256=fold_sha,
                    source=science, environment_sha256=environment_sha)
    run_id = digest(identity)
    output = Path(workspace) / "experiments" / "C1-E20" / ("run_" + run_id[:20])
    output.mkdir(parents=True, exist_ok=True)
    lineage_path = Path(root) / "provenance/lineage.json"
    lineage_sha = sha256(lineage_path)
    snapshot = output / "provenance" / ("lineage_" + lineage_sha + ".json")
    if not snapshot.exists():
        publish_bytes(snapshot, lineage_path.read_bytes())
    if sha256(snapshot) != lineage_sha:
        raise ValueError("Preserved full lineage snapshot changed")
    plan_path = output / "run_manifest.json"
    if plan_path.exists():
        if read_json(plan_path)["identity"] != identity:
            raise ValueError("Output identity collision; never mix runs")
    else:
        write_csv(output / "fold_manifest.csv", folds)
        write_json(plan_path, dict(schema_version=1, run_id=run_id, identity=identity,
                                 fold_file_sha256=sha256(output / "fold_manifest.csv"),
                                 status="FROZEN_EXECUTION_PLAN", engine="prefect_self_hosted_local",
                                 planned_fits=len(config["modalities"])*len(config["models"])*config["n_splits"],
                                 new_simulations=0, publication_qualified=False))
    expected_sha = read_json(plan_path)["fold_file_sha256"]
    if sha256(output / "fold_manifest.csv") != expected_sha:
        raise ValueError("Saved fold manifest changed")
    return dict(frame=dataset.frame, folds=folds, config=config, run_id=run_id,
                output=output, identity=identity,
                lineage_snapshot=snapshot.relative_to(output).as_posix(), lineage_sha256=lineage_sha)
