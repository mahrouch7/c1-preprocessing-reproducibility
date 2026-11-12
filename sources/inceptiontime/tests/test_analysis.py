from pathlib import Path
import numpy as np
import pytest
import c1inception.analysis as analysis
from c1inception.data import LABELS
from c1inception.io import digest, sha256, write_json


def fake_run(root, config, rows, *, folds=(1, 2, 3, 4, 5)):
    config["training"]["seeds"] = [42]
    write_json(root / "run_manifest.json", {"config": config, "profile": "clean"})
    for fold in folds:
        directory = root / "jobs/clean" / f"fold{fold:02d}" / "seed42"
        (directory / "predictions").mkdir(parents=True)
        identity = {"synthetic_test": True, "fold": fold}
        write_json(directory / "identity.json", identity)
        ids = np.array([r["Id"] for r in rows if r["fold"] == fold])
        truth = np.array([LABELS.index(rows[int(i)]["fault"]) for i in ids])
        path = directory / "predictions/clean.npz"
        np.savez_compressed(path, row_ids=ids, truth=truth, predicted=truth, probabilities=np.eye(12)[truth])
        write_json(directory / "prediction_catalogue.json", [{"condition": "clean", "prediction_file": path.name,
                    "prediction_sha256": sha256(path)}])
        artifacts = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob("*") if p.is_file()}
        write_json(directory / "completed.json", {"status": "COMPLETE", "identity_hash": digest(identity),
                                                  "fit_procedures": 2, "artifacts": artifacts})


def test_full_oof_aggregation_is_not_mean_fold_score(tmp_path, config, synthetic_rows, monkeypatch):
    root = tmp_path / "run"
    fake_run(root, config, synthetic_rows)
    monkeypatch.setattr(analysis, "load_source", lambda *a, **k: (synthetic_rows, None))
    monkeypatch.setattr(analysis, "baseline_comparisons", lambda *a, **k: [])
    report = analysis.aggregate(root, tmp_path / "analysis")
    assert report["status"] == "ALL_EXPECTED_JOBS_COMPLETE"
    assert report["complete_oof_rows"] == 1
    with np.load(tmp_path / "analysis/oof/seed42_shared_clean_clean.npz") as data:
        np.testing.assert_array_equal(data["row_ids"], np.arange(len(synthetic_rows)))


def test_missing_folds_cannot_be_called_complete(tmp_path, config, synthetic_rows, monkeypatch):
    root = tmp_path / "run"
    fake_run(root, config, synthetic_rows, folds=(1,))
    monkeypatch.setattr(analysis, "load_source", lambda *a, **k: (synthetic_rows, None))
    monkeypatch.setattr(analysis, "baseline_comparisons", lambda *a, **k: [])
    with pytest.raises(ValueError, match="jobs missing"):
        analysis.aggregate(root, tmp_path / "rejected")
    report = analysis.aggregate(root, tmp_path / "partial", allow_partial=True)
    assert report["status"] == "PARTIAL_NOT_PUBLICATION_READY"
    assert report["complete_oof_rows"] == 0


def test_modified_prediction_is_rejected(tmp_path, config, synthetic_rows, monkeypatch):
    root = tmp_path / "run"
    fake_run(root, config, synthetic_rows)
    monkeypatch.setattr(analysis, "load_source", lambda *a, **k: (synthetic_rows, None))
    path = root / "jobs/clean/fold01/seed42/predictions/clean.npz"
    with path.open("ab") as stream:
        stream.write(b"CORRUPTED")
    with pytest.raises(ValueError, match="artifact changed"):
        analysis.aggregate(root, tmp_path / "rejected")


def test_comparisons_read_case_keyed_index_and_keep_all_24(tmp_path, config, synthetic_rows, monkeypatch):
    baseline_root = tmp_path / "baseline"
    final = baseline_root / "final_analysis"
    (final / "oof").mkdir(parents=True)
    ids = np.arange(len(synthetic_rows))
    truth = np.array([LABELS.index(r["fault"]) for r in synthetic_rows])
    probabilities = np.eye(12)[truth]
    entries = {}
    for number in range(24):
        path = final / "oof" / f"fixture{number:02d}.npz"
        np.savez_compressed(path, row_ids=ids, probabilities=probabilities)
        entries[f"case{number:02d}"] = {
            "case": {"study":"robustness", "preprocessor":"none", "anchor_only":False,
                     "density_balanced":False, "representation":("RMS6","RAW12","MULTI18")[number % 3],
                     "classifier":f"classifier{number // 3}", "condition":"clean", "protocol":"deployment", "normalizer":"none"},
            "path":path.relative_to(final).as_posix(), "sha256":sha256(path), "eligible_for_inference":number != 0}
    index = final / "pooled_prediction_index.json"
    write_json(index, entries)
    monkeypatch.setattr(analysis, "INDEX_SHA", sha256(index))
    monkeypatch.setattr(analysis, "resolve_input", lambda *args: baseline_root)
    inception = analysis.score(synthetic_rows, ids, probabilities)
    result = analysis.baseline_comparisons(config, synthetic_rows, {(42,"shared_clean","clean"):inception})
    assert len(result) == 24 and len({x["baseline_case_id"] for x in result}) == 24
    assert sum(not x["E32_inferential_eligibility"] for x in result) == 1
    assert all(x["difference_macro_f1_pp"] == x["difference_balanced_accuracy_pp"] == x["difference_mcc"] == 0 for x in result)


def test_index_list_is_rejected_instead_of_silently_misread(tmp_path, config, synthetic_rows, monkeypatch):
    baseline_root=tmp_path/"baseline"
    index=baseline_root/"final_analysis/pooled_prediction_index.json"
    write_json(index,[{"unexpected":"list schema"}])
    monkeypatch.setattr(analysis,"INDEX_SHA",sha256(index))
    monkeypatch.setattr(analysis,"resolve_input",lambda *args:baseline_root)
    with pytest.raises(ValueError,match="keyed E32 index mapping"):
        analysis.baseline_comparisons(config,synthetic_rows,{(42,"shared_clean","clean"): {}})
