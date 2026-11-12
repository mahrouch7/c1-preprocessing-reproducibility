import numpy as np
import pytest
import torch
import c1inception.engine as engine
from c1inception.data import make_split
from c1inception.io import exclusive_lock, read_json, write_json


def test_full_profile_has_declared_fit_budget(config):
    assert len(list(engine.job_definitions(config, "paper"))) == 390
    assert len(list(engine.job_definitions(config, "pilot"))) == 45
    assert len(list(engine.job_definitions(config, "clean"))) == 15


def test_initialization_matches_across_noise_conditions():
    assert engine.derived_seed(42, 1, "selection") == engine.derived_seed(42, 1, "selection")
    assert engine.derived_seed(42, 1, "selection") != engine.derived_seed(42, 1, "refit")


def test_atomic_json_never_overwrites_conflicting_identity(tmp_path):
    path = tmp_path / "identity.json"
    write_json(path, {"x": 1})
    write_json(path, {"x": 1})
    with pytest.raises(ValueError):
        write_json(path, {"x": 2})
    assert read_json(path) == {"x": 1}


def test_lock_prevents_duplicate_training(tmp_path):
    with exclusive_lock(tmp_path / "running.lock"):
        with pytest.raises(RuntimeError):
            with exclusive_lock(tmp_path / "running.lock"):
                pass
    assert not (tmp_path / "running.lock").exists()


def test_epoch_resume_is_bitwise_identical(tmp_path, tiny_config, synthetic_rows, synthetic_values, monkeypatch):
    device, _ = engine.environment("cpu", cpu_threads=2)
    split = make_split(synthetic_rows, 1)
    baseline = tmp_path / "baseline/selection"
    reference, _, _ = engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split,
                                         42, device, baseline, "selection", "identity", resume=False)
    interrupted = tmp_path / "interrupted/selection"
    original = engine.atomic_checkpoint
    calls = []
    def crash_after_committed_epoch(path, payload):
        original(path, payload)
        calls.append(payload["epoch"])
        raise RuntimeError("SIMULATED_INTERRUPTION")
    monkeypatch.setattr(engine, "atomic_checkpoint", crash_after_committed_epoch)
    with pytest.raises(RuntimeError, match="SIMULATED_INTERRUPTION"):
        engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split,
                           42, device, interrupted, "selection", "identity", resume=False)
    assert calls == [1]
    monkeypatch.setattr(engine, "atomic_checkpoint", original)
    resumed, _, _ = engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split,
                                      42, device, interrupted, "selection", "identity", resume=True)
    assert all(torch.equal(reference.state_dict()[k], resumed.state_dict()[k]) for k in reference.state_dict())
    assert read_json(baseline / "history.json") == read_json(interrupted / "history.json")


def test_refit_uses_all_outer_training_rows(tmp_path, tiny_config, synthetic_rows, synthetic_values):
    device, _ = engine.environment("cpu", cpu_threads=2)
    split = make_split(synthetic_rows, 2)
    write_json(tmp_path / "selection/summary.json", {"best_epoch": 1})
    _, scaler, summary = engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split,
                                           42, device, tmp_path / "refit", "refit", "identity", resume=False)
    assert scaler["fit_row_ids"] == split["outer_train"]
    assert summary["epochs_completed"] == 1
    assert not set(scaler["fit_row_ids"]) & set(split["test"])


def test_resume_rejects_another_checkpoint_identity(tmp_path, tiny_config, synthetic_rows, synthetic_values):
    device, _ = engine.environment("cpu", cpu_threads=2)
    split = make_split(synthetic_rows, 1)
    path = tmp_path / "selection"
    engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split,
                       42, device, path, "selection", "identityA", resume=False)
    with pytest.raises(ValueError, match="identity changed"):
        engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split,
                           42, device, path, "selection", "identityB", resume=True)


def test_whole_job_receipt_and_resume_skip(tmp_path, tiny_config, synthetic_rows, synthetic_values, monkeypatch):
    device, _ = engine.environment("cpu", cpu_threads=2)
    write_json(tmp_path / "run_manifest.json", {"profile": "clean"})
    job = {"training_condition": "clean", "fold": 1, "seed": 42}
    cached = {"sha256": "SYNTHETIC_TEST_CACHE_ONLY"}
    all_values = lambda condition: (synthetic_values, cached)
    engine.run_job(tiny_config, synthetic_rows, None, synthetic_values, cached,
                   all_values, job, device, tmp_path, "SYNTHETIC_MANIFEST", resume=False)
    directory = tmp_path / "jobs/clean/fold01/seed42"
    assert read_json(directory / "completed.json")["fit_procedures"] == 2
    before = (directory / "completed.json").read_bytes()
    def forbidden(*a, **kw):
        raise AssertionError("A completed job must not train again")
    monkeypatch.setattr(engine, "train_phase", forbidden)
    engine.run_job(tiny_config, synthetic_rows, None, synthetic_values, cached,
                   all_values, job, device, tmp_path, "SYNTHETIC_MANIFEST", resume=True)
    assert (directory / "completed.json").read_bytes() == before
