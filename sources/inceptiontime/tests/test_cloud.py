import time
import zipfile
import pytest
import torch
from c1inception import engine
from c1inception.cloud import safe_members, save_resume_archive, restore_archive, runtime_key
from c1inception.data import make_split
from c1inception.io import read_json, write_json


@pytest.mark.parametrize("name", ["../bad", "/absolute", "C:/drive", "folder\\bad"])
def test_archive_path_safety(tmp_path, name):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, "invalid")
    with zipfile.ZipFile(path) as archive, pytest.raises(ValueError):
        # Windows ZipInfo normalizes backslashes on construction; emulate the
        # raw member name a foreign/malicious archive could actually contain.
        archive.filelist[0].filename = name
        safe_members(archive)


def test_archive_roundtrip_excludes_locks(tmp_path):
    run = tmp_path / "run"
    write_json(run / "run_manifest.json", {"experiment_id": "C1-E43"})
    write_json(run / "phase/checkpoint.json", {"epoch": 1})
    (run / "coordinator.lock").touch()
    archive = tmp_path / "c1e43_resume.zip"
    save_resume_archive(run, archive)
    restored = tmp_path / "restored"
    restore_archive(archive, restored)
    assert read_json(restored / "phase/checkpoint.json") == {"epoch": 1}
    assert not (restored / "coordinator.lock").exists()
    with pytest.raises(ValueError):
        restore_archive(archive, restored)


def test_gpu_requires_cuda_no_fallback():
    with pytest.raises(ValueError, match="GPU execution required"):
        runtime_key({"device": "cpu"})


def test_budget_pause_is_bitwise_resumable(tmp_path, tiny_config, synthetic_rows, synthetic_values):
    device, _ = engine.environment("cpu", cpu_threads=2)
    split = make_split(synthetic_rows, 1)
    baseline = tmp_path / "baseline/selection"
    reference, _, _ = engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split,
                                        42, device, baseline, "selection", "identity", resume=False)
    paused = tmp_path / "paused/selection"
    with pytest.raises(engine.SessionBudgetReached):
        engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split, 42, device,
                           paused, "selection", "identity", resume=False, deadline=time.monotonic() - 1)
    assert len(read_json(paused / "history.json")) == 1
    resumed, _, _ = engine.train_phase(tiny_config, synthetic_values, synthetic_rows, split,
                                     42, device, paused, "selection", "identity", resume=True)
    assert all(torch.equal(reference.state_dict()[k], resumed.state_dict()[k]) for k in reference.state_dict())
    assert read_json(paused / "history.json") == read_json(baseline / "history.json")
