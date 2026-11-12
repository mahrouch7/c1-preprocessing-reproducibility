"""Kaggle driver: exact frozen CPU input bytes, a separate GPU run identity.

Does not regenerate noise on Linux or import CPU checkpoints. Does not upload,
publish, purchase, change account settings or control the local CPU process.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import stat
import time
import uuid
import zipfile

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from .data import LABELS, ROOT, load_config, make_split, validate_rows
from .engine import (SessionBudgetReached, WaveformDataset, environment, job_definitions,
                     run_job, seed_all)
from .data import fit_scaler
from .io import digest, exclusive_lock, git_state, now, read_json, require, sha256, write_json
from .model import InceptionTime1D


def safe_members(archive):
    names = set()
    for member in archive.infolist():
        parts = PurePosixPath(member.filename)
        require(not parts.is_absolute() and ".." not in parts.parts and "\\" not in member.filename
                and ":" not in member.filename and member.filename not in names, "Unsafe/duplicate archive entry")
        require(not stat.S_ISLNK(member.external_attr >> 16), "Symlink in archive")
        names.add(member.filename)
    return archive.infolist()


def verify_handoff(payload):
    payload = Path(payload).resolve()
    manifest = read_json(payload / "handoff_manifest.json")
    require(manifest["schema_version"] == 1 and manifest["experiment_id"] == "C1-E43", "Wrong handoff")
    require(manifest["kind"] == "PRIVATE_KAGGLE_FROZEN_INPUTS_V1", "Wrong payload kind")
    config = load_config()
    require(digest(config) == manifest["config_sha256"], "Scientific configuration differs")
    for relative, expected in manifest["files"].items():
        path = (payload / relative).resolve()
        require(payload in path.parents and path.is_file() and not path.is_symlink(), "Invalid payload input")
        require(sha256(path) == expected, "Payload SHA256 mismatch: " + relative)
    require(manifest["code_commit"] == git_state(ROOT)["commit"], "Wrong code checkout")
    rows = validate_rows(read_json(payload / "data/metadata.json"))
    require(sha256(payload / "data/metadata.json") == config["source_pins"]["metadata.json"], "Metadata drift")
    loaded = {}
    for condition in config["profiles"][manifest["profile"]]:
        entry = manifest["conditions"][condition]
        relative = entry["path"]
        require(relative in manifest["files"] and entry["sha256"] == manifest["files"][relative], "Unbound cache")
        values = np.load(payload / relative, mmap_mode="r", allow_pickle=False)
        require(values.shape == (1839, 6, 8000) and values.dtype == np.float32, "Frozen NCT schema changed")
        # Hashes already verify every byte; check finiteness without copying the full array.
        for start in range(0, len(rows), 64):
            require(np.isfinite(values[start:start + 64]).all(), "Nonfinite frozen inputs")
        loaded[condition] = (values, {"sha256": entry["sha256"]})
    for fold in range(1, 6):
        make_split(rows, fold, config["training"]["validation_family_fraction"], config["training"]["inner_split_seed"])
    return manifest, config, rows, loaded


def runtime_key(env):
    """Require the same numerical stack/GPU to resume; record session OS separately."""
    require(env["device"] == "cuda", "GPU execution required; no CPU fallback")
    return {key: env[key] for key in ("python", "versions", "device", "cuda_runtime", "gpu",
            "cpu_threads", "deterministic_algorithms", "mixed_precision", "tf32", "cublas_workspace_config")} | {
            "cudnn": torch.backends.cudnn.version(), "compute_capability": list(torch.cuda.get_device_capability(0))}


def benchmark(config, rows, values, device):
    """Discarded model, training-only inputs. No outer scores or model selection."""
    split = make_split(rows, 1)
    one_per_class = [next(i for i in split["inner_train"] if rows[i]["fault"] == label) for label in LABELS]
    ids = [one_per_class[i % len(one_per_class)] for i in range(config["training"]["batch_size"])]
    scaler = fit_scaler(values, split["inner_train"])
    dataset = WaveformDataset(values, rows, ids, scaler)
    x = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    y = torch.tensor([dataset[i][1] for i in range(len(dataset))], device=device)
    seed_all(20261029)
    model = InceptionTime1D(config["architecture"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"],
                                  weight_decay=config["training"]["weight_decay"])
    times = []
    for step in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        require(torch.isfinite(loss).item(), "Nonfinite GPU benchmark loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["training"]["gradient_clip_norm"], error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize()
        if step >= 2:
            times.append(time.perf_counter() - started)
    model.eval()
    with torch.no_grad():
        require(torch.isfinite(model(x)).all().item(), "Nonfinite GPU prediction")
    report = {"status": "GPU_SOFTWARE_BENCHMARK_NOT_SCIENTIFIC_PERFORMANCE", "training_row_ids": ids,
              "outer_test_rows_used": 0, "warmup_steps": 2, "timed_steps": 3,
              "batch_size": len(ids), "seconds_per_training_step": times,
              "median_seconds_per_training_step": float(np.median(times)), "device": str(device),
              "discarded_benchmark_weights": True, "new_model_seeded_for_every_scientific_phase": True}
    del model, optimizer, x, y
    torch.cuda.empty_cache()
    return report


def save_resume_archive(output, destination):
    """Only scientific run files; never env, inputs, credentials, locks or partials."""
    output, destination = Path(output).resolve(), Path(destination).resolve()
    require(output not in destination.parents, "Resume archive must be outside its input tree")
    partial = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.suffix != ".lock" and not path.name.endswith(".partial"):
                archive.write(path, path.relative_to(output).as_posix())
    os.replace(partial, destination)
    write_json(destination.with_suffix(".receipt.json"), {"sha256": sha256(destination),
                "created_utc": now(), "contains": "run artifacts only, no lock files", "archive": destination.name}, replace=True)


def restore_archive(archive_path, output):
    output = Path(output).resolve()
    require(not output.exists(), "Restore destination must not exist")
    with zipfile.ZipFile(archive_path) as archive:
        members = safe_members(archive)
        require("run_manifest.json" in archive.namelist(), "Not a C1-E43 run archive")
        require(all(not m.filename.endswith(".lock") for m in members), "Stale lock in run archive")
        output.mkdir(parents=True)
        archive.extractall(output)


def run_cloud(payload, output, *, hours=9.0, resume_archive=None, qualify_only=False):
    require(0 < hours <= 9, "Use a maximum 9-hour compute budget to leave Kaggle session/export headroom")
    require(not git_state(ROOT)["dirty"], "Cloud source checkout must be clean")
    handoff, config, rows, loaded = verify_handoff(payload)
    device, env = environment("cuda", cpu_threads=config["training"]["cpu_threads"])
    key = runtime_key(env)
    output = Path(output).resolve()
    payload = Path(payload).resolve()
    require(output != payload and payload not in output.parents and output not in payload.parents,
            "Run outputs must be separate from frozen inputs")
    archive_path = output.parent / "c1e43_resume.zip"
    if resume_archive is not None and not output.exists():
        restore_archive(resume_archive, output)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest = {"experiment_id": "C1-E43", "profile": handoff["profile"], "config": config,
                "config_sha256": digest(config), "git": git_state(ROOT), "runtime_key": key,
                "handoff_sha256": sha256(Path(payload) / "handoff_manifest.json"),
                "input_policy": "EXACT_FROZEN_CPU_NCT_FLOAT32_BYTES; NO_LINUX_NOISE_REGENERATION",
                "model": InceptionTime1D(config["architecture"]).description()}
    if manifest_path.exists():
        require(read_json(manifest_path) == manifest, "GPU/run/data/version identity changed; refuse cross-device or CPU resume")
    else:
        require(not list(output.iterdir()), "New cloud output must be empty")
        write_json(manifest_path, manifest)
    session_id = uuid.uuid4().hex
    write_json(output / "sessions" / (session_id + ".json"), {"started_utc": now(), "environment": env,
                "hours_budget": hours, "code_commit": handoff["code_commit"]})
    result = benchmark(config, rows, loaded["clean"][0], device)
    write_json(output / "sessions" / (session_id + "_benchmark.json"), result)
    print("[GPU QUALIFIED] " + str(key), flush=True)
    print("[BENCHMARK] median seconds/training step=" + str(result["median_seconds_per_training_step"]), flush=True)
    if qualify_only:
        save_resume_archive(output, archive_path)
        return "QUALIFICATION_ONLY"
    deadline = time.monotonic() + hours * 3600 - 300
    jobs = list(job_definitions(config, handoff["profile"]))
    status = "COMPLETE"
    try:
        with exclusive_lock(output / "coordinator.lock"):
            for number, job in enumerate(jobs, 1):
                if time.monotonic() >= deadline:
                    status = "SESSION_PAUSED"
                    break
                print(f"[GPU JOB {number}/{len(jobs)}] {job}", flush=True)
                values, receipt = loaded[job["training_condition"]]
                run_job(config, rows, None, values, receipt, loaded.__getitem__, job, device, output,
                        sha256(manifest_path), resume=True, deadline=deadline)
                save_resume_archive(output, archive_path)
    except SessionBudgetReached:
        status = "SESSION_PAUSED"
        print("[SESSION PAUSE] Last completed epoch saved; resume next Kaggle version with its output attached.", flush=True)
    finally:
        save_resume_archive(output, archive_path)
    write_json(output / "sessions" / (session_id + "_finished.json"), {"finished_utc": now(), "status": status})
    save_resume_archive(output, archive_path)
    print("[SESSION] " + status + "; archive=" + str(archive_path), flush=True)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hours", type=float, default=9.0)
    parser.add_argument("--resume-archive", type=Path)
    parser.add_argument("--qualify-only", action="store_true")
    args = parser.parse_args(argv)
    run_cloud(args.payload, args.output, hours=args.hours, resume_archive=args.resume_archive, qualify_only=args.qualify_only)


if __name__ == "__main__":
    main()
