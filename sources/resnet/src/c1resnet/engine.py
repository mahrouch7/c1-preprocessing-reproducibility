"""Deterministic nested stopping, full-outer refit and resumable scientific jobs."""
from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time
import traceback
import uuid
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data import LABELS, ROOT, condition_cache, fit_scaler, load_source, make_split, verify_feature_parity
from .io import digest, exclusive_lock, git_state, now, read_json, require, sha256, source_tree, write_json
from .metrics import score
from .model import ResNet1D


def environment(device_name="auto", *, cpu_threads=4):
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "torch")}
    expected = {"numpy": "2.3.5", "scipy": "1.18.1", "scikit-learn": "1.9.1", "torch": "2.8.0"}
    require({k: v.split("+")[0] for k, v in versions.items()} == expected, "Scientific dependency version drift")
    require(platform.python_version_tuple()[:2] == ("3", "12"), "Python 3.12 is required")
    require(device_name in ("auto", "cpu", "cuda"), "Invalid device choice")
    if device_name == "cuda":
        require(torch.cuda.is_available(), "CUDA requested but unavailable; no silent CPU fallback")
    chosen = "cuda" if device_name in ("cuda", "auto") and torch.cuda.is_available() else "cpu"
    torch.set_num_threads(cpu_threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions()}
    return torch.device(chosen), {"python": platform.python_version(), "platform": platform.platform(),
        "versions": versions, "device": chosen, "cuda_runtime": torch.version.cuda,
        "installed_packages": dict(sorted(packages.items())),
        "gpu": torch.cuda.get_device_name(0) if chosen == "cuda" else None,
        "cpu_threads": cpu_threads, "deterministic_algorithms": True, "mixed_precision": False,
        "tf32": False, "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def derived_seed(seed, fold, phase):
    # Same initialization/order across training conditions; corruption RNG is separate.
    return int(digest({"seed": int(seed), "fold": int(fold), "phase": phase})[:8], 16)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WaveformDataset(Dataset):
    def __init__(self, values, rows, ids, scaler):
        self.values, self.rows = values, rows
        self.ids = np.asarray(ids, dtype=np.int64)
        self.mean = np.asarray(scaler["mean"], dtype=np.float64)[:, None]
        self.std = np.asarray(scaler["std"], dtype=np.float64)[:, None]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        row_id = int(self.ids[index])
        # Out-of-place arithmetic cannot mutate the read-only mapped cache.
        x = ((np.asarray(self.values[row_id], dtype=np.float64) - self.mean) / self.std).astype(np.float32)
        return torch.from_numpy(x), LABELS.index(self.rows[row_id]["fault"]), row_id


@torch.no_grad()
def predict(model, dataset, batch_size, device):
    model.eval()
    probabilities, ids = [], []
    started = time.perf_counter()
    for x, _, row_ids in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
        logits = model(x.to(device))
        require(torch.isfinite(logits).all().item(), "Nonfinite inference logits")
        p = logits.softmax(dim=1).cpu().numpy().astype(np.float64)
        # Float32 softmax may sum off one by one ulp; align the publication values.
        p /= p.sum(axis=1, keepdims=True)
        probabilities.append(p)
        ids.extend(row_ids.tolist())
    return np.asarray(ids, dtype=np.int64), np.concatenate(probabilities), time.perf_counter() - started


def atomic_checkpoint(path, payload):
    path = Path(path)
    partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with partial.open("xb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def rng_state(generator):
    return {"python": random.getstate(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader": generator.get_state()}


def restore_rng(state, generator):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        require(torch.cuda.is_available(), "Cannot resume CUDA RNG state on CPU")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    generator.set_state(state["loader"])


def train_phase(config, values, rows, split, seed, device, directory, phase, identity_hash, *, resume):
    """One stopping fit or one full-outer refit. Checkpoints are epoch-atomic."""
    t = config["training"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    phase_seed = derived_seed(seed, split["fold"], phase)
    seed_all(phase_seed)
    ids = split["inner_train"] if phase == "selection" else split["outer_train"]
    scaler_path = directory / "scaler.json"
    if scaler_path.exists():
        require(resume, "Existing phase requires --resume")
        scaler = read_json(scaler_path)
        require(scaler["fit_row_ids"] == ids, "Scaler fit partition changed")
    else:
        scaler = fit_scaler(values, ids)
        write_json(scaler_path, scaler)
    dataset = WaveformDataset(values, rows, ids, scaler)
    generator = torch.Generator().manual_seed(phase_seed)
    loader = DataLoader(dataset, batch_size=t["batch_size"], shuffle=True, generator=generator,
                        num_workers=0, drop_last=False)
    model = ResNet1D(config["architecture"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    if phase == "selection":
        epochs = t["max_epochs"]
        validation_data = WaveformDataset(values, rows, split["validation"], scaler)
    else:
        epochs = read_json(directory.parent / "selection/summary.json")["best_epoch"]
        validation_data = None
    checkpoint_path = directory / "last.pt"
    history, best, best_epoch, bad_epochs, elapsed_prior = [], -1.0, 0, 0, 0.0
    if checkpoint_path.exists():
        require(resume, "Existing checkpoint requires --resume")
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        require(state["identity_hash"] == identity_hash and state["phase"] == phase, "Checkpoint identity changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        for group in optimizer.state.values():
            for key, value in group.items():
                if isinstance(value, torch.Tensor):
                    group[key] = value.to(device)
        history, best, best_epoch, bad_epochs = (state[k] for k in ("history", "best", "best_epoch", "bad_epochs"))
        elapsed_prior = state["elapsed_seconds"]
        restore_rng(state["rng"], generator)
    started = time.perf_counter()
    for epoch in range(len(history) + 1, epochs + 1):
        if phase == "selection" and bad_epochs >= t["patience"]:
            break
        model.train()
        total_loss, examples = 0.0, 0
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            require(torch.isfinite(loss).item(), "Nonfinite training loss; no automatic retry")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), t["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(y)
            examples += len(y)
        item = {"epoch": epoch, "training_cross_entropy": total_loss / examples}
        if validation_data is not None:
            val_ids, probabilities, _ = predict(model, validation_data, t["evaluation_batch_size"], device)
            value = score(rows, val_ids, probabilities)["macro_f1"]
            item["validation_macro_f1"] = value
            if value > best + t["min_delta"]:
                best, best_epoch, bad_epochs = value, epoch, 0
            else:
                bad_epochs += 1
        else:
            best_epoch = epochs
        history.append(item)
        elapsed = elapsed_prior + time.perf_counter() - started
        atomic_checkpoint(checkpoint_path, {"identity_hash": identity_hash, "phase": phase,
            "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "history": history, "best": best, "best_epoch": best_epoch, "bad_epochs": bad_epochs,
            "rng": rng_state(generator), "elapsed_seconds": elapsed})
        write_json(directory / "history.json", history, replace=True)
        print(f"[TRAIN] fold={split['fold']} seed={seed} {phase} epoch={epoch}/{epochs} "
              f"loss={item['training_cross_entropy']:.5f}" +
              (f" valF1={item['validation_macro_f1']:.5f}" if validation_data is not None else ""), flush=True)
    require(len(history) > 0 and best_epoch >= 1, "No valid stopping/refit epoch")
    write_json(directory / "history.json", history, replace=True)
    summary = {"phase": phase, "seed": phase_seed, "epochs_completed": len(history), "best_epoch": best_epoch,
               "best_validation_macro_f1": best if phase == "selection" else None,
               "stopping_reason": "patience" if phase == "selection" and bad_epochs >= t["patience"] else "epoch_budget",
               "elapsed_seconds": elapsed_prior + time.perf_counter() - started,
               "training_rows": len(ids), "training_families": len({rows[i]["family_id"] for i in ids}),
               "scaler_sha256": sha256(scaler_path), "checkpoint_sha256": sha256(checkpoint_path)}
    # A completed phase is immutable; on resumed completion return its original timings.
    summary_path = directory / "summary.json"
    if summary_path.exists():
        prior = read_json(summary_path)
        require(prior["checkpoint_sha256"] == summary["checkpoint_sha256"], "Completed phase checkpoint changed")
        summary = prior
    else:
        write_json(summary_path, summary)
    return model, scaler, summary


def job_definitions(config, profile):
    for condition in config["profiles"][profile]:
        for fold in config["training"]["outer_folds"]:
            for seed in config["training"]["seeds"]:
                yield {"training_condition": condition, "fold": fold, "seed": seed}


def verify_receipt(directory, identity_hash):
    receipt = read_json(Path(directory) / "completed.json")
    require(receipt["identity_hash"] == identity_hash and receipt["status"] == "COMPLETE", "Job receipt identity/status changed")
    for relative, expected in receipt["artifacts"].items():
        path = (Path(directory) / relative).resolve()
        require(Path(directory).resolve() in path.parents and sha256(path) == expected, "Completed artifact changed: " + relative)
    return receipt


def run_job(config, rows, clean, values, cache_receipt, all_values, job, device, output, manifest_hash, *, resume):
    name = f"{job['training_condition']}/fold{job['fold']:02d}/seed{job['seed']}"
    directory = Path(output) / "jobs" / name
    directory.mkdir(parents=True, exist_ok=True)
    split = make_split(rows, job["fold"], config["training"]["validation_family_fraction"], config["training"]["inner_split_seed"])
    identity = {"job": job, "run_manifest_sha256": manifest_hash, "split_sha256": digest(split),
                "training_cache_sha256": cache_receipt["sha256"]}
    identity_hash = digest(identity)
    if (directory / "completed.json").exists():
        require(resume, "Completed job exists; use --resume or choose a new output")
        verify_receipt(directory, identity_hash)
        print("[SKIP VERIFIED COMPLETE] " + name, flush=True)
        return
    with exclusive_lock(directory / "running.lock"):
        write_json(directory / "identity.json", identity)
        write_json(directory / "split.json", split)
        try:
            train_phase(config, values, rows, split, job["seed"], device, directory / "selection", "selection", identity_hash, resume=resume)
            model, scaler, _ = train_phase(config, values, rows, split, job["seed"], device, directory / "refit", "refit", identity_hash, resume=resume)
            conditions = config["profiles"][read_json(Path(output) / "run_manifest.json")["profile"]] if job["training_condition"] == "clean" else [job["training_condition"]]
            predictions = directory / "predictions"
            predictions.mkdir(exist_ok=True)
            catalogue = []
            for condition in conditions:
                prediction_path = predictions / (condition + ".npz")
                score_path = predictions / (condition + ".json")
                if prediction_path.exists() and score_path.exists() and resume:
                    report = read_json(score_path)
                    require(sha256(prediction_path) == report["prediction_sha256"], "Partial prediction changed")
                    catalogue.append(report)
                    continue
                require(not prediction_path.exists() and not score_path.exists(), "Unpaired prediction artifacts; inspect before recovery")
                data, _ = all_values(condition)
                dataset = WaveformDataset(data, rows, split["test"], scaler)
                ids, probabilities, elapsed = predict(model, dataset, config["training"]["evaluation_batch_size"], device)
                truth = np.asarray([LABELS.index(rows[int(i)]["fault"]) for i in ids], dtype=np.int64)
                with prediction_path.open("xb") as stream:
                    np.savez_compressed(stream, row_ids=ids, truth=truth, predicted=probabilities.argmax(axis=1),
                                        probabilities=probabilities, family_id=np.array([rows[int(i)]["family_id"] for i in ids]))
                report = {"condition": condition, "metrics": score(rows, ids, probabilities),
                          "prediction_sha256": sha256(prediction_path), "prediction_file": prediction_path.name,
                          "elapsed_prediction_seconds": elapsed,
                          "timing_scope": "includes batching, channel normalization, transfer and prediction; not relay latency; no warmup exclusion"}
                write_json(score_path, report)
                catalogue.append(report)
            write_json(directory / "prediction_catalogue.json", catalogue)
            artifacts = {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob("*"))
                         if p.is_file() and p.name not in ("running.lock", "completed.json") and not p.name.endswith(".partial")}
            write_json(directory / "completed.json", {"status": "COMPLETE", "identity_hash": identity_hash,
                "completed_utc": now(), "fit_procedures": 2, "artifacts": artifacts})
        except BaseException:
            write_json(directory / ("failure_" + uuid.uuid4().hex + ".json"),
                       {"failed_utc": now(), "traceback": traceback.format_exc(), "job": job})
            raise


def run(config, profile, output, cache_root, device_name, *, resume=False, allow_cpu=False,
        only_fold=None, only_seed=None, only_condition=None):
    state = git_state(ROOT)
    require(not state["dirty"], "Commit all source/protocol changes before scientific execution")
    device, env = environment(device_name, cpu_threads=config["training"]["cpu_threads"])
    require(device.type == "cuda" or allow_cpu, "Scientific CPU training requires --allow-cpu (cost can be substantial)")
    output, cache_root = Path(output).resolve(), Path(cache_root).resolve()
    for name in ("source_root", "baseline_root"):
        from .data import resolve_input
        source = resolve_input(config, name)
        require(output != source and source not in output.parents and cache_root != source and source not in cache_root.parents,
                "Outputs/cache must not be inside read-only E32 sources")
    rows, clean = load_source(config)
    manifest = {"experiment_id": "C1-E33", "profile": profile, "config": config,
                "config_sha256": digest(config), "git": state, "source_tree": source_tree(ROOT),
                "environment": env, "cohort": {"rows": len(rows), "families": 39},
                "model": ResNet1D(config["architecture"]).description(config["sampling_rate_hz"])}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run_manifest.json"
    if path.exists():
        require(resume, "Run exists; --resume is mandatory")
        require(read_json(path) == manifest, "Run manifest/config/code/environment changed; use a separate run directory")
    else:
        require(not list(output.iterdir()), "New output directory must be empty")
        parity = verify_feature_parity(config, rows, clean)
        write_json(path, manifest)
        write_json(output / "noise_parity.json", parity)
    manifest_hash = sha256(path)
    loaded = {}
    def all_values(condition):
        if condition not in loaded:
            loaded[condition] = condition_cache(config, rows, clean, condition, cache_root)
        return loaded[condition]
    selected = [j for j in job_definitions(config, profile)
                if (only_fold is None or j["fold"] == only_fold)
                and (only_seed is None or j["seed"] == only_seed)
                and (only_condition is None or j["training_condition"] == only_condition)]
    require(bool(selected), "Empty execution selection")
    print(f"[RUN] {len(selected)} jobs / {2 * len(selected)} fit procedures; device={device}; output={output}", flush=True)
    with exclusive_lock(output / "coordinator.lock"):
        for number, job in enumerate(selected, 1):
            print(f"[JOB {number}/{len(selected)}] {job}", flush=True)
            values, receipt = all_values(job["training_condition"])
            run_job(config, rows, clean, values, receipt, all_values, job, device, output, manifest_hash, resume=resume)
    print("[DONE] Requested jobs verified. Aggregate only after required folds/seeds are complete.", flush=True)
