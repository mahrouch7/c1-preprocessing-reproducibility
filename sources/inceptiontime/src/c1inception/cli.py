"""Explicit commands; merely importing this module cannot start an experiment."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Required before creating a CUDA context; never silently relax determinism.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from .data import ROOT, load_config, load_source, make_split, verify_feature_parity
from .io import digest, now, read_json, require, sha256, write_json


def plan(config, profile):
    from .engine import job_definitions
    from .model import InceptionTime1D
    jobs = list(job_definitions(config, profile))
    model = InceptionTime1D(config["architecture"]).description(config["sampling_rate_hz"])
    return {"experiment_id": config["experiment_id"], "profile": profile,
            "conditions": config["profiles"][profile], "outer_folds": config["training"]["outer_folds"],
            "initialization_seeds": config["training"]["seeds"], "model_jobs": len(jobs),
            "fit_procedures": 2 * len(jobs), "maximum_epochs_per_fit": config["training"]["max_epochs"],
            "full_cache_GiB_approximately": len(config["profiles"][profile]) * 1839 * 6 * 8000 * 4 / 1024 ** 3,
            "model": model, "note": "selection fit + full outer-training refit per job; clean shared between exposures; no fitting in plan"}


def preflight(config, output, device):
    from .engine import environment
    _, env = environment(device, cpu_threads=config["training"]["cpu_threads"])
    require(not Path(output).exists(), "Preflight destination must be new")
    rows, clean = load_source(config)
    splits = [make_split(rows, fold, config["training"]["validation_family_fraction"],
                         config["training"]["inner_split_seed"]) for fold in range(1, 6)]
    print("[PREFLIGHT] full source hashes and 1,839 numeric window hashes verified; zero fits", flush=True)
    parity = verify_feature_parity(config, rows, clean)
    report = {"status": "SOURCE_SPLITS_NOISE_PARITY_PASS_NOT_SCIENTIFIC_RESULTS", "created_utc": now(),
              "config_sha256": digest(config), "source_pins": config["source_pins"], "environment": env,
              "rows": len(rows), "families": len({r["family_id"] for r in rows}),
              "all_five_splits_family_disjoint": True, "splits": splits,
              "E32_feature_parity": parity, "fits": 0, "simulations": 0,
              "plans": {p: plan(config, p) for p in config["profiles"]}}
    write_json(Path(output) / "preflight.json", report)
    print("[PREFLIGHT PASS] " + str(Path(output).resolve()), flush=True)
    return report


def smoke(config, output, device):
    """Two optimizer updates on outer-TRAINING examples only; never scientific scores."""
    import numpy as np
    import torch
    from .data import fit_scaler
    from .engine import WaveformDataset, atomic_checkpoint, environment, seed_all
    from .model import InceptionTime1D
    require(not Path(output).exists(), "Smoke destination must be new")
    device, env = environment(device, cpu_threads=config["training"]["cpu_threads"])
    rows, clean = load_source(config, verify_rows=False)
    split = make_split(rows, 1)
    ids = [next(i for i in split["inner_train"] if rows[i]["fault"] == label) for label in config["labels"]]
    # A tiny in-memory NCT array; external inputs are never written.
    array = np.asarray(clean[ids]).transpose(0, 2, 1).astype(np.float32)
    tiny_rows = [{**rows[i], "Id": j} for j, i in enumerate(ids)]
    scaler = fit_scaler(array, list(range(len(ids))))
    dataset = WaveformDataset(array, tiny_rows, list(range(len(ids))), scaler)
    seed_all(12345)
    model = InceptionTime1D(config["architecture"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["training"]["learning_rate"])
    losses = []
    for step in range(2):
        batch = [dataset[(step * 2 + j) % len(dataset)] for j in range(2)]
        x = torch.stack([item[0] for item in batch]).to(device)
        y = torch.tensor([item[1] for item in batch], device=device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        require(torch.isfinite(loss).item(), "Nonfinite smoke loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    Path(output).mkdir(parents=True)
    checkpoint = Path(output) / "SMOKE_ONLY.pt"
    atomic_checkpoint(checkpoint, {"state": model.state_dict(), "SMOKE_ONLY": True})
    restored = InceptionTime1D(config["architecture"]).to(device)
    restored.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["state"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        require(torch.equal(model(x), restored(x)), "Checkpoint reload mismatch")
    report = {"status": "SOFTWARE_SMOKE_ONLY_NOT_A_SCIENTIFIC_RESULT", "created_utc": now(),
        "outer_training_row_ids": ids, "outer_test_rows_used": 0, "optimizer_updates": 2,
        "losses_not_performance_estimates": losses, "checkpoint_reload_exact": True,
        "checkpoint_sha256": sha256(checkpoint), "environment": env,
        "model": model.description(config["sampling_rate_hz"])}
    write_json(Path(output) / "smoke_receipt.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="C1-E43 adapted InceptionTime comparator; no implicit training")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiment.json")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Report architecture and declared fit budget; no source data access or fits")
    p.add_argument("--profile", choices=("clean", "pilot", "paper"), default="pilot")
    for name in ("preflight", "smoke"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    p = sub.add_parser("run")
    p.add_argument("--profile", choices=("clean", "pilot", "paper"), default="pilot")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, default=ROOT / "cache")
    p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-cpu", action="store_true")
    p.add_argument("--fold", type=int, choices=range(1, 6))
    p.add_argument("--seed", type=int)
    p.add_argument("--condition")
    p = sub.add_parser("analyze")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-partial", action="store_true")
    p = sub.add_parser("unlock", help="Remove ONLY an explicitly named C1-E43 lock after operator verification")
    p.add_argument("--path", type=Path, required=True)
    p.add_argument("--confirmed-process-stopped", action="store_true", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "plan":
        print(json.dumps(plan(config, args.profile), indent=2))
    elif args.command == "preflight":
        preflight(config, args.output, args.device)
    elif args.command == "smoke":
        smoke(config, args.output, args.device)
    elif args.command == "run":
        from .engine import run
        run(config, args.profile, args.output, args.cache, args.device, resume=args.resume,
            allow_cpu=args.allow_cpu, only_fold=args.fold, only_seed=args.seed, only_condition=args.condition)
    elif args.command == "analyze":
        from .analysis import aggregate
        aggregate(args.run, args.output, allow_partial=args.allow_partial)
    else:
        path = args.path.resolve()
        require(path.suffix == ".lock" and path.is_file(), "Explicit existing lock required")
        require(ROOT in path.parents, "Unlock is limited to this experiment root")
        record = read_json(path)
        require("pid" in record and "started_utc" in record, "Not a C1-E43 lock")
        print("Operator confirmed stopped process; removing lock only: " + str(path))
        path.unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
