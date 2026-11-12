"""Prepare without fitting, then externally launch one automatic C1-E26 campaign."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import threading
import time
import zipfile

import c1e25 as base
import c1e26 as science
from c1_progress import Progress
from r07.distributed.contracts import assignment_sha256, unit_identity_sha256, validate_assignment
from r07.distributed.coordinator import stage_authorized_assignment
from r07.distributed.runtime import run_pc1_local
from r07.distributed.submitter import submit_pc2_plan
from run_c1e25_pipeline import returned_archive, require_worker_online, scoped_barrier

SOURCE_ANALYSIS_SHA = "a66bf79b860497de0edf0cb1bec8386b8fbbb39686d5e3aaaaafc0073f643735"
SOURCE_ASSIGNMENT_SHA = "a8ec78bd5c4160f96f0d5c2f6da8a0dbeebaf21d681b611acca63106cb533051"
SOURCE_LEDGER_SHA = "6d26d834a974275f18e590f35f3887e416e7b79c024bca9ebf1c9c56a8f16471"


def verify_sources(package):
    report = base.read_json(package / "provenance/c1_e26_lineage.json")
    for relative, digest in report["files"].items():
        if base.sha(package / relative) != digest:
            raise ValueError("C1-E26 implementation changed without provenance: " + relative)
    return report


def source_features(source_run):
    """Read-only provenance validation, no feature regeneration or model fitting."""
    if base.sha(source_run / "final_analysis/analysis_receipt.json") != SOURCE_ANALYSIS_SHA:
        raise ValueError("Unpinned E25 analysis receipt")
    if base.sha(source_run / "stage_b/completion_ledger.json") != SOURCE_LEDGER_SHA:
        raise ValueError("Unpinned E25 complete ledger")
    assignment = validate_assignment(base.read_json(source_run / "stage_b/assignment.json"), require_authorized=True)
    if assignment["assignment_sha256"] != SOURCE_ASSIGNMENT_SHA:
        raise ValueError("E25 assignment changed")
    metadata = source_run.parent / "base/metadata.json"
    paths = {}
    for unit in assignment["units"]:
        task = source_run / "stage_b_tasks" / (unit["task_archive_sha256"] + ".zip")
        if base.sha(task) != unit["task_archive_sha256"]:
            raise ValueError("E25 feature-task archive changed")
        with zipfile.ZipFile(task) as bundle:
            raw = bundle.read("configuration.json")
        if hashlib.sha256(raw).hexdigest() != unit["configuration_sha256"]:
            raise ValueError("E25 feature-task configuration changed")
        config = json.loads(raw)
        prep = config["preprocessor"]
        for condition in base.CONDITIONS:
            path = source_run / "verified_features" / (condition + "_" + prep + ".npy")
            if base.sha(path) != config["inputs"][condition + "_features"]["sha256"]:
                raise ValueError("E25 returned feature table changed")
            paths[(condition, prep)] = path
        if base.sha(metadata) != config["inputs"]["metadata"]["sha256"]:
            raise ValueError("E25 window metadata changed")
    if len(paths) != 12:
        raise ValueError("Incomplete source feature matrix")
    analysis = base.read_json(source_run / "final_analysis/analysis_receipt.json")
    baseline = source_run / "final_analysis/summary_metrics.csv"
    primary = source_run / "final_analysis/primary_endpoint.json"
    for path in (baseline, primary):
        if base.sha(path) != analysis["files"][path.name]:
            raise ValueError("E25 baseline artifact changed")
    return paths, metadata, baseline, primary


def configuration(preprocessor, fold, paths):
    return {"experiment_id": science.EXPERIMENT, "protocol": science.PROTOCOL,
        "preprocessor": preprocessor, "fold": fold, "expected_fits": 18,
        "row_count": 858, "model_contracts": science.MODEL_CONTRACTS,
        "scaled_classifiers": list(science.SCALED), "conditions": list(base.CONDITIONS),
        "feature_columns": base.FEATURES, "scientific_limits": base.LIMITS,
        "inputs": {key: {"path": "data/" + key + path.suffix, "sha256": base.sha(path)}
                   for key, path in paths.items()}}


def make_task(package, directory, config, paths):
    temporary = directory / "task.zip"
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        payload = {"configuration.json": base.canonical(config),
                   "run_unit.py": (package / "scripts/run_c1e26_unit.py").read_bytes(),
                   "c1e26.py": (package / "scripts/c1e26.py").read_bytes(),
                   "c1e25.py": (package / "scripts/c1e25.py").read_bytes()}
        payload.update({config["inputs"][key]["path"]: path.read_bytes() for key, path in paths.items()})
        for name, data in sorted(payload.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o100644 << 16
            bundle.writestr(info, data)
    target = directory / (base.sha(temporary) + ".zip")
    if target.exists():
        raise ValueError("Task archive already exists")
    temporary.rename(target)
    return target


def make_unit(task, config, role, commit, package):
    identity = {"experiment_id": science.EXPERIMENT, "role": role,
        "expected_host": "MAHER" if role == "PC1" else "DESKTOP-URB44PU",
        "source_commit": commit, "stage": "CLASSIFIER_EXTENSION",
        "preprocessor": config["preprocessor"], "fold": config["fold"],
        "inputs": [{"name": key, "sha256": pin["sha256"]} for key, pin in sorted(config["inputs"].items())],
        "expected_fits": 18, "task_archive_sha256": base.sha(task),
        "configuration_sha256": hashlib.sha256(base.canonical(config)).hexdigest(),
        "runner_sha256": base.sha(package / "scripts/run_c1e26_unit.py")}
    return {"unit_id": unit_identity_sha256(identity), "identity": identity,
        **{key: identity[key] for key in ("role", "expected_host", "expected_fits",
            "task_archive_sha256", "configuration_sha256", "runner_sha256")},
        "task_archive_bytes": task.stat().st_size, "runner_path": "run_unit.py",
        "configuration_path": "configuration.json"}


def prepare(package, source_run, output):
    report = verify_sources(package)
    if not output.is_absolute() or output.exists() or output.resolve().is_relative_to(source_run.parent.resolve()) or output.resolve().is_relative_to(package.parents[2].resolve()):
        raise ValueError("Use a new absolute output outside the repository and source capsule")
    paths, metadata, baseline, primary = source_features(source_run)
    rows = base.validate_metadata(base.read_json(metadata))
    if len(rows) != 858:
        raise ValueError("Source sample count changed")
    output.mkdir(parents=True)
    tasks = output / "tasks"
    tasks.mkdir()
    commit = subprocess.check_output(["git", "-c", "safe.directory=" + package.parents[2].as_posix(),
        "-C", str(package.parents[2]), "rev-parse", "HEAD"], text=True).strip()
    candidates = []
    for prep in ("none", "savgol"):
        for fold in range(1, 6):
            selected = {"metadata": metadata, **{c + "_features": paths[(c, prep)] for c in base.CONDITIONS}}
            config = configuration(prep, fold, selected)
            science.validate_configuration(config, 18)
            task = make_task(package, tasks, config, selected)
            candidates.append((hashlib.sha256(base.canonical(config)).hexdigest(), task, config))
    units = [make_unit(task, config, "PC1" if index % 2 == 0 else "PC2", commit, package)
             for index, (_, task, config) in enumerate(sorted(candidates, key=lambda item: item[0]))]
    if len(units) != 10 or any(sum(u["expected_fits"] for u in units if u["role"] == role) != 90 for role in ("PC1", "PC2")):
        raise ValueError("Exact 90/90 fit allocation failed")
    for source, name in ((metadata, "metadata.json"), (baseline, "reused_e25_summary.csv"), (primary, "reused_e25_primary.json")):
        shutil.copyfile(source, output / name)
    receipt = {"protocol": science.PROTOCOL, "status": "PREPARED_NOT_EXECUTED",
        "source_commit": commit, "source_files": report["files"], "source_analysis_sha256": SOURCE_ANALYSIS_SHA,
        "units": units, "planned_fits": 180, "reused_fits": 60, "fits_executed": 0,
        "base_files": {name: base.sha(output / name) for name in ("metadata.json", "reused_e25_summary.csv", "reused_e25_primary.json")}}
    base.write_json(output / "preparation_receipt.json", receipt)
    print("[C1-E26 PREPARED; NEW FITS=0; 180 FITS AWAIT EXTERNAL INITIAL LAUNCH]", flush=True)
    print("preparation_receipt_sha256=" + base.sha(output / "preparation_receipt.json"), flush=True)
    return receipt


def freeze(units, preparation_hash):
    assignment = {"schema_version": 2, "status": "FROZEN", "experiment_id": science.EXPERIMENT,
        "execution_authorized": True, "authorization": {"authorized_by": getpass.getuser(),
            "authorized_utc": datetime.now(timezone.utc).isoformat(), "decision_id": "C1-D61",
            "scope": "One C1-E26 campaign, exactly 180 new fits; automatic units and final analysis",
            "preparation_sha256": preparation_hash},
        "total_fit_budget": 180, "scientific_limits": base.LIMITS, "units": units}
    assignment["assignment_sha256"] = assignment_sha256(assignment)
    return validate_assignment(assignment, require_authorized=True)


def remote_status(receipts):
    from prefect.client.orchestration import get_client
    states, durations = {}, []
    with get_client(sync_client=True) as client:
        for receipt in receipts:
            run = client.read_flow_run(receipt["flow_run_id"])
            state = run.state.type.value if run.state else "UNKNOWN"
            states[receipt["unit_id"]] = state
            if state in {"FAILED", "CRASHED", "CANCELLED", "PAUSED", "SUSPENDED"}:
                raise RuntimeError(f"PC2 stopped: {receipt['unit_id']} {state}; no automatic retry")
            if state == "COMPLETED" and run.start_time and run.end_time:
                durations.append((run.end_time - run.start_time).total_seconds())
    return states, durations


def persist_progress(root, snapshot):
    temporary = root / "progress.partial"
    temporary.write_bytes(base.canonical(snapshot))
    temporary.replace(root / "progress.json")


def launch(package, output, expected_hash, timeout=14400, transport_root=None, poll_interval=10):
    current = verify_sources(package)
    if platform.node().casefold() != "maher" or base.sha(output / "preparation_receipt.json") != expected_hash:
        raise ValueError("Run on qualified PC1 with the exact preparation hash")
    preparation = base.read_json(output / "preparation_receipt.json")
    if (preparation["protocol"] != science.PROTOCOL or preparation["source_files"] != current["files"] or
            preparation["planned_fits"] != 180 or preparation["fits_executed"] != 0):
        raise ValueError("Preparation source identity or budget differs")
    for name, expected in preparation["base_files"].items():
        if base.sha(output / name) != expected:
            raise ValueError("Prepared reference artifact changed")
    for unit in preparation["units"]:
        if base.sha(output / "tasks" / (unit["task_archive_sha256"] + ".zip")) != unit["task_archive_sha256"]:
            raise ValueError("Prepared task archive changed")
    root = output / "execution_v1"
    if root.exists():
        raise ValueError("Execution already exists; reconcile, never retry automatically")
    for key in tuple(os.environ):
        if key.startswith("PREFECT_"):
            os.environ.pop(key, None)
    os.environ.update(PREFECT_API_URL="http://127.0.0.1:4200/api", PREFECT_HOME=str(root / "coordinator_client"),
        PREFECT_SERVER_EPHEMERAL_ENABLED="false", PREFECT_SERVER_ANALYTICS_ENABLED="false", DO_NOT_TRACK="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="42")
    require_worker_online()
    # Prefect may create its isolated client directory during the read-only
    # preflight. Accept only that owned directory, never an existing execution.
    root.mkdir(exist_ok=True)
    if any(path.name != "coordinator_client" for path in root.iterdir()):
        raise ValueError("Execution directory acquired unexpected artifacts during preflight")
    transport = Path(transport_root) if transport_root is not None else Path(r"F:\Backup\R07_Dispatch")
    assignment = freeze(preparation["units"], expected_hash)
    path = root / "assignment.json"
    base.write_json(path, assignment)
    plan = root / "dispatch_plan.json"
    progress = Progress()
    print("[PHASE] Publication des taches; aucun nouveau traitement des signaux bruts", flush=True)
    stage_authorized_assignment(assignment=path, task_directory=output / "tasks", transport_tasks=transport / "tasks", output_plan=plan)
    receipts = submit_pc2_plan(plan_path=plan, assignment_path=path, work_pool="C1-R07-8658432b19cb",
        work_queue="PC2", node_home=r"C:\r07node_e24_v2r1", ssh_exe=r"C:\Windows\System32\OpenSSH\ssh.exe",
        identity_file=r"C:\r07node\transport\c1_r07_dispatch_ed25519", known_hosts=r"C:\Users\Maher\.ssh\known_hosts",
        pc1_host="Maher369@172.16.1.178", receipt_directory=root / "submission_receipts", confirm=True)
    stopped, lock = threading.Event(), threading.Lock()
    local = {"durations": [], "started": None, "unit_id": None}

    def local_work():
        for unit in assignment["units"]:
            if unit["role"] != "PC1":
                continue
            if stopped.is_set():
                return
            started = time.monotonic()
            with lock:
                local.update(started=started, unit_id=unit["unit_id"])
            run_pc1_local(manifest_path=path, task_archive=output / "tasks" / (unit["task_archive_sha256"] + ".zip"),
                unit_id=unit["unit_id"], attempt=root / "pc1_attempts" / unit["unit_id"],
                results=transport / "results", markers=transport / "markers")
            with lock:
                local["durations"].append(time.monotonic() - started)
                local.update(started=None, unit_id=None)

    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(local_work)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if future.done() and future.exception() is not None:
                raise future.exception()
            states, remote_durations = remote_status(receipts)
            counts = {role: sum((transport / "markers" / (u["unit_id"] + ".json")).is_file()
                               for u in assignment["units"] if u["role"] == role) for role in ("PC1", "PC2")}
            with lock:
                durations = {"PC1": list(local["durations"]), "PC2": remote_durations}
                active = {"PC1": time.monotonic() - local["started"] if local["started"] else 0}
                local_active = local["started"] is not None
            pc_states = {"PC1": "EN_COURS" if local_active else "TERMINE" if counts["PC1"] == 5 else "EN_ATTENTE",
                         "PC2": "EN_COURS" if "RUNNING" in states.values() else "TERMINE" if all(v == "COMPLETED" for v in states.values()) else "EN_ATTENTE"}
            snapshot = progress.update("ENTRAINEMENT", counts, {"PC1": 5, "PC2": 5}, durations, active, pc_states)
            snapshot["pc2_states"] = states
            persist_progress(root, snapshot)
            if sum(counts.values()) == 10 and future.done() and all(s == "COMPLETED" for s in states.values()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Incomplete campaign; preserve artifacts and reconcile, no automatic retry")
            stopped.wait(poll_interval)
        print("[PHASE] Verification complete des retours PC1 et PC2", flush=True)
        persist_progress(root, progress.update("VERIFICATION_DES_RETOURS", counts, {"PC1": 5, "PC2": 5}, durations))
        ledger = scoped_barrier(path, transport, root)
        from aggregate_c1e26 import aggregate
        print("[PHASE] Analyse finale automatique; aucune validation de lot requise", flush=True)
        persist_progress(root, progress.update("ANALYSE_FINALE", counts, {"PC1": 5, "PC2": 5}, durations))
        aggregate(assignment, transport, output, root / "final_analysis")
        base.write_json(root / "timing_receipt.json", {"elapsed_seconds": progress.clock() - progress.started,
            "pc1_unit_seconds": durations["PC1"], "pc2_unit_seconds": durations["PC2"], "speedup_measured": None,
            "serial_baseline": "UNAVAILABLE", "complete_units": ledger["unit_count"]})
        snapshot = progress.update("TERMINE_VALIDATION_ET_ANALYSE_PASSEES", {"PC1": 5, "PC2": 5}, {"PC1": 5, "PC2": 5}, durations)
        snapshot.update(execution_completed=True, analysis_complete=True)
        persist_progress(root, snapshot)
        print("[C1-E26 COMPLETE] 180 nouveaux fits; 60 reutilises; 8 classifieurs AWGN; analyse finale sauvegardee", flush=True)
    except BaseException as exc:
        persist_progress(root, {"phase": "INTERROMPU" if isinstance(exc, KeyboardInterrupt) else "ERREUR_ARRET",
            "error": str(exc), "execution_completed": False, "analysis_complete": False,
            "automatic_retry": False, "elapsed_seconds": progress.clock() - progress.started})
        raise
    finally:
        stopped.set()
        pool.shutdown(wait=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, default=Path(r"G:\IEEE9_Analysis\R07\experiments\C1-E25\capsule_grid_v1\execution_v1"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--expected-preparation-sha256")
    parser.add_argument("--confirm-campaign-180-fits", action="store_true")
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    if args.prepare_only:
        if args.confirm_campaign_180_fits:
            raise ValueError("Preparation and scientific launch are separate actions")
        prepare(package, args.source_run, args.output)
    else:
        if not args.confirm_campaign_180_fits or not args.expected_preparation_sha256:
            raise ValueError("External initial launch requires exact preparation hash and 180-fit campaign confirmation")
        launch(package, args.output, args.expected_preparation_sha256)


if __name__ == "__main__":
    main()
