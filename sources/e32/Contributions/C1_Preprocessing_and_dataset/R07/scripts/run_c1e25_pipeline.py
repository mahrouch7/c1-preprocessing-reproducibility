"""Researcher-run bounded two-PC pipeline; no task is launched on import."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import getpass
import hashlib
import io
import os
from pathlib import Path
import platform
import shutil
import time
import zipfile

import numpy as np

import c1e25 as science
from prepare_c1e25 import (freeze_assignment, make_task, task_configuration,
                           unit_from_task, verify_science_sources)
from r07.distributed.barrier import verify_barrier
from r07.distributed.contracts import validate_result_archive
from r07.distributed.coordinator import stage_authorized_assignment
from r07.distributed.runtime import publish_write_once, run_pc1_local
from r07.distributed.submitter import submit_pc2_plan


def returned_archive(unit, assignment, transport):
    marker_path = transport / "markers" / (unit["unit_id"] + ".json")
    marker = science.read_json(marker_path)
    for key in ("unit_id", "role", "expected_host", "expected_fits", "configuration_sha256"):
        if marker.get(key) != unit[key]:
            raise ValueError("Returned marker mismatch: " + key)
    if marker.get("assignment_sha256") != assignment["assignment_sha256"]:
        raise ValueError("Returned assignment mismatch")
    archive = transport / "results" / (marker["result_archive_sha256"] + ".zip")
    if archive.stat().st_size != marker["result_archive_bytes"] or science.sha(archive) != marker["result_archive_sha256"]:
        raise ValueError("Returned result archive mismatch")
    receipt = validate_result_archive(archive, marker)
    if receipt.get("fits_completed") != unit["expected_fits"] or receipt.get("simulations") != 0:
        raise ValueError("Returned executed-fit/simulation counters differ")
    return archive


def check_remote_runs(receipts):
    from prefect.client.orchestration import get_client
    states = {}
    with get_client(sync_client=True) as client:
        for receipt in receipts:
            run = client.read_flow_run(receipt["flow_run_id"])
            state = run.state.type.value if run.state is not None else "UNKNOWN"
            states[receipt["unit_id"]] = state
            if state in {"FAILED", "CRASHED", "CANCELLED", "PAUSED", "SUSPENDED"}:
                raise RuntimeError(f"PC2 unit stopped: {receipt['unit_id']} state={state}; no automatic retry")
    return states


def require_worker_online():
    from prefect.client.orchestration import get_client
    with get_client(sync_client=True) as client:
        pool = client.read_work_pool("C1-R07-8658432b19cb")
        if pool.type != "process" or pool.is_paused:
            raise ValueError("Dedicated PC2 process pool is absent, paused or has the wrong type")
        matches = [w for w in client.read_workers_for_work_pool(pool.name) if w.name == "PC2"]
    if len(matches) != 1 or matches[0].last_heartbeat_time is None:
        raise ValueError("No uniquely identifiable PC2 worker heartbeat")
    age = (datetime.now(timezone.utc) - matches[0].last_heartbeat_time).total_seconds()
    if age < -5 or age > 90:
        raise ValueError("PC2 heartbeat is stale or clock review is required")


def scoped_barrier(assignment_path, transport, stage_root):
    assignment = science.read_json(assignment_path)
    units = {unit["unit_id"]: unit for unit in assignment["units"]}
    # The fixed inbox retains older assignments. Reject unexpected units within
    # this assignment, while preserving unrelated assignments without deletion.
    for path in (transport / "markers").glob("*.json"):
        marker = science.read_json(path)
        if marker.get("assignment_sha256") == assignment["assignment_sha256"] and marker.get("unit_id") not in units:
            raise ValueError("Unexpected marker for the current assignment")
    markers = stage_root / "verified_markers"
    markers.mkdir()
    for unit in units.values():
        returned_archive(unit, assignment, transport)
        publish_write_once(transport / "markers" / (unit["unit_id"] + ".json"), markers / (unit["unit_id"] + ".json"))
    return verify_barrier(assignment=assignment_path, markers=markers,
                          results=transport / "results", output=stage_root / "completion_ledger.json")


def run_stage(stage, units, task_directory, root, transport, args, authorization_utc):
    stage_root = root / ("stage_" + stage.lower())
    stage_root.mkdir()
    assignment = freeze_assignment(units, getpass.getuser(), authorization_utc, stage)
    expected_budget = 0 if stage == "A" else 60
    if assignment["total_fit_budget"] != expected_budget:
        raise ValueError("Unexpected stage fit budget")
    path = stage_root / "assignment.json"
    science.write_json(path, assignment)
    plan_path = stage_root / "dispatch_plan.json"
    stage_authorized_assignment(assignment=path, task_directory=task_directory,
        transport_tasks=transport / "tasks", output_plan=plan_path)
    receipts = submit_pc2_plan(plan_path=plan_path, assignment_path=path,
        work_pool="C1-R07-8658432b19cb", work_queue="PC2", node_home=r"C:\r07node_e24_v2r1",
        ssh_exe=r"C:\Windows\System32\OpenSSH\ssh.exe",
        identity_file=r"C:\r07node\transport\c1_r07_dispatch_ed25519",
        known_hosts=r"C:\Users\Maher\.ssh\known_hosts", pc1_host="Maher369@172.16.1.178",
        receipt_directory=stage_root / "submission_receipts", confirm=True)
    print(f"[STAGE {stage}] submitted {len(receipts)} distinct PC2 units", flush=True)
    for unit in units:
        if unit["role"] == "PC1":
            run_pc1_local(manifest_path=path, task_archive=task_directory / (unit["task_archive_sha256"] + ".zip"),
                unit_id=unit["unit_id"], attempt=stage_root / "pc1_attempts" / unit["unit_id"],
                results=transport / "results", markers=transport / "markers")
            check_remote_runs(receipts)
    deadline, previous = time.monotonic() + args.timeout_per_stage, None
    while True:
        states = check_remote_runs(receipts)
        arrived = sum((transport / "markers" / (unit["unit_id"] + ".json")).is_file() for unit in units)
        status = (arrived, tuple(sorted(states.items())))
        if status != previous:
            print(f"[STAGE {stage}] received={arrived}/{len(units)} PC2_states={list(states.values())}", flush=True)
            previous = status
        if arrived == len(units) and all(state == "COMPLETED" for state in states.values()):
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("Incomplete stage; preserve artifacts and reconcile UNKNOWN units, no automatic retry")
        time.sleep(10)
    ledger = scoped_barrier(path, transport, stage_root)
    print(f"[STAGE {stage} BARRIER PASS] units={ledger['unit_count']}", flush=True)
    return assignment, stage_root


def assemble_features(assignment, preparation, capsule, root, transport):
    output = root / "verified_features"
    output.mkdir()
    reference = np.load(capsule / "base/reference_features.npy", allow_pickle=False)
    features = {}
    for unit in assignment["units"]:
        archive = returned_archive(unit, assignment, transport)
        identity = unit["identity"]
        condition, preprocessor = identity["condition_or_fold"], identity["preprocessor"]
        with zipfile.ZipFile(archive) as bundle:
            array = np.load(io.BytesIO(bundle.read("features.npy")), allow_pickle=False)
            diagnostics = __import__("json").loads(bundle.read("feature_diagnostics.json"))
        if array.shape != (858, 18) or array.dtype != np.float64 or not np.isfinite(array).all():
            raise ValueError("Returned feature schema mismatch")
        if diagnostics["condition"] != condition or diagnostics["preprocessor"] != preprocessor:
            raise ValueError("Returned feature condition mismatch")
        source_name = condition + ".npy"
        if diagnostics["source_waveforms_sha256"] != preparation["base_files"][source_name]:
            raise ValueError("Different frozen waveform realization across preprocessors")
        if condition == "clean" and preprocessor == "none" and not np.allclose(array, reference, atol=1e-8, rtol=1e-6):
            raise ValueError("Clean/no-preprocessing features disagree with C1-E20/E23")
        path = output / (condition + "_" + preprocessor + ".npy")
        np.save(path, array, allow_pickle=False)
        features[(condition, preprocessor)] = path
    if set(features) != {(condition, preprocessor) for condition in science.CONDITIONS for preprocessor in ("none", "savgol")}:
        raise ValueError("Incomplete preprocessing matrix")
    return features


def prepare_stage_b(package, preparation, capsule, root, features):
    tasks = root / "stage_b_tasks"
    tasks.mkdir()
    metadata = capsule / "base/metadata.json"
    candidates = []
    for preprocessor in ("none", "savgol"):
        for fold in range(1, 6):
            paths = {"metadata": metadata, **{condition + "_features": features[(condition, preprocessor)]
                     for condition in science.CONDITIONS}}
            pins = {name: {"path": "data/" + name + (".json" if name == "metadata" else ".npy"),
                           "sha256": science.sha(path)} for name, path in paths.items()}
            config = task_configuration("B", preprocessor, pins, fold=fold)
            task = make_task(package=package, output=tasks, config=config,
                input_files={pins[name]["path"]: path for name, path in paths.items()})
            # Allocation key excludes role/host; final unit identity includes both.
            key = hashlib.sha256(science.canonical({"configuration": config,
                "source_commit": preparation["source_commit"]})).hexdigest()
            candidates.append((key, task, config))
    units = []
    for index, (_, task, config) in enumerate(sorted(candidates, key=lambda item: item[0])):
        role = "PC1" if index % 2 == 0 else "PC2"
        units.append(unit_from_task(task, config, role, "MAHER" if role == "PC1" else "DESKTOP-URB44PU",
                                    preparation["source_commit"], package))
    if len(units) != 10 or any(sum(u["expected_fits"] for u in units if u["role"] == role) != 30 for role in ("PC1", "PC2")):
        raise ValueError("Exact 30/30 fit allocation failed")
    return units, tasks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--transport-root", type=Path, default=Path(r"F:\Backup\R07_Dispatch"))
    parser.add_argument("--timeout-per-stage", type=int, default=7200)
    parser.add_argument("--confirm-60-fits", action="store_true")
    parser.add_argument("--confirm-final-analysis", action="store_true")
    args = parser.parse_args()
    if not args.confirm_60_fits:
        raise ValueError("Explicit exact 60-fit confirmation required")
    if args.timeout_per_stage < 60:
        raise ValueError("The explicit stage timeout must be at least 60 seconds")
    if platform.node().casefold() != "maher":
        raise ValueError("Run the coordinator on PC1 MAHER")
    package = Path(__file__).resolve().parents[1]
    current = verify_science_sources(package)
    receipt = args.capsule / "preparation_receipt.json"
    if science.sha(receipt) != args.expected_preparation_sha256:
        raise ValueError("Prepared capsule identity mismatch")
    preparation = science.read_json(receipt)
    if preparation["protocol"] != science.PROTOCOL or preparation["source_files"] != current["files"]:
        raise ValueError("Prepared sources differ from the current reviewed implementation")
    if (preparation["planned_stage_b_fits"] != 60 or preparation["fits_executed"] != 0 or
            preparation["snr_levels_db"] != list(science.SNR_LEVELS) or
            preparation["evaluation_conditions"] != list(science.CONDITIONS) or
            len(preparation["stage_a_units"]) != 12):
        raise ValueError("Preparation budget or counters differ")
    for name, expected in preparation["base_files"].items():
        if science.sha(args.capsule / "base" / name) != expected:
            raise ValueError("Prepared input changed: " + name)
    root = args.capsule / "execution_v1"
    root.mkdir(exist_ok=False)
    # This is a child process context; no persistent profile/environment changes.
    for key in tuple(os.environ):
        if key.startswith("PREFECT_"):
            os.environ.pop(key, None)
    os.environ.update(PREFECT_API_URL="http://127.0.0.1:4200/api", PREFECT_HOME=str(root / "coordinator_client"),
        PREFECT_SERVER_EPHEMERAL_ENABLED="false", PREFECT_SERVER_ANALYTICS_ENABLED="false", DO_NOT_TRACK="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="42")
    require_worker_online()
    authorization = datetime.now(timezone.utc).isoformat()
    science.write_json(root / "authorization.json", {"authorized_by": getpass.getuser(), "authorized_utc": authorization,
        "decision_id": "C1-D60", "preparation_sha256": args.expected_preparation_sha256,
        "stage_a_units": 12, "stage_b_units": 10, "total_fit_budget": 60, "fits_per_pc": 30,
        "snr_levels_db": list(science.SNR_LEVELS),
        "final_analysis_authorized": args.confirm_final_analysis})
    a, _ = run_stage("A", preparation["stage_a_units"], args.capsule / "stage_a/tasks",
                      root, args.transport_root, args, authorization)
    features = assemble_features(a, preparation, args.capsule, root, args.transport_root)
    units, tasks = prepare_stage_b(package, preparation, args.capsule, root, features)
    b, _ = run_stage("B", units, tasks, root, args.transport_root, args, authorization)
    if args.confirm_final_analysis:
        from aggregate_c1e25 import aggregate
        aggregate(b, args.transport_root, args.capsule / "base/metadata.json", root / "final_analysis")
        print("[C1-E25 COMPLETE] stage A/B barriers passed; 60 fits; final analysis saved", flush=True)
    else:
        print("[C1-E25 FITS COMPLETE] 60 fits; final analysis awaits its explicit confirmation", flush=True)


if __name__ == "__main__":
    main()
