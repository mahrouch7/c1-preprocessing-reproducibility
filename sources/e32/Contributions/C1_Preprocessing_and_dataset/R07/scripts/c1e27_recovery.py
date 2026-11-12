"""Explicit one-off storage recovery of the interrupted C1-E27 campaign.

Original science, assignments, failed attempts and completion markers are immutable.
Imports never submit work. Only an externally confirmed main() can resume jobs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import getpass
import io
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import shutil
import statistics
import subprocess
import threading
import time
import zipfile

import numpy as np
import c1e25 as base
import c1e27_spec as spec
import c1e27_campaign as original
from c1_progress import Progress, duration_text
from c1e26_campaign import remote_status, persist_progress
from r07.distributed.contracts import assignment_sha256, validate_assignment, unit_identity_sha256
from r07.distributed.coordinator import stage_authorized_assignment
from r07.distributed.runtime import run_pc1_local, publish_write_once
from r07.distributed.submitter import submit_pc2_plan
from run_c1e25_pipeline import returned_archive, scoped_barrier, require_worker_online

PREPARATION_SHA = "1786e42bcc099e8d12b71fe9e2e988c0ef6a743f2d30dc5d67c59e3b205cc6d9"
ORIGINAL_ASSIGNMENT_ARTIFACT_SHA = "aa020d69e9066ae10c625b97f856bd8767d93b36ea7a64659c51453b57e7b524"
ORIGINAL_ASSIGNMENT_SHA = "cf4e966364bc04e0e968c9d3ec878614cee435f5f4658354807999c8e799ae73"
INTERRUPTED_PLAN_SHA = "3d8b560befef87aa927235f8ccf6f0b14ac1a64475f31b9a8779a70072cc80a2"
INTERRUPTED_STAGE_A_SHA = "83ec8122b0411d627b2ccba2e961f20b44953dab1fd7215bdc61079eb07a3e30"
INTERRUPTED_RELEASE_SHA = "c67a22b1318f04b56f1c3e5bc4f4bef2eb70c2f002efd28b03b63f007fde106f"
TERMINAL = {"COMPLETED", "FAILED", "CRASHED", "CANCELLED"}
GIB = 1024**3
cancel_pending = original.cancel_pending
comparable_eta = original.comparable_eta


def verify_release(package):
    report = base.read_json(package / "provenance/c1_e27_recovery_lineage.json")
    for name, expected in report["files"].items():
        if base.sha(package / name) != expected:
            raise ValueError("Recovery source changed without provenance: " + name)
    current = original.verify_sources(package)
    if base.sha(package / "provenance/c1_e27_lineage.json") != report["original_lineage_sha256"]:
        raise ValueError("Original scientific lineage changed")
    return current, report


def validate_workspace(value):
    path = PureWindowsPath(str(value))
    if path.drive.upper() != "H:" or not path.is_absolute() or ".." in path.parts or len(path.parts) < 3:
        raise ValueError("Use a dedicated absolute PC2 H: workspace, never a drive root or C:")
    return str(path)


def split_completed(units, assignment, transport):
    kept, missing = [], []
    for unit in units:
        if (transport / "markers" / (unit["unit_id"] + ".json")).exists():
            archive = returned_archive(unit, assignment, transport)
            kept.append({"unit": unit, "assignment_sha256": assignment["assignment_sha256"],
                         "result_archive_sha256": base.sha(archive)})
        else:
            missing.append(unit)
    return kept, missing


def inspect_saved(package, prepared, transport, *, verify_tasks=True):
    current, release = verify_release(package)
    path = prepared / "preparation_receipt.json"
    if base.sha(path) != PREPARATION_SHA:
        raise ValueError("Not the exact interrupted C1-E27 capsule")
    receipt = base.read_json(path)
    if (receipt["source_files"] != current["files"] or receipt["new_fit_budget"] != 16240 or
            receipt["scope_sha256"] != spec.digest(spec.scope()) or receipt["fits_executed"] != 0):
        raise ValueError("Frozen science or fit ceiling changed")
    for name, expected in receipt["base_files"].items():
        if base.sha(prepared / name) != expected:
            raise ValueError("Prepared input changed: " + name)
    old_root = prepared / "execution_v1"
    assignment_path = old_root / "stage_a/assignment.json"
    if base.sha(assignment_path) != ORIGINAL_ASSIGNMENT_ARTIFACT_SHA:
        raise ValueError("Original assignment artifact changed")
    assignment = validate_assignment(base.read_json(assignment_path), require_authorized=True)
    if (assignment["assignment_sha256"] != ORIGINAL_ASSIGNMENT_SHA or
            assignment["units"] != receipt["stage_a_units"] or assignment["total_fit_budget"] != 0):
        raise ValueError("Original Stage A identity changed")
    if (old_root / "stage_b").exists() or (old_root / "completion_receipt.json").exists():
        raise ValueError("This adapter is only for the interrupted zero-fit preprocessing stage")
    if base.read_json(old_root / "stage_a/progress.json").get("phase") != "ERREUR_ARRET":
        raise ValueError("Original coordinator is not recorded stopped")
    kept, missing = split_completed(assignment["units"], assignment, transport)
    expected_kept = {"clean", "awgn__10", "awgn__20"}
    if {u["unit"]["identity"]["condition"] for u in kept} != expected_kept or len(kept) != 3 or len(missing) != 23:
        raise ValueError("Completion coverage changed; reconcile instead of guessing")
    kept_local = {u["unit"]["unit_id"] for u in kept if u["unit"]["role"] == "PC1"}
    attempts = old_root / "stage_a/pc1_attempts"
    if {p.name for p in attempts.iterdir()} != kept_local:
        raise ValueError("Unexpected local attempt; verify that no PC1 task is still running")
    for uid in kept_local:
        if base.read_json(attempts / uid / "completion_marker.json") != base.read_json(transport / "markers" / (uid + ".json")):
            raise ValueError("Local completed attempt and returned marker differ")
    receipts = [base.read_json(p) for p in sorted((old_root / "stage_a/submission_receipts").glob("*.json"))]
    expected_remote = {u["unit_id"] for u in assignment["units"] if u["role"] == "PC2"}
    if len(receipts) != 13 or {r["unit_id"] for r in receipts} != expected_remote:
        raise ValueError("Original submission coverage differs")
    for r in receipts:
        if r["assignment_sha256"] != assignment["assignment_sha256"]:
            raise ValueError("Foreign submission receipt")
    task_bytes = 0
    for unit in missing:
        task = prepared / "stage_a_tasks" / (unit["task_archive_sha256"] + ".zip")
        if task.stat().st_size != unit["task_archive_bytes"] or (verify_tasks and base.sha(task) != unit["task_archive_sha256"]):
            raise ValueError("Prepared task changed")
        task_bytes += task.stat().st_size
    return receipt, assignment, kept, missing, receipts, release, task_bytes


def validate_prior_states(receipts, states):
    if set(states) != {r["unit_id"] for r in receipts} or any(s not in TERMINAL for s in states.values()):
        raise ValueError("Original PC2 runs are not all terminal; no overlapping recovery allowed")


def prior_states(receipts):
    from prefect.client.orchestration import get_client
    states = {}
    with get_client(sync_client=True) as client:
        for receipt in receipts:
            run = client.read_flow_run(receipt["flow_run_id"])
            if run.parameters.get("unit_id") != receipt["unit_id"]:
                raise ValueError("Original remote run identity differs")
            states[receipt["unit_id"]] = run.state.type.value if run.state else "UNKNOWN"
    validate_prior_states(receipts, states)
    return states


def freeze(units, preparation_sha):
    assignment = {"schema_version": 2, "status": "FROZEN", "experiment_id": spec.EXPERIMENT,
        "execution_authorized": True, "authorization": {"authorized_by": getpass.getuser(),
            "authorized_utc": datetime.now(timezone.utc).isoformat(), "decision_id": "C1-D67",
            "scope": "Explicit external recovery only; keep 3 original results; maximum 16240 new fits; no automatic retry",
            "preparation_sha256": PREPARATION_SHA, "recovery_plan_sha256": preparation_sha},
        "scientific_limits": spec.LIMITS, "total_fit_budget": sum(u["expected_fits"] for u in units), "units": units}
    assignment["assignment_sha256"] = assignment_sha256(assignment)
    return validate_assignment(assignment, require_authorized=True)


def require_identical(path, digest, size):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size or base.sha(path) != digest:
        raise ValueError("Existing content-addressed artifact differs: " + str(path))


def stage_reusable_assignment(*, assignment, task_directory, transport_tasks, output_plan):
    """Reuse only byte-identical artifacts; never overwrite or silently repair."""
    manifest = validate_assignment(base.read_json(assignment), require_authorized=True)
    if output_plan.exists():
        raise ValueError("Dispatch plan exists; reconcile before continuation")
    artifacts = {}
    for unit in manifest["units"]:
        digest = unit["task_archive_sha256"]
        source = task_directory / (digest + ".zip")
        require_identical(source, digest, unit["task_archive_bytes"])
        artifacts[digest + ".zip"] = (source, digest, unit["task_archive_bytes"])
    assignment_digest = base.sha(assignment)
    artifacts[assignment_digest + ".json"] = (assignment, assignment_digest, assignment.stat().st_size)
    # Check ALL existing targets before publishing even the first absent target.
    for name, (_, digest, size) in artifacts.items():
        target = transport_tasks / name
        if target.exists() or target.is_symlink():
            require_identical(target, digest, size)
    reused = []
    for name, (source, digest, size) in artifacts.items():
        target = transport_tasks / name
        if target.exists():
            require_identical(target, digest, size)
            reused.append(name)
            continue
        try:
            publish_write_once(source, target)
        except ValueError:
            # Another publisher may have won the immutable-name race.
            require_identical(target, digest, size)
            reused.append(name)
    plan = {"schema_version": 1, "status": "STAGED_NOT_SUBMITTED",
        "assignment_sha256": manifest["assignment_sha256"],
        "assignment_artifact_sha256": assignment_digest,
        "total_fit_budget": manifest["total_fit_budget"],
        "pc2_submissions": [{"unit_id": u["unit_id"], "assignment_sha256": manifest["assignment_sha256"],
            "assignment_artifact_sha256": assignment_digest, "task_archive_sha256": u["task_archive_sha256"],
            "expected_host": u["expected_host"]} for u in manifest["units"] if u["role"] == "PC2"],
        "prefect_contacted": False, "scientific_units_executed": 0,
        "existing_artifacts_reused_after_full_hash_check": reused}
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    with output_plan.open("xb") as stream:
        stream.write(base.canonical(plan))
    return plan


def dispatch_stage(units, tasks, root, transport, preparation_sha, timeout, pc2_workspace,
                   poll_interval=15, continue_unsubmitted=False):
    path = root / "assignment.json"
    plan = root / "dispatch_plan.json"
    if continue_unsubmitted:
        assignment = validate_assignment(base.read_json(path), require_authorized=True)
        if (set(p.name for p in root.iterdir()) != {"assignment.json"} or
                assignment["units"] != units or
                assignment["authorization"]["recovery_plan_sha256"] != preparation_sha):
            raise ValueError("Only the exact never-submitted staging checkpoint can continue")
    else:
        root.mkdir()
        assignment = freeze(units, preparation_sha)
        base.write_json(path, assignment)
    try:
        stage_reusable_assignment(assignment=path, task_directory=tasks,
            transport_tasks=transport / "tasks", output_plan=plan)
        receipts = submit_pc2_plan(plan_path=plan, assignment_path=path, work_pool="C1-R07-8658432b19cb", work_queue="PC2",
            node_home=str(pc2_workspace), ssh_exe=r"C:\Windows\System32\OpenSSH\ssh.exe",
            identity_file=r"C:\r07node\transport\c1_r07_dispatch_ed25519", known_hosts=r"C:\Users\Maher\.ssh\known_hosts",
            pc1_host="Maher369@172.16.1.178", receipt_directory=root / "submission_receipts", confirm=True)
    except BaseException as exc:
        saved = [base.read_json(p) for p in (root / "submission_receipts").glob("*.json")]
        try:
            cancel_pending(saved)
        except Exception as cancel_error:
            print(f"[ATTENTION] Partial submission needs manual reconciliation: {cancel_error}", flush=True)
        persist_progress(root, {"phase": "ERREUR_ARRET", "error": str(exc), "automatic_retry": False,
            "execution_completed": False, "analysis_complete": False,
            "submission_warning": "A server-accepted request with a lost reply may lack a receipt; reconcile this assignment before any retry"})
        raise
    totals = {r: sum(u["role"] == r for u in units) for r in ("PC1", "PC2")}
    local, stopped, lock = {"durations": [], "observations": [], "started": None}, threading.Event(), threading.Lock()
    progress = Progress(emit=lambda *args, **kwargs: None)
    def local_work():
        for unit in units:
            if unit["role"] != "PC1" or stopped.is_set():
                continue
            started = time.monotonic()
            with lock:
                local["started"] = started
            run_pc1_local(manifest_path=path, task_archive=tasks / (unit["task_archive_sha256"] + ".zip"), unit_id=unit["unit_id"],
                attempt=root / "pc1_attempts" / unit["unit_id"], results=transport / "results", markers=transport / "markers")
            with lock:
                local["durations"].append(time.monotonic() - started)
                local["observations"].append((unit["unit_id"], time.monotonic() - started))
                local["started"] = None
    pool = ThreadPoolExecutor(max_workers=1)
    future = pool.submit(local_work)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if future.done() and future.exception():
                raise future.exception()
            states, remote_durations = remote_status(receipts)
            counts = {r: sum((transport / "markers" / (u["unit_id"] + ".json")).is_file() for u in units if u["role"] == r) for r in totals}
            with lock:
                durations = {"PC1": list(local["durations"]), "PC2": remote_durations}
                active = {"PC1": time.monotonic() - local["started"] if local["started"] else 0}
                local_active = local["started"] is not None
                observations = list(local["observations"])
            pc_states = {"PC1": "EN_COURS" if local_active else "TERMINE" if counts["PC1"] == totals["PC1"] else "EN_ATTENTE",
                         "PC2": "EN_COURS" if "RUNNING" in states.values() else "TERMINE" if all(s == "COMPLETED" for s in states.values()) else "EN_ATTENTE"}
            phase = {"A": "PRETRAITEMENT", "B": "ENTRAINEMENT", "STORAGE": "CONTROLE_STOCKAGE"}[units[0]["identity"]["stage"]]
            snapshot = progress.update(phase, counts, totals, durations, active, pc_states)
            completed_ids = {u["unit_id"] for u in units if (transport / "markers" / (u["unit_id"] + ".json")).is_file()}
            completed_remote = [r for r in receipts if states[r["unit_id"]] == "COMPLETED"]
            if len(completed_remote) == len(remote_durations):
                observations += [(r["unit_id"], d) for r, d in zip(completed_remote, remote_durations)]
            eta = None if units[0]["identity"]["stage"] == "STORAGE" else comparable_eta(units, completed_ids, observations)
            snapshot["estimated_remaining_execution_seconds"] = eta
            snapshot["estimate_assumption"] = "Same-host same-classifier (or preprocessing noise) observed unit seconds per new fit; final inference excluded; provisional"
            print(f"[AVANCEMENT] {phase} | PC1 {counts['PC1']}/{totals['PC1']} [{pc_states['PC1']}]"
                  f" | PC2 {counts['PC2']}/{totals['PC2']} [{pc_states['PC2']}] | ecoule {duration_text(snapshot['elapsed_seconds'])}"
                  f" | restant ~{duration_text(eta)} hors analyse finale | gain non mesure", flush=True)
            persist_progress(root, snapshot)
            if sum(counts.values()) == len(units) and future.done() and all(s == "COMPLETED" for s in states.values()):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Stage timeout; preserve artifacts, no automatic retry")
            stopped.wait(poll_interval)
        ledger = scoped_barrier(path, transport, root)
        base.write_json(root / "timing_receipt.json", {"elapsed_seconds": progress.clock() - progress.started,
            "unit_seconds": durations, "complete_units": ledger["unit_count"], "speedup_measured": None,
            "serial_baseline": "UNAVAILABLE"})
        return assignment
    except BaseException as exc:
        stopped.set()
        try:
            cancel_pending(receipts)
        except Exception as cancel_error:
            print(f"[ATTENTION] Pending-job cancellation needs review: {cancel_error}", flush=True)
        persist_progress(root, {"phase": "ERREUR_ARRET", "error": str(exc), "automatic_retry": False,
            "execution_completed": False, "analysis_complete": False,
            "already_running_units": "May finish and preserve their results; subsequent stages do not start"})
        raise
    finally:
        stopped.set()
        pool.shutdown(wait=True)



def storage_unit(package, root, workspace, required_bytes, phase, source_commit):
    root.mkdir()
    config = {"phase": phase, "workspace": validate_workspace(workspace),
              "required_free_bytes": required_bytes, "control_drive_minimum_bytes": 512 * 1024**2,
              "fits": 0, "simulations": 0}
    runner = package / "scripts/c1e27_storage_guard.py"
    archive = root / "task.zip"
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("configuration.json", base.canonical(config))
        bundle.writestr("run_unit.py", runner.read_bytes())
    identity = {"experiment_id": spec.EXPERIMENT, "stage": "STORAGE", "role": "PC2",
        "expected_host": "DESKTOP-URB44PU", "source_commit": source_commit,
        "inputs": [{"name": "storage_policy", "sha256": spec.digest(config)}], "expected_fits": 0,
        "task_archive_sha256": base.sha(archive), "configuration_sha256": spec.digest(config),
        "runner_sha256": base.sha(runner)}
    unit = {"unit_id": unit_identity_sha256(identity), "identity": identity,
        **{k: identity[k] for k in ("role", "expected_host", "expected_fits", "task_archive_sha256",
                                    "configuration_sha256", "runner_sha256")},
        "task_archive_bytes": archive.stat().st_size, "configuration_path": "configuration.json", "runner_path": "run_unit.py"}
    archive.rename(root / (unit["task_archive_sha256"] + ".zip"))
    return unit


def require_storage(package, execution, workspace, transport, plan_sha, phase, required_bytes, commit, timeout):
    unit = storage_unit(package, execution / ("storage_" + phase + "_tasks"), workspace, required_bytes, phase, commit)
    assignment = dispatch_stage([unit], execution / ("storage_" + phase + "_tasks"),
        execution / ("storage_" + phase), transport, plan_sha, timeout, workspace)
    with zipfile.ZipFile(returned_archive(unit, assignment, transport)) as bundle:
        evidence = json.loads(bundle.read("storage_receipt.json"))
    if (evidence["workspace"].casefold() != validate_workspace(workspace).casefold() or
            evidence["free_bytes"] < required_bytes or evidence["required_free_bytes"] != required_bytes or
            evidence["host"].casefold() != "desktop-urb44pu" or evidence["fits_completed"] != 0):
        raise ValueError("Invalid PC2 storage receipt")
    return evidence


def ownership_union(original_assignment, recovered_assignment, kept):
    original_units = {u["unit_id"]: u for u in original_assignment["units"]}
    kept_ids = {r["unit"]["unit_id"] for r in kept}
    new_units = {u["unit_id"]: u for u in recovered_assignment["units"]}
    if kept_ids & set(new_units) or kept_ids | set(new_units) != set(original_units):
        raise ValueError("Preprocessing union contains duplicate, omitted or foreign units")
    if len(kept_ids) != len(kept) or len(new_units) != len(recovered_assignment["units"]):
        raise ValueError("Duplicate preprocessing ownership")
    for uid, unit in new_units.items():
        if unit != original_units[uid]:
            raise ValueError("Scientific unit identity changed during recovery")
    for item in kept:
        if item["unit"] != original_units[item["unit"]["unit_id"]]:
            raise ValueError("Adopted scientific unit changed")
    return [(u, original_assignment if u["unit_id"] in kept_ids else recovered_assignment)
            for u in original_assignment["units"]]


def assemble_union(old, new, kept, prepared, execution, transport):
    ownership = ownership_union(old, new, kept)
    features, diagnostics = execution / "verified_features", execution / "verified_diagnostics"
    features.mkdir()
    diagnostics.mkdir()
    ledger = []
    for unit, assignment in ownership:
        archive = returned_archive(unit, assignment, transport)
        condition = unit["identity"]["condition"]
        with zipfile.ZipFile(archive) as bundle:
            for prep in spec.PREPROCESSORS:
                payload = bundle.read("features_" + prep + ".npy")
                data = np.load(io.BytesIO(payload), allow_pickle=False)
                if data.shape != (858, 36) or data.dtype != np.float64 or not np.isfinite(data).all():
                    raise ValueError("Returned feature matrix schema invalid")
                old_condition = "clean" if condition == "clean" else condition.replace("__", "")
                reference = prepared / "base" / (old_condition + "_" + prep + ".npy")
                if reference.exists() and not np.array_equal(data[:, :18], np.load(reference, allow_pickle=False)):
                    raise ValueError("Reused feature equality failed; no automatic refit")
                (features / (condition + "_" + prep + ".npy")).write_bytes(payload)
                (diagnostics / (condition + "_" + prep + ".npz")).write_bytes(bundle.read("diagnostics_" + prep + ".npz"))
        ledger.append({"unit_id": unit["unit_id"], "condition": condition, "role": unit["role"],
            "assignment_sha256": assignment["assignment_sha256"], "result_archive_sha256": base.sha(archive),
            "adopted_from_original": assignment is old, "fits_completed": 0})
    if {r["condition"] for r in ledger} != set(spec.CONDITIONS) or len(ledger) != 26:
        raise ValueError("Full 26-condition preprocessing barrier failed")
    base.write_json(execution / "stage_a_union_ledger.json",
        {"status": "ALL_26_ORIGINAL_IDENTITIES_COMPLETE_VERIFIED", "units": ledger,
         "adopted_units": len(kept), "original_markers_rewritten": False, "fits_completed": 0})


def configure_client(execution, *, reuse_existing=False):
    # Create the client-only home BEFORE importing Prefect settings or clients.
    home = execution / "coordinator_client"
    if reuse_existing:
        if not home.is_dir() or home.is_symlink():
            raise ValueError("Existing coordinator client home is invalid")
    else:
        home.mkdir(parents=True, exist_ok=False)
    for key in tuple(os.environ):
        if key.startswith("PREFECT_"):
            os.environ.pop(key)
    os.environ.update(PREFECT_API_URL="http://127.0.0.1:4200/api", PREFECT_HOME=str(home),
        PREFECT_SERVER_EPHEMERAL_ENABLED="false", PREFECT_SERVER_ANALYTICS_ENABLED="false", DO_NOT_TRACK="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="42")


def interrupted_staging(prepared, execution, transport, workspace, kept, missing):
    """Read-only reconciliation of this exact pre-submission failure."""
    plan_path, stage_path = execution / "recovery_plan.json", execution / "stage_a/assignment.json"
    if base.sha(plan_path) != INTERRUPTED_PLAN_SHA or base.sha(stage_path) != INTERRUPTED_STAGE_A_SHA:
        raise ValueError("Not the exact interrupted staging checkpoint")
    plan = base.read_json(plan_path)
    claim = base.read_json(prepared / "recovery_claim.json")
    if (claim != {"execution": "recovery_v1", "recovery_plan_sha256": INTERRUPTED_PLAN_SHA} or
            plan["recovery_release_sha256"] != INTERRUPTED_RELEASE_SHA or
            plan["pc2_workspace"] != workspace or plan["missing_units"] != missing or plan["kept"] != kept):
        raise ValueError("Interrupted recovery ownership or workspace changed")
    if {p.name for p in execution.iterdir()} != {"coordinator_client", "stage_a", "storage_A", "storage_A_tasks", "recovery_plan.json"}:
        raise ValueError("Unexpected recovery artifacts; reconcile rather than retry")
    if {p.name for p in (execution / "stage_a").iterdir()} != {"assignment.json"}:
        raise ValueError("Stage A may already have been staged/submitted; continuation refused")
    if any((transport / "markers" / (u["unit_id"] + ".json")).exists() for u in missing):
        raise ValueError("A supposedly unexecuted scientific unit has returned")
    stage = validate_assignment(base.read_json(stage_path), require_authorized=True)
    if stage["units"] != missing or stage["authorization"]["recovery_plan_sha256"] != INTERRUPTED_PLAN_SHA:
        raise ValueError("Interrupted Stage A unit identities differ")
    guard = validate_assignment(base.read_json(execution / "storage_A/assignment.json"), require_authorized=True)
    if len(guard["units"]) != 1 or guard["units"][0]["identity"]["stage"] != "STORAGE" or guard["total_fit_budget"] != 0:
        raise ValueError("Original storage gate identity differs")
    u = guard["units"][0]
    if guard["authorization"]["recovery_plan_sha256"] != INTERRUPTED_PLAN_SHA:
        raise ValueError("Original storage gate has foreign authorization")
    with zipfile.ZipFile(returned_archive(u, guard, transport)) as bundle:
        evidence = json.loads(bundle.read("storage_receipt.json"))
    if (evidence["workspace"].casefold() != workspace.casefold() or evidence["phase"] != "A" or
            evidence["required_free_bytes"] != 80 * GIB or evidence["free_bytes"] < 80 * GIB or
            evidence["control_drive_free_bytes"] < 512 * 1024**2 or evidence["fits_completed"] != 0 or
            evidence["simulations"] != 0 or evidence["host"].casefold() != "desktop-urb44pu"):
        raise ValueError("The completed H: storage gate is invalid")
    submission = base.read_json(execution / "storage_A/submission_receipts" / (u["unit_id"] + ".json"))
    return plan, stage, evidence, submission


def require_unsubmitted_remote(stage, storage_submission):
    """Read only: refuse any already-created remote run for this assignment."""
    from prefect.client.orchestration import get_client
    with get_client(sync_client=True) as client:
        storage = client.read_flow_run(storage_submission["flow_run_id"])
        if not storage.state or storage.state.type.value != "COMPLETED":
            raise ValueError("Original storage run is not completed")
        for offset in range(0, 10000, 200):
            runs = client.read_flow_runs(limit=200, offset=offset)
            if any(r.parameters.get("assignment_sha256") == stage["assignment_sha256"] for r in runs):
                raise ValueError("Interrupted assignment has a remote run; reconcile instead of resubmitting")
            if len(runs) < 200:
                return
    raise ValueError("Cannot exhaust remote-run inventory within bound; no submission")


def resume(package, prepared, execution, transport, workspace, timeout=86400, continue_after_staging_fix=False):
    workspace = validate_workspace(workspace)
    if platform.node().casefold() != "maher":
        raise ValueError("Recovery must run externally on qualified PC1")
    if execution != prepared / "recovery_v1" or (execution.exists() and not continue_after_staging_fix):
        raise ValueError("Use the new recovery_v1 child; existing evidence is never overwritten")
    if (prepared / "recovery_claim.json").exists() and not continue_after_staging_fix:
        raise ValueError("Recovery has already been claimed; no automatic retry")
    receipt, old, kept, missing, receipts, release, task_bytes = inspect_saved(package, prepared, transport)
    if shutil.disk_usage(prepared).free < 30 * GIB or shutil.disk_usage(transport).free < 30 * GIB:
        raise ValueError("At least 30 GiB free required on PC1 analysis and transport drives; retained outputs require monitoring")
    if continue_after_staging_fix:
        plan, pending_stage, storage, storage_submission = interrupted_staging(
            prepared, execution, transport, workspace, kept, missing)
        configure_client(execution, reuse_existing=True)
        require_unsubmitted_remote(pending_stage, storage_submission)
    else:
        execution.mkdir()
        configure_client(execution)
    states = prior_states(receipts)
    require_worker_online()
    new_plan = {"status": "EXTERNALLY_CONFIRMED_RECOVERY_NOT_COMPLETE", "decision_id": "C1-D67",
        "original_preparation_sha256": PREPARATION_SHA, "original_assignment_sha256": ORIGINAL_ASSIGNMENT_SHA,
        "original_source_commit": receipt["source_commit"], "recovery_release_sha256": spec.digest(release),
        "pc2_workspace": workspace, "prior_remote_states": states, "kept": kept,
        "missing_units": missing, "remaining_stage_a_task_archive_bytes": task_bytes,
        "new_fit_budget": 16240, "reused_fit_budget": 240, "scientific_scope_sha256": spec.digest(spec.scope()),
        "storage_allowance_note": "80 GiB provisional reservation per phase, not a proven peak bound; original per-unit 5 GiB guard remains",
        "automatic_retry": False, "created_utc": datetime.now(timezone.utc).isoformat()}
    plan_path = execution / "recovery_plan.json"
    if not continue_after_staging_fix:
        base.write_json(plan_path, new_plan)
        with (prepared / "recovery_claim.json").open("xb") as stream:
            stream.write(base.canonical({"recovery_plan_sha256": base.sha(plan_path), "execution": execution.name}))
    plan_sha = base.sha(plan_path)
    started = time.monotonic()
    print("[REPRISE] 3 resultats conserves; 23 pretraitements manquants; PC2 sur H:; aucun ancien dossier deplace", flush=True)
    if continue_after_staging_fix:
        age = (datetime.now(timezone.utc) - datetime.fromisoformat(storage["checked_utc"])).total_seconds()
        if age < -5:
            raise ValueError("Storage receipt clock is in the future")
        if age > 1800:
            require_storage(package, execution, workspace, transport, plan_sha, "A_staging_fix",
                80 * GIB, receipt["source_commit"], timeout)
        with (execution / "staging_fix_receipt.json").open("xb") as stream:
            stream.write(base.canonical({"status": "STAGING_FIX_VALIDATED_NOT_COMPLETE",
                "decision_id": "C1-D68", "previous_plan_sha256": plan_sha,
                "current_recovery_release_sha256": spec.digest(release), "old_artifacts_overwritten": False,
                "old_stage_a_assignment_preserved": True, "fresh_storage_recheck": age > 1800,
                "recorded_utc": datetime.now(timezone.utc).isoformat()}))
    else:
        require_storage(package, execution, workspace, transport, plan_sha, "A", 80 * GIB, receipt["source_commit"], timeout)
    # No bulk scientific capsule is sent until the H: receipt passes.
    dispatch_options = {"continue_unsubmitted": True} if continue_after_staging_fix else {}
    stage_a = dispatch_stage(missing, prepared / "stage_a_tasks", execution / "stage_a",
        transport, plan_sha, timeout, workspace, **dispatch_options)
    assemble_union(old, stage_a, kept, prepared, execution, transport)
    if original.verify_sources(package)["files"] != receipt["source_files"]:
        raise ValueError("Original science changed during recovery")
    units, tasks = original.prepare_stage_b(package, prepared, execution, receipt["source_commit"])
    if sum(u["expected_fits"] for u in units) != 16240 or len(units) != 200:
        raise ValueError("Original fit or learning-unit budget changed")
    for role in ("PC1", "PC2"):
        if sum(u["expected_fits"] for u in units if u["role"] == role) != 8120:
            raise ValueError("Host fit allocation changed")
    if any((transport / "markers" / (u["unit_id"] + ".json")).exists() for u in units):
        raise ValueError("Unexpected completed learning unit; reconcile instead of refitting")
    require_storage(package, execution, workspace, transport, plan_sha, "B", 80 * GIB, receipt["source_commit"], timeout)
    if shutil.disk_usage(execution).free < 30 * GIB or shutil.disk_usage(transport).free < 30 * GIB:
        raise ValueError("PC1 storage guard failed before learning")
    stage_b = dispatch_stage(units, tasks, execution / "stage_b", transport, plan_sha, timeout, workspace)
    if original.verify_sources(package)["files"] != receipt["source_files"]:
        raise ValueError("Original science changed during learning")
    print("[PHASE] Barriere PC1/PC2 complete; analyse scientifique finale", flush=True)
    from c1e27_analysis import aggregate
    aggregate(stage_b, transport, prepared, execution, execution / "final_analysis")
    base.write_json(execution / "completion_receipt.json", {
        "status": "COMPLETE_SCIENTIFIC_CAMPAIGN_NOT_PAPER_CLOSURE", "new_fits": 16240, "reused_fits": 240,
        "adopted_preprocessing_units": 3, "recovered_preprocessing_units": 23,
        "pc2_workspace": workspace, "recovery_plan_sha256": plan_sha,
        "elapsed_recovery_seconds": time.monotonic() - started, "speedup_measured": None,
        "scientific_limits": spec.LIMITS})
    print("[C1-E27 TERMINE] Calcul scientifique termine; manuscrit et reponses aux reviewers restent a verifier", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capsule", type=Path, required=True)
    parser.add_argument("--transport", type=Path, default=Path(r"F:\Backup\R07_Dispatch"))
    parser.add_argument("--pc2-workspace", default=r"H:\R07Work\C1-E27\recovery_v1")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--continue-after-staging-fix", action="store_true")
    parser.add_argument("--confirm-new-fits", type=int)
    args = parser.parse_args()
    package = Path(__file__).resolve().parents[1]
    if args.inspect_only:
        if args.confirm_new_fits is not None:
            raise ValueError("Inspection cannot authorize execution")
        _, _, kept, missing, _, _, size = inspect_saved(package, args.capsule, args.transport)
        if args.continue_after_staging_fix:
            interrupted_staging(args.capsule, args.capsule / "recovery_v1", args.transport,
                validate_workspace(args.pc2_workspace), kept, missing)
        print(json.dumps({"status": "INSPECTED_NOT_RESUMED", "kept": len(kept), "missing": len(missing),
                          "missing_by_role": {r: sum(u["role"] == r for u in missing) for r in ("PC1", "PC2")},
                          "missing_task_archive_bytes": size, "fits_executed": 0,
                          "pc2_workspace": validate_workspace(args.pc2_workspace)}))
        return
    if args.confirm_new_fits != 16240:
        raise ValueError("External explicit recovery confirmation of the unchanged 16240-fit ceiling is required")
    resume(package, args.capsule, args.capsule / "recovery_v1", args.transport, args.pc2_workspace,
        continue_after_staging_fix=args.continue_after_staging_fix)


if __name__ == "__main__":
    main()
