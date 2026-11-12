"""One external launch: reuse completed fits, preprocess, train and analyze R07."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import getpass
import hashlib
import io
import os
from pathlib import Path
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
from c1_progress import Progress, duration_text
from c1e26_campaign import source_features, remote_status, persist_progress
from r07.distributed.contracts import assignment_sha256, unit_identity_sha256, validate_assignment
from r07.distributed.coordinator import stage_authorized_assignment
from r07.distributed.runtime import run_pc1_local
from r07.distributed.submitter import submit_pc2_plan
from run_c1e25_pipeline import returned_archive, scoped_barrier, require_worker_online

E25_PREPARATION = "647d06f2ec6b103ceae3d87d4fb20c3d6cfc3d8737631a55d469c8c35a0ab376"
E26_LEDGER = "3eadfbfd95c498e320d0483ecaa29f2b9b81330cfa37aa99336eea344a58f4d5"
E26_ANALYSIS = "3a77e743fb03facf99c4705f7ba4aeb3f7b925a9ee82e530589ffe250e868b9d"
E26_ASSIGNMENT = "74a7cea976857a41c3391832ac5af57f51a84ecb6ceee82a7c403bce91ba984a"
TASK_SCRIPTS = ("c1e25.py", "c1e26.py", "c1e27_spec.py", "c1e27_signal.py", "c1e27_worker.py")


def verify_sources(package):
    report = base.read_json(package / "provenance/c1_e27_lineage.json")
    for relative, expected in report["files"].items():
        if base.sha(package / relative) != expected:
            raise ValueError("Unrecorded C1-E27 implementation change: " + relative)
    for relative, expected in report["legacy_sources"].items():
        if base.sha(package.parents[2] / relative) != expected:
            raise ValueError("Historical R03 source identity changed: " + relative)
    if report["scope_sha256"] != spec.digest(spec.scope()):
        raise ValueError("Frozen scope changed")
    return report


def make_task(package, directory, config, inputs):
    temporary = directory / "task.zip"
    contents = {"configuration.json": base.canonical(config),
                "run_unit.py": (package / "scripts/run_c1e27_unit.py").read_bytes()}
    contents.update({name: (package / "scripts" / name).read_bytes() for name in TASK_SCRIPTS})
    with zipfile.ZipFile(temporary, "x", zipfile.ZIP_DEFLATED, compresslevel=1) as bundle:
        for name, payload in sorted(contents.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o100644 << 16
            bundle.writestr(info, payload)
        for name, source in sorted(inputs.items()):
            info = zipfile.ZipInfo(config["inputs"][name]["path"], (1980, 1, 1, 0, 0, 0))
            info.compress_type, info.external_attr = zipfile.ZIP_DEFLATED, 0o100644 << 16
            with source.open("rb") as stream, bundle.open(info, "w", force_zip64=True) as target:
                shutil.copyfileobj(stream, target, 1024 * 1024)
    path = directory / (base.sha(temporary) + ".zip")
    if path.exists():
        raise ValueError("Immutable task archive already exists")
    temporary.rename(path)
    return path


def input_pins(paths):
    return {name: {"path": "data/" + name + path.suffix, "sha256": base.sha(path)} for name, path in paths.items()}


def make_unit(package, archive, config, role, commit):
    identity = {"experiment_id": spec.EXPERIMENT, "stage": config["stage"], "role": role,
        "expected_host": "MAHER" if role == "PC1" else "DESKTOP-URB44PU", "source_commit": commit,
        "inputs": [{"name": k, "sha256": v["sha256"]} for k, v in sorted(config["inputs"].items())],
        "expected_fits": config["expected_fits"], "task_archive_sha256": base.sha(archive),
        "configuration_sha256": spec.digest(config), "runner_sha256": base.sha(package / "scripts/run_c1e27_unit.py")}
    for key in ("condition", "preprocessor", "fold", "classifier"):
        if key in config:
            identity[key] = config[key]
    return {"unit_id": unit_identity_sha256(identity), "identity": identity,
        **{k: identity[k] for k in ("role", "expected_host", "expected_fits", "task_archive_sha256", "configuration_sha256", "runner_sha256")},
        "task_archive_bytes": archive.stat().st_size, "configuration_path": "configuration.json", "runner_path": "run_unit.py"}


def collect_cached_models(e25, e26, transport, output, rows):
    if base.sha(e26 / "completion_ledger.json") != E26_LEDGER or base.sha(e26 / "final_analysis/analysis_receipt.json") != E26_ANALYSIS:
        raise ValueError("Unpinned E26 completion evidence")
    e26_assignment = validate_assignment(base.read_json(e26 / "assignment.json"), require_authorized=True)
    if e26_assignment["assignment_sha256"] != E26_ASSIGNMENT:
        raise ValueError("E26 source assignment changed")
    assignments = [validate_assignment(base.read_json(e25 / "stage_b/assignment.json"), require_authorized=True), e26_assignment]
    cache, origins = {}, []
    for assignment in assignments:
        for unit in assignment["units"]:
            archive = returned_archive(unit, assignment, transport)
            with zipfile.ZipFile(archive) as bundle:
                runtime = __import__("json").loads(bundle.read("runtime.json"))
                if runtime["versions"] != base.VERSIONS:
                    raise ValueError("Historical fitted-model runtime changed")
                records = __import__("json").loads(bundle.read("fit_records.json"))["records"]
                for record in records:
                    prep, fold, model, representation = (record[k] for k in ("preprocessor", "fold", "classifier", "representation"))
                    key = (prep, fold, model, representation)
                    expected_train = spec.digest([r["window_id"] for r in rows if r["fold"] != fold])
                    if (key in cache or record["model_contract"] != spec.MODEL_CONTRACTS[model] or
                            record["training_window_ids_sha256"] != expected_train or
                            record["feature_columns"] != [base.FEATURES[i] for i in base.REPRESENTATIONS[representation]]):
                        raise ValueError("Historical fit contract, training rows or feature identity changed")
                    data = bundle.read(record["saved_model"]["path"])
                    if hashlib.sha256(data).hexdigest() != record["saved_model"]["sha256"]:
                        raise ValueError("Historical fitted model hash changed")
                    name = f"{prep}_{fold}_{model}_{representation}.pickle"
                    path = output / "cache" / name
                    path.write_bytes(data)
                    predictions = {}
                    for condition in base.CONDITIONS:
                        member = f"predictions_{representation}_{model}_{condition}.csv"
                        data = bundle.read(member)
                        prediction_path = output / "cache" / f"{prep}_{fold}_{model}_{representation}__{condition}.csv"
                        prediction_path.write_bytes(data)
                        predictions[condition] = {"path": prediction_path.name, "sha256": hashlib.sha256(data).hexdigest()}
                    cache[key] = (path, {**record, "prediction_artifacts": predictions})
            origins.append({"unit_id": unit["unit_id"], "assignment_sha256": assignment["assignment_sha256"], "archive_sha256": base.sha(archive)})
    if len(cache) != 240:
        raise ValueError("All 240 compatible completed fits are required; do not silently refit")
    for prep in ("none", "savgol"):
        for fold in range(1, 6):
            for model in spec.MODELS:
                records = {rep: cache[(prep, fold, model, rep)][1] for rep in base.REPRESENTATIONS}
                base.write_json(output / "cache" / f"index_{prep}_{fold}_{model}.json", records)
    base.write_json(output / "cache_origins.json", origins)
    return cache


def prepare(package, e25, e26, wheel, output, transport):
    started = time.monotonic()
    report = verify_sources(package)
    if (output.exists() or not output.is_absolute() or output.resolve().is_relative_to(package.parents[2].resolve())
            or output.resolve().is_relative_to(e25.parent.resolve()) or output.resolve().is_relative_to(e26.parent.resolve())):
        raise ValueError("Use a new absolute analysis directory outside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    if base.sha(wheel) != spec.WHEEL_SHA256:
        raise ValueError("Expected approved PyWavelets wheel is missing or changed")
    if base.sha(e25.parent / "preparation_receipt.json") != E25_PREPARATION:
        raise ValueError("E25 saved-waveform preparation changed")
    features, metadata, baseline, primary = source_features(e25)
    rows = base.validate_metadata(base.read_json(metadata))
    saved_preparation = base.read_json(e25.parent / "preparation_receipt.json")
    source_clean = e25.parent / "base/clean.npy"
    if base.sha(source_clean) != saved_preparation["base_files"]["clean.npy"]:
        raise ValueError("Frozen clean waveform capsule changed")
    if shutil.disk_usage(output.parent).free < 30 * 1024**3:
        raise ValueError("At least 30 GiB free required before packaging; full retained-model storage is additional")
    output.mkdir(parents=True)
    for folder in ("base", "cache", "stage_a_tasks"):
        (output / folder).mkdir()
    shutil.copyfile(metadata, output / "metadata.json")
    shutil.copyfile(source_clean, output / "base/clean.npy")
    shutil.copyfile(wheel, output / "base/pywavelets.whl")
    shutil.copyfile(baseline, output / "base/reused_e25_summary.csv")
    shutil.copyfile(primary, output / "base/reused_e25_primary_endpoint.json")
    for (condition, prep), path in features.items():
        shutil.copyfile(path, output / "base" / (condition + "_" + prep + ".npy"))
    collect_cached_models(e25, e26, transport, output, rows)
    commit = subprocess.check_output(["git", "-c", "safe.directory=" + package.parents[2].as_posix(), "-C", str(package.parents[2]), "rev-parse", "HEAD"], text=True).strip()
    units = []
    paths = {"metadata": output / "metadata.json", "clean": output / "base/clean.npy", "pywavelets_wheel": output / "base/pywavelets.whl"}
    for index, condition in enumerate(spec.CONDITIONS):
        config = spec.configuration("A", input_pins(paths), condition=condition)
        archive = make_task(package, output / "stage_a_tasks", config, paths)
        units.append(make_unit(package, archive, config, "PC1" if index % 2 == 0 else "PC2", commit))
        print(f"[PACKAGING] {index + 1}/{len(spec.CONDITIONS)} condition capsules; fits=0", flush=True)
    base.write_json(output / "scope.json", spec.scope())
    pins = {p.relative_to(output).as_posix(): base.sha(p) for p in output.rglob("*") if p.is_file() and p.parent.name != "stage_a_tasks"}
    receipt = {"status": "PREPARED_NOT_EXECUTED", "protocol": spec.PROTOCOL, "source_commit": commit,
        "prepared_utc": datetime.now(timezone.utc).isoformat(), "packaging_elapsed_seconds": time.monotonic() - started,
        "scope_sha256": spec.digest(spec.scope()), "source_files": report["files"], "base_files": pins,
        "stage_a_units": units, "new_fit_budget": 16240, "reused_fits": 240, "fits_executed": 0}
    base.write_json(output / "preparation_receipt.json", receipt)
    return receipt


def freeze(units, preparation_sha):
    assignment = {"schema_version": 2, "status": "FROZEN", "experiment_id": spec.EXPERIMENT,
        "execution_authorized": True, "authorization": {"authorized_by": getpass.getuser(),
            "authorized_utc": datetime.now(timezone.utc).isoformat(), "decision_id": "C1-D66",
            "scope": "One externally confirmed C1-E27 campaign; maximum 16240 new fits, automatic stages and final inference",
            "preparation_sha256": preparation_sha}, "scientific_limits": spec.LIMITS,
        "total_fit_budget": sum(u["expected_fits"] for u in units), "units": units}
    assignment["assignment_sha256"] = assignment_sha256(assignment)
    return validate_assignment(assignment, require_authorized=True)


def cancel_pending(receipts):
    """Failure containment applies only to this campaign's not-yet-running jobs."""
    from prefect.client.orchestration import get_client
    from prefect.states import Cancelled
    with get_client(sync_client=True) as client:
        for receipt in receipts:
            run = client.read_flow_run(receipt["flow_run_id"])
            if run.state and run.state.type.value in {"SCHEDULED", "PENDING"}:
                client.set_flow_run_state(run.id, Cancelled(message="C1-E27 sibling failure; no retry"), force=True)


def comparable_eta(units, completed_ids, observations):
    """Use only observed same-host, same-classifier/noise units; no extra fits."""
    bins = {}
    by_id = {u["unit_id"]: u for u in units}
    def cost_key(unit):
        identity = unit["identity"]
        return (unit["role"], identity["stage"], identity.get("classifier") if identity["stage"] == "B" else spec.condition(identity["condition"])[0])
    for uid, seconds in observations:
        unit = by_id[uid]
        if seconds > 0:
            bins.setdefault(cost_key(unit), []).append(seconds / max(1, unit["expected_fits"]))
    totals = {"PC1": 0.0, "PC2": 0.0}
    for unit in units:
        if unit["unit_id"] in completed_ids:
            continue
        values = bins.get(cost_key(unit))
        if not values:
            return None
        totals[unit["role"]] += statistics.median(values) * max(1, unit["expected_fits"])
    return max(totals.values())


def dispatch_stage(units, tasks, root, transport, preparation_sha, timeout, poll_interval=15):
    root.mkdir()
    assignment = freeze(units, preparation_sha)
    path = root / "assignment.json"
    base.write_json(path, assignment)
    plan = root / "dispatch_plan.json"
    stage_authorized_assignment(assignment=path, task_directory=tasks, transport_tasks=transport / "tasks", output_plan=plan)
    receipts = submit_pc2_plan(plan_path=plan, assignment_path=path, work_pool="C1-R07-8658432b19cb", work_queue="PC2",
        node_home=r"C:\r07node_e24_v2r1", ssh_exe=r"C:\Windows\System32\OpenSSH\ssh.exe",
        identity_file=r"C:\r07node\transport\c1_r07_dispatch_ed25519", known_hosts=r"C:\Users\Maher\.ssh\known_hosts",
        pc1_host="Maher369@172.16.1.178", receipt_directory=root / "submission_receipts", confirm=True)
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
            phase = "PRETRAITEMENT" if units[0]["identity"]["stage"] == "A" else "ENTRAINEMENT"
            snapshot = progress.update(phase, counts, totals, durations, active, pc_states)
            completed_ids = {u["unit_id"] for u in units if (transport / "markers" / (u["unit_id"] + ".json")).is_file()}
            completed_remote = [r for r in receipts if states[r["unit_id"]] == "COMPLETED"]
            if len(completed_remote) == len(remote_durations):
                observations += [(r["unit_id"], d) for r, d in zip(completed_remote, remote_durations)]
            eta = comparable_eta(units, completed_ids, observations)
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


def collect_stage_a(assignment, transport, prepared, execution):
    features, diagnostics = execution / "verified_features", execution / "verified_diagnostics"
    features.mkdir()
    diagnostics.mkdir()
    for unit in assignment["units"]:
        condition = unit["identity"]["condition"]
        with zipfile.ZipFile(returned_archive(unit, assignment, transport)) as bundle:
            for prep in spec.PREPROCESSORS:
                payload = bundle.read("features_" + prep + ".npy")
                data = np.load(io.BytesIO(payload), allow_pickle=False)
                if data.shape != (858, 36) or not np.isfinite(data).all():
                    raise ValueError("Returned feature matrix incomplete")
                old = "clean" if condition == "clean" else condition.replace("__", "")
                reference = prepared / "base" / (old + "_" + prep + ".npy")
                if reference.exists() and not np.array_equal(data[:, :18], np.load(reference, allow_pickle=False)):
                    raise ValueError("Reused E25 features differ; do not refit automatically")
                (features / (condition + "_" + prep + ".npy")).write_bytes(payload)
                (diagnostics / (condition + "_" + prep + ".npz")).write_bytes(bundle.read("diagnostics_" + prep + ".npz"))


def prepare_stage_b(package, prepared, execution, commit):
    tasks = execution / "stage_b_tasks"
    tasks.mkdir()
    units = []
    for definition in spec.learning_plan():
        prep, fold, model = (definition[k] for k in ("preprocessor", "fold", "classifier"))
        paths = {"metadata": prepared / "metadata.json", **{c: execution / "verified_features" / (c + "_" + prep + ".npy") for c in spec.CONDITIONS}}
        if prep in ("none", "savgol"):
            records = {}
            for rep in base.REPRESENTATIONS:
                key = f"{prep}_{fold}_{model}_{rep}"
                paths["cache_" + rep] = prepared / "cache" / (key + ".pickle")
            # Origin records were extracted from validated, hash-pinned source archives.
            index_path = prepared / "cache" / f"index_{prep}_{fold}_{model}.json"
            if not index_path.exists():
                raise ValueError("Verified historical model index is missing")
            paths["cache_index"] = index_path
            index = base.read_json(index_path)
            for rep in base.REPRESENTATIONS:
                for condition, pin in index[rep]["prediction_artifacts"].items():
                    paths[f"cachepred_{rep}_{condition}"] = prepared / "cache" / pin["path"]
            paths["reuse_base_features"] = prepared / "base" / ("clean_" + prep + ".npy")
        config = spec.configuration("B", input_pins(paths), preprocessor=prep, fold=fold, classifier=model)
        archive = make_task(package, tasks, config, paths)
        units.append(make_unit(package, archive, config, definition["role"], commit))
    if sum(u["expected_fits"] for u in units) != 16240:
        raise ValueError("Global new-fit budget changed")
    return units, tasks


def launch(package, prepared, transport, preparation_sha, timeout=86400):
    current = verify_sources(package)
    receipt = base.read_json(prepared / "preparation_receipt.json")
    if (platform.node().casefold() != "maher" or base.sha(prepared / "preparation_receipt.json") != preparation_sha or
            receipt["source_files"] != current["files"] or receipt["new_fit_budget"] != 16240 or
            receipt["status"] != "PREPARED_NOT_EXECUTED" or receipt["protocol"] != spec.PROTOCOL or
            receipt["scope_sha256"] != spec.digest(spec.scope()) or receipt["fits_executed"] != 0):
        raise ValueError("Wrong host, release, prepared capsule or fit budget")
    for name, digest in receipt["base_files"].items():
        if base.sha(prepared / name) != digest:
            raise ValueError("Prepared artifact changed: " + name)
    for key in tuple(os.environ):
        if key.startswith("PREFECT_"):
            os.environ.pop(key)
    execution = prepared / "execution_v1"
    if execution.exists():
        raise ValueError("Execution exists: reconcile; no automatic retry")
    os.environ.update(PREFECT_API_URL="http://127.0.0.1:4200/api", PREFECT_HOME=str(execution / "coordinator_client"),
        PREFECT_SERVER_EPHEMERAL_ENABLED="false", PREFECT_SERVER_ANALYTICS_ENABLED="false", DO_NOT_TRACK="1",
        OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONHASHSEED="42")
    require_worker_online()
    execution.mkdir(exist_ok=True)
    if any(p.name != "coordinator_client" for p in execution.iterdir()):
        raise ValueError("Unexpected artifacts appeared before launch")
    started = time.monotonic()
    stage_a = dispatch_stage(receipt["stage_a_units"], prepared / "stage_a_tasks", execution / "stage_a", transport, preparation_sha, timeout)
    collect_stage_a(stage_a, transport, prepared, execution)
    if verify_sources(package)["files"] != receipt["source_files"]:
        raise ValueError("Scientific release changed during preprocessing")
    units, tasks = prepare_stage_b(package, prepared, execution, receipt["source_commit"])
    stage_b = dispatch_stage(units, tasks, execution / "stage_b", transport, preparation_sha, timeout)
    if verify_sources(package)["files"] != receipt["source_files"]:
        raise ValueError("Scientific release changed during training")
    print("[PHASE] Both-host barrier passed; automatic final scientific analysis", flush=True)
    from c1e27_analysis import aggregate
    aggregate(stage_b, transport, prepared, execution, execution / "final_analysis")
    base.write_json(execution / "completion_receipt.json", {"status": "COMPLETE_SCIENTIFIC_CAMPAIGN_NOT_PAPER_CLOSURE",
        "new_fits": 16240, "reused_fits": 240, "elapsed_seconds": time.monotonic() - started,
        "speedup_measured": None, "scientific_limits": spec.LIMITS})
    print("[C1-E27 COMPLETE] 16240 new fits; 240 reused; inference and figures saved; paper revision remains", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--e25", type=Path, default=Path(r"G:\IEEE9_Analysis\R07\experiments\C1-E25\capsule_grid_v1\execution_v1"))
    parser.add_argument("--e26", type=Path, default=Path(r"G:\IEEE9_Analysis\R07\experiments\C1-E26\campaign_v2\execution_v1"))
    package = Path(__file__).resolve().parents[1]
    parser.add_argument("--wheel", type=Path, default=package / "deployment/scientific_dependencies/pywavelets-1.8.0-cp312-cp312-win_amd64.whl")
    parser.add_argument("--transport", type=Path, default=Path(r"F:\Backup\R07_Dispatch"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--confirm-new-fits", type=int)
    args = parser.parse_args()
    if not args.prepare_only and args.confirm_new_fits != 16240:
        raise ValueError("One external campaign-level confirmation of exactly 16240 new fits is required")
    if args.prepare_only and args.confirm_new_fits is not None:
        raise ValueError("Preparation-only cannot authorize scientific execution")
    prepare(package, args.e25, args.e26, args.wheel, args.output, args.transport)
    if not args.prepare_only:
        launch(package, args.output, args.transport, base.sha(args.output / "preparation_receipt.json"))


if __name__ == "__main__":
    main()
