"""Shared verified execution path for PC1-local and PC2 worker units."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import os
import platform
import shutil
import subprocess
import sys
import uuid
import zipfile

from .contracts import (canonical_bytes, digest_file, read_json, safe_relative,
                        validate_assignment, validate_result_archive)

MAX_TASK_UNPACKED_BYTES = 50 * 1024 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def extract_safe(archive: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive) as bundle:
        names: set[str] = set()
        total = 0
        for info in bundle.infolist():
            path = PurePosixPath(info.filename)
            if (path.is_absolute() or ".." in path.parts or "\\" in info.filename or
                    info.filename in names):
                raise ValueError("Unsafe or duplicate task archive member")
            names.add(info.filename)
            total += info.file_size
            if total > MAX_TASK_UNPACKED_BYTES:
                raise ValueError("Task archive exceeds unpacked size limit")
            if ((info.external_attr >> 16) & 0o170000) == 0o120000:
                raise ValueError("Task archive symlink is forbidden")
            if not info.is_dir():
                target = destination.joinpath(*path.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)


def find_unit(manifest: dict, unit_id: str, role: str, host: str) -> dict:
    validate_assignment(manifest, require_authorized=True)
    matches = [unit for unit in manifest["units"] if unit["unit_id"] == unit_id]
    if len(matches) != 1 or matches[0]["role"] != role:
        raise ValueError("Assigned unit is absent or belongs to another role")
    unit = matches[0]
    if host.casefold() != unit["expected_host"].casefold():
        raise ValueError("Execution hostname does not match the frozen assignment")
    return unit


def execute_unit(*, manifest: dict, unit_id: str, role: str, archive: Path,
                 attempt: Path, host: str | None = None,
                 python_executable: str | None = None) -> tuple[Path, dict]:
    """Execute exactly one authorized unit and create its result before marker."""
    host = host or platform.node()
    unit = find_unit(manifest, unit_id, role, host)
    archive = Path(archive)
    if (not archive.is_file() or archive.is_symlink() or
            archive.stat().st_size != unit["task_archive_bytes"] or
            digest_file(archive) != unit["task_archive_sha256"]):
        raise ValueError("Task archive does not match the frozen unit")
    attempt = Path(attempt)
    attempt.mkdir(parents=True, exist_ok=False)
    unpacked = attempt / "task"
    unpacked.mkdir()
    extract_safe(archive, unpacked)
    runner = unpacked.joinpath(*PurePosixPath(safe_relative(unit["runner_path"], "runner_path")).parts)
    configuration = unpacked.joinpath(*PurePosixPath(
        safe_relative(unit["configuration_path"], "configuration_path")).parts)
    if not runner.is_file() or digest_file(runner) != unit["runner_sha256"]:
        raise ValueError("Unit runner hash mismatch")
    if not configuration.is_file() or digest_file(configuration) != unit["configuration_sha256"]:
        raise ValueError("Unit configuration hash mismatch")
    output = attempt / "output"
    output.mkdir()
    completed = subprocess.run([
        python_executable or sys.executable, str(runner), "--unit-id", unit_id,
        "--role", role, "--assignment-sha256", manifest["assignment_sha256"],
        "--configuration-sha256", unit["configuration_sha256"],
        "--expected-fits", str(unit["expected_fits"]),
        "--configuration", str(configuration), "--output", str(output),
    ], cwd=unpacked, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Unit stopped with exit code {completed.returncode}")
    science = read_json(output / "scientific_receipt.json")
    if science.get("expected_fits") != unit["expected_fits"]:
        raise ValueError("Scientific receipt fit count differs from frozen budget")
    result_path = Path(shutil.make_archive(str(attempt / "result"), "zip", root_dir=output))
    marker = {
        "schema_version": 2, "status": "COMPLETED", "completed_utc": now(),
        "unit_id": unit_id, "role": role, "expected_host": unit["expected_host"],
        "configuration_sha256": unit["configuration_sha256"],
        "expected_fits": unit["expected_fits"],
        "assignment_sha256": manifest["assignment_sha256"],
        "result_archive_sha256": digest_file(result_path),
        "result_archive_bytes": result_path.stat().st_size,
    }
    validate_result_archive(result_path, marker)
    marker_path = attempt / "completion_marker.json"
    marker_path.write_bytes(canonical_bytes(marker))
    return result_path, marker


def publish_write_once(source: Path, destination: Path) -> None:
    """Atomically publish a file without replacing a prior artifact."""
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".partial")
    try:
        shutil.copyfile(source, temporary)
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise ValueError(f"Publication target already exists: {destination}") from exc
    finally:
        if temporary.exists():
            temporary.unlink()


def run_pc1_local(*, manifest_path: Path, task_archive: Path, unit_id: str,
                  attempt: Path, results: Path, markers: Path,
                  host: str | None = None) -> dict:
    manifest = read_json(manifest_path)
    result, marker = execute_unit(manifest=manifest, unit_id=unit_id, role="PC1",
                                  archive=task_archive, attempt=attempt, host=host)
    publish_write_once(result, Path(results) / (marker["result_archive_sha256"] + ".zip"))
    marker_source = Path(attempt) / "completion_marker.json"
    publish_write_once(marker_source, Path(markers) / (unit_id + ".json"))
    return marker

