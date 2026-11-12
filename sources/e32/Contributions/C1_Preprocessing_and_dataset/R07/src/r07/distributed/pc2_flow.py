"""PC2 lifecycle: fetch one authorized frozen unit, return result, then marker."""
from __future__ import annotations

from pathlib import Path

from prefect import flow

from .contracts import canonical_bytes, read_json, validate_assignment
from .runtime import execute_unit
from .transport import base_ssh, fetch_task, push


@flow(name="C1-R07-distributed-unit-v2", retries=0, persist_result=False,
      log_prints=True, description="One authorized frozen unit; verified marker-last return")
def pc2_unit_flow(*, unit_id: str, assignment_sha256: str,
                  assignment_artifact_sha256: str, task_archive_sha256: str,
                  node_home: str, ssh_exe: str, identity_file: str,
                  known_hosts: str, pc1_host: str) -> dict:
    unit_root = Path(node_home).resolve() / "distributed_units_v2" / unit_id
    unit_root.mkdir(parents=True, exist_ok=False)
    command = base_ssh(Path(ssh_exe), Path(identity_file), Path(known_hosts), pc1_host)
    manifest_path = unit_root / "assignment.json"
    fetch_task(command, assignment_artifact_sha256, manifest_path, ".json")
    manifest = validate_assignment(read_json(manifest_path), require_authorized=True)
    if manifest["assignment_sha256"] != assignment_sha256:
        raise ValueError("Submitted assignment hash mismatch")
    archive = unit_root / "task.zip"
    fetch_task(command, task_archive_sha256, archive)
    result, marker = execute_unit(manifest=manifest, unit_id=unit_id, role="PC2",
                                  archive=archive, attempt=unit_root / "execution")
    push(command, "put-result", result, marker["result_archive_sha256"] + ".zip")
    marker_path = unit_root / "returned_completion_marker.json"
    marker_path.write_bytes(canonical_bytes(marker))
    push(command, "put-marker", marker_path, unit_id + ".json")
    print(f"[RETURNED] unit_id={unit_id} result_sha256={marker['result_archive_sha256']}",
          flush=True)
    return marker

