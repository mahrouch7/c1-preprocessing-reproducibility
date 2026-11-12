"""Explicit Prefect submission adapter for already-staged authorized PC2 units."""
from __future__ import annotations

from pathlib import Path

from .contracts import (canonical_bytes, digest_file, read_json, require_hex,
                        validate_assignment)
from .runtime import publish_write_once


def submit_pc2_plan(*, plan_path: Path, assignment_path: Path,
                    work_pool: str, work_queue: str,
                    node_home: str, ssh_exe: str, identity_file: str,
                    known_hosts: str, pc1_host: str, receipt_directory: Path,
                    confirm: bool) -> list[dict]:
    if not confirm:
        raise ValueError("Explicit PC2 submission confirmation is required")
    plan = read_json(plan_path)
    if plan.get("status") != "STAGED_NOT_SUBMITTED" or plan.get("prefect_contacted") is not False:
        raise ValueError("Only a fresh staged plan can be submitted")
    assignment = require_hex(plan.get("assignment_sha256"), "assignment_sha256")
    artifact = require_hex(plan.get("assignment_artifact_sha256"),
                           "assignment_artifact_sha256")
    submissions = plan.get("pc2_submissions")
    if not isinstance(submissions, list) or not submissions:
        raise ValueError("Staged plan has no PC2 units")
    manifest = validate_assignment(read_json(assignment_path), require_authorized=True)
    if (manifest["assignment_sha256"] != assignment or
            digest_file(assignment_path) != artifact or
            plan.get("total_fit_budget") != manifest["total_fit_budget"]):
        raise ValueError("Staged plan does not match the authorized assignment")
    expected = {unit["unit_id"]: unit for unit in manifest["units"] if unit["role"] == "PC2"}
    validated = []
    seen: set[str] = set()
    for item in submissions:
        if item.get("assignment_sha256") != assignment or item.get(
                "assignment_artifact_sha256") != artifact:
            raise ValueError("PC2 submission does not match its staged plan")
        unit_id = require_hex(item.get("unit_id"), "unit_id")
        task_hash = require_hex(item.get("task_archive_sha256"), "task_archive_sha256")
        if unit_id in seen or unit_id not in expected or task_hash != expected[unit_id]["task_archive_sha256"]:
            raise ValueError("Unexpected, duplicate or mismatched PC2 submission")
        seen.add(unit_id)
        validated.append((unit_id, task_hash))
    if seen != set(expected):
        raise ValueError("Staged plan omits an authorized PC2 unit")
    receipt_directory = Path(receipt_directory)
    for unit_id, _ in validated:
        if (receipt_directory / (unit_id + ".json")).exists():
            raise ValueError("Submission receipt already exists; reconcile before retry: " + unit_id)
    # All local validation is complete before the first Prefect mutation.
    from prefect.client.orchestration import get_client
    from prefect.types.entrypoint import EntrypointType
    from .pc2_flow import pc2_unit_flow
    deployment = pc2_unit_flow.to_deployment(
        name="authorized-" + assignment[:12], work_pool_name=work_pool,
        work_queue_name=work_queue, version=assignment,
        entrypoint_type=EntrypointType.MODULE_PATH,
        description="Explicitly authorized C1-R07 immutable units; retries=0",
        tags=["R07", "authorized-scientific-unit", assignment[:12]])
    deployment_id = deployment.apply()
    receipts = []
    with get_client(sync_client=True) as client:
        for unit_id, task_hash in validated:
            parameters = dict(unit_id=unit_id, assignment_sha256=assignment,
                              assignment_artifact_sha256=artifact,
                              task_archive_sha256=task_hash, node_home=node_home,
                              ssh_exe=ssh_exe, identity_file=identity_file,
                              known_hosts=known_hosts, pc1_host=pc1_host)
            run = client.create_flow_run_from_deployment(
                deployment_id=deployment_id, parameters=parameters,
                idempotency_key="C1-R07-v2-" + assignment + "-" + unit_id)
            receipt = {"schema_version": 1, "status": "SUBMITTED_NOT_COMPLETED",
                       "unit_id": unit_id, "assignment_sha256": assignment,
                       "deployment_id": str(deployment_id), "flow_run_id": str(run.id),
                       "idempotency_key": "C1-R07-v2-" + assignment + "-" + unit_id,
                       "automatic_retry": False}
            temporary = receipt_directory / (unit_id + ".receipt-source")
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_bytes(canonical_bytes(receipt))
            try:
                publish_write_once(temporary, receipt_directory / (unit_id + ".json"))
            finally:
                temporary.unlink(missing_ok=True)
            receipts.append(receipt)
    return receipts


