"""Prepare content-addressed tasks and a PC2 submission plan.

Staging is deliberately impossible until the frozen assignment contains an
explicit authorization record. This module does not contact Prefect.
"""
from __future__ import annotations

from pathlib import Path

from .contracts import canonical_bytes, digest_file, read_json, validate_assignment
from .runtime import publish_write_once


def stage_authorized_assignment(*, assignment: Path, task_directory: Path,
                                transport_tasks: Path, output_plan: Path) -> dict:
    manifest = validate_assignment(read_json(assignment), require_authorized=True)
    output_plan = Path(output_plan)
    if output_plan.exists():
        raise ValueError("Dispatch plan exists; overwrite refused")
    assignment_artifact = Path(assignment)
    assignment_digest = digest_file(assignment_artifact)
    pc2 = []
    staged_archives: set[str] = set()
    for unit in manifest["units"]:
        archive = Path(task_directory) / (unit["task_archive_sha256"] + ".zip")
        if (not archive.is_file() or archive.stat().st_size != unit["task_archive_bytes"] or
                digest_file(archive) != unit["task_archive_sha256"]):
            raise ValueError("Missing or mismatched frozen task: " + unit["unit_id"])
        if unit["task_archive_sha256"] not in staged_archives:
            publish_write_once(archive, Path(transport_tasks) / archive.name)
            staged_archives.add(unit["task_archive_sha256"])
        if unit["role"] == "PC2":
            pc2.append({"unit_id": unit["unit_id"],
                        "assignment_sha256": manifest["assignment_sha256"],
                        "assignment_artifact_sha256": assignment_digest,
                        "task_archive_sha256": unit["task_archive_sha256"],
                        "expected_host": unit["expected_host"]})
    # Publish the assignment last so it acts as the ready signal for staged tasks.
    publish_write_once(assignment_artifact, Path(transport_tasks) / (assignment_digest + ".json"))
    plan = {"schema_version": 1, "status": "STAGED_NOT_SUBMITTED",
            "assignment_sha256": manifest["assignment_sha256"],
            "assignment_artifact_sha256": assignment_digest,
            "total_fit_budget": manifest["total_fit_budget"], "pc2_submissions": pc2,
            "prefect_contacted": False, "scientific_units_executed": 0}
    output_plan.parent.mkdir(parents=True, exist_ok=True)
    output_plan.write_bytes(canonical_bytes(plan))
    return plan

