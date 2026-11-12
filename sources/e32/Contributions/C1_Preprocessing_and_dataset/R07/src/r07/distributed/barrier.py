"""Validate the complete two-node barrier; never execute aggregation."""
from __future__ import annotations

from pathlib import Path

from .contracts import (canonical_bytes, digest_file, read_json,
                        validate_completion_barrier, validate_result_archive)


def verify_barrier(*, assignment: Path, markers: Path, results: Path, output: Path) -> dict:
    output = Path(output)
    if output.exists():
        raise ValueError("Barrier output exists; overwrite refused")
    manifest = read_json(assignment)
    receipts = [read_json(path) for path in sorted(Path(markers).glob("*.json"))]
    ledger = validate_completion_barrier(manifest, receipts)
    for receipt in receipts:
        result = Path(results) / (receipt["result_archive_sha256"] + ".zip")
        if (not result.is_file() or result.is_symlink() or
                result.stat().st_size != receipt["result_archive_bytes"] or
                digest_file(result) != receipt["result_archive_sha256"]):
            raise ValueError("Returned artifact mismatch: " + receipt["unit_id"])
        validate_result_archive(result, receipt)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_bytes(ledger))
    return ledger

