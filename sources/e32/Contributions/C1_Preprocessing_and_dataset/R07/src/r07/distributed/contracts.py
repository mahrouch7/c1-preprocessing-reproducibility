"""Version-2 immutable assignment and completion-barrier contracts.

The module is standard-library only. It performs no network access and starts no
scientific process. Version 1 remains preserved in the deployment evidence tree;
new integrated lifecycle assignments must use this stricter version 2 contract.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable
import zipfile

HEX64 = re.compile(r"^[0-9a-f]{64}$")
HOST = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
ROLES = {"PC1", "PC2"}
MAX_RECEIPT_BYTES = 1024 * 1024
MAX_ARCHIVE_BYTES = 100 * 1024 * 1024 * 1024

REQUIRED_LIMITS = {
    "initial_detection_requires_clearing": False,
    "true_b7_dijkstra_distance_is_fault_type_predictor": False,
    "e16_voltage_failures_preserved": 14,
    "abc_abcg_ambiguity_preserved": True,
    "model_validity_limits_unresolved": True,
    "clean_pilot_accuracy_percent": 93.473193,
    "clean_pilot_scope": "grouped_interpolation_only",
}


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def safe_relative(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a relative POSIX path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or value in {"", "."}:
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return value


def require_hex(value: Any, field: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def unit_identity_sha256(identity: dict[str, Any]) -> str:
    if not isinstance(identity, dict) or not identity:
        raise ValueError("Unit identity must be a non-empty object")
    return digest_bytes(canonical_bytes(identity))


def assignment_sha256(manifest: dict[str, Any]) -> str:
    body = {key: value for key, value in manifest.items() if key != "assignment_sha256"}
    return digest_bytes(canonical_bytes(body))


def validate_assignment(manifest: dict[str, Any], *, require_authorized: bool) -> dict[str, Any]:
    if manifest.get("schema_version") != 2 or manifest.get("status") != "FROZEN":
        raise ValueError("Assignment must use frozen schema version 2")
    experiment_id = manifest.get("experiment_id")
    if not isinstance(experiment_id, str) or not ID.fullmatch(experiment_id):
        raise ValueError("Invalid experiment_id")
    if manifest.get("scientific_limits") != REQUIRED_LIMITS:
        raise ValueError("Required C1 scientific limitations are absent or changed")
    authorized = manifest.get("execution_authorized")
    if not isinstance(authorized, bool):
        raise ValueError("execution_authorized must be Boolean")
    authorization = manifest.get("authorization")
    if authorized:
        if not isinstance(authorization, dict):
            raise ValueError("Authorized assignment lacks an authorization record")
        for field in ("authorized_by", "authorized_utc", "decision_id"):
            if not isinstance(authorization.get(field), str) or not authorization[field]:
                raise ValueError(f"Authorization field is missing: {field}")
    elif authorization is not None:
        raise ValueError("Unauthorized assignment must not contain authorization metadata")
    if require_authorized and not authorized:
        raise ValueError("Scientific execution is not explicitly authorized")
    fit_budget = manifest.get("total_fit_budget")
    if not isinstance(fit_budget, int) or fit_budget < 0:
        raise ValueError("total_fit_budget must be a non-negative integer")
    units = manifest.get("units")
    if not isinstance(units, list) or not units:
        raise ValueError("Assignment must contain at least one unit")
    seen: set[str] = set()
    observed_budget = 0
    for unit in units:
        if not isinstance(unit, dict):
            raise ValueError("Every unit must be an object")
        uid = unit.get("unit_id")
        if not isinstance(uid, str) or not HEX64.fullmatch(uid) or uid in seen:
            raise ValueError("Invalid or duplicate unit_id")
        seen.add(uid)
        if unit_identity_sha256(unit.get("identity")) != uid:
            raise ValueError(f"unit_id does not match canonical identity: {uid}")
        identity = unit["identity"]
        if unit.get("role") not in ROLES:
            raise ValueError(f"Invalid role: {uid}")
        host = unit.get("expected_host")
        if not isinstance(host, str) or not HOST.fullmatch(host):
            raise ValueError(f"Invalid expected_host: {uid}")
        for field in ("task_archive_sha256", "configuration_sha256", "runner_sha256"):
            require_hex(unit.get(field), f"{uid}.{field}")
        if identity.get("experiment_id") != experiment_id:
            raise ValueError(f"Unit experiment identity mismatch: {uid}")
        for field in ("role", "expected_host", "task_archive_sha256",
                      "configuration_sha256", "runner_sha256", "expected_fits"):
            if identity.get(field) != unit.get(field):
                raise ValueError(f"Unit identity field mismatch ({field}): {uid}")
        source_commit = identity.get("source_commit")
        if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", source_commit):
            raise ValueError(f"Unit identity source_commit is invalid: {uid}")
        inputs = identity.get("inputs")
        if not isinstance(inputs, list) or not inputs:
            raise ValueError(f"Unit identity must pin at least one input: {uid}")
        input_names: set[str] = set()
        for item in inputs:
            if (not isinstance(item, dict) or not isinstance(item.get("name"), str) or
                    not item["name"] or item["name"] in input_names):
                raise ValueError(f"Unit identity input is invalid or duplicate: {uid}")
            input_names.add(item["name"])
            require_hex(item.get("sha256"), f"{uid}.identity.inputs.sha256")
        safe_relative(unit.get("configuration_path"), f"{uid}.configuration_path")
        safe_relative(unit.get("runner_path"), f"{uid}.runner_path")
        size = unit.get("task_archive_bytes")
        fits = unit.get("expected_fits")
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"Invalid task_archive_bytes: {uid}")
        if not isinstance(fits, int) or fits < 0:
            raise ValueError(f"Invalid expected_fits: {uid}")
        observed_budget += fits
    if observed_budget != fit_budget:
        raise ValueError("total_fit_budget does not equal the frozen unit sum")
    require_hex(manifest.get("assignment_sha256"), "assignment_sha256")
    if assignment_sha256(manifest) != manifest["assignment_sha256"]:
        raise ValueError("Assignment manifest hash mismatch")
    return manifest


def validate_completion_barrier(manifest: dict[str, Any], receipts: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Return a complete verified ledger; never tolerate partial aggregation."""
    validate_assignment(manifest, require_authorized=True)
    expected = {unit["unit_id"]: unit for unit in manifest["units"]}
    observed: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        uid = receipt.get("unit_id")
        if uid not in expected or uid in observed:
            raise ValueError(f"Unexpected or duplicate receipt: {uid}")
        unit = expected[uid]
        if receipt.get("status") != "COMPLETED":
            raise ValueError(f"Non-completed receipt: {uid}")
        for field in ("role", "expected_host", "configuration_sha256", "expected_fits"):
            if receipt.get(field) != unit[field]:
                raise ValueError(f"Receipt {field} mismatch: {uid}")
        if receipt.get("assignment_sha256") != manifest["assignment_sha256"]:
            raise ValueError(f"Receipt assignment mismatch: {uid}")
        require_hex(receipt.get("result_archive_sha256"), f"{uid}.result_archive_sha256")
        if not isinstance(receipt.get("result_archive_bytes"), int) or receipt["result_archive_bytes"] <= 0:
            raise ValueError(f"Invalid result archive size: {uid}")
        observed[uid] = receipt
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError("Barrier incomplete; missing: " + ", ".join(missing))
    return {"schema_version": 2, "status": "ALL_UNITS_COMPLETE_VERIFIED",
            "assignment_sha256": manifest["assignment_sha256"],
            "unit_count": len(expected),
            "units": [observed[uid] for uid in sorted(observed)]}


def validate_result_archive(path: Path, completion: dict[str, Any]) -> dict[str, Any]:
    with zipfile.ZipFile(path) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos if not info.is_dir()]
        if len(names) != len(set(names)) or "scientific_receipt.json" not in names:
            raise ValueError("Result archive has duplicate members or no scientific receipt")
        if (len(infos) > 100000 or sum(info.file_size for info in infos) > MAX_ARCHIVE_BYTES or
                any(_unsafe_zip_name(name) for name in names) or
                any(((info.external_attr >> 16) & 0o170000) == 0o120000 for info in infos)):
            raise ValueError("Result archive has an unsafe member")
        receipt_info = next(info for info in infos if info.filename == "scientific_receipt.json")
        if receipt_info.file_size > MAX_RECEIPT_BYTES:
            raise ValueError("Scientific receipt exceeds size limit")
        science = json.loads(bundle.read("scientific_receipt.json"))
        if not isinstance(science, dict) or science.get("status") != "COMPLETED":
            raise ValueError("Scientific receipt is not COMPLETED")
        if science.get("unit_id") != completion.get("unit_id"):
            raise ValueError("Scientific receipt unit mismatch")
        for field in ("role", "assignment_sha256", "configuration_sha256", "expected_fits"):
            if science.get(field) != completion.get(field):
                raise ValueError(f"Scientific receipt {field} mismatch")
        artifacts = science.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            raise ValueError("Scientific receipt has no artifacts")
        declared: set[str] = set()
        by_name = {info.filename: info for info in infos if not info.is_dir()}
        for artifact in artifacts:
            name = artifact.get("path") if isinstance(artifact, dict) else None
            if not isinstance(name, str) or _unsafe_zip_name(name) or name in declared or name not in by_name:
                raise ValueError("Invalid, duplicate or missing result artifact")
            declared.add(name)
            payload = bundle.read(name)
            if len(payload) != artifact.get("bytes") or digest_bytes(payload) != artifact.get("sha256"):
                raise ValueError("Scientific result artifact integrity mismatch")
        if set(names) != declared | {"scientific_receipt.json"}:
            raise ValueError("Result archive contains undeclared artifacts")
        return science


def _unsafe_zip_name(name: str) -> bool:
    value = PurePosixPath(name)
    return value.is_absolute() or ".." in value.parts or "\\" in name or name in {"", "."}

