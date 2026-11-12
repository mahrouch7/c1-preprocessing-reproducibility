"""Small explicit provenance and atomic-output helpers."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import uuid


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical(value)
    if path.exists() and not replace:
        require(path.read_bytes() == data, f"Conflicting immutable JSON: {path}")
        return
    partial = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with partial.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    if replace:
        os.replace(partial, path)
    else:
        try:
            os.link(partial, path)
        finally:
            partial.unlink()


def write_csv(path, rows):
    rows = list(rows)
    require(bool(rows), "Refusing empty scientific CSV")
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def exclusive_lock(path):
    """No automatic removal of another process's lock, even after a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        handle = path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RuntimeError(f"Lock exists: {path}. Check that its process stopped; use unlock explicitly.") from exc
    with handle:
        handle.write(json.dumps({"pid": os.getpid(), "started_utc": now()}))
        handle.flush()
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def git_state(root):
    def call(*args):
        result = subprocess.run(["git", "-c", f"safe.directory={Path(root).as_posix()}", "-C", str(root), *args],
                                check=True, text=True, capture_output=True)
        return result.stdout.strip()
    return {"commit": call("rev-parse", "HEAD"), "dirty": bool(call("status", "--porcelain"))}


def source_tree(root):
    """Hash executable/config inputs as well as documentation, not ignored results."""
    root = Path(root)
    paths = []
    for directory in ("src", "configs", "tests", "provenance", "docs", "research_record"):
        paths.extend(p for p in (root / directory).rglob("*") if p.is_file()
                     and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo")
                     and not any(part.endswith(".egg-info") for part in p.parts))
    paths.extend(p for p in root.iterdir() if p.is_file() and p.suffix in (".toml", ".txt", ".ps1", ".md"))
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(paths)}
