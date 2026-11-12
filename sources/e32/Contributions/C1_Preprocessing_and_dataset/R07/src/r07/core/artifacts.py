"""New artifact identity and publication boundary. No scheduler lives here."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import uuid

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def child(root, relative):
    # Both slash conventions are checked even when running on a different OS.
    text = str(relative).replace("\\", "/")
    parts = text.split("/")
    if ":" in text or text.startswith("/") or any(p in ("", ".", "..") for p in parts):
        raise ValueError("Unsafe relative artifact path")
    root = Path(root).resolve()
    target = root.joinpath(*PurePosixPath(text).parts).resolve()
    if root not in target.parents:
        raise ValueError("Artifact path escapes its root")
    return target

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def publish_bytes(path, payload):
    """Single-writer, same-directory atomic replace; partial attempts stay separate."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".partial")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)

def write_json(path, value):
    publish_bytes(path, (json.dumps(value, indent=2, sort_keys=True,
                                   allow_nan=False) + "\n").encode("utf-8"))

def write_csv(path, frame):
    publish_bytes(path, frame.to_csv(index=False, lineterminator="\n").encode("utf-8"))

@contextmanager
def exclusive_lock(path):
    """OS-held lock, automatically released on process death; file is not deleted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Another R07 controller holds this workspace") from exc
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
