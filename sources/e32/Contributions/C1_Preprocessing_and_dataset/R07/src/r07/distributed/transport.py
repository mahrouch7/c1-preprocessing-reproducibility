"""Non-interactive client for the restricted PC1 artifact gate."""
from __future__ import annotations

from pathlib import Path
import subprocess

from .contracts import digest_file


def base_ssh(ssh: Path, identity: Path, known_hosts: Path, host: str) -> list[str]:
    return [str(ssh), "-T", "-a", "-x", "-o", "BatchMode=yes", "-o",
            "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=yes", "-o",
            f"UserKnownHostsFile={known_hosts}", "-i", str(identity), host]


def fetch_task(command: list[str], digest: str, destination: Path, suffix: str = ".zip") -> None:
    if suffix not in {".zip", ".json"}:
        raise ValueError("Unsupported task artifact suffix")
    destination = Path(destination)
    if destination.exists():
        raise ValueError("Destination exists; overwrite refused")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise ValueError("Partial destination exists; review required")
    try:
        with temporary.open("xb") as output:
            completed = subprocess.run(command + [f"get-task {digest}{suffix} {digest}"],
                                       stdout=output, stderr=subprocess.PIPE, check=False)
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
        if digest_file(temporary) != digest:
            raise ValueError("Downloaded task hash mismatch")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def push(command: list[str], operation: str, source: Path, remote_name: str) -> str:
    if operation not in {"put-result", "put-marker"}:
        raise ValueError("Unsupported transport upload operation")
    source = Path(source)
    if not source.is_file() or source.is_symlink():
        raise ValueError("Upload source is missing or unsafe")
    digest = digest_file(source)
    size = source.stat().st_size
    with source.open("rb") as input_stream:
        completed = subprocess.run(command + [f"{operation} {remote_name} {digest} {size}"],
                                   stdin=input_stream, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.decode("utf-8", "replace"))
    return digest

