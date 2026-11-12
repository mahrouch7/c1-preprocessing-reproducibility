"""Targeted zero-fit storage receipt, executed only by external recovery launch."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def check_space(workspace, required_bytes, control_minimum_bytes, *, disk_usage=shutil.disk_usage):
    free = disk_usage(workspace).free
    if free < required_bytes:
        raise RuntimeError(f"PC2 H: storage insufficient: free={free}, required={required_bytes}; no scientific capsule sent")
    control = Path(os.environ.get("SystemDrive", "C:") + "\\")
    control_free = disk_usage(control).free
    if control_free < control_minimum_bytes:
        raise RuntimeError(f"PC2 control drive needs {control_minimum_bytes} free bytes for worker metadata; observed {control_free}")
    return free, control_free


def execute(args):
    configuration = Path(args.configuration)
    if hashlib.sha256(configuration.read_bytes()).hexdigest() != args.configuration_sha256:
        raise ValueError("Storage policy hash changed")
    config = json.loads(configuration.read_text("utf-8"))
    workspace = Path(config["workspace"]).resolve()
    if (workspace.drive.upper() != "H:" or not workspace.is_absolute() or len(workspace.parts) < 3 or
            not configuration.resolve().is_relative_to(workspace) or args.role != "PC2" or
            args.expected_fits != 0 or config["fits"] != 0 or config["simulations"] != 0 or
            platform.node().casefold() != "desktop-urb44pu"):
        raise ValueError("Wrong storage path, host, role or fit budget")
    free, control_free = check_space(workspace, config["required_free_bytes"], config["control_drive_minimum_bytes"])
    probe = Path(args.output) / "write_probe"
    with probe.open("xb") as stream:
        stream.write(b"C1-E27-storage-only\n")
        stream.flush()
        os.fsync(stream.fileno())
    probe.unlink()
    receipt = {"status": "PC2_H_STORAGE_CHECK_PASSED", "phase": config["phase"],
        "workspace": str(workspace), "host": platform.node(), "checked_utc": datetime.now(timezone.utc).isoformat(),
        "free_bytes": free, "required_free_bytes": config["required_free_bytes"], "control_drive_free_bytes": control_free,
        "allowance_is_provisional_not_proven_peak": True, "fits_completed": 0, "simulations": 0}
    path = Path(args.output) / "storage_receipt.json"
    path.write_bytes(canonical(receipt))
    science = {"status": "COMPLETED", "unit_id": args.unit_id, "role": args.role,
        "assignment_sha256": args.assignment_sha256, "configuration_sha256": args.configuration_sha256,
        "expected_fits": 0, "fits_completed": 0, "simulations": 0,
        "artifacts": [{"path": path.name, "bytes": path.stat().st_size,
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}]}
    (Path(args.output) / "scientific_receipt.json").write_bytes(canonical(science))
    print(f"[STOCKAGE PC2] H: libre={free / 1024**3:.1f} GiB; reserve demandee={config['required_free_bytes'] / 1024**3:.1f} GiB; fits=0", flush=True)


def main():
    parser = argparse.ArgumentParser()
    for name in ("unit-id", "role", "assignment-sha256", "configuration-sha256", "configuration", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--expected-fits", required=True, type=int)
    execute(parser.parse_args())


if __name__ == "__main__":
    main()
