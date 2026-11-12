"""User-facing entrypoint. No MATLAB or generation command is reachable."""
import argparse
import json
from pathlib import Path
import sys
from r07.core.artifacts import exclusive_lock
from r07.core.config import package_root
from r07.core.runtime import configure_local_engine, require_environment

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("environment", "engine", "check", "pilot"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--confirm-pilot", action="store_true")
    args = parser.parse_args(argv)
    root = package_root()
    workspace = args.workspace.resolve()
    if workspace == workspace.parent or workspace in root.parents or root in workspace.parents or workspace == root:
        raise ValueError("Use a separate named analysis workspace, not code/root")
    if args.bundle is not None:
        bundle = args.bundle.resolve()
        if workspace == bundle or workspace in bundle.parents or bundle in workspace.parents:
            raise ValueError("Analysis output must be separate from the acquisition store")
    with exclusive_lock(workspace / "controller.lock"):
        inventory, fingerprint = require_environment(workspace)
        print("[ENVIRONMENT] " + inventory["status"], flush=True)
        if args.stage == "environment":
            return 0
        if args.stage == "engine":
            from r07.core.engine import check_local_engine
            check_local_engine(workspace, root)
            return 0
        if args.bundle is None:
            raise ValueError("Dataset bundle required")
        registry = args.dataset_config or root / "configs/datasets/e19_reference.json"
        config = args.config or root / "configs/experiments/clean_pilot.json"
        parameters = (args.bundle, registry, config, root, args.repo, workspace, fingerprint)
        if args.stage == "check":
            from r07.workflows.preparation import prepare
            prepared = prepare(*parameters)
            print("[PREFLIGHT PASS] 429 paired acquisitions; no fitting or simulation.", flush=True)
            print("saved_directory=" + str(prepared["output"]), flush=True)
            return 0
        if not args.confirm_pilot:
            raise ValueError("Pilot requires explicit confirmation: 30 bounded clean fold fits")
        from r07.core.engine import check_local_engine
        check_local_engine(workspace, root)
        configure_local_engine(workspace, root)
        from r07.workflows.clean import clean_pilot
        clean_pilot(*parameters)
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("[STOP] " + str(exc), file=sys.stderr, flush=True)
        raise
