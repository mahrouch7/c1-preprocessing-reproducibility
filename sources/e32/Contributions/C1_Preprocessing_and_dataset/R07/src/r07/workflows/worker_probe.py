"""Explicit diagnostic deployment only. Never load data or fit models."""
from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import platform
import re
import socket
from prefect import flow
from r07.core.artifacts import write_json, utc_now

@flow(name="C1-R07-worker-probe",retries=0,persist_result=False,log_prints=True)
def worker_probe(nonce: str, expected_kit_id: str, expected_worker_name: str):
    if not re.fullmatch(r"[0-9a-f]{32}",nonce):
        raise ValueError("Invalid diagnostic nonce")
    root = Path(os.environ["R07_NODE_KIT_ROOT"])
    spec = spec_from_file_location("r07_node_common_probe",root/"node_common.py")
    common = module_from_spec(spec)
    spec.loader.exec_module(common)
    manifest = common.verify_kit(root)
    if manifest["kit_id"]!=expected_kit_id:
        raise ValueError("Probe reached a different code kit")
    if os.environ.get("R07_NODE_WORKER_NAME")!=expected_worker_name:
        raise ValueError("Probe reached an unexpected worker queue")
    home = Path(os.environ["R07_NODE_HOME"])
    ready = common.audit(root,home)
    receipt = dict(status="WORKER_PROBE_PASS_NOT_DISTRIBUTED_SCIENCE_QUALIFIED",
                   observed_utc=utc_now(),nonce=nonce,hostname=socket.gethostname(),
                   worker=expected_worker_name,kit_id=expected_kit_id,
                   python=platform.python_version(),
                   python_patch_matches_baseline=ready["python_patch_matches_baseline"],
                   environment_sha256=ready["environment_sha256"],
                   datasets_read=0,model_fits=0,simulations=0)
    target = home/"probe_results"/(nonce+".json")
    write_json(target,receipt)
    print(receipt,flush=True)
    print("probe_receipt="+str(target),flush=True)
    return receipt
