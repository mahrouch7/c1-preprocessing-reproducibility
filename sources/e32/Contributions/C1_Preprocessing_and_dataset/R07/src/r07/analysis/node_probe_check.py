"""Verify copied no-data probe receipts; never contacts or starts a server."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import re
from r07.core.artifacts import read_json,write_json
from r07.analysis.core import require

KIT="2df225d2474d1b0fca12c5120e96185580ae40799f5515ce2fc3ddbf03b741ff"

def verify(submitted, observed, expected_hostname):
    require(submitted.get("status")=="PROBE_SUBMITTED_NOT_COMPLETED","Not a submitted diagnostic")
    require(observed.get("status")=="WORKER_PROBE_PASS_NOT_DISTRIBUTED_SCIENCE_QUALIFIED",
            "Worker did not report diagnostic completion")
    require(submitted["kit_id"]==observed["kit_id"]==KIT,"Unexpected or mismatched kit")
    require(re.fullmatch(r"[0-9a-f]{32}",submitted["nonce"]) is not None and
            observed["nonce"]==submitted["nonce"],"Probe nonce mismatch")
    require(observed["worker"]==submitted["expected_worker_name"],"Wrong worker queue")
    require(observed["hostname"].casefold()==expected_hostname.casefold(),"Unexpected physical hostname")
    require(submitted["no_training"] is True,"Submission is not a no-training diagnostic")
    require(all(observed[k]==0 for k in ("datasets_read","model_fits","simulations")),
            "Probe exceeded no-science scope")
    def timestamp(value):
        parsed=datetime.fromisoformat(value.replace("Z","+00:00"))
        require(parsed.tzinfo is not None,"Receipt timestamp lacks timezone")
        return parsed.astimezone(timezone.utc)
    a,b=timestamp(submitted["submitted_utc"]),timestamp(observed["observed_utc"])
    require(b>=a-timedelta(seconds=5),"Probe clock disagreement exceeds five seconds")
    require(b<=datetime.now(timezone.utc)+timedelta(seconds=5),"Worker clock is in the future")
    require(re.fullmatch(r"[0-9a-f]{64}",observed["environment_sha256"]) is not None,"Invalid environment digest")
    return dict(status="NO_DATA_WORKER_EXECUTION_VERIFIED_SCIENTIFIC_EQUIVALENCE_PENDING",
                hostname=observed["hostname"],worker=observed["worker"],nonce=observed["nonce"],
                kit_id=KIT,python=observed["python"],
                python_patch_matches_baseline=observed["python_patch_matches_baseline"],
                scientific_equivalence_established=False,datasets_read=0,model_fits=0,simulations=0)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--submission",type=Path,required=True)
    parser.add_argument("--worker-receipt",type=Path,required=True)
    parser.add_argument("--expected-hostname",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    inputs=[args.submission.resolve(),args.worker_receipt.resolve()]
    target=args.output.resolve()
    require(not target.exists() and target not in inputs,"Use a new explicit verification output")
    result=verify(read_json(inputs[0]),read_json(inputs[1]),args.expected_hostname)
    # Label input roles: both original receipts may have the same nonce.json name.
    result["submission_sha256"]=hashlib.sha256(inputs[0].read_bytes()).hexdigest()
    result["worker_receipt_sha256"]=hashlib.sha256(inputs[1].read_bytes()).hexdigest()
    write_json(target,result)
    print(json.dumps(result),flush=True)

if __name__=="__main__":
    main()
