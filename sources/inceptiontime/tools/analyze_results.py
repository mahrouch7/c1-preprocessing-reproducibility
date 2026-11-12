"""Explicit analysis entry point; no calculation on import."""
from pathlib import Path
import argparse
import sys
import traceback
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from c1inception.comparison import analyze


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',type=Path,default=ROOT/'configs/comparative_analysis.json')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    try:analyze(args.policy,args.output)
    except Exception:
        if args.output.is_dir() and not (args.output/'analysis_receipt.json').exists():
            (args.output/'analysis_failure.json').write_text(json.dumps({'status':'FAILED_DO_NOT_INTERPRET_PARTIAL_OUTPUTS',
                'traceback':traceback.format_exc()},indent=2)+'\n',encoding='utf-8')
        raise
