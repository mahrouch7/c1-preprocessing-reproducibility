"""Canonical-tree bootstrap. No execution on import."""
from pathlib import Path
import sys
if __name__=='__main__':
    repo=Path(__file__).resolve().parents[4]
    sys.path.insert(0,str(repo/'Contributions/C1_Preprocessing_and_dataset/R07/scripts'))
    from c1e32_campaign import main
    main()
