"""Canonical-tree recovery bootstrap; no scientific execution on import.

Set both local script roots explicitly so an ordinary or isolated Python process
does not depend on the working directory, PYTHONPATH, or a test runner's sys.path.
"""
from pathlib import Path
import sys


if __name__ == '__main__':
    scripts = Path(__file__).resolve().parent
    repo = scripts.parents[3]
    upstream = repo / 'Contributions/C1_Preprocessing_and_dataset/R07/scripts'
    if not (upstream / 'c1e25.py').is_file():
        raise SystemExit('Canonical R07 script root missing; no installation or scientific execution')
    sys.path[:0] = [str(scripts), str(upstream)]
    from c1e32_recovery import main
    main()
