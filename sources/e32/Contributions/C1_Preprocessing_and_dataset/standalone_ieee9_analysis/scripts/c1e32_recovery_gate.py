"""One zero-fit PC2 gate: compare stable posteriors on the retained failed model."""
from __future__ import annotations
import argparse
import importlib.metadata
from pathlib import Path
import platform
import numpy as np
import c1e32_common as c
import c1e32_probabilities as posterior
import c1e32_prediction_diagnostic as diagnostic


def compare(root, expected_model_sha):
    original = diagnostic.diagnose(root)
    c.require(original['status'] == 'ORIGINAL_CONTRACT_REJECTION_REPRODUCED' and
              original['first_rejected_condition_in_original_order'] == 'awgn__20',
              'Retained failure differs; do not launch fits')
    saved = original['artifacts']['retained_model']
    c.require(saved['sha256'] == expected_model_sha, 'Retained model changed since researcher diagnostic')
    model_path = diagnostic.confined_file(root, 'execution/output/model_' + diagnostic.JOB + '.pickle.gz')
    model = diagnostic.load_local_model(model_path)
    task = root / 'execution/task'
    config = c.read(task / 'configuration.json')
    rows = c.read(diagnostic.confined_file(task, config['inputs']['metadata']['path']))
    test = np.array([r['fold'] == 1 for r in rows]); ids = np.arange(len(rows))[test]
    comparisons = []
    for condition in diagnostic.CONDITIONS:
        path = diagnostic.confined_file(task, config['inputs'][condition]['path'])
        x = np.load(path, allow_pickle=False)[test, 6:18]
        native, stable = posterior.checked_prediction(model, x)
        raw = model.predict_proba(x)[:, [list(model.classes_).index(v) for v in diagnostic.LABELS]]
        checked = diagnostic.contract(native, stable, diagnostic.LABELS, ids)
        c.require(not checked['violations'], 'Stable posterior failed original strict contract')
        disagreements = int(np.count_nonzero(stable.argmax(axis=1) != raw.argmax(axis=1)))
        c.require(disagreements == 0, 'Main classification decisions changed; review before fits')
        comparisons.append({'condition': condition, 'contract': checked,
            'maximum_absolute_probability_change': float(np.max(abs(raw - stable))),
            'main_argmax_disagreements': disagreements})
    c.require(c.sha(model_path) == expected_model_sha, 'Retained model changed during gate')
    return {'status': 'STABLE_POSTERIOR_VERIFIED_ON_RETAINED_MODEL', 'fits': 0, 'simulations': 0,
        'policy': posterior.POLICY, 'original_diagnostic': original, 'comparisons': comparisons,
        'scope_limit': 'Numerical validation only; does not prove robustness, calibration or physical validity'}


def main():
    parser = argparse.ArgumentParser()
    for key in ('unit-id', 'role', 'assignment-sha256', 'configuration-sha256', 'configuration', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--expected-fits', required=True, type=int)
    args = parser.parse_args(); output = Path(args.output); configuration = Path(args.configuration)
    c.require(args.role == 'PC2' and args.expected_fits == 0 and c.sha(configuration) == args.configuration_sha256,
              'Gate identity/configuration differs')
    versions = {k: importlib.metadata.version(k) for k in diagnostic.VERSIONS}
    c.require(versions == diagnostic.VERSIONS, 'Gate versions differ; no automatic installation')
    config = c.read(configuration); c.inputs(config, configuration.parent)
    root = Path(config['retained_unit_root']).resolve(strict=True)
    c.require(not output.resolve().is_relative_to(root), 'Gate must not write into retained failed unit')
    result = compare(root, config['retained_model_sha256'])
    c.write(output / 'posterior_gate.json', result)
    c.write(output / 'runtime.json', {'host': platform.node(), 'python': platform.python_version(),
        'versions': versions, 'fits_completed': 0, 'simulations': 0, 'stage': 'NUMERIC'})
    artifacts = [{'path': p.name, 'sha256': c.sha(p), 'bytes': p.stat().st_size}
                 for p in sorted(output.iterdir()) if p.is_file()]
    c.write(output / 'scientific_receipt.json', {'status': 'COMPLETED', 'unit_id': args.unit_id,
        'role': args.role, 'assignment_sha256': args.assignment_sha256,
        'configuration_sha256': args.configuration_sha256, 'expected_fits': 0,
        'fits_completed': 0, 'simulations': 0, 'artifacts': artifacts})
    print('[NUMERICAL GATE PASS; FITS=0; RETAINED MODEL UNCHANGED]', flush=True)


if __name__ == '__main__': main()
