"""Externally authorized E32 recovery: preserve full returns, replace only missing units."""
from __future__ import annotations
import argparse
import io
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import zipfile
import numpy as np
import c1e32_campaign as campaign
import c1e32_common as c
import c1e32_spec as spec
from c1e32_pdf import DEFAULT_PYTHON, DEFAULT_FONT

OLD_ASSIGNMENT_FILE = 'c5ab7307d4b58521c29ed5fea56064102f282ec4fb675cb5add73b20fe15c949'
MODEL_SHA = '4426a4f192aea45b49bdb63ec4380fc1871808ca371891a3c757c61156258cba'
FAILED = 'a709bff00cb7e27c14e41815072dfc4d5be09092e629a62c278f4737fe43621e'
KEPT = {
    'ea879ac7e61300d33e4d340875042f9acd22d0e89ee31b71cef195f5c33cae61',
    '53b8e76c2c71869856f696e6f8c40f3240cc995581a2ae28c94b60366f85fe84',
    '659683a2bfb0d6fcaced4850e46f521c1a10d792f9d60c7cbd3f0611596274d6',
    'b0479706251115f2f6f4f6d2d5f9c9c3a6daa8790acff7cd3fcb39b0f6a0cda9',
}
NEW_FITS = 18286
PRIOR_PARTIAL_FITS = 97


def logical(unit):
    v = unit['identity']
    return v['preprocessor'], v['fold'], v['classifier']


def reconcile(original, transport):
    from r07.distributed.contracts import validate_assignment
    from run_c1e25_pipeline import returned_archive
    path = original / 'stage_b/assignment.json'
    c.require(c.sha(path) == OLD_ASSIGNMENT_FILE, 'Original assignment changed')
    assignment = validate_assignment(c.read(path), require_authorized=True)
    present = {u['unit_id'] for u in assignment['units'] if (transport/'markers'/(u['unit_id']+'.json')).exists()}
    c.require(present == KEPT, 'Original returns changed; reconcile rather than repeat more fits')
    for unit in assignment['units']:
        if unit['unit_id'] in KEPT:
            returned_archive(unit, assignment, transport)
            c.require(unit['identity']['classifier'] != 'naive_bayes', 'Unexpected old GaussianNB return')
    c.require(sum(u['expected_fits'] for u in assignment['units'] if u['unit_id'] in KEPT) == 644,
              'Verified reused-fit count differs')
    keys = {logical(u) for u in assignment['units'] if u['unit_id'] in KEPT}
    missing = [v for v in spec.learning_plan() if (v['preprocessor'],v['fold'],v['classifier']) not in keys]
    c.require(len(missing) == 201 and sum(v['expected_fits'] for v in missing) == NEW_FITS,
              'Recovery fit/unit budget differs')
    return assignment, missing


def require_old_remote_terminal(original):
    from prefect.client.orchestration import get_client
    receipts = [c.read(p) for p in (original/'stage_b/submission_receipts').glob('*.json')]
    c.require(len(receipts) == 102, 'Original remote submission receipts incomplete')
    with get_client(sync_client=True) as client:
        for receipt in receipts:
            run = client.read_flow_run(receipt['flow_run_id'])
            state = run.state.type.value if run.state is not None else 'UNKNOWN'
            c.require(state in {'COMPLETED','FAILED','CRASHED','CANCELLED'},
                      'An original PC2 run is still nonterminal; no recovery submission')


def verify_features(original, assignment):
    metadata_hashes = {entry['sha256'] for unit in assignment['units']
                       for entry in unit['identity']['inputs'] if entry['name']=='metadata'}
    c.require(metadata_hashes=={c.sha(original/'metadata.json')}, 'Frozen row/fold metadata changed')
    expected = {}
    for unit in assignment['units']:
        prep = unit['identity']['preprocessor']
        for entry in unit['identity']['inputs']:
            if entry['name'] == 'metadata': continue
            name = entry['name'] + '_' + prep + '.npy'
            c.require(name not in expected or expected[name] == entry['sha256'], 'Frozen feature hashes conflict')
            expected[name] = entry['sha256']
    manifest = c.read(original/'verified_features/feature_manifest.json')
    c.require(manifest == expected and len(expected) == 150, 'Assembled feature grid differs')
    for name, digest in expected.items():
        c.require(c.sha(original/'verified_features'/name) == digest, 'Assembled feature bytes changed: ' + name)
    return expected


def verify_diagnostics(original, transport, rows):
    """Read-only comparison with all ten original verified chunk returns, not recomputation."""
    from r07.distributed.contracts import validate_assignment
    from run_c1e25_pipeline import returned_archive
    assignment = validate_assignment(c.read(original/'stage_a/assignment.json'), require_authorized=True)
    ledger = c.read(original/'stage_a/completion_ledger.json')
    c.require(assignment['units']==c.read(original/'preparation_receipt.json')['stage_a_units'],
              'Original preprocessing units differ from pinned preparation')
    c.require(ledger['status'] == 'ALL_UNITS_COMPLETE_VERIFIED' and ledger['unit_count'] == 10 and
              ledger['assignment_sha256'] == assignment['assignment_sha256'], 'Original preprocessing barrier differs')
    root = original/'verified_diagnostics'
    tensors = {k: np.load(root/(k+'.npy'),mmap_mode='r',allow_pickle=False)
               for k in ('tpi','relative_l2','noise')}
    seen = set()
    for unit in assignment['units']:
        with zipfile.ZipFile(returned_archive(unit,assignment,transport)) as archive:
            with np.load(io.BytesIO(archive.read('signal_diagnostics.npz')),allow_pickle=False) as data:
                ids = data['row_ids']
                c.require(not seen & set(ids.tolist()), 'Overlapping diagnostic chunks')
                seen.update(ids.tolist())
                for name in tensors:
                    chunk = data[name] if name != 'relative_l2' else data['relative_l2']
                    actual = tensors[name][:,ids] if name == 'noise' else tensors[name][:,:,ids]
                    c.require(np.array_equal(actual,chunk,equal_nan=True), 'Original assembled diagnostics changed: '+name)
            name = 'examples_' + str(unit['identity']['chunk']) + '.npz'
            c.require((root/name).read_bytes() == archive.read('waveform_examples.npz'), 'Waveform diagnostic example changed')
    c.require(seen == set(range(len(rows))), 'Original diagnostic coverage differs')


def copy_analysis_inputs(original, output):
    files = ('metadata.json','comparison_plan.json','scope.json','dataset_view.json','fold_design.json',
             'leakage_audit.json','checklist.json','graphics_runtime.json')
    for name in files:
        shutil.copyfile(original/name, output/name)
        c.require(c.sha(original/name) == c.sha(output/name), 'Copied analysis identity changed')
    for name in ('verified_features','verified_diagnostics','dataset_analysis'):
        shutil.copytree(original/name, output/name)
        for source in (original/name).rglob('*'):
            if source.is_file():
                c.require(not source.is_symlink() and c.sha(source) == c.sha(output/source.relative_to(original)),
                          'Copied analysis input changed')


def prepare_gate(package,repo,output,commit,workspace):
    directory = output/'numeric_gate_tasks'; directory.mkdir()
    retained = str(Path(workspace)/'distributed_units_v2'/FAILED)
    config = {'stage':'NUMERIC','expected_fits':0,'retained_unit_root':retained,'retained_model_sha256':MODEL_SHA,
              'inputs':{'identity':{'path':'data/identity.json','sha256':spec.digest({'failed_unit':FAILED,'model_sha256':MODEL_SHA})}}}
    scripts = repo/'Contributions/C1_Preprocessing_and_dataset/R07/scripts'
    diagnostic = repo/'research_record/tools/c1e32_prediction_diagnostic.py'
    c.require(c.sha(diagnostic) == '007c5ac9a32d8c62ee0e27b76ea13ef0940ec63f35549071047ed283aba7e7ab',
              'Original non-fitting diagnostic changed')
    files = {'configuration.json':spec.canonical(config),'data/identity.json':spec.canonical({'failed_unit':FAILED,'model_sha256':MODEL_SHA}),
        'run_unit.py':(package/'scripts/c1e32_recovery_gate.py').read_bytes(),
        'c1e32_prediction_diagnostic.py':diagnostic.read_bytes()}
    files.update({v:(scripts/v).read_bytes() for v in campaign.UPSTREAM_TASK})
    files.update({v:(package/'scripts'/v).read_bytes() for v in campaign.OWN_TASK})
    temporary = directory/'building.zip'
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name,payload in sorted(files.items()):
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16
            archive.writestr(info,payload)
    archive = directory/(c.sha(temporary)+'.zip'); temporary.rename(archive)
    unit = campaign.unit(archive,c.sha(package/'scripts/c1e32_recovery_gate.py'),config,'PC2',commit)
    return unit,directory


def prepare_missing(package,repo,output,definitions,commit):
    tasks = output/'stage_b_tasks'; tasks.mkdir(); units=[]
    for index,definition in enumerate(definitions,1):
        prep=definition['preprocessor']; paths={'metadata':output/'metadata.json'}
        paths.update({v['name']:output/'verified_features'/(v['name']+'_'+prep+'.npy') for v in spec.CONDITIONS})
        cfg=campaign.config('B',paths,**{k:definition[k] for k in ('preprocessor','fold','classifier')})
        archive,runner=campaign.task(package,repo,tasks,cfg,paths)
        units.append(campaign.unit(archive,runner,cfg,definition['role'],commit))
        if index%10==0: print(f'[RECOVERY PACKAGING] {index}/{len(definitions)}; fits=0',flush=True)
    c.require(sum(u['expected_fits'] for u in units)==NEW_FITS,'New fit budget differs')
    return units,tasks


def resume(package,repo,original,output,transport,pc2_workspace,graphics_python,graphics_font):
    c.require(platform.node().casefold()=='maher','Run recovery externally on qualified PC1')
    c.require(not output.exists() and output.parent==original.parent and output!=original,
              'Fresh sibling recovery directory required; never reuse an attempt')
    campaign.verify_release(package,repo); c.capacity(output.parent,35)
    for name in list(os.environ):
        if name.startswith('PREFECT_'): os.environ.pop(name)
    client=output.parent/(output.name+'_client'); client.mkdir(exist_ok=True)
    os.environ.update(PREFECT_API_URL='http://127.0.0.1:4200/api',PREFECT_HOME=str(client),
        PREFECT_SERVER_EPHEMERAL_ENABLED='false',PREFECT_SERVER_ANALYTICS_ENABLED='false',
        DO_NOT_TRACK='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    from run_c1e25_pipeline import require_worker_online,returned_archive
    print('[REPRISE] Verification en lecture seule des retours, des features et des diagnostics existants; fits=0',flush=True)
    require_worker_online(); require_old_remote_terminal(original)
    assignment,missing=reconcile(original,transport)
    c.require(c.sha(original/'preparation_receipt.json')==assignment['authorization']['preparation_sha256'],
              'Original preparation receipt changed')
    receipt=c.read(original/'preparation_receipt.json'); campaign.verify_prepared(original,receipt)
    rows=spec.validate_rows(c.read(original/'metadata.json'))
    verify_features(original,assignment); verify_diagnostics(original,transport,rows)
    graphics=campaign.check_graphics(package,graphics_python,graphics_font)
    commit=subprocess.check_output(['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    print('[REPRISE] Copie verifiee des entrees deja calculees; aucun nouveau pretraitement',flush=True)
    output.mkdir(); copy_analysis_inputs(original,output)
    c.write(output/'recovery_plan.json',{'status':'EXTERNALLY_AUTHORIZED_RECOVERY','decision_id':'C1-D91','source_commit':commit,
        'release_sha256':c.sha(package/'provenance/release.json'),'original_assignment_file_sha256':OLD_ASSIGNMENT_FILE,
        'original_assignment_sha256':assignment['assignment_sha256'],'retained_units':sorted(KEPT),
        'same_campaign_verified_fits_reused':644,'new_fits_authorized':NEW_FITS,'prior_partial_fits_acknowledged':PRIOR_PARTIAL_FITS,
        'unique_final_fit_budget':18930,'maximum_cumulative_attempted_fits':19027,'missing_units':missing,
        'partial_policy':'Retain 97 unreceipted NB fits; do not reuse them as verified scientific records; repeat this unit once explicitly',
        'automatic_retry':False,'new_native_simulations':0,'graphics':graphics,'scientific_limits':spec.LIMITS})
    plan_sha=c.sha(output/'recovery_plan.json'); started=time.monotonic()
    print('[REPRISE] 644 fits verifies conserves; 18286 fits restants; 97 essais partiels archives et explicitement repetes',flush=True)
    gate,tasks=prepare_gate(package,repo,output,commit,pc2_workspace)
    gate_assignment=campaign.dispatch([gate],tasks,output/'numeric_gate',transport,plan_sha,pc2_workspace,decision_id='C1-D91')
    with zipfile.ZipFile(returned_archive(gate,gate_assignment,transport)) as archive:
        report=__import__('json').loads(archive.read('posterior_gate.json'))
        c.require(report['status']=='STABLE_POSTERIOR_VERIFIED_ON_RETAINED_MODEL' and report['fits']==0,
                  'Specific numerical gate not passed; no fits')
        c.write(output/'posterior_gate.json',report)
    campaign.verify_release(package,repo)
    campaign.storage(package,repo,output,transport,commit,pc2_workspace,'before_recovery_B')
    units,tasks=prepare_missing(package,repo,output,missing,commit)
    new_assignment=campaign.dispatch(units,tasks,output/'stage_b',transport,plan_sha,pc2_workspace,decision_id='C1-D91')
    campaign.verify_release(package,repo); campaign.verify_prepared(original,receipt)
    c.require(c.sha(original/'stage_b/assignment.json')==OLD_ASSIGNMENT_FILE,'Original assignment changed during recovery')
    from c1e32_analysis import aggregate
    aggregate([(assignment,KEPT),(new_assignment,None)],transport,output,rows,graphics_python,graphics_font)
    c.write(output/'completion_receipt.json',{'status':'SCIENTIFIC_CAMPAIGN_COMPLETE_NOT_PAPER_CLOSURE',
        'unique_completed_fits':18930,'same_campaign_fits_reused':644,'new_fits':NEW_FITS,
        'prior_partial_attempts':97,'cumulative_attempted_fits':19027,'simulations':0,
        'elapsed_recovery_seconds':time.monotonic()-started,'speedup_measured':None,'scientific_limits':spec.LIMITS})
    print('[C1-E32 RECOVERY COMPLETE] Figures PDF et analyse terminees; ce n est pas la fin du papier',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original',type=Path,default=Path(r'F:\Backup\C1_E32\campaign_v1'))
    parser.add_argument('--output',type=Path,default=Path(r'F:\Backup\C1_E32\recovery_numeric_v1'))
    parser.add_argument('--transport',type=Path,default=Path(r'F:\Backup\R07_Dispatch'))
    parser.add_argument('--pc2-workspace',default=r'H:\r07node_e32\workspace')
    parser.add_argument('--graphics-python',default=DEFAULT_PYTHON); parser.add_argument('--graphics-font',default=DEFAULT_FONT)
    parser.add_argument('--confirm-new-fits',type=int,required=True)
    parser.add_argument('--acknowledge-prior-partial-fits',type=int,required=True)
    args=parser.parse_args()
    c.require(args.confirm_new_fits==NEW_FITS and args.acknowledge_prior_partial_fits==PRIOR_PARTIAL_FITS,
              'Explicit 18286 new fits and 97 prior unreceipted attempts must be acknowledged')
    package=Path(__file__).resolve().parents[1]; repo=package.parents[2]
    resume(package,repo,args.original.resolve(),args.output.resolve(),args.transport.resolve(),
           campaign.workspace(args.pc2_workspace),args.graphics_python,args.graphics_font)


if __name__=='__main__': main()
