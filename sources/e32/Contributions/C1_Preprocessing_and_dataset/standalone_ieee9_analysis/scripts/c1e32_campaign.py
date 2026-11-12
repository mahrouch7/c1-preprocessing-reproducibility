"""External-only launch: new cohort readback, shared preprocessing, fits, inference."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import getpass
import io
import json
import os
from pathlib import Path, PureWindowsPath
import platform
import shutil
import subprocess
import threading
import time
import zipfile
import numpy as np
import c1e25 as base
import c1e32_spec as spec
import c1e32_common as c
from c1e32_pdf import DEFAULT_PYTHON as GRAPHICS_PYTHON, DEFAULT_FONT
from c1e32_dataset import metadata, extract, characterize, leakage_audit

UPSTREAM_TASK=('c1e25.py','c1e26.py','c1e27_spec.py','c1e27_signal.py','c1e27_worker.py')
OWN_TASK=('c1e32_spec.py','c1e32_common.py','c1e32_probabilities.py','c1e32_worker.py','run_c1e32_unit.py')

def verify_release(package,repo):
    manifest=c.read(package/'provenance/release.json')
    own={p.relative_to(package).as_posix():c.sha(p) for p in package.rglob('*') if p.is_file() and p.suffix in ('.py','.ps1','.md')}
    c.require(own==manifest['files'],'Analysis source set/hash differs from release')
    for relative,digest in manifest['upstream'].items():
        c.require(c.sha(repo/relative)==digest,'Pinned upstream implementation changed: '+relative)
    c.require(manifest['scope_sha256']==spec.digest(spec.scope()),'Prospective scientific scope changed')
    return manifest

def config(stage,paths,**identity):
    value={'experiment_id':spec.EXPERIMENT,'stage':stage,'scope_sha256':spec.digest(spec.scope()),
           'inputs':c.pins(paths),**identity}
    value['expected_fits']=0 if stage=='A' else len(spec.jobs(value['preprocessor'],value['fold'],value['classifier']))
    return value

def task(package,repo,directory,configuration,paths,storage=False):
    directory.mkdir(parents=True,exist_ok=True); temporary=directory/'building.zip'
    scripts=repo/'Contributions/C1_Preprocessing_and_dataset/R07/scripts'
    files={'configuration.json':spec.canonical(configuration)}
    if storage:
        files['run_unit.py']=(scripts/'c1e27_storage_guard.py').read_bytes()
    else:
        files['run_unit.py']=(package/'scripts/run_c1e32_unit.py').read_bytes()
        files.update({v:(scripts/v).read_bytes() for v in UPSTREAM_TASK})
        files.update({v:(package/'scripts'/v).read_bytes() for v in OWN_TASK})
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as archive:
        for name,payload in sorted(files.items()):
            info=zipfile.ZipInfo(name,(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16
            archive.writestr(info,payload)
        for name,path in sorted(paths.items()):
            info=zipfile.ZipInfo(configuration['inputs'][name]['path'],(1980,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o100644<<16
            with Path(path).open('rb') as source,archive.open(info,'w',force_zip64=True) as target:
                shutil.copyfileobj(source,target,1024*1024)
    archive=directory/(c.sha(temporary)+'.zip'); temporary.rename(archive)
    return archive,c.sha(package/'scripts/run_c1e32_unit.py') if not storage else c.sha(scripts/'c1e27_storage_guard.py')

def unit(archive,runner_sha,configuration,role,commit):
    from r07.distributed.contracts import unit_identity_sha256
    identity={'experiment_id':spec.EXPERIMENT,'stage':configuration['stage'],'role':role,
        'expected_host':'MAHER' if role=='PC1' else 'DESKTOP-URB44PU','source_commit':commit,
        'inputs':[{'name':k,'sha256':v['sha256']} for k,v in sorted(configuration.get('inputs',{}).items())] or
                 [{'name':'storage_policy','sha256':spec.digest(configuration)}],
        'expected_fits':configuration.get('expected_fits',0),'task_archive_sha256':c.sha(archive),
        'configuration_sha256':spec.digest(configuration),'runner_sha256':runner_sha}
    identity.update({k:configuration[k] for k in ('chunk','preprocessor','fold','classifier') if k in configuration})
    return {'unit_id':unit_identity_sha256(identity),'identity':identity,
        **{k:identity[k] for k in ('role','expected_host','expected_fits','task_archive_sha256','configuration_sha256','runner_sha256')},
        'task_archive_bytes':archive.stat().st_size,'configuration_path':'configuration.json','runner_path':'run_unit.py'}

def freeze(units,preparation_sha,decision_id='C1-D87'):
    from r07.distributed.contracts import assignment_sha256,validate_assignment
    value={'schema_version':2,'status':'FROZEN','experiment_id':spec.EXPERIMENT,'execution_authorized':True,
        'authorization':{'authorized_by':getpass.getuser(),'authorized_utc':c.now(),'decision_id':decision_id,
            'scope':('One externally confirmed recovery; 18286 new fits, 97 acknowledged prior partial attempts; no native simulation or automatic retry'
                     if decision_id=='C1-D91' else 'One externally confirmed new-cohort campaign; no native simulation or automatic retry'),
            'preparation_sha256':preparation_sha},'scientific_limits':spec.LIMITS,
        'total_fit_budget':sum(u['expected_fits'] for u in units),'units':units}
    value['assignment_sha256']=assignment_sha256(value)
    return validate_assignment(value,require_authorized=True)

def workspace(value):
    path=PureWindowsPath(str(value))
    c.require(path.drive.upper()=='H:' and path.is_absolute() and '..' not in path.parts and len(path.parts)>=3,'Dedicated PC2 H: directory required')
    return str(path)

def dispatch(units,tasks,root,transport,preparation_sha,pc2_workspace,timeout=172800,poll_interval=15,decision_id='C1-D87'):
    from c1e27_recovery import stage_reusable_assignment
    from c1e27_campaign import cancel_pending
    from c1e26_campaign import remote_status,persist_progress
    from c1_progress import Progress,duration_text
    from r07.distributed.runtime import run_pc1_local
    from r07.distributed.submitter import submit_pc2_plan
    from run_c1e25_pipeline import scoped_barrier
    c.require(not any((transport/'markers'/(u['unit_id']+'.json')).exists() for u in units),
        'One or more immutable units already returned; reconcile evidence instead of resubmitting')
    root.mkdir(); assignment=freeze(units,preparation_sha,decision_id); path=root/'assignment.json'; c.write(path,assignment)
    plan=root/'dispatch_plan.json'; receipts=[]; stop=threading.Event(); lock=threading.Lock()
    state={'durations':[],'observations':[],'started':None}
    try:
        stage_reusable_assignment(assignment=path,task_directory=tasks,transport_tasks=transport/'tasks',output_plan=plan)
        receipts=submit_pc2_plan(plan_path=plan,assignment_path=path,work_pool='C1-R07-8658432b19cb',work_queue='PC2',
            node_home=workspace(pc2_workspace),ssh_exe=r'C:\Windows\System32\OpenSSH\ssh.exe',
            identity_file=r'C:\r07node\transport\c1_r07_dispatch_ed25519',known_hosts=r'C:\Users\Maher\.ssh\known_hosts',
            pc1_host='Maher369@172.16.1.178',receipt_directory=root/'submission_receipts',confirm=True)
        def local():
            for item in units:
                if item['role']!='PC1' or stop.is_set(): continue
                started=time.monotonic()
                with lock: state['started']=started
                run_pc1_local(manifest_path=path,task_archive=tasks/(item['task_archive_sha256']+'.zip'),unit_id=item['unit_id'],
                    attempt=root/'pc1_attempts'/item['unit_id'],results=transport/'results',markers=transport/'markers')
                with lock:
                    elapsed=time.monotonic()-started; state['durations'].append(elapsed)
                    state['observations'].append((item['unit_id'],elapsed)); state['started']=None
        pool=ThreadPoolExecutor(max_workers=1); future=pool.submit(local)
        progress=Progress(emit=lambda *a,**k:None); started=time.monotonic()
        try:
            while True:
                if future.done() and future.exception(): raise future.exception()
                states,remote_times=remote_status(receipts)
                totals={r:sum(u['role']==r for u in units) for r in ('PC1','PC2')}
                counts={r:sum((transport/'markers'/(u['unit_id']+'.json')).exists() for u in units if u['role']==r) for r in totals}
                with lock:
                    durations={'PC1':list(state['durations']),'PC2':remote_times}
                    active={'PC1':time.monotonic()-state['started'] if state['started'] else 0}
                    observations=list(state['observations'])
                pc_states={'PC1':'TERMINE' if counts['PC1']==totals['PC1'] else 'EN_COURS',
                           'PC2':'TERMINE' if counts['PC2']==totals['PC2'] else 'EN_COURS' if 'RUNNING' in states.values() else 'EN_ATTENTE'}
                phase={'STORAGE':'CONTROLE_STOCKAGE','NUMERIC':'CONTROLE_NUMERIQUE','A':'PRETRAITEMENT','B':'ENTRAINEMENT'}[units[0]['identity']['stage']]
                snapshot=progress.update(phase,counts,totals,durations,active,pc_states)
                completed={u['unit_id'] for u in units if (transport/'markers'/(u['unit_id']+'.json')).exists()}
                remote_complete=[r for r in receipts if states[r['unit_id']]=='COMPLETED']
                if len(remote_complete)==len(remote_times): observations += [(r['unit_id'],t) for r,t in zip(remote_complete,remote_times)]
                by_id={u['unit_id']:u for u in units}; bins={}
                def cost_key(u):
                    ident=u['identity']; return (u['role'],ident['stage'],ident.get('classifier'),ident.get('preprocessor')=='none')
                for uid,seconds in observations:
                    u=by_id[uid]; bins.setdefault(cost_key(u),[]).append(seconds/max(1,u['expected_fits']))
                estimates={'PC1':0.,'PC2':0.}; available=True
                for u in units:
                    if u['unit_id'] in completed: continue
                    values=bins.get(cost_key(u))
                    if not values: available=False; break
                    estimates[u['role']]+=float(np.median(values))*max(1,u['expected_fits'])
                eta=max(estimates.values()) if available else None
                snapshot['estimated_remaining_execution_seconds']=eta
                snapshot['estimate_assumption']='Provisional same-host/classifier/control-profile observed seconds per fit; final inference/rendering excluded'
                snapshot['completed_fits']={r:sum(u['expected_fits'] for u in units if u['role']==r and u['unit_id'] in completed) for r in totals}
                print(f"[AVANCEMENT] {phase} | PC1 {counts['PC1']}/{totals['PC1']} [{pc_states['PC1']}] | PC2 {counts['PC2']}/{totals['PC2']} [{pc_states['PC2']}] | ecoule {duration_text(snapshot['elapsed_seconds'])} | restant ~{duration_text(eta)} hors analyse finale | gain non mesure",flush=True)
                persist_progress(root,snapshot)
                if sum(counts.values())==len(units) and future.done() and all(v=='COMPLETED' for v in states.values()): break
                if time.monotonic()-started>=timeout: raise TimeoutError('Stage deadline exceeded; preserve results, no retry')
                stop.wait(poll_interval)
            ledger=scoped_barrier(path,transport,root)
            c.write(root/'timing_receipt.json',{'elapsed_seconds':time.monotonic()-started,'unit_seconds':durations,
                'unit_count':ledger['unit_count'],'speedup_measured':None,'serial_baseline':'UNAVAILABLE'})
            return assignment
        finally:
            stop.set(); pool.shutdown(wait=True)
    except BaseException as error:
        stop.set()
        saved=[c.read(p) for p in (root/'submission_receipts').glob('*.json')]
        try: cancel_pending(saved)
        except Exception as cancellation: print('[ATTENTION] Reconcile pending runs: '+str(cancellation),flush=True)
        persist_progress(root,{'phase':'ERREUR_ARRET','error':str(error),'automatic_retry':False,
            'execution_completed':False,'analysis_complete':False,'running_units_may_finish':True})
        raise

def storage(package,repo,output,transport,commit,pc2_workspace,phase):
    configuration={'stage':'STORAGE','workspace':workspace(pc2_workspace),'required_free_bytes':40*1024**3,
        'control_drive_minimum_bytes':3*1024**3,'phase':phase,'fits':0,'simulations':0}
    directory=output/('storage_'+phase+'_tasks')
    archive,runner=task(package,repo,directory,configuration,{},storage=True)
    definition=unit(archive,runner,configuration,'PC2',commit)
    return dispatch([definition],directory,output/('storage_'+phase),transport,spec.digest(spec.scope()),pc2_workspace)

def check_graphics(package,python,font):
    c.require(Path(python).is_file() and Path(font).is_file(),'Existing graphics runtime/font missing; no installation')
    completed=subprocess.run([str(python),'-B',str(package/'scripts/c1e32_pdf.py'),'--check','--font',str(font)],
        check=True,capture_output=True,text=True,timeout=60)
    result=json.loads(completed.stdout)
    return {**result,'python_executable_sha256':c.sha(python),'purpose':'NONSCIENTIFIC_VECTOR_EXPORT_ONLY'}

def prepare(package,repo,bundle,output,graphics=None):
    release=verify_release(package,repo); c.capacity(output.parent,80)
    c.require(not output.exists(),'Fresh analysis output required; never overwrite/retry an attempt')
    rows,view,fold_info=metadata(bundle)
    plan=c.read(bundle/'plan.json')
    for pin in plan['sources']:
        c.require(c.sha(repo/pin['relative_path'])==pin['sha256'],'Acquisition source identity changed')
    output.mkdir(parents=True); (output/'base').mkdir(); (output/'dataset_analysis').mkdir()
    c.write(output/'scope.json',spec.scope()); c.write(output/'metadata.json',rows)
    from c1e32_statistics import comparison_plan
    c.write(output/'comparison_plan.json',comparison_plan()); c.write(output/'fold_design.json',fold_info)
    c.write(output/'dataset_view.json',view)
    c.write(output/'leakage_audit.json',leakage_audit(rows))
    c.write(output/'checklist.json',c.read(repo/'research_record/planning/C1_scientific_results_generation_checklist_20261026.json'))
    if graphics is not None: c.write(output/'graphics_runtime.json',graphics)
    clean_path=extract(bundle,output/'base',rows); clean=np.load(clean_path,mmap_mode='r',allow_pickle=False)
    characterize(rows,clean,output/'dataset_analysis')
    wheel=repo/'Contributions/C1_Preprocessing_and_dataset/R07/deployment/scientific_dependencies/pywavelets-1.8.0-cp312-cp312-win_amd64.whl'
    c.require(c.sha(wheel)==spec.old.WHEEL_SHA256,'Approved task-local PyWavelets wheel changed')
    shutil.copyfile(wheel,output/'base/pywavelets.whl')
    commit=subprocess.check_output(['git','-c','safe.directory='+repo.as_posix(),'-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    chunks=np.array_split(np.arange(len(rows)),spec.scope()['preprocessing_chunks']); units=[]
    directory=output/'stage_a_tasks'; directory.mkdir()
    for i,ids in enumerate(chunks):
        chunk=output/'base'/f'chunk_{i:02}.npy'; np.save(chunk,np.asarray(clean[ids]),allow_pickle=False)
        paths={'metadata':output/'metadata.json','clean':chunk,'pywavelets_wheel':output/'base/pywavelets.whl'}
        cfg=config('A',paths,chunk=i,row_ids=ids.tolist())
        archive,runner=task(package,repo,directory,cfg,paths)
        units.append(unit(archive,runner,cfg,'PC1' if i%2==0 else 'PC2',commit))
        print(f'[TASK PREPARED] waveform chunk {i+1}/{len(chunks)}; fits=0',flush=True)
    receipt={'status':'PREPARED_NOT_EXECUTED','experiment_id':spec.EXPERIMENT,'source_commit':commit,
        'release_sha256':c.sha(package/'provenance/release.json'),'scope_sha256':spec.digest(spec.scope()),
        'dataset_view_sha256':spec.DATASET_SHA,'metadata_sha256':c.sha(output/'metadata.json'),
        'comparison_plan_sha256':c.sha(output/'comparison_plan.json'),'fit_budget':spec.scope()['fit_budget'],
        'base_files':{p.relative_to(output).as_posix():c.sha(p) for p in (output/'base').iterdir() if p.is_file()},
        'stage_a_units':units,'fits_executed':0,'scientific_limits':spec.LIMITS,'prepared_utc':c.now()}
    c.write(output/'preparation_receipt.json',receipt)
    return receipt

def collect_a(assignment,transport,output,rows):
    from run_c1e25_pipeline import returned_archive
    target=output/'verified_features'; target.mkdir(); diagnostic=output/'verified_diagnostics'; diagnostic.mkdir()
    values=np.empty((len(spec.CONDITIONS),len(spec.PREPS),len(rows),36)); seen=set()
    tpi=np.lib.format.open_memmap(diagnostic/'tpi.npy',mode='w+',dtype=np.float64,shape=(30,5,len(rows),6,3,9))
    errors=np.empty((30,5,len(rows),6)); noise=np.empty((30,len(rows),6,5))
    for item in assignment['units']:
        with zipfile.ZipFile(returned_archive(item,assignment,transport)) as archive:
            with np.load(io.BytesIO(archive.read('features.npz')),allow_pickle=False) as data:
                ids=data['row_ids']; features=data['features']
                c.require(not (seen & set(ids.tolist())) and features.shape==(30,5,len(ids),36) and np.isfinite(features).all(),'Chunk overlap/incomplete features')
                values[:,:,ids,:]=features; seen.update(ids.tolist())
            with np.load(io.BytesIO(archive.read('signal_diagnostics.npz')),allow_pickle=False) as data:
                c.require(np.array_equal(data['row_ids'],ids),'Diagnostic row identity differs')
                tpi[:,:,ids]=data['tpi']; errors[:,:,ids]=data['relative_l2']; noise[:,ids]=data['noise']
            with (diagnostic/('examples_'+str(item['identity']['chunk'])+'.npz')).open('xb') as stream:
                stream.write(archive.read('waveform_examples.npz'))
    c.require(seen==set(range(len(rows))),'Missing preprocessing rows')
    reference=np.load(output/'base/clean_base_features.npy',allow_pickle=False)
    c.require(np.allclose(values[0,0,:,:18],reference,atol=1e-8,rtol=1e-6),'Clean feature readback mismatch')
    for ci,entry in enumerate(spec.CONDITIONS):
        for pi,prep in enumerate(spec.PREPS): np.save(target/(entry['name']+'_'+prep+'.npy'),values[ci,pi],allow_pickle=False)
    tpi.flush(); del tpi
    np.save(diagnostic/'relative_l2.npy',errors,allow_pickle=False); np.save(diagnostic/'noise.npy',noise,allow_pickle=False)
    c.write(target/'feature_manifest.json',{p.name:c.sha(p) for p in sorted(target.glob('*.npy'))})

def verify_prepared(output,receipt):
    c.require(c.sha(output/'metadata.json')==receipt['metadata_sha256'] and c.sha(output/'comparison_plan.json')==receipt['comparison_plan_sha256'],'Prepared metadata/comparison plan changed')
    c.require(c.read(output/'scope.json')==__import__('json').loads(spec.canonical(spec.scope())),'Prepared scope changed')

def prepare_b(package,repo,output,commit):
    tasks=output/'stage_b_tasks'; tasks.mkdir(); units=[]
    for i,definition in enumerate(spec.learning_plan()):
        prep=definition['preprocessor']; paths={'metadata':output/'metadata.json'}
        paths.update({v['name']:output/'verified_features'/(v['name']+'_'+prep+'.npy') for v in spec.CONDITIONS})
        cfg=config('B',paths,**{k:definition[k] for k in ('preprocessor','fold','classifier')})
        archive,runner=task(package,repo,tasks,cfg,paths)
        units.append(unit(archive,runner,cfg,definition['role'],commit))
        if i%10==0: print(f'[LEARNING TASK PREPARED] {i+1}/{len(spec.learning_plan())}',flush=True)
    c.require(sum(u['expected_fits'] for u in units)==spec.scope()['fit_budget'],'Global fit ceiling drift')
    return units,tasks

def launch(package,repo,bundle,output,transport,pc2_workspace,graphics_python,graphics_font,graphics):
    c.require(platform.node().casefold()=='maher','External launch must run on qualified PC1')
    for name in list(os.environ):
        if name.startswith('PREFECT_'): os.environ.pop(name)
    # Create the client home before Prefect imports initialize settings.
    client=output.parent/(output.name+'_client'); client.mkdir(parents=True,exist_ok=True)
    os.environ.update(PREFECT_API_URL='http://127.0.0.1:4200/api',PREFECT_HOME=str(client),
        PREFECT_SERVER_EPHEMERAL_ENABLED='false',PREFECT_SERVER_ANALYTICS_ENABLED='false',
        DO_NOT_TRACK='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    from run_c1e25_pipeline import require_worker_online
    require_worker_online()
    receipt=prepare(package,repo,bundle,output,graphics); started=time.monotonic()
    preparation_sha=c.sha(output/'preparation_receipt.json'); commit=receipt['source_commit']
    verify_prepared(output,receipt)
    storage(package,repo,output,transport,commit,pc2_workspace,'before_A')
    a=dispatch(receipt['stage_a_units'],output/'stage_a_tasks',output/'stage_a',transport,preparation_sha,pc2_workspace)
    verify_release(package,repo); verify_prepared(output,receipt)
    rows=spec.validate_rows(c.read(output/'metadata.json')); collect_a(a,transport,output,rows)
    storage(package,repo,output,transport,commit,pc2_workspace,'before_B'); c.capacity(output,35)
    units,tasks=prepare_b(package,repo,output,commit)
    b=dispatch(units,tasks,output/'stage_b',transport,preparation_sha,pc2_workspace)
    verify_release(package,repo); verify_prepared(output,receipt)
    print('[PHASE] Both-host barrier passed; aggregate and perform family-aware inference',flush=True)
    from c1e32_analysis import aggregate
    aggregate(b,transport,output,rows,graphics_python,graphics_font)
    c.write(output/'completion_receipt.json',{'status':'SCIENTIFIC_CAMPAIGN_COMPLETE_NOT_PAPER_CLOSURE',
        'fit_budget':spec.scope()['fit_budget'],'fits_reused':0,'simulations':0,'elapsed_execution_seconds':time.monotonic()-started,
        'scientific_limits':spec.LIMITS,'speedup_measured':None})
    print('[C1-E32 COMPLETE] New-cohort science and figures saved; manuscript/reviewer work remains',flush=True)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=['check','prepare','all'],default='check')
    parser.add_argument('--dataset',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--transport',type=Path,default=Path(r'F:\Backup\R07_Dispatch'))
    parser.add_argument('--pc2-workspace',default=r'H:\r07node_e32\workspace'); parser.add_argument('--confirm-fits',type=int)
    parser.add_argument('--graphics-python',default=GRAPHICS_PYTHON); parser.add_argument('--graphics-font',default=DEFAULT_FONT)
    args=parser.parse_args(); package=Path(__file__).resolve().parents[1]; repo=package.parents[2]
    verify_release(package,repo)
    c.require(args.dataset.is_absolute() and args.output.is_absolute(),'Absolute dataset/output required')
    dataset=args.dataset.resolve(); output=args.output.resolve()
    c.require(output.parent!=output and not output.is_relative_to(repo) and not output.is_relative_to(dataset) and not dataset.is_relative_to(output),'Unsafe/overlapping output')
    c.require(c.sha(dataset/'views/paper_reduced/view_manifest.json')==spec.DATASET_SHA,'Wrong dataset identity')
    graphics=check_graphics(package,args.graphics_python,args.graphics_font)
    if args.stage=='check':
        rows,_,folds=metadata(dataset); print('[CHECK ONLY; NO FIT/SIMULATION/SUBMISSION] '+str(spec.scope()['fit_budget'])+' fits planned; '+str(folds)); return
    if args.stage=='prepare':
        c.require(args.confirm_fits is None,'Prepare-only cannot authorize fits'); prepare(package,repo,dataset,output,graphics); return
    c.require(args.confirm_fits==spec.scope()['fit_budget'],'Exact initial external fit-budget confirmation required')
    launch(package,repo,dataset,output,args.transport,workspace(args.pc2_workspace),args.graphics_python,args.graphics_font,graphics)

if __name__=='__main__': main()
