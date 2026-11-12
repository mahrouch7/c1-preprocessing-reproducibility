"""Verify/extract the pinned archives into a new code copy, then reproduce tables."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import re
import shutil
import stat
import subprocess
import sys
import uuid
import zipfile

BUFFER=1024*1024
PINS={
    'C1_predictions_metadata.zip':{'sha256':'b40be8567c85924c940d9de034fc2aeb21cfd1ae39323dad04819b0ba6c7e7f0','group':'predictions_metadata','members':13050,'source_bytes':1325817082},
    'C1_clean_windows_features.zip':{'sha256':'07d82a9a0991717943e7ee5494e0c36d0cf942fc69f624f801480eccc80f3c7d','group':'clean_windows_features','members':152,'source_bytes':785654845}}


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(BUFFER),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
def inside(path,root):return path==root or root in path.parents


def safe(root,name):
    value=str(name).replace('\\','/')
    if ':' in value or value.startswith('/') or PureWindowsPath(value).drive or any(p in ('','.','..') for p in value.split('/')):
        raise ValueError('Unsafe archive/code path')
    path=root
    for part in value.split('/'):
        path=path/part
        if path.exists() or path.is_symlink():
            info=path.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('Symlink/reparse point refused')
    return path


def restore_zip(archive,pin,workspace,restored):
    print('[VERIFY ZIP] '+archive.name,flush=True)
    if sha_file(archive)!=pin['sha256']:raise ValueError('ZIP differs from independently verified archive pin: '+archive.name)
    print('[EXTRACT/HASH] '+archive.name,flush=True)
    with zipfile.ZipFile(archive) as z:
        names=z.namelist()
        if len(names)!=len({n.casefold() for n in names}):raise ValueError('Duplicate/case-colliding archive members')
        manifest_name='archive_info/'+pin['group']+'/MANIFEST.json'
        readme_name='archive_info/'+pin['group']+'/README.md'
        if z.getinfo(manifest_name).file_size>64*1024*1024:raise ValueError('Oversized archive manifest')
        manifest=json.loads(z.read(manifest_name));rows=manifest['files']
        if len(rows)!=pin['members'] or sum(r['bytes'] for r in rows)!=pin['source_bytes']:raise ValueError('Archive plan counts/bytes differ from verified receipt')
        lookup={r['path']:r for r in rows}
        if len(lookup)!=len(rows) or set(names)!=set(lookup)|{manifest_name,readme_name}:raise ValueError('Archive membership mismatch')
        for info in z.infolist():
            name=info.filename
            path=safe(workspace,name)
            if not name.startswith(('data/','reference/','archive_info/')):raise ValueError('Archive may not replace source code')
            if info.is_dir() or stat.S_IFMT(info.external_attr>>16) not in (0,stat.S_IFREG):raise ValueError('Non-regular ZIP member refused')
            if path.exists():raise ValueError('Refusing to overwrite a restored/code file')
            expected=lookup.get(name)
            if expected and (not re.fullmatch('[0-9a-f]{64}',expected['sha256']) or expected['bytes']!=info.file_size):raise ValueError('Invalid member size/hash contract')
            path.parent.mkdir(parents=True,exist_ok=True)
            h=hashlib.sha256();count=0
            with z.open(info) as source,path.open('xb') as target:
                for block in iter(lambda:source.read(BUFFER),b''):
                    count+=len(block);h.update(block);target.write(block)
            if expected:
                if h.hexdigest()!=expected['sha256'] or count!=expected['bytes']:raise ValueError('Extracted member hash mismatch: '+name)
                restored[name]=expected['sha256']
                if len(restored)%1000==0:print('[EXTRACT/HASH] '+str(len(restored))+' data files',flush=True)
        return {'file':archive.name,'sha256':pin['sha256'],'verified_data_members':len(rows),'source_bytes':pin['source_bytes']}


def copy_code(package,workspace):
    with safe(package,'selected_sources.csv').open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    copied=[]
    for row in rows:
        name=row['destination']
        if not name.startswith('sources/'):continue
        source=safe(package,name);target=safe(workspace,name)
        if sha_file(source)!=row['sha256']:raise ValueError('Selected code changed: '+name)
        target.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as src,target.open('xb') as dst:shutil.copyfileobj(src,dst,BUFFER)
        if sha_file(target)!=row['sha256']:raise ValueError('Code copy mismatch')
        copied.append({'path':name,'sha256':row['sha256']})
    # Never copy data, outputs, local paths, or CSV reference values from the live package.
    for name in ['run.py','Run-C1.ps1','README.md','DATA.md','entrypoints.json','data_requirements.json']:
        source=safe(package,name);target=safe(workspace,name)
        if target.exists():raise ValueError('Duplicate code/adapter destination')
        shutil.copy2(source,target);copied.append({'path':name,'sha256':sha_file(target)})
    # Preserve later licence/citation sidecars; old verified packages may lack them.
    for name in ['LICENSE','LICENSE-DATA-CC-BY-4.0.txt','LICENSING.md','THIRD_PARTY_NOTICES.md','CITATION.cff']:
        source=safe(package,name)
        if not source.exists():continue
        target=safe(workspace,name)
        if target.exists():raise ValueError('Duplicate licence/citation destination')
        shutil.copy2(source,target);copied.append({'path':name,'sha256':sha_file(target)})
    return copied


def local_paths(workspace,example):
    result=dict(example)
    required=['campaign_root','ml_analysis_root','resnet_run','resnet_audit','inceptiontime_run','inceptiontime_audit']
    for key in required:
        value=result[key]
        path=safe(workspace,value).resolve()
        if not inside(path,workspace/'data'):raise ValueError('Configured study input outside restored data')
    result.update(output_root='outputs',comparison_analysis='',graphics_deps='',python_graphics='')
    return result


def policy_for(workspace,config):
    policy=read(safe(workspace,'sources/inceptiontime/configs/comparative_analysis.json'))
    roots={k:(workspace/config[k]).resolve() for k in ['campaign_root','ml_analysis_root','resnet_run','resnet_audit','inceptiontime_run','inceptiontime_audit']}
    policy['metadata_relative']=str(roots['campaign_root']/'metadata.json')
    policy['ml_root_relative']=str(roots['ml_analysis_root'])
    for model,key in [('ResNet','resnet'),('InceptionTime','inceptiontime')]:
        policy['neural'][model]['run_relative']=str(roots[key+'_run'])
        policy['neural'][model]['audit_relative']=str(roots[key+'_audit'])
    policy['resnet_reference_relative']=str(workspace/'reference/resnet_seeds_verified.csv')
    policy['python_relative']=sys.executable
    policy['graphics_deps_relative']=str(workspace/'optional_graphics_dependencies')
    policy['output_root_relative']=str(workspace/'outputs')
    policy['protected_roots_relative']=[str(roots[k]) for k in ['campaign_root','ml_analysis_root','resnet_run','inceptiontime_run']]
    return policy


def table_checks(workspace,result,restored):
    expected={n:h for n,h in restored.items() if n.startswith('reference/comparison/') and n.endswith('.csv')}
    if len(expected)!=17:raise ValueError('Expected all 17 archived reference tables')
    actual={p.name for p in result.glob('*.csv')}
    if actual!={Path(n).name for n in expected}:raise ValueError('Generated table set differs from reference')
    checks=[]
    for name,digest in sorted(expected.items()):
        if sha_file(safe(workspace,name))!=digest:raise ValueError('Restored reference table changed')
        found=sha_file(result/Path(name).name)
        checks.append({'file':Path(name).name,'expected_sha256':digest,'actual_sha256':found,'byte_identical':digest==found})
    write(result/'restoration_table_checks.json',checks)
    if not all(c['byte_identical'] for c in checks):raise ValueError('One or more tables differ from archived reference')
    return checks


def check_recorded_inputs(workspace,result,roots,restored):
    records=read(result/'inputs_manifest.json')['inputs'];checks=[]
    for item in records:
        role=item['role']
        if role=='metadata':p=Path(item['path']).resolve()
        elif role=='ML_predictions':p=(roots['ml_analysis_root']/item['path']).resolve()
        elif role in ('ResNet_predictions','InceptionTime_predictions'):
            p=(roots['resnet_run' if role.startswith('ResNet') else 'inceptiontime_run']/item['path']).resolve()
        else:raise ValueError('Unknown comparator input role')
        if not inside(p,workspace):raise ValueError('Recorded input outside restored workspace')
        member=p.relative_to(workspace).as_posix()
        if restored.get(member)!=item['sha256']:raise ValueError('Recorded study input not matched by restored archive')
        checks.append({'role':role,'restored_member':member,'sha256':item['sha256']})
    if len(checks)!=271:raise ValueError('Expected 271 recorded study inputs')
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root',required=True,type=Path)
    p.add_argument('--archive-root',required=True,type=Path)
    p.add_argument('--output-root',required=True,type=Path)
    a=p.parse_args();package=a.package_root.resolve();archives=a.archive_root.resolve();output=a.output_root.resolve()
    protected=[package,archives]
    # Original path config is read only to forbid putting the test inside an old data tree.
    old=read(safe(package,'paths.local.json'))
    for key in ('campaign_root','ml_analysis_root','resnet_run','inceptiontime_run'):
        original=(package/old[key]).resolve()
        protected.append(original.parent if key!='campaign_root' else original)
    if any(inside(output,r) or inside(r,output) for r in protected):raise ValueError('Restoration root overlaps original inputs')
    run=output/('restore_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);workspace=run/'workspace'
    workspace.mkdir(parents=True,exist_ok=False)
    print('[OUTPUT] '+str(run),flush=True)
    receipt={'created_local':datetime.now().astimezone().isoformat(timespec='seconds'),'status':'IN_PROGRESS','new_training':False,'new_model_inference':False,'new_hypothesis_tests':False,'original_sources_modified':False,'publication_ready':False,'fresh_environment_installation':False}
    write(run/'restoration_receipt.json',receipt)
    try:
        receipt['code_copies']=copy_code(package,workspace)
        restored={};receipt['archives']=[]
        for name,pin in PINS.items():receipt['archives'].append(restore_zip(safe(archives,name),pin,workspace,restored))
        if len(restored)!=13202:raise ValueError('Restored archive member count differs from verified set')
        # All original comparator provenance files must still match the archived successful run.
        reference_manifest=read(safe(workspace,'reference/comparison/inputs_manifest.json'))
        for name,digest in reference_manifest['source_files'].items():
            if sha_file(safe(workspace,'sources/inceptiontime/'+name))!=digest:raise ValueError('Comparator source differs from archived provenance')
        config=local_paths(workspace,read(safe(archives,'paths.archived.example.json')))
        write(workspace/'paths.local.json',config)
        policy=policy_for(workspace,config);runtime=workspace/'runtime';runtime.mkdir()
        write(runtime/'comparison.paths.json',policy)
        guard_dir=workspace/'guard';guard_dir.mkdir()
        shutil.copy2(Path(__file__).with_name('guarded_compare.py'),guard_dir/'guarded_compare.py')
        destination=workspace/'outputs'/'isolated_comparison';guard_report=workspace/'guard_report.json'
        env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONSTARTUP',None)
        print('[COMPARE] Frozen comparator; restored inputs only; Python file-access guard enabled.',flush=True)
        with (run/'comparison.log').open('wb') as log:
            completed=subprocess.run([sys.executable,'-I','-B',str(guard_dir/'guarded_compare.py'),'--workspace',str(workspace),'--policy',str(runtime/'comparison.paths.json'),'--output',str(destination),'--report',str(guard_report)],cwd=workspace,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,close_fds=True)
        if completed.returncode!=0:raise ValueError('Guarded comparison failed; see comparison.log and workspace/guard_report.json')
        guard=read(guard_report)
        if guard['status']!='GUARDED_COMPARISON_COMPLETE' or guard['blocked_operations']:raise ValueError('File access guard reported an external operation')
        scientific=read(destination/'analysis_receipt.json')
        if scientific['status']!='COMPLETE_DESCRIPTIVE_ANALYSIS_FOR_AUTHOR_REVIEW' or scientific['new_training'] or scientific['new_hypothesis_tests']:raise ValueError('Comparison did not complete as declared')
        receipt['tables']=table_checks(workspace,destination,restored)
        roots={k:(workspace/config[k]).resolve() for k in ('ml_analysis_root','resnet_run','inceptiontime_run')}
        inputs=check_recorded_inputs(workspace,destination,roots,restored)
        write(run/'restored_input_checks.json',inputs)
        receipt.update(status='ISOLATED_RESTORATION_REPRODUCED_17_TABLES_BYTE_IDENTICAL',restored_data_files=len(restored),matching_tables=17,recorded_study_inputs_matched_to_archive=len(inputs),blocked_python_file_accesses=0,guard_scope=guard['guard_scope'],comparison_relative='workspace/outputs/isolated_comparison')
    except Exception as exc:
        receipt.update(status='FAILED_RESTORATION_OR_COMPARISON_NOT_VALIDATED',error_type=type(exc).__name__,error=str(exc))
        write(run/'restoration_receipt.json',receipt)
        print('[FAILED] '+str(run/'restoration_receipt.json'),flush=True)
        raise
    write(run/'restoration_receipt.json',receipt)
    print(json.dumps({'output':str(run),'status':receipt['status'],'matching_tables':17,'new_training':False}))


if __name__=='__main__':main()
