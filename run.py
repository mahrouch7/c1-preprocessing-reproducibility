"""Generated package launcher. Default action inspects paths; never trains models."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def resolve(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT/p).resolve()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def paths_from(config):
    return {k:resolve(v) for k,v in config.items() if v and k not in ('python_graphics',)}


def neural_config(original, paths):
    result=json.loads(json.dumps(original))
    result['source_root']=str(paths['campaign_root'])
    result['baseline_root']=str(paths['ml_analysis_root'].parent)
    return result


def comparison_policy(original, paths):
    result=json.loads(json.dumps(original))
    result['metadata_relative']=str(paths['campaign_root']/'metadata.json')
    result['ml_root_relative']=str(paths['ml_analysis_root'])
    for name,key in [('ResNet','resnet'),('InceptionTime','inceptiontime')]:
        result['neural'][name]['run_relative']=str(paths[key+'_run'])
        result['neural'][name]['audit_relative']=str(paths[key+'_audit'])
    result['resnet_reference_relative']=str(ROOT/'reference/resnet_seeds_verified.csv')
    result['python_relative']=sys.executable
    result['graphics_deps_relative']=str(paths.get('graphics_deps',ROOT/'optional_graphics_dependencies'))
    result['output_root_relative']=str(paths['output_root'])
    result['protected_roots_relative']=[str(paths[k]) for k in ('campaign_root','ml_analysis_root','resnet_run','inceptiontime_run')]
    return result


def inspect(paths, requirements, hash_inputs=False):
    rows=[]
    for item in requirements:
        p=paths[item['root']]/item['relative']
        exists=p.is_file() if item['kind']=='file' else p.is_dir()
        result={**item,'resolved_path':str(p),'exists':exists,'hash_check':'NOT_REQUESTED'}
        if hash_inputs and exists and item.get('sha256'):
            result['hash_check']='PASS' if file_hash(p)==item['sha256'] else 'MISMATCH'
        rows.append(result)
    return rows


def output_directory(paths, action):
    output=paths['output_root']
    for key in ('campaign_root','ml_analysis_root','resnet_run','inceptiontime_run'):
        source=paths[key]
        if output==source or source in output.parents or output in source.parents:
            raise ValueError('Outputs must not overlap inputs: '+key)
    destination=output/(action+'_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    destination.mkdir(parents=True,exist_ok=False)
    return destination


def call(command, python_paths, cwd=ROOT):
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
    env['PYTHONPATH']=os.pathsep.join(str(p) for p in python_paths)
    subprocess.run([str(x) for x in command],check=True,cwd=cwd,env=env)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'paths.local.json')
    parser.add_argument('--action',choices=['inspect','compare','export-ml','plan-resnet','plan-inception','plot-comparison'],default='inspect')
    parser.add_argument('--verify-hashes',action='store_true')
    args=parser.parse_args()
    cfg=read(args.config); paths=paths_from(cfg); requirements=read(ROOT/'data_requirements.json')
    if args.action=='inspect':
        rows=inspect(paths,requirements,args.verify_hashes)
        print(json.dumps({'action':'inspect','data':rows,'scientific_code_executed':False},indent=2))
        return 2 if any(not r['exists'] or r['hash_check']=='MISMATCH' for r in rows if 'compare' in r['needed_for']) else 0
    if args.action.startswith('plan-'):
        name='resnet' if args.action=='plan-resnet' else 'inceptiontime'
        package='c1resnet' if name=='resnet' else 'c1inception'
        dest=output_directory(paths,args.action)
        runtime=neural_config(read(ROOT/'sources'/name/'configs/experiment.json'),paths)
        write(dest/'experiment.paths.json',runtime)
        call([sys.executable,'-m',package+'.cli','--config',dest/'experiment.paths.json','plan','--profile','pilot'],[ROOT/'sources'/name/'src'])
        print('PLAN_ONLY_NO_TRAINING: '+str(dest)); return 0
    if args.action=='compare':
        # Frozen comparator still enforces original labels, folds, seed identities,
        # run/receipt hashes, probability alignment, source commits and scientific versions.
        dest=output_directory(paths,'compare')
        runtime=comparison_policy(read(ROOT/'sources/inceptiontime/configs/comparative_analysis.json'),paths)
        write(dest/'comparison.paths.json',runtime)
        call([sys.executable,ROOT/'sources/inceptiontime/tools/analyze_results.py','--policy',dest/'comparison.paths.json','--output',dest/'analysis'],[ROOT/'sources/inceptiontime/src'])
        print('COMPARISON: '+str(dest/'analysis')); return 0
    if args.action=='export-ml':
        src=paths['ml_analysis_root']; receipt=read(src/'analysis_receipt.json')
        names=['metrics.csv','per_class_metrics.csv','paired_family_inference.csv','pooled_prediction_index.json']
        for name in names:
            expected=receipt['artifacts'].get(name)
            if not expected or file_hash(src/name)!=expected:
                raise ValueError('Frozen ML artifact identity missing or changed: '+name)
        dest=output_directory(paths,'export_ml')
        for name in names: shutil.copy2(src/name,dest/name)
        write(dest/'export_receipt.json',{'status':'SAVED_TABLES_EXPORTED_NOT_RECOMPUTED','files':{n:receipt['artifacts'][n] for n in names},'new_training':False})
        print('ML_TABLES: '+str(dest)); return 0
    analysis=paths.get('comparison_analysis')
    if analysis is None or not (analysis/'analysis_receipt.json').is_file():
        raise ValueError('Set comparison_analysis to a completed comparator analysis folder')
    graphics_python=cfg.get('python_graphics') or sys.executable
    # Graphics tool writes into the explicitly selected completed analysis folder.
    if any(analysis==paths[k] or paths[k] in analysis.parents for k in ('campaign_root','ml_analysis_root','resnet_run','inceptiontime_run')):
        raise ValueError('Plot output cannot be inside frozen scientific inputs')
    dest=output_directory(paths,'plots')
    for source in analysis.iterdir():
        if source.is_symlink():raise ValueError('Analysis symlink refused')
        if source.is_file() and source.suffix in ('.csv','.json'):
            shutil.copy2(source,dest/source.name)
    command=[graphics_python,ROOT/'sources/inceptiontime/tools/plot_comparison.py','--analysis',dest]
    if 'graphics_deps' in paths:command+=['--graphics-deps',paths['graphics_deps']]
    call(command,[])
    print('PLOTS_REQUIRE_ACTUAL_PDF_VISUAL_REVIEW: '+str(dest))
    return 0


if __name__=='__main__':
    sys.exit(main())
