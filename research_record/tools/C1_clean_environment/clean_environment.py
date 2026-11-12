"""Author-run clean installation, frozen archive/table test, optional figure export."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from urllib.parse import urlparse
from urllib.request import Request,urlopen
import uuid

BUFFER=1024*1024
BOOTSTRAP=Path(__file__).resolve().parent


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def inside(p,r):return p==r or r in p.parents


def validate_pins(config):
    if config.get('schema')!=1 or set(config['profiles'])!={'comparison','graphics'}:raise ValueError('Invalid dependency profiles')
    for pins in config['profiles'].values():
        if not pins:raise ValueError('Empty dependency profile')
        for name,version in pins.items():
            if not re.fullmatch('[a-z0-9]+(?:-[a-z0-9]+)*',name) or not re.fullmatch(r'\d+(?:\.\d+){1,3}',version):
                raise ValueError('Only explicit package/version pins are allowed')


def wheel_rank(filename):
    if not filename.endswith('.whl') or Path(filename).name!=filename:return None
    parts=filename[:-4].rsplit('-',3)
    if len(parts)!=4:return None
    _,python,abi,platform=parts
    if python=='cp312' and abi=='cp312' and platform=='win_amd64':return 0
    if abi=='none' and platform=='win_amd64' and 'py3' in python.split('.'):return 1
    if abi=='none' and platform=='any' and 'py3' in python.split('.'):return 1
    return None


def choose_wheel(document,name,version):
    if document['info']['version']!=version:raise ValueError('PyPI version mismatch')
    normalize=lambda n:re.sub('[-_.]+','-',n).lower()
    if normalize(document['info']['name'])!=name:raise ValueError('PyPI package mismatch')
    candidates=[]
    for item in document['urls']:
        rank=wheel_rank(item['filename'])
        if rank is None or item.get('yanked') or item['packagetype']!='bdist_wheel':continue
        url=urlparse(item['url'])
        if url.scheme!='https' or url.hostname!='files.pythonhosted.org' or url.username or url.password:
            raise ValueError('Unexpected wheel download host')
        if not re.fullmatch('[0-9a-f]{64}',item['digests']['sha256']):raise ValueError('Invalid wheel SHA256')
        if not isinstance(item['size'],int) or not 0<item['size']<512*1024*1024:raise ValueError('Invalid wheel size')
        candidates.append((rank,item['filename'],item))
    if not candidates:raise ValueError('No pinned CPython 3.12 Windows x64/pure-Python wheel: '+name+'=='+version)
    return sorted(candidates,key=lambda x:x[:2])[0][2]


def download_stream(source,target,digest,size):
    partial=target.with_suffix(target.suffix+'.partial');h=hashlib.sha256();count=0
    with partial.open('xb') as f:
        for block in iter(lambda:source.read(BUFFER),b''):
            count+=len(block)
            if count>size:raise ValueError('Wheel download exceeds declared size')
            h.update(block);f.write(block)
    if count!=size or h.hexdigest()!=digest:raise ValueError('Downloaded wheel size/SHA256 mismatch')
    if target.exists():raise ValueError('Wheel destination already exists')
    partial.rename(target)


def download_profile(pins,folder):
    folder.mkdir();lock=[];manifest=[]
    for name,version in pins.items():
        print('[DOWNLOAD/VERIFY] '+name+'=='+version,flush=True)
        request=Request('https://pypi.org/pypi/'+name+'/'+version+'/json',headers={'User-Agent':'C1-reproducibility-environment/1'})
        with urlopen(request,timeout=60) as response:
            raw=response.read(32*1024*1024+1)
            if len(raw)>32*1024*1024:raise ValueError('Oversized PyPI metadata')
        document=json.loads(raw);item=choose_wheel(document,name,version)
        (folder/(name+'.pypi.json')).write_bytes(raw)
        with urlopen(Request(item['url'],headers={'User-Agent':'C1-reproducibility-environment/1'}),timeout=60) as source:
            download_stream(source,folder/item['filename'],item['digests']['sha256'],item['size'])
        manifest.append({'name':name,'version':version,'wheel':item['filename'],'sha256':item['digests']['sha256'],
                         'bytes':item['size'],'url':item['url'],'metadata_sha256':hashlib.sha256(raw).hexdigest()})
        lock.append(name+'=='+version+' --hash=sha256:'+item['digests']['sha256'])
    (folder/'requirements.lock.txt').write_text('\n'.join(lock)+'\n',encoding='utf-8')
    write(folder/'wheel_manifest.json',manifest)
    return manifest


def run_command(command,log,cwd,extra_env=None):
    env=dict(os.environ)
    for key in list(env):
        if key.upper().startswith(('PYTHON','PIP_')) or key.upper()=='VIRTUAL_ENV':env.pop(key,None)
    env['PIP_CONFIG_FILE']=os.devnull
    if extra_env:env.update(extra_env)
    with log.open('wb') as f:
        process=subprocess.run([str(x) for x in command],cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    if process.returncode:raise RuntimeError('Step failed; see '+str(log))


def install_profile(profile,run,tools,config):
    prefix=run/('venv_'+profile);python=prefix/'Scripts/python.exe'
    print('[CREATE ENVIRONMENT] '+profile,flush=True)
    run_command([sys.executable,'-I','-B','-m','venv',prefix],run/(profile+'_create.log'),run)
    probe=[python,'-I','-B',tools/'environment_probe.py','--config',tools/'dependencies.json','--profile',profile]
    run_command(probe+['--phase','before','--output',run/(profile+'_before.json')],run/(profile+'_before.log'),run)
    wheels=run/('wheels_'+profile);manifest=download_profile(config['profiles'][profile],wheels)
    print('[INSTALL] '+profile+'; local verified wheels only',flush=True)
    run_command([python,'-I','-B','-m','pip','--isolated','--require-virtualenv','--disable-pip-version-check','install',
                 '--no-index','--find-links',wheels,'--no-cache-dir','--only-binary=:all:','--require-hashes','--no-deps',
                 '--report',run/(profile+'_pip_install.json'),'-r',wheels/'requirements.lock.txt'],run/(profile+'_install.log'),run)
    run_command([python,'-I','-B','-m','pip','--isolated','check'],run/(profile+'_pip_check.log'),run)
    run_command(probe+['--phase','after','--output',run/(profile+'_after.json')],run/(profile+'_after.log'),run)
    run_command([python,'-I','-B','-m','pip','--isolated','freeze','--all'],run/(profile+'_freeze.txt'),run)
    return python,manifest


def figure_checks(folder,tables):
    figures=folder/'figures';manifest=read(figures/'figures_manifest.json')
    if len(manifest['figures'])!=3 or manifest['scientific_predictions_or_metrics_recomputed']:
        raise ValueError('Unexpected figure export scope')
    expected={r['file']:r['actual_sha256'] for r in tables};outputs=[]
    for name,digest in manifest['source_tables_sha256'].items():
        if expected.get(name)!=digest or sha(folder/name)!=digest:raise ValueError('Figure input tables changed')
    for item in manifest['figures']:
        name=item['name']
        if not re.fullmatch('[A-Za-z0-9_]+',name):raise ValueError('Unsafe figure basename')
        for suffix in ('pdf','svg','png'):
            p=figures/(name+'.'+suffix)
            if p.stat().st_size<100:raise ValueError('Empty figure output')
            with p.open('rb') as f:header=f.read(300)
            if suffix=='pdf' and not header.startswith(b'%PDF-'):raise ValueError('Invalid PDF header')
            if suffix=='png' and not header.startswith(b'\x89PNG\r\n\x1a\n'):raise ValueError('Invalid PNG header')
            if suffix=='svg' and b'<svg' not in header:raise ValueError('Invalid SVG header')
            outputs.append({'path':'figures/'+p.name,'sha256':sha(p),'bytes':p.stat().st_size})
    return {'status':'THREE_FIGURES_EXPORTED_ACTUAL_PDF_VISUAL_QA_PENDING','files':outputs,'actual_PDF_visual_QA':'PENDING'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package-root',type=Path,required=True);p.add_argument('--archive-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True);p.add_argument('--include-figures',action='store_true')
    p.add_argument('--paper-figures',action='store_true')
    a=p.parse_args();package=a.package_root.resolve();archives=a.archive_root.resolve();output=a.output_root.resolve()
    if os.name!='nt' or sys.implementation.name!='cpython' or sys.version_info[:2]!=(3,12) or struct.calcsize('P')!=8:
        raise ValueError('Use CPython 3.12 Windows x64. No interpreter is installed automatically.')
    protected=[package,archives,BOOTSTRAP,BOOTSTRAP.parent/'C1_archive_restoration',Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
    old=read(package/'paths.local.json')
    for key in ('campaign_root','ml_analysis_root','resnet_run','inceptiontime_run'):
        original=(package/old[key]).resolve();protected.append(original if key=='campaign_root' else original.parent)
    if any(inside(output,r) or inside(r,output) for r in protected):raise ValueError('Output overlaps original code/data/environments')
    run=output/('env_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]);run.mkdir(parents=True,exist_ok=False)
    print('[OUTPUT] '+str(run),flush=True)
    receipt={'created_local':datetime.now().astimezone().isoformat(timespec='seconds'),'status':'IN_PROGRESS',
             'new_training':False,'new_model_inference':False,'new_hypothesis_tests':False,'publication_ready':False,
             'requested_figures':a.include_figures,'requested_processing_charts':a.paper_figures,
             'base_python_reused':True,'new_environments':True}
    write(run/'fresh_environment_receipt.json',receipt)
    try:
        tools=run/'tools';tools.mkdir()
        for name in ('clean_environment.py','environment_probe.py','dependencies.json'):
            shutil.copy2(BOOTSTRAP/name,tools/name)
        for name in ('restore_validate.py','guarded_compare.py','test_restore_validate.py'):
            shutil.copy2(BOOTSTRAP.parent/'C1_archive_restoration'/name,tools/name)
        receipt['helper_sha256']={p.name:sha(p) for p in tools.iterdir()}
        config=read(tools/'dependencies.json');validate_pins(config)
        python,wheels=install_profile('comparison',run,tools,config);receipt['comparison_wheels']=wheels
        print('[TEST] Restoration tests in the newly installed environment',flush=True)
        run_command([python,'-I','-B','-m','unittest','discover','-s',tools,'-p','test_restore_validate.py','-v'],run/'restoration_tests.log',run)
        print('[RESTORE/COMPARE] Fresh archives and 17-table reproduction; see restoration.log',flush=True)
        restore_root=output/'restorations'
        run_command([python,'-I','-B',tools/'restore_validate.py','--package-root',package,'--archive-root',archives,
                     '--output-root',restore_root],run/'restoration.log',run)
        lines=(run/'restoration.log').read_text(encoding='utf-8-sig').strip().splitlines()
        summary=json.loads(lines[-1]);restored=Path(summary['output']).resolve()
        if not inside(restored,restore_root.resolve()):raise ValueError('Restoration result outside declared root')
        restored_receipt=read(restored/'restoration_receipt.json')
        if restored_receipt['status']!='ISOLATED_RESTORATION_REPRODUCED_17_TABLES_BYTE_IDENTICAL' or restored_receipt['matching_tables']!=17:
            raise ValueError('All 17 tables must reproduce identically')
        receipt['restoration_relative']=os.path.relpath(restored,run).replace('\\','/')
        receipt['restoration_receipt_sha256']=sha(restored/'restoration_receipt.json')
        receipt['matching_tables']=17
        receipt['numerical_validation']='PASS'
        receipt['figures']={'status':'NOT_REQUESTED'}
        receipt['paper_figures']={'status':'NOT_REQUESTED'}
        if a.include_figures or a.paper_figures:
            graphics,graphics_wheels=install_profile('graphics',run,tools,config);receipt['graphics_wheels']=graphics_wheels
        if a.include_figures:
            plots=run/'plots';plots.mkdir();ws=restored/'workspace';result=ws/'outputs/isolated_comparison'
            for table in restored_receipt['tables']:
                if sha(result/table['file'])!=table['actual_sha256']:raise ValueError('Verified table changed before plotting')
                shutil.copy2(result/table['file'],plots/table['file'])
            shutil.copy2(result/'analysis_receipt.json',plots/'analysis_receipt.json')
            cache=run/'mplconfig';cache.mkdir()
            plot=ws/'sources/inceptiontime/tools/plot_comparison.py'
            print('[PLOT] Three figures, PDF/SVG/PNG; visual review remains pending',flush=True)
            run_command([graphics,'-I','-B',plot,'--analysis',plots],run/'graphics_plot.log',run,{'MPLCONFIGDIR':str(cache),'MPLBACKEND':'Agg'})
            receipt['figures']=figure_checks(plots,restored_receipt['tables'])
        if a.paper_figures:
            paper_source=BOOTSTRAP.parent/'C1_paper_figures';paper_tools=tools/'paper_figures'
            shutil.copytree(paper_source,paper_tools,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            receipt['paper_tool_sha256']={p.relative_to(paper_tools).as_posix():sha(p) for p in paper_tools.iterdir() if p.is_file()}
            work=run/'paper_figures';cache=run/'paper_mplconfig';cache.mkdir()
            print('[PAPER CHARTS] Rebuild charts 3--7 only; figures 1 and 2 excluded. Reference PDFs are QA material only.',flush=True)
            run_command([graphics,'-I','-B',paper_tools/'reproduce_paper_figures.py','--bundle',paper_tools/'payload',
                         '--comparison',restored/'workspace/outputs/isolated_comparison','--output',work,
                         '--guard-source',tools/'guarded_compare.py'],run/'paper_figures.log',run,
                         {'MPLCONFIGDIR':str(cache),'MPLBACKEND':'Agg'})
            paper_receipt=read(work/'paper_figures_receipt.json')
            if paper_receipt['status']!='FIVE_PROCESSING_CHARTS_REGENERATED_ACTUAL_PDF_QA_PENDING' or paper_receipt['figures_generated']!=5:
                raise ValueError('Five processing charts did not complete')
            receipt['paper_figures']={'status':paper_receipt['status'],'figures':5,'paper_numbers':[3,4,5,6,7],
                                     'receipt_relative':'paper_figures/paper_figures_receipt.json',
                                     'receipt_sha256':sha(work/'paper_figures_receipt.json'),'actual_PDF_visual_QA':'PENDING'}
        receipt.update(status='CLEAN_ENVIRONMENT_17_TABLES_REPRODUCED',completed_local=datetime.now().astimezone().isoformat(timespec='seconds'),
                       numerical_validation='PASS',figure_visual_approval='PENDING' if a.include_figures or a.paper_figures else 'NOT_REQUESTED')
    except Exception as exc:
        receipt.update(status='FAILED_CLEAN_ENVIRONMENT_NOT_VALIDATED',error_type=type(exc).__name__,error=str(exc))
        write(run/'fresh_environment_receipt.json',receipt);print('[FAILED] '+str(run/'fresh_environment_receipt.json'),flush=True);raise
    write(run/'fresh_environment_receipt.json',receipt)
    print(json.dumps({'output':str(run),'status':receipt['status'],'matching_tables':17,
                      'figures':receipt['figures']['status'],'processing_charts':receipt['paper_figures']['status']}))


if __name__=='__main__':main()
