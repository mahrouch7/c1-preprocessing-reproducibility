"""Regenerate processing/result charts 3--7; exclude network/protocol and acquisition."""
import argparse
import csv
from datetime import datetime
import hashlib
import io
import json
import os
from pathlib import Path,PureWindowsPath
import shutil
import stat
import subprocess
import sys


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def safe(root,name):
    name=str(name).replace('\\','/')
    if ':' in name or name.startswith('/') or PureWindowsPath(name).drive or any(p in ('','.','..') for p in name.split('/')):
        raise ValueError('Unsafe payload path')
    p=root
    for part in name.split('/'):
        p=p/part
        if p.exists() or p.is_symlink():
            info=p.lstat()
            if stat.S_ISLNK(info.st_mode) or getattr(info,'st_file_attributes',0)&0x400:raise ValueError('Link/reparse point refused')
    return p


def verify_payload(bundle,manifest):
    if manifest.get('schema')!=1 or [r['number'] for r in manifest['figures']]!=list(range(1,8)):
        raise ValueError('Expected seven explicitly numbered paper figures')
    seen=set()
    for row in manifest['files']:
        name=row['path'];p=safe(bundle,name)
        if name.casefold() in seen:raise ValueError('Payload collision')
        seen.add(name.casefold())
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Figure payload changed: '+name)


def verify_comparison(folder,manifest):
    receipt=read(folder/'analysis_receipt.json')
    if receipt['status']!='COMPLETE_DESCRIPTIVE_ANALYSIS_FOR_AUTHOR_REVIEW' or receipt['new_training'] or receipt['new_hypothesis_tests']:
        raise ValueError('Completed frozen comparison required')
    for name,digest in manifest['dynamic_comparison_tables_sha256'].items():
        if sha(safe(folder,name))!=digest:raise ValueError('Reproduced comparison table differs: '+name)


def chart_specs(manifest):
    selected=[r for r in manifest['figures'] if r['number'] in (3,4,5,6,7)]
    if [r['number'] for r in selected]!=[3,4,5,6,7]:raise ValueError('Expected processing/result charts 3--7 only')
    return selected


def verify_data(work):
    audit=read(work/'data/abc_abcg_audit_summary.json')
    if audit['pairs']!=153 or audit['exact_point_equal_pairs']!=0:raise ValueError('ABC/ABCG audit scope changed')
    with (work/'data/e42_figure_data.csv').open(encoding='utf-8-sig',newline='') as f:medians=list(csv.DictReader(f))
    if len(medians)!=120:raise ValueError('Expected all 120 frozen transient medians')
    summary=read(work/'data/waveforms/summary.json')
    if summary['synthetic_fixture'] or summary['selection']['Id']!=963:raise ValueError('Frozen real waveform example changed')


def configuration_environment(work,inherited):
    """Redirect optional personal configuration, without weakening the I/O guard."""
    env=dict(inherited);profile=work/'isolated_profile'
    for key in list(env):
        if key.upper().startswith(('RL_','PYTHON')):env.pop(key,None)
    env.update(USERPROFILE=str(profile),HOME=str(profile),HOMEDRIVE=profile.drive,
               HOMEPATH=str(profile)[len(profile.drive):],
               APPDATA=str(profile/'AppData/Roaming'),LOCALAPPDATA=str(profile/'AppData/Local'),
               XDG_CONFIG_HOME=str(profile/'config'),XDG_CACHE_HOME=str(profile/'cache'),
               MPLCONFIGDIR=str(work/'mplcache'),MPLBACKEND='Agg',MPL_IGNORE_SYSTEM_FONTS='1')
    return env


def run_driver(work,name,arguments):
    report=work/'qa'/(name+'.guard.json');log=work/'logs'/(name+'.log')
    command=[sys.executable,'-I','-B',str(work/'guarded_driver.py'),'--workspace',str(work),
             '--driver',str(work/'drivers'/name),'--report',str(report),'--']+[str(x) for x in arguments]
    env=configuration_environment(work,os.environ)
    print('[PAPER DRIVER] '+name,flush=True)
    with log.open('wb') as f:
        result=subprocess.run(command,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError('Figure driver failed: '+str(log))
    guard=read(report)
    if guard['status']!='COMPLETE' or guard['blocked_operations']:raise ValueError('Figure driver violated declared inputs')


def title_band(source,target,title):
    from pypdf import PdfReader,PdfWriter,PageObject
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    import matplotlib
    font=Path(matplotlib.get_data_path())/'fonts/ttf/DejaVuSans-Bold.ttf'
    pdfmetrics.registerFont(TTFont('C1FigureTitle',str(font)))
    page=PdfReader(source).pages[0];w,h=float(page.mediabox.width),float(page.mediabox.height)
    size=9.5 if w<350 else 11.5
    while pdfmetrics.stringWidth(title,'C1FigureTitle',size)>w-20:size-=.25
    if size<8:raise ValueError('Title would be illegible')
    buf=io.BytesIO();c=Canvas(buf,pagesize=(w,h+24),invariant=1)
    c.setFont('C1FigureTitle',size);c.setFillColorRGB(.08,.14,.19);c.drawCentredString(w/2,h+8,title);c.save();buf.seek(0)
    final=PageObject.create_blank_page(width=w,height=h+24);final.merge_page(page);final.merge_page(PdfReader(buf).pages[0])
    writer=PdfWriter();writer.add_page(final);writer.add_metadata({'/Title':title})
    with target.open('xb') as f:writer.write(f)


def render_image(pdf,scale):
    import pypdfium2 as pdfium
    document=pdfium.PdfDocument(str(pdf))
    try:
        if len(document)!=1:raise ValueError('Expected a one-page figure PDF')
        page=document[0]
        try:
            bitmap=page.render(scale=scale)
            try:return bitmap.to_pil().convert('RGB').copy()
            finally:bitmap.close()
        finally:page.close()
    finally:document.close()


def final_exports(work,manifest,generated):
    from pypdf import PdfReader
    from PIL import Image,ImageChops,ImageStat,ImageDraw
    entries=[];thumbnails=[]
    selected=chart_specs(manifest)
    if len(generated)!=len(selected):raise ValueError('Incomplete processing chart outputs')
    for spec,source in zip(selected,generated):
        number=spec['number'];name=Path(spec['paper_filename']).stem+'_reproduced_v1'
        target=work/'figures'/(name+'.pdf');shutil.copy2(source,target)
        page=PdfReader(target).pages[0];size=[float(page.mediabox.width),float(page.mediabox.height)]
        if any(abs(a-b)>.05 for a,b in zip(size,spec['expected_page_pt'])):raise ValueError('Figure canvas changed: '+str(number))
        png=target.with_suffix('.png');actual=render_image(target,200/72);actual.save(png)
        if max(ImageStat.Stat(actual).var)<.01:raise ValueError('Blank figure render')
        reference=render_image(work/'reference'/('figure'+str(number)+'.pdf'),2)
        rebuilt=render_image(target,2);reference.save(work/'qa'/('reference_'+str(number)+'.png'))
        rebuilt.save(work/'qa'/('reproduced_'+str(number)+'.png'))
        comparison={'same_raster_dimensions':reference.size==rebuilt.size}
        if comparison['same_raster_dimensions']:
            diff=ImageChops.difference(reference,rebuilt)
            comparison['maximum_channel_mean_absolute_difference_8bit']=max(ImageStat.Stat(diff).mean)
            comparison['pixels_identical']=diff.getbbox() is None
        for title,image in [('Reference '+str(number),reference),('Reproduced '+str(number),rebuilt)]:
            image.thumbnail((650,450));panel=Image.new('RGB',(680,490),'white')
            ImageDraw.Draw(panel).text((10,5),title,fill='black');panel.paste(image,((680-image.width)//2,30));thumbnails.append(panel)
        entries.append({'number':number,'paper_filename':spec['paper_filename'],'pdf':'figures/'+target.name,'png':'figures/'+png.name,
                        'pdf_sha256':sha(target),'png_sha256':sha(png),'page_pt':size,'reference_comparison':comparison,
                        'actual_PDF_visual_QA':'PENDING_AUTHOR_AND_ASSISTANT_REVIEW'})
    sheet=Image.new('RGB',(1360,490*len(selected)),'#dddddd')
    for i,image in enumerate(thumbnails):sheet.paste(image,((i%2)*680,(i//2)*490))
    sheet.save(work/'qa/contact_sheet.png')
    return entries


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--guard-source',type=Path,required=True)
    a=p.parse_args();bundle=a.bundle.resolve();comparison=a.comparison.resolve();work=a.output.resolve()
    if any(work==r or r in work.parents or work in r.parents for r in (bundle,comparison)):
        raise ValueError('Figure workspace must be separate from inputs')
    manifest=read(bundle/'manifest.json');verify_payload(bundle,manifest);verify_comparison(comparison,manifest)
    work.mkdir(parents=True,exist_ok=False)
    print('[OUTPUT] '+str(work),flush=True)
    receipt={'created_local':datetime.now().astimezone().isoformat(timespec='seconds'),'status':'IN_PROGRESS',
             'new_training':False,'new_noise_or_filtering':False,'new_scientific_scores_or_tests':False,
             'input_manifest_sha256':sha(bundle/'manifest.json'),'actual_PDF_visual_QA':'PENDING',
             'requested_paper_chart_numbers':[3,4,5,6,7],'excluded_paper_figure_numbers':[1,2]}
    write(work/'paper_figures_receipt.json',receipt)
    try:
        for row in manifest['files']:
            target=safe(work,row['path']);target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(safe(bundle,row['path']),target)
        for name in ('guarded_driver.py',):shutil.copy2(Path(__file__).with_name(name),work/name)
        shutil.copy2(a.guard_source,work/'guarded_io.py')
        for directory in ('intermediate','qa','logs','verification','figures','empty_dependencies'):(work/directory).mkdir(exist_ok=True)
        for directory in ('isolated_profile','isolated_profile/AppData/Roaming','isolated_profile/AppData/Local',
                          'isolated_profile/config','isolated_profile/cache','mplcache'):
            (work/directory).mkdir(parents=True,exist_ok=True)
        # This process also uses ReportLab for title bands: isolate it before imports.
        isolated=configuration_environment(work,os.environ)
        for key in list(os.environ):
            if key.upper().startswith(('RL_','PYTHON')):os.environ.pop(key,None)
        os.environ.update(isolated)
        receipt['personal_configuration']='Redirected to empty workspace/isolated_profile; original user configuration not read or modified'
        for name in manifest['dynamic_comparison_tables_sha256']:
            shutil.copy2(comparison/name,work/'data'/name)
        verify_data(work)
        receipt['font_policy']='Matplotlib bundled fonts only; system/user font discovery disabled. No Arial-dependent diagram generated.'
        run_driver(work,'waveforms.py',['--project-root',work,'--numeric',work/'data/waveforms','--output',work/'intermediate/waveforms',
                                       '--graphics-dependencies',work/'empty_dependencies'])
        run_driver(work,'snr.py',[])
        run_driver(work,'fusion.py',[])
        title_band(work/'intermediate/fusion_snr_and_constituents.pdf',work/'intermediate/fusion_titled.pdf','Fusion Effects Relative to Both Constituent Representations')
        run_driver(work,'fusion_crop.py',[])
        run_driver(work,'audit_panel.py',[])
        run_driver(work,'temporal_top.py',[])
        run_driver(work,'temporal_merge.py',[])
        run_driver(work,'transients.py',[])
        generated=[work/'intermediate/waveforms/waveform_processing_examples_compact_v3.pdf',
                   work/'intermediate/s10_fig02_snr_exposure_filtering_v2.pdf',work/'intermediate/fusion_final.pdf',
                   work/'intermediate/o15_fig06_temporal_comparison_e43_v1.pdf',work/'intermediate/o15_fig07_transient_preservation_noise_v2.pdf']
        receipt['figures']=final_exports(work,manifest,generated)
        verify_payload(work,manifest)  # frozen data and staged sources must remain unchanged
        verify_comparison(comparison,manifest)
        receipt.update(status='FIVE_PROCESSING_CHARTS_REGENERATED_ACTUAL_PDF_QA_PENDING',figures_generated=5,
                       outputs='Five vector PDFs and five PNGs for charts 3--7; figures 1 and 2 excluded; native intermediate SVGs retained.',
                       completed_local=datetime.now().astimezone().isoformat(timespec='seconds'))
    except Exception as exc:
        receipt.update(status='FAILED_PAPER_FIGURE_REPRODUCTION_NOT_VALIDATED',error_type=type(exc).__name__,error=str(exc))
        write(work/'paper_figures_receipt.json',receipt);raise
    write(work/'paper_figures_receipt.json',receipt)
    print(json.dumps({'output':str(work),'status':receipt['status'],'processing_charts':5,'paper_numbers':[3,4,5,6,7],'actual_PDF_visual_QA':'PENDING'}))


if __name__=='__main__':main()
