"""Dependency-free scientific SVG atlas with explicit cases and missing results."""
from __future__ import annotations
import csv
from html import escape
import math
import textwrap
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import c1e32_spec as spec
import c1e32_common as c
import c1e32_statistics as stats

COLORS=('#0072B2','#D55E00','#009E73','#CC79A7','#E69F00','#56B4E9','#222222','#7570B3')

class SVG:
    def __init__(self,path,title,width=1450,height=1100):
        self.path,self.w,self.h=path,width,height
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>','<g font-family="Arial,Helvetica,sans-serif" fill="#172b3a">']
        self.text(45,35,title,22)
    def text(self,x,y,value,size=14,color='#172b3a',anchor='start'):
        c.require(math.isfinite(float(x)+float(y)) and 0<=x<=self.w and 0<=y<=self.h,'Text anchor outside figure')
        self.parts.append(f'<text x="{x:.3f}" y="{y:.3f}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>')
    def line(self,points,color='#b8c5d0',width=1,dash=''):
        c.require(all(math.isfinite(float(x)+float(y)) for x,y in points),'Nonfinite line')
        self.parts.append(f'<polyline points="'+ ' '.join(f'{x:.3f},{y:.3f}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
    def rect(self,x,y,w,h,fill,stroke='none'):
        c.require(all(math.isfinite(float(v)) for v in (x,y,w,h)) and w>=0 and h>=0,'Invalid rectangle')
        self.parts.append(f'<rect x="{x:.3f}" y="{y:.3f}" width="{w:.3f}" height="{h:.3f}" fill="{fill}" stroke="{stroke}"/>')
    def dot(self,x,y,color,r=3):
        c.require(math.isfinite(float(x)+float(y)),'Nonfinite marker')
        self.parts.append(f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r}" fill="{color}"/>')
    def finish(self,footer):
        wrapped=textwrap.wrap(footer,width=max(35,int((self.w-90)/6.7)))
        for i,line in enumerate(wrapped): self.text(45,self.h-24-(len(wrapped)-1-i)*17,line,12)
        self.parts.append('</g></svg>'); self.path.write_text('\n'.join(self.parts),encoding='utf-8')
        ET.parse(self.path)

def metric_lookup(metrics):
    return {r['case_id']:r for r in metrics if r['population']=='family_class_standardized' and r['view']=='full'}

def axes(fig,x,y,w,h,title):
    fig.text(x,y-18,title,17)
    for tick in (0,.25,.5,.75,1):
        yy=y+h*(1-tick); fig.line([(x,yy),(x+w,yy)],'#dce3e8'); fig.text(x-12,yy+5,f'{tick:g}',12,anchor='end')

def curves(directory,lookup):
    for protocol in ('deployment','matched'):
        for prep in spec.PREPS:
            for rep in ('RMS6','RAW12','MULTI18','LATE_SIMPLE','LATE_ORACLE_SNR'):
                fig=SVG(directory/f'curves_{protocol}_{prep}_{rep}.svg',f'Macro-F1 | {protocol} | {prep} | {rep}')
                for panel,noise in enumerate(spec.old.NOISES):
                    x=65+(panel%2)*710; y=90+(panel//2)*315; w=590; h=225
                    axes(fig,x,y,w,h,noise)
                    levels=list(spec.base.SNR_LEVELS)
                    for snr in levels:
                        xx=x+(snr-10)/110*(w-70); fig.text(xx,y+h+23,snr,12,anchor='middle')
                    fig.text(x+w-15,y+h+23,'Clean',12,anchor='middle')
                    for mi,model in enumerate(spec.MODELS):
                        points=[]
                        for snr in levels:
                            key=spec.digest(stats.robust(model,noise+'__'+str(snr),rep,prep,protocol)); row=lookup[key]
                            if not row['eligible_for_inference']:
                                if points: fig.line(points,COLORS[mi],2); points=[]
                                continue
                            xx=x+(snr-10)/110*(w-70); yy=y+h*(1-row['macro_f1']); points.append((xx,yy)); fig.dot(xx,yy,COLORS[mi])
                        if points: fig.line(points,COLORS[mi],2)
                        reference=lookup[spec.digest(stats.robust(model,'clean',rep,prep,protocol))]
                        if reference['eligible_for_inference']: fig.dot(x+w-15,y+h*(1-reference['macro_f1']),COLORS[mi],4)
                for mi,model in enumerate(spec.MODELS): fig.text(810,790+mi*26,model,15,COLORS[mi])
                fig.finish('Family/class standardized held-out estimates. Clean is separate; gaps indicate exclusions. True-SNR fusion is ORACLE only.')

def radars(directory,lookup):
    for protocol in ('deployment','matched'):
        for prep in spec.PREPS:
            for rep in ('RMS6','RAW12','MULTI18','LATE_SIMPLE'):
                fig=SVG(directory/f'radar_{protocol}_{prep}_{rep}.svg',f'Severe-noise Macro-F1 | {protocol} | {prep} | {rep}',1300,850)
                cx,cy,radius=430,420,270
                def point(i,value):
                    a=-math.pi/2+i*2*math.pi/5; return cx+radius*value*math.cos(a),cy+radius*value*math.sin(a)
                for tick in (.2,.4,.6,.8,1):
                    points=[point(i,tick) for i in range(5)]; fig.line(points+[points[0]])
                    fig.text(cx+8,cy-radius*tick+15,f'{tick:.1f}',11)
                for i,noise in enumerate(spec.old.NOISES):
                    fig.line([(cx,cy),point(i,1)]); xx,yy=point(i,1.15); fig.text(xx,yy,noise,12,anchor='middle')
                for mi,model in enumerate(spec.MODELS):
                    rows=[lookup[spec.digest(stats.robust(model,n+'__10',rep,prep,protocol))] for n in spec.old.NOISES]
                    reference=lookup[spec.digest(stats.robust(model,'clean',rep,prep,protocol))]
                    if all(r['eligible_for_inference'] for r in rows):
                        points=[point(i,r['macro_f1']) for i,r in enumerate(rows)]
                        fig.line(points+[points[0]],COLORS[mi],2)
                        label=f"{model}: clean {reference['macro_f1']:.3f}" if reference['eligible_for_inference'] else model+': clean excluded'
                    else: label=model+': incomplete/excluded profile'
                    fig.text(805,195+mi*42,label,15,COLORS[mi])
                fig.finish('Five synthetic noise mechanisms at 10 dB additive SNR; identical 0-1 axes. Descriptive profile, not equal physical field severity.')

def publication_curves(directory,lookup):
    """One noise per 7-inch panel, minimum embedded PDF font 8.4 pt."""
    for protocol in ('deployment','matched'):
        for prep in spec.PREPS:
            for rep in ('RMS6','RAW12','MULTI18','LATE_SIMPLE','LATE_ORACLE_SNR'):
                for noise in spec.old.NOISES:
                    fig=SVG(directory/f'publication_curve_{protocol}_{prep}_{rep}_{noise}.svg',
                        'Macro-F1 under synthetic noise',720,580)
                    fig.text(45,65,f'{protocol} | {prep} | {rep} | {noise}',14)
                    x,y,w,h=75,110,590,220; axes(fig,x,y,w,h,'')
                    for snr in spec.base.SNR_LEVELS: fig.text(x+(snr-10)/110*(w-70),y+h+24,snr,13,anchor='middle')
                    fig.text(x+w-15,y+h+24,'Clean',13,anchor='middle')
                    fig.text(290,383,'Injected additive SNR (dB)',14,anchor='middle')
                    for mi,model in enumerate(spec.MODELS):
                        points=[]
                        for snr in spec.base.SNR_LEVELS:
                            row=lookup[spec.digest(stats.robust(model,noise+'__'+str(snr),rep,prep,protocol))]
                            if not row['eligible_for_inference']:
                                if points: fig.line(points,COLORS[mi],2); points=[]
                                continue
                            p=(x+(snr-10)/110*(w-70),y+h*(1-row['macro_f1'])); points.append(p); fig.dot(*p,COLORS[mi],3)
                        if points: fig.line(points,COLORS[mi],2)
                        clean=lookup[spec.digest(stats.robust(model,'clean',rep,prep,protocol))]
                        if clean['eligible_for_inference']: fig.dot(x+w-15,y+h*(1-clean['macro_f1']),COLORS[mi],4)
                        xx,yy=70+(mi%2)*330,415+(mi//2)*23
                        fig.line([(xx,yy-4),(xx+24,yy-4)],COLORS[mi],2); fig.text(xx+34,yy,model,13)
                    fig.finish('Family/class standardized. Clean is separate; exclusions leave gaps. True-SNR fusion is oracle, not deployable.')

def publication_radars(directory,lookup):
    for protocol in ('deployment','matched'):
        for prep in spec.PREPS:
            for rep in ('RMS6','RAW12','MULTI18','LATE_SIMPLE'):
                fig=SVG(directory/f'publication_radar_{protocol}_{prep}_{rep}.svg','Severe-noise Macro-F1 profiles',720,680)
                fig.text(45,65,f'{protocol} | {prep} | {rep} | additive SNR 10 dB',14)
                cx,cy,radius=350,275,150
                def point(i,value):
                    a=-math.pi/2+i*2*math.pi/5; return cx+radius*value*math.cos(a),cy+radius*value*math.sin(a)
                for tick in (.2,.4,.6,.8,1):
                    p=[point(i,tick) for i in range(5)]; fig.line(p+[p[0]])
                    fig.text(cx+9,cy-radius*tick+5,f'{tick:.1f}',12)
                for i,noise in enumerate(spec.old.NOISES):
                    fig.line([(cx,cy),point(i,1)]); xx,yy=point(i,1.28); fig.text(xx,yy,noise,12,anchor='middle')
                for mi,model in enumerate(spec.MODELS):
                    values=[lookup[spec.digest(stats.robust(model,n+'__10',rep,prep,protocol))] for n in spec.old.NOISES]
                    if all(r['eligible_for_inference'] for r in values):
                        p=[point(i,r['macro_f1']) for i,r in enumerate(values)]; fig.line(p+[p[0]],COLORS[mi],2)
                    xx,yy=70+(mi%2)*330,500+(mi//2)*24
                    fig.line([(xx,yy-4),(xx+24,yy-4)],COLORS[mi],2); fig.text(xx+34,yy,model,13)
                fig.finish('Identical 0-1 axes. Synthetic noise types need not have equal field severity. Incomplete profiles are omitted, not zero-filled.')

def confusion(directory,lookup,index,output):
    from c1e32_analysis import selected,pooled
    rows=c.read(output.parent/'metadata.json')
    for identity,entry in sorted(index.items()):
        if not selected(entry['case']): continue
        data=pooled(output/entry['path']); _,matrix=stats.score(rows,data['truth'],data['predicted'])
        normalized=matrix/matrix.sum(axis=1,keepdims=True)
        case=entry['case']; title=f"{case['classifier']} | {case['condition']} | {case['representation']}"
        fig=SVG(directory/('publication_confusion_'+identity+'.svg'),title,720,780)
        fig.text(45,64,'Family/class standardized row percentages',14)
        if not entry['eligible_for_inference']: fig.text(45,85,'EXCLUDED numerical warning',14,'#D55E00')
        x0,y0,cell=100,140,45
        for i,label in enumerate(spec.LABELS):
            fig.text(x0+i*cell+cell/2,y0-18,label,13,anchor='middle'); fig.text(x0-15,y0+i*cell+cell/2+5,label,14,anchor='end')
            for j in range(12):
                value=float(normalized[i,j]); shade=255-int(190*value)
                fig.rect(x0+j*cell,y0+i*cell,cell-1,cell-1,f'rgb({shade},{min(255,shade+18)},245)')
                fig.text(x0+j*cell+cell/2,y0+i*cell+cell/2+5,f'{100*value:.0f}',12,anchor='middle')
        fig.finish('Rows: true label. Columns: probability-argmax prediction. All numerical matrices and class supports are saved separately.')

def forests(directory,tests,index):
    for family in sorted({t['family'] for t in tests}):
        members=[t for t in tests if t['family']==family]
        for page,start in enumerate(range(0,len(members),30),1):
            entries=members[start:start+30]; fig=SVG(directory/f'effects_{family}_{page:02}.svg',family,1500,1150)
            left,width=810,580; center=left+width/2
            fig.line([(center,80),(center,1050)],'#333',1,'4,4')
            for i,t in enumerate(entries):
                y=105+i*30; case=index[t['left_case_id']]['case']
                label=f"{case['classifier']} / {case['condition']} / {case['preprocessor']} / {case['representation']} / {case['normalizer']}"
                fig.text(40,y,label,12)
                if t['eligible_for_inference']:
                    color='#0072B2' if t['difference']>=0 else '#D55E00'
                    xx=lambda v:left+(v+1)*width/2
                    fig.line([(xx(t['lower_95']),y-4),(xx(t['upper_95']),y-4)],color,2)
                    fig.dot(xx(t['difference']),y-4,color,4)
                    fig.text(1410,y,f"{100*t['difference']:+.1f}",12,anchor='middle')
                else: fig.text(850,y,'EXCLUDED',12,'#666')
            fig.text(left,1080,'-100',12); fig.text(center,1080,'0',12,anchor='middle'); fig.text(left+width,1080,'+100 pp',12,anchor='end')
            fig.finish('Macro-F1 difference and pointwise 95% family-cluster interval. Holm-adjusted tests and Monte Carlo precision flags are in the paired table.')

def scaling(directory,lookup):
    for rep in stats.base_reps():
        fig=SVG(directory/f'scaling_clean_{rep}.svg','Train-only scaling on clean data | '+rep,1450,800)
        for j,scale in enumerate(spec.NORMALIZERS): fig.text(280+j*185,100,scale,12)
        for i,model in enumerate(spec.MODELS):
            yy=130+i*67; fig.text(35,yy+35,model,15)
            for j,scale in enumerate(spec.NORMALIZERS):
                case=stats.robust(model,'clean',rep); case['normalizer']=scale
                if scale!=spec.default_scale(model): case['study']='scaling'
                row=lookup[spec.digest(case)]; shade=245-int(130*row['macro_f1'])
                color=f'rgb({shade},{min(255,shade+20)},245)' if row['eligible_for_inference'] else '#ddd'
                fig.rect(270+j*185,yy,175,57,color)
                fig.text(355+j*185,yy+35,f"{row['macro_f1']:.3f}" if row['eligible_for_inference'] else 'EXCLUDED',15,anchor='middle')
        fig.finish('Same family folds and learner settings. Every scaler is fitted only within the training partition; no test-set selection.')

def dataset_figures(root,directory,rows):
    counts={label:sum(r['fault']==label for r in rows) for label in spec.LABELS}
    fig=SVG(directory/'class_counts.svg','Acquired class counts',1300,650)
    for i,label in enumerate(spec.LABELS):
        xx=85+i*95; height=counts[label]*2.5; fig.rect(xx,500-height,65,height,'#0072B2'); fig.text(xx+32,525,label,14,anchor='middle'); fig.text(xx+32,485-height,counts[label],14,anchor='middle')
    fig.finish('Dedicated healthy acquisitions; 156 healthy and 153 per fault class. These are not independent replicated networks.')
    profiles={tuple(r['factors']):(r['fault']=='noFault' and not r['healthy_anchor']) for r in rows}
    fig=SVG(directory/'load_coverage.svg','Load factors and dedicated healthy enrichment',1450,650)
    for panel,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
        x,y,w,h=80+panel*470,120,350,350
        fig.rect(x,y,w,h,'white','#aaa'); fig.text(x,y-25,f'B{[5,6,8][a]} versus B{[5,6,8][b]}',18)
        for factors,extra in profiles.items(): fig.dot(x+(factors[a]-.75)/.5*w,y+h-(factors[b]-.75)/.5*h,'#D55E00' if extra else '#0072B2',3)
        fig.text(x,y+h+30,'.75',13); fig.text(x+w,y+h+30,'1.25',13,anchor='end')
    fig.text(80,555,'Blue: shared anchor support. Orange: healthy-only parameter points. Projections can overlap.',15)
    fig.finish('Three load ratings vary from 75% to 125% at fixed Q/P; this is a sensitivity envelope, not certified network capacity.')
    with np.load(root/'dataset_analysis/healthy_diagnostic_examples.npz',allow_pickle=False) as d:
        wave=d['waveforms']
        fig=SVG(directory/'healthy_waveform_overlays.svg','Dedicated healthy waveforms | fixed first eight cases',1450,950)
        for channel in range(6):
            x=80+(channel%2)*710; y=110+(channel//2)*255; w=580; h=170
            values=wave[:,:,channel]; lo=float(values.min()); hi=float(values.max()); span=max(hi-lo,1e-12)
            fig.text(x,y-20,spec.base.CHANNELS[channel],16)
            for i,v in enumerate(values):
                dec=v[::16]; fig.line([(x+j/(len(dec)-1)*w,y+h-(value-lo)/span*h) for j,value in enumerate(dec)],COLORS[i],1)
            fig.text(x,y+h+22,'0 ms',12); fig.text(x+w,y+h+22,'200 ms',12,anchor='end')
        fig.finish('Fixed case selection, not chosen for separation. Raw phase-sensitive and RMS nearest-neighbor distances are reported for all healthy cases.')
    with np.load(root/'final_analysis/exploratory_pca.npz',allow_pickle=False) as d:
        scores=d['scores']; fig=SVG(directory/'exploratory_feature_pca.svg','Exploratory descriptor projection',1300,850)
        lo=scores[:,:2].min(axis=0); span=np.maximum(scores[:,:2].max(axis=0)-lo,1e-12)
        for r,value in zip(rows,scores):
            label=spec.LABELS.index(r['fault']); fig.dot(90+(value[0]-lo[0])/span[0]*900,710-(value[1]-lo[1])/span[1]*580,COLORS[label%8],2 if label<8 else 4)
        for i,label in enumerate(spec.LABELS): fig.text(1040,110+i*40,label,15,COLORS[i%8])
        fig.finish('Descriptive full-cohort PCA only. It is not fed into any trained model and does not validate independent test generalization.')

def tpi_figure(directory,lookup,summaries):
    fig=SVG(directory/'energy_ratio_and_classification.svg','Signal energy ratios and held-out classification effects',1450,1000)
    for panel,noise in enumerate(('lowfreq','mixed_gain_additive')):
        x=80+panel*710; y=150; w=580; h=280; points=[]
        for pi,prep in enumerate(spec.PREPS):
            for snr in spec.base.SNR_LEVELS:
                condition=noise+'__'+str(snr)
                vals=[r['median_tpi_noisy'] for r in summaries if r['condition']==condition and r['preprocessor']==prep and r['operator']=='DWT_D3_2500_5000Hz']
                effects=[]
                for model in spec.MODELS:
                    a=lookup[spec.digest(stats.robust(model,condition,prep=prep))]; b=lookup[spec.digest(stats.robust(model,condition))]
                    if a['eligible_for_inference'] and b['eligible_for_inference']: effects.append(a['macro_f1']-b['macro_f1'])
                if vals and effects: points.append((math.log10(max(float(np.median(vals)),1e-12)),float(np.median(effects)),pi,snr))
        if not points: fig.text(x,y,'No eligible paired effect',16); continue
        xmin=min(p[0] for p in points); xmax=max(p[0] for p in points); ymin=min(-.01,min(p[1] for p in points)); ymax=max(.01,max(p[1] for p in points))
        if xmax-xmin<1e-12: xmax=xmin+1
        fig.rect(x,y,w,h,'white','#aaa'); fig.text(x,y-35,noise,18)
        yy=lambda v:y+h-(v-ymin)/(ymax-ymin)*h
        fig.line([(x,yy(0)),(x+w,yy(0))],'#888',1,'4,4')
        for a,b,pi,snr in points:
            xx=x+(a-xmin)/(xmax-xmin)*w; yv=yy(b); fig.dot(xx,yv,COLORS[pi],4); fig.text(xx+5,yv-5,str(snr),10)
        fig.text(x,y+h+30,f'log10 energy ratio [{xmin:.2g}, {xmax:.2g}]',13)
        fig.text(x,y+h+55,f'Filter-minus-none Macro-F1 range [{ymin:.3f}, {ymax:.3f}]',13)
    for pi,prep in enumerate(spec.PREPS): fig.text(80+pi*260,650,prep,15,COLORS[pi])
    fig.text(80,725,'DWT D3 primary operator. Points label SNR; other operators and clean/error references are in the diagnostic tables.',15)
    fig.text(80,765,'Shared noise severity and condition dependence confound this descriptive association.',15)
    fig.finish('Energy retention is not diagnostic-information preservation. This plot does not justify selecting a denoiser by maximizing TPI.')

def render(root,output,metrics,index,rows,tests,summaries):
    directory=output/'figures'; directory.mkdir(); lookup=metric_lookup(metrics)
    curves(directory,lookup); radars(directory,lookup); publication_curves(directory,lookup); publication_radars(directory,lookup)
    confusion(directory,lookup,index,output)
    forests(directory,tests,index); scaling(directory,lookup); dataset_figures(root,directory,rows); tpi_figure(directory,lookup,summaries)
    coverage=[]
    for p in sorted(directory.glob('*.svg')):
        tree=ET.parse(p); c.require(tree.getroot().tag.endswith('svg'),'Invalid SVG')
        coverage.append({'file':p.name,'sha256':c.sha(p),'format':'SVG','status':'GENERATED_XML_CHECKED_REQUIRES_VISUAL_PUBLICATION_REVIEW'})
    c.table(output/'figure_manifest.csv',coverage)
    with (output/'figure_atlas.html').open('x',encoding='utf-8') as stream:
        stream.write('<!doctype html><meta charset="utf-8"><title>Scientific figure atlas</title><h1>Scientific figure atlas</h1><p>Fixed configurations, family-based evaluation. Visual publication review remains required.</p>')
        for row in coverage: stream.write('<h2>'+escape(row['file'])+'</h2><object width="100%" data="figures/'+escape(row['file'])+'" type="image/svg+xml"></object>')
    checklist=c.read(root/'checklist.json')
    for row in checklist['rows']:
        if row['destination']=='C': continue
        row['status']='PENDING_EVIDENCE_MAPPING_AND_JOINT_SCIENTIFIC_REVIEW'
        row['artifacts']=['final_analysis/analysis_receipt.json','final_analysis/metrics.csv','final_analysis/figure_manifest.csv']
        row['verified']=False
    checklist['pipeline_launched']=True; checklist['new_cohort_fits_executed']=spec.scope()['fit_budget']
    checklist['status']='COMPUTATIONAL_OUTPUTS_COMPLETE_NOT_ALL_82_ITEMS_VERIFIED'
    c.write(output/'checklist_execution_status.json',checklist)
