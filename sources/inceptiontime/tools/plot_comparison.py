"""Graphics-only process: read verified tables; never load model predictions or fit."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys

CASES=[('shared_clean','clean'),('clean_trained','awgn__20'),('noise_matched','awgn__20'),
       ('clean_trained','mixed_gain_additive__10'),('noise_matched','mixed_gain_additive__10')]
LABELS=['Clean','AWGN20\nClean-trained','AWGN20\nNoise-matched','Mixed10\nClean-trained','Mixed10\nNoise-matched']
CLASSES=['noFault','ab','ac','bc','abc','ag','bg','cg','abg','acg','bcg','abcg']


def read(folder,name):
    with (folder/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def plot(folder,graphics_deps):
    if graphics_deps:sys.path.insert(0,str(graphics_deps))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    folder=Path(folder)
    receipt=json.loads((folder/'analysis_receipt.json').read_text())
    if receipt['status']!='COMPLETE_DESCRIPTIVE_ANALYSIS_FOR_AUTHOR_REVIEW':
        raise ValueError('Plots require a complete analysis receipt')
    out=folder/'figures';out.mkdir(exist_ok=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'legend.fontsize':7,'pdf.fonttype':42,'svg.fonttype':'none'})
    figures=[]
    def save(fig,name,sources):
        for extension in ('pdf','svg','png'):fig.savefig(out/(name+'.'+extension),dpi=220,bbox_inches='tight')
        plt.close(fig)
        figures.append({'name':name,'sources':sources,'actual_PDF_visual_QA':'PENDING_AUTHOR_OUTPUT_REVIEW'})
    rows=read(folder,'metrics_per_run.csv');summary=read(folder,'metrics_summary.csv')
    colors={'RMS6':'#888888','RAW12':'#3182bd','MULTI18':'#56a85b'}
    neural={'ResNet':('#b24a33','D',-.13),'InceptionTime':('#6a3d9a','*',.13)}
    fig,axes=plt.subplots(3,1,figsize=(7.25,6.5),sharex=True)
    for ax,metric,title in zip(axes,['macro_f1','healthy_false_alarm_rate','missed_fault_rate'],
        ['(a) Macro-F1 (%) — higher is better','(b) Healthy false alarms (%) — lower is better','(c) Missed faults (%) — lower is better']):
        for x,(exposure,condition) in enumerate(CASES):
            subset=[r for r in rows if r['exposure']==exposure and r['condition']==condition]
            ml=sorted([r for r in subset if r['model_family']=='ML'],key=lambda r:r['model'])
            for j,r in enumerate(ml):
                rep=r['representation'];offset=(j-11.5)*.020
                ax.scatter(x+offset,100*float(r[metric]),s=11,c=colors[rep],alpha=.65,zorder=2,linewidths=0)
            for model,(color,marker,offset) in neural.items():
                values=[100*float(r[metric]) for r in subset if r['model']==model]
                for j,value in enumerate(values):ax.scatter(x+offset+(j-1)*.028,value,s=16,facecolors='none',edgecolors=color,zorder=4)
                mean=next(r for r in summary if (r['model'],r['exposure'],r['condition'])==(model,exposure,condition))
                ax.scatter(x+offset,100*float(mean[metric+'_mean']),s=55 if marker=='*' else 24,c=color,marker=marker,zorder=5)
        ax.set_title(title,loc='left');ax.set_ylim(-2,102);ax.set_yticks([0,25,50,75,100]);ax.grid(axis='y',alpha=.2)
        if metric=='missed_fault_rate':
            upper=max(1,min(100,max(100*float(r[metric]) for r in rows)*1.12))
            ax.set_ylim(-.03*upper,upper);ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.spines[['top','right']].set_visible(False)
    axes[-1].set_xticks(range(5),LABELS);axes[-1].tick_params(axis='x',labelsize=7)
    legend=[Line2D([0],[0],marker='o',ls='',color=c,label='ML '+r,markersize=4) for r,c in colors.items()]
    legend += [Line2D([0],[0],marker=m,ls='',color=c,label=n+' mean',markersize=6) for n,(c,m,_) in neural.items()]
    legend += [Line2D([0],[0],marker='o',ls='',markerfacecolor='none',color='#444444',label='Individual seed',markersize=4)]
    fig.suptitle('Waveform models and fixed descriptor pipelines',fontsize=11,y=.995)
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.96),ncol=3,frameon=False)
    fig.subplots_adjust(top=.86,hspace=.28,bottom=.12)
    save(fig,'c1e43_fig01_model_comparison_v1',['metrics_per_run.csv','metrics_summary.csv'])
    errors=read(folder,'error_partition_group_means.csv')
    parts=['healthy_to_fault_mass_pp','fault_to_healthy_mass_pp','abc_pair_mass_pp','other_fault_error_mass_pp']
    names=['Healthy → fault','Fault → healthy','ABC ↔ ABCG','Other fault errors']
    palette=['#d6616b','#fdae61','#8073ac','#80cdc1']
    fig,ax=plt.subplots(figsize=(7.25,4.0))
    for i,(exposure,condition) in enumerate(CASES):
        for j,model in enumerate(['ML','ResNet','InceptionTime']):
            row=next(r for r in errors if (r['model_group'],r['exposure'],r['condition'])==(model,exposure,condition));bottom=0
            for p,color in zip(parts,palette):
                value=float(row[p]);ax.bar(i+(j-1)*.24,value,width=.21,bottom=bottom,color=color);bottom+=value
            if abs(bottom-float(row['total_ba_error_pp']))>1e-8:raise ValueError('Error parts no longer sum to total')
            ax.text(i+(j-1)*.24,bottom+1,{'ML':'ML','ResNet':'R','InceptionTime':'I'}[model],ha='center',fontsize=6)
    ax.set_xticks(range(5),LABELS);ax.tick_params(axis='x',labelsize=7)
    ax.set_ylim(0,100);ax.set_ylabel('Contribution to balanced-accuracy error (pp)')
    ax.set_title('Where classification errors arise',loc='left',fontsize=11)
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.legend([plt.Rectangle((0,0),1,1,color=c) for c in palette],names,ncol=2,loc='upper left',frameon=False)
    fig.text(.5,.01,'ML: mean of 24 pipelines. R: ResNet mean of 3 seeds. I: InceptionTime mean of 3 seeds.',ha='center',fontsize=7)
    fig.subplots_adjust(bottom=.22,top=.9)
    save(fig,'c1e43_fig02_error_partition_v1',['error_partition_group_means.csv'])
    classes=read(folder,'class_recall_difference.csv');matrix=np.zeros((12,5))
    for i,label in enumerate(CLASSES):
        for j,(exposure,condition) in enumerate(CASES):
            r=next(r for r in classes if (r['exposure'],r['condition'],r['class'])==(exposure,condition,label))
            matrix[i,j]=float(r['recall_difference_pp'])
    fig,ax=plt.subplots(figsize=(6.1,5.0));limit=max(1,float(np.abs(matrix).max()))
    im=ax.imshow(matrix,cmap='RdBu',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(5),LABELS);ax.set_yticks(range(12),['Healthy' if x=='noFault' else x.upper() for x in CLASSES]);ax.tick_params(labelsize=7)
    for i in range(12):
        for j in range(5):ax.text(j,i,f'{matrix[i,j]:+.1f}',ha='center',va='center',fontsize=7,color='white' if abs(matrix[i,j])>.65*limit else 'black')
    ax.set_title('Class recall: InceptionTime minus ResNet',loc='left',fontsize=10)
    fig.colorbar(im,ax=ax,pad=.025,label='Difference of three-seed mean recalls (pp)')
    fig.tight_layout()
    save(fig,'c1e43_fig03_class_recall_difference_v1',['class_recall_difference.csv'])
    (out/'figures_manifest.json').write_text(json.dumps({'status':'EXPORTED_ACTUAL_PDF_VISUAL_QA_PENDING',
        'figures':figures,'matplotlib':matplotlib.__version__,'numpy_graphics_only':np.__version__,
        'scientific_predictions_or_metrics_recomputed':False,
        'source_tables_sha256':{name:hashlib.sha256((folder/name).read_bytes()).hexdigest()
            for name in sorted({s for f in figures for s in f['sources']})}},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'figures':3,'formats':['pdf','svg','png'],'actual_PDF_visual_QA':'PENDING'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--graphics-deps',type=Path)
    args=parser.parse_args()
    plot(args.analysis,args.graphics_deps)
