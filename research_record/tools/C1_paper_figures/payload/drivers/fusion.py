"""Seven vector composites from saved, reconciled E32/E33 results only."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
from statistics import median

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--deps',type=Path);args=parser.parse_args()
if args.deps:sys.path.insert(0,str(args.deps))
os.environ['MPLCONFIGDIR']=str(ROOT/'qa/mplconfig')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm

OUT=ROOT/'intermediate';OUT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,
                     'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':7,'pdf.fonttype':42,
                     'svg.fonttype':'none','svg.hashsalt':'C1SectionIV20261103','axes.spines.top':False,
                     'axes.spines.right':False,'axes.linewidth':.65,'grid.linewidth':.45})
NOISE=['awgn','lowfreq','impulsive','gain_plus_awgn','mixed_gain_additive']
NAMES=['AWGN','Low-frequency','Impulsive','Gain + AWGN','Mixed']
SHORT=['AWGN','LF','Imp','Gain','Mix']
SNRS=[10,20,40,60,120]
REPS=['RMS6','RAW12','MULTI18']
COLORS=['#0072B2','#D55E00','#009E73']
PREPS=['median','savgol','wavelet_soft','wavelet_hard']
PNAMES=['Median','SG','Soft','Hard'];PCOLORS=['#0072B2','#D55E00','#009E73','#CC79A7']
MODELS=['logistic_regression','random_forest','naive_bayes','knn','decision_tree','svm','gradient_boosting','mlp']
FIGURES=[]



def rows(name):
    with (ROOT/'data'/name).open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))

def save(fig,name,sources):
    for extension in ['pdf','svg']:
        metadata={'CreationDate':None,'ModDate':None} if extension=='pdf' else {'Date':None}
        fig.savefig(OUT/(name+'.'+extension),metadata=metadata,facecolor='white')
    svg=OUT/(name+'.svg');svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8',newline='\n')
    FIGURES.append({'name':name,'width_inches':float(fig.get_figwidth()),'height_inches':float(fig.get_figheight()),
                    'sources':sources,'pdf_sha256':hashlib.sha256((OUT/(name+'.pdf')).read_bytes()).hexdigest(),
                    'svg_sha256':hashlib.sha256(svg.read_bytes()).hexdigest(),'actual_pdf_visual_QA':'PENDING'})
    plt.close(fig)

def fusion():
    data=rows('fusion_effects_verified.csv');contrasts=rows('paired_contrasts_verified.csv')
    fig=plt.figure(figsize=(7.25,3.75));gs=fig.add_gridspec(2,5,height_ratios=[1.45,1.2],left=.12,right=.98,top=.86,bottom=.12,hspace=.56,wspace=.20)
    curves=[('early','RMS6','#0072B2','-'),('early','RAW12','#0072B2','--'),('late','RMS6','#D55E00','-'),('late','RAW12','#D55E00','--'),('oracle','RMS6','#666666',':')]
    for j,(n,title) in enumerate(zip(NOISE,SHORT)):
        ax=fig.add_subplot(gs[0,j])
        for method,constituent,color,style in curves:
            values=[next(float(r['median_paired_change_pp']) for r in data if r['noise']==n and int(r['snr'])==s and r['fusion']==method and r['constituent']==constituent) for s in SNRS]
            ax.plot(SNRS,values,color=color,ls=style,lw=1.15)
        ax.axhline(0,color='black',lw=.7);ax.set_title(title);ax.set_xlim(8,122);ax.set_xticks([10,60,120]);ax.set_ylim(-80,80);ax.grid(axis='y',color='#dddddd')
        if j:ax.set_yticklabels([])
        else:ax.set_ylabel('Fusion minus constituent\nMacro-F1 (points)')
    combos=[('early','RMS6','early_vs_RMS6','Early - RMS6'),('early','RAW12','early_vs_RAW12','Early - RAW12'),('late','RMS6','late_vs_RMS6','Late - RMS6'),('late','RAW12','late_vs_RAW12','Late - RAW12')]
    z=np.array([[next(float(r['median_paired_change_pp']) for r in data if r['noise']==n and r['snr']=='10' and r['fusion']==m and r['constituent']==c) for n in NOISE] for m,c,_,_ in combos])
    ax=fig.add_subplot(gs[1,:]);im=ax.imshow(z,cmap='RdBu',vmin=-80,vmax=80,aspect='auto');ax.set_yticks(range(4),[v[3] for v in combos]);ax.set_xticks(range(5),SHORT);ax.set_title('At 10 dB: paired medians and supported individual directions',fontsize=8,pad=6)
    for i,(_,_,family,_) in enumerate(combos):
        for j,n in enumerate(NOISE):
            rs=[r for r in contrasts if r['family']==family and r['left_condition']==f'{n}__10'];assert len(rs)==8
            pos=sum(r['interpretation']=='SUPPORTED_POSITIVE_DIFFERENCE' for r in rs);neg=sum(r['interpretation']=='SUPPORTED_NEGATIVE_DIFFERENCE' for r in rs);mc=sum(r['interpretation']=='MONTE_CARLO_PRECISION_REVIEW' for r in rs)
            value=z[i,j];label=f'{value:+.1f}\n{pos}+ / {neg}-'+(f'  ?{mc}' if mc else '')
            ax.text(j,i,label,ha='center',va='center',fontsize=6.7,color='white' if abs(value)>50 else 'black',linespacing=1.15)
    handles=[Line2D([],[],color=c,ls=s,label=('Oracle - RMS6' if m=='oracle' else m.title()+' - '+r),lw=1.2) for m,r,c,s in curves]
    fig.legend(handles=handles,ncol=3,loc='upper center',frameon=False,bbox_to_anchor=(.55,1.015),fontsize=6.8)
    fig.text(.54,.012,'Upper panels: additive SNR (dB). Lower counts: supported positive/negative cases; ? = precision review.',ha='center',fontsize=6.7)
    save(fig,'fusion_snr_and_constituents',['fusion_effects_verified.csv','paired_contrasts_verified.csv'])

fusion()
