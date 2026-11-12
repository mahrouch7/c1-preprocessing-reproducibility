"""Plot frozen verified scores; no fitting, simulation or hypothesis testing.

Run with the science Python and a graphics-only dependency directory. The
source CSVs are retained beside this script's project. No test-driven selection.
"""
from pathlib import Path
import sys, csv, json, hashlib, argparse

parser=argparse.ArgumentParser()
parser.add_argument('--graphics-deps',type=Path)
args=parser.parse_args()
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P=Path(__file__).resolve().parents[1]
DATA=P/'data';OUT=P/'intermediate'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,
    'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7.5,
    'ytick.labelsize':7.5,'legend.fontsize':7.5,'pdf.fonttype':42,
    'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})

def read(name):
    with (DATA/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def export(fig,name):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/(name+'.'+ext),dpi=180,facecolor='white')
    plt.close(fig)

noises=['awgn','lowfreq','impulsive','gain_plus_awgn','mixed_gain_additive']
names=['AWGN','Low-frequency','Impulsive','Gain + AWGN','Mixed']
colors={'RMS6':'#0072B2','RAW12':'#D55E00','MULTI18':'#009E73'}
rows=read('robustness_curves_verified.csv')
den=read('denoising_effects_verified.csv')
fig=plt.figure(figsize=(7.25,3.45))
gs=fig.add_gridspec(2,5,left=.075,right=.992,top=.76,bottom=.13,
    hspace=.60,wspace=.23,height_ratios=[1.15,1])
fig.suptitle('Classification, Training Exposure and Filtering across Noise Levels',
             y=.985,fontsize=10,fontweight='bold')
rep_handles=[Line2D([],[],color=v,lw=1.6,label=k) for k,v in colors.items()]
rep_handles += [Line2D([],[],color='.25',lw=1.2,label='Clean-trained'),
                Line2D([],[],color='.25',ls='--',lw=1.2,label='Noise-matched')]
fig.legend(handles=rep_handles,loc='upper center',bbox_to_anchor=(.52,.955),ncol=5,
           frameon=False,handlelength=1.8,columnspacing=1.2)
fig.text(.012,.845,'(a) Unfiltered performance',fontsize=8.5,fontweight='bold')
filter_colors={'median':'#0072B2','savgol':'#D55E00','wavelet_soft':'#009E73','wavelet_hard':'#CC79A7'}
filter_labels={'median':'Median','savgol':'SG','wavelet_soft':'Soft','wavelet_hard':'Hard'}
for i,noise in enumerate(noises):
    ax=fig.add_subplot(gs[0,i]);ax.set_title(names[i],pad=4)
    for rep,col in colors.items():
        for protocol,ls in [('deployment','-'),('matched','--')]:
            rr=sorted((r for r in rows if r['noise']==noise and r['representation']==rep and r['protocol']==protocol),key=lambda x:int(x['snr']))
            assert len(rr)==5 and all(r['n']=='8' for r in rr)
            ax.plot([int(r['snr']) for r in rr],[float(r['median_macro_f1_pct']) for r in rr],ls=ls,c=col,lw=1.25,marker='o',ms=2)
    ax.set_ylim(0,103);ax.set_xlim(8,123);ax.set_xticks([10,60,120]);ax.set_yticks([0,50,100]);ax.grid(axis='y',lw=.35,alpha=.4)
    ax.tick_params(labelbottom=False)
    if i==0:ax.set_ylabel('Macro-F1 (%)')
    else:ax.tick_params(labelleft=False)
    ax=fig.add_subplot(gs[1,i])
    for fil,col in filter_colors.items():
        rr=sorted((r for r in den if r['noise']==noise and r['preprocessor']==fil),key=lambda x:int(x['snr']))
        assert len(rr)==5 and all(r['n']=='24' for r in rr)
        ax.plot([int(r['snr']) for r in rr],[float(r['median_paired_change_pp']) for r in rr],c=col,lw=1.25,marker='o',ms=2)
    ax.axhline(0,c='.25',lw=.5);ax.set_xlim(8,123);ax.set_ylim(-6,44);ax.set_xticks([10,60,120]);ax.set_yticks([0,20,40]);ax.grid(axis='y',lw=.35,alpha=.4)
    if i==0:ax.set_ylabel('Macro-F1 change\n(points)')
    else:ax.tick_params(labelleft=False)
fig.text(.012,.425,'(b) Filtering gain after clean training',fontsize=8.5,fontweight='bold')
fig.legend(handles=[Line2D([],[],color=c,lw=1.5,label=filter_labels[k]) for k,c in filter_colors.items()],
    loc='center right',bbox_to_anchor=(.998,.44),ncol=4,frameon=False,handlelength=1.5,columnspacing=1)
fig.text(.54,.025,'Additive SNR (dB); gain error persists at high SNR',ha='center',fontsize=8)
export(fig,'s10_fig02_snr_exposure_filtering_v2')

