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

fig=plt.figure(figsize=(7.25,3.55));gs=fig.add_gridspec(2,2,left=.16,right=.98,top=.82,bottom=.12,
    width_ratios=[1,1],height_ratios=[3.5,1],wspace=.20,hspace=.65)
ax=fig.add_subplot(gs[1,:]);audit=json.loads((DATA/'abc_abcg_audit_summary.json').read_text(encoding='utf-8-sig'))
for i,(key,label) in enumerate([('full_I_relative_l2','Currents'),('full_V_relative_l2','Voltages')]):
    d=audit['distributions'][key]
    ax.plot([d['min'],d['max']],[i,i],c='#0072B2',lw=2)
    ax.plot([d['median']],[i],marker='D',c='#0072B2',ms=4)
ax.set_xscale('log');ax.set_xlim(2e-12,5e-10);ax.set_ylim(-.6,1.6);ax.set_yticks([0,1]);ax.set_yticklabels(['Currents','Voltages'])
ax.set_xticks([1e-11,1e-10]);ax.grid(axis='x',alpha=.3,lw=.4)
ax.set_title('(c) ABC–ABCG: paired relative difference (153 pairs)',loc='left',fontsize=8.5,fontweight='bold',pad=6)
export(fig,'s10_fig04_resnet_errors_audit_v1')
