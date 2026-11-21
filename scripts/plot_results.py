"""Plot the current deposited aggregate AUCs; no clinical records needed."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from npc_eitn.io import output_dir

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=Path(__file__).resolve().parents[1]
    d=pd.read_csv(r/'results/prediction_results.csv');d=d.loc[d['repeat'].eq(0)]
    out=output_dir(a.output);fig,axes=plt.subplots(1,2,figsize=(8,3.2),sharey=True)
    for ax,cohort,title in zip(axes,['internal','external'],['Internal OOF','Frozen external']):
        for h,offset,color in [(36,-.08,'#0072B2'),(60,.08,'#D55E00')]:
            s=d.loc[d.cohort.eq(cohort)&d.horizon_months.eq(h)].set_index('model').reindex(['M1','M2','M3','M4'])
            x=[v+offset for v in range(4)];y=s.ipcw_auc.to_numpy()
            ax.errorbar(x,y,yerr=[y-s.auc_ci_low.to_numpy(),s.auc_ci_high.to_numpy()-y],fmt='o',capsize=3,color=color,label=f'{h} months')
        ax.set(xticks=range(4),xticklabels=['M1','M2','M3','M4'],title=title,ylim=(.3,1.02))
        ax.spines[['top','right']].set_visible(False)
        ax.legend(frameon=False,loc='lower right')
    axes[0].set_ylabel('IPCW AUC');fig.tight_layout();fig.savefig(out/'auc_summary.png',dpi=220);fig.savefig(out/'auc_summary.svg');plt.close(fig)
