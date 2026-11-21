"""Fixed-prediction evaluation; stratified bootstrap re-estimates censoring."""
import numpy as np
import pandas as pd
from ._ridge import weighted_metrics,horizon_labels_weights
from .splits import fast_auc

PAIRS=(('M4','M3'),('M2','M3'),('M4','M2'),('M2','M1'),('M4','M1'))

def evaluate(frame,predictions,horizon,*,bootstrap=1000,seed=20261107):
    names=list(predictions)
    matrix=np.column_stack([predictions[n] for n in names])
    if matrix.shape!=(len(frame),len(names)) or not np.isfinite(matrix).all() or ((matrix<0)|(matrix>1)).any():
        raise ValueError('Complete finite probabilities in [0,1] are required')
    results=pd.DataFrame([dict(model=n,**weighted_metrics(frame,matrix[:,j],horizon)) for j,n in enumerate(names)])
    differences=[]
    if bootstrap:
        rng=np.random.default_rng(seed)
        strata=[np.flatnonzero(frame.center.to_numpy()==c) for c in sorted(frame.center.unique())]
        aa=np.full((bootstrap,len(names)),np.nan);bb=np.full_like(aa,np.nan)
        for b in range(bootstrap):
            ii=np.concatenate([rng.choice(g,len(g),replace=True) for g in strata])
            sample=frame.iloc[ii].reset_index(drop=True);y,w=horizon_labels_weights(sample,horizon);known=np.isfinite(y)
            for j in range(len(names)):
                p=matrix[ii,j][known]
                aa[b,j]=fast_auc(y[known],p,w[known]);bb[b,j]=np.sum(w[known]*(y[known]-p)**2)/len(sample)
        for j,n in enumerate(names):
            results.loc[results.model.eq(n),['auc_ci_low','auc_ci_high']]=np.nanquantile(aa[:,j],[.025,.975])
            results.loc[results.model.eq(n),['brier_ci_low','brier_ci_high']]=np.nanquantile(bb[:,j],[.025,.975])
        lookup=results.set_index('model')
        for high,low in PAIRS:
            if high not in names or low not in names:continue
            j,k=names.index(high),names.index(low)
            aci=np.nanquantile(aa[:,j]-aa[:,k],[.025,.975]);bci=np.nanquantile(bb[:,j]-bb[:,k],[.025,.975])
            differences.append(dict(comparison=high+' minus '+low,interpretation='paired_EITN_increment' if (high,low)==('M4','M3') else 'whole_pipeline_comparison',
                delta_auc=lookup.loc[high,'ipcw_auc']-lookup.loc[low,'ipcw_auc'],auc_ci_low=aci[0],auc_ci_high=aci[1],
                delta_brier=lookup.loc[high,'ipcw_brier']-lookup.loc[low,'ipcw_brier'],brier_ci_low=bci[0],brier_ci_high=bci[1],
                bootstrap_replicates=bootstrap,valid_auc_replicates=int(np.isfinite(aa[:,j]-aa[:,k]).sum())))
    return results,pd.DataFrame(differences)
