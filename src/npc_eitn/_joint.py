"""Current M2: EITN enters screening and Elastic Net as an ordinary candidate."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy import special, stats
from sklearn.metrics import log_loss
from . import _elastic as legacy
EITN=legacy.FIXED_EITN
class JointPrep:
    def __init__(self,columns,horizon):self.columns=list(columns);self.horizon=horizon

    def fit(self,frame):
        self.preprocessor=legacy.FoldPreprocessor(self.columns,legacy.ModelSpec('joint_M2',include_eitn=True,image_mode='ridge')).fit(frame)
        x=self.preprocessor.transform(frame);assert self.preprocessor.summary['fixed_columns'][4]==EITN
        y,w=legacy.horizon_labels_weights(frame,self.horizon,frame)
        self.clinical_initial=legacy.clinical_mle(x[:,:4],y,w)
        names=[EITN]+list(self.preprocessor.image_cols_)
        raw_e=pd.to_numeric(frame[EITN],errors='coerce').replace([np.inf,-np.inf],np.nan)
        eitn_eligible=raw_e.isna().mean()<=.30 and raw_e.notna().any() and raw_e.fillna(raw_e.median()).var(ddof=0)>1e-8
        eligible=[i for i in sorted(range(len(names)),key=lambda i:names[i]) if i!=0 or eitn_eligible]
        frequency=[i for i in eligible if pd.Series(x[:,4+i]).value_counts(normalize=True).iloc[0]<.995]
        rank=stats.rankdata(x[:,np.array(frequency)+4],axis=0)
        rank-=rank.mean(axis=0);rank/=np.sqrt(np.sum(rank**2,axis=0))
        corr=np.clip(rank.T@rank,-1,1);kept=[]
        for j in range(len(frequency)):
            if not kept or np.max(abs(corr[j,kept]))<.9:kept.append(j)
        decorrelated=np.asarray([frequency[j] for j in kept],dtype=int)
        ok=np.isfinite(y);weight=w[ok]/w[ok].sum()
        residual=weight*(special.expit(self.clinical_initial[0]+x[ok,:4]@self.clinical_initial[1:])-y[ok])
        gradient=x[ok][:,decorrelated+4].T@residual
        selected=np.argsort(-abs(gradient),kind='stable')[:30]
        self.selected_indices=decorrelated[selected]
        self.selected_names=[names[i] for i in self.selected_indices]
        self.max_gradient=float(np.max(abs(gradient[selected])))
        self.training_patient_ids=frame.patient_id.tolist()
        self.summary=dict(raw_penalized_candidates=len(self.columns)+1,after_missing_variance=len(eligible),
            after_frequency=len(frequency),after_correlation=len(decorrelated),after_screen=len(self.selected_indices),
            eitn_after_missing_variance=bool(eitn_eligible),eitn_after_frequency=0 in frequency,
            eitn_after_correlation=bool(0 in decorrelated),eitn_screened=EITN in self.selected_names,
            selected_candidate_names='|'.join(self.selected_names),
            training_patient_hash=hashlib.sha256('\n'.join(frame.patient_id).encode()).hexdigest())
        assert len(self.selected_indices)==30 and self.max_gradient>0
        assert all(c==EITN or not c.startswith(('formal__','legacy__')) for c in self.selected_names)
        return self

    def transform(self,frame,include=False):
        x=self.preprocessor.transform(frame)
        return x[:,:4],x[:,self.selected_indices+4]


def tune(frame,columns,h,splits, *, alpha_grid=legacy.ALPHAS, ratio_grid=legacy.RATIOS):
    losses={(a,r):[] for a in alpha_grid for r in ratio_grid}
    for ti,vi in splits:
        t,v=frame.iloc[ti].reset_index(drop=True),frame.iloc[vi].reset_index(drop=True)
        assert not set(t.patient_id)&set(v.patient_id)
        prep=JointPrep(columns,h).fit(t);tc,tx=prep.transform(t);vc,vx=prep.transform(v)
        ty,tw=legacy.horizon_labels_weights(t,h,t);vy,vw=legacy.horizon_labels_weights(v,h,t);ok=np.isfinite(vy)
        for a in alpha_grid:
            initial=None
            for ratio in ratio_grid:
                if initial is None:
                    initial=np.zeros(5+2*tx.shape[1]);initial[:5]=prep.clinical_initial
                fit=legacy.fit_elastic(tc,tx,ty,tw,ratio*prep.max_gradient/a,a,initial=initial)
                initial=fit['split_theta'];coef=fit['coef']
                p=special.expit(coef[0]+vc@coef[1:5]+vx@coef[5:])
                losses[(a,ratio)].append(float(log_loss(vy[ok].astype(int),np.clip(p[ok],1e-12,1-1e-12),sample_weight=vw[ok],labels=[0,1])))
    minimum=min(np.mean(v) for v in losses.values())
    choice=max((k for k,v in losses.items() if np.mean(v)<=minimum+1e-4),key=lambda z:(z[0],z[1]))
    rows=[dict(alpha=a,lambda_ratio=r,mean_inner_log_loss=float(np.mean(v)),selected=int((a,r)==choice),
        **{f'inner_loss_{j+1}':x for j,x in enumerate(v)}) for (a,r),v in losses.items()]
    return choice,rows


def fitted(frame,columns,h,choice):
    prep=JointPrep(columns,h).fit(frame)
    model=legacy.estimate(prep,frame,*choice)
    model['eitn_handling']='ordinary_penalized_candidate';model['report_model']='M2'
    return model


def diagnostic(model):
    prep=model['preprocessor'];fit=model['estimator'];coef=fit['coef'][5:];active=abs(coef)>1e-6
    ei=prep.selected_names.index(EITN) if EITN in prep.selected_names else None
    return dict(prep.summary,alpha=fit['alpha'],lambda_ratio=model['ratio'],lambda_absolute=fit['lam'],max_kkt=fit['max_kkt'],
        nonzero_conventional_images=int(sum(a and n!=EITN for a,n in zip(active,prep.selected_names))),
        eitn_coefficient=float(coef[ei]) if ei is not None else 0.,eitn_nonzero=bool(ei is not None and active[ei]))

