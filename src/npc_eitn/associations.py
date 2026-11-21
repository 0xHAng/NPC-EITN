"""Cause-specific Cox/landmark associations and tied-time Aalen-Johansen."""
import warnings
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test
ADJ=['age','t_stage','n_stage','log_gtv_volume']
def association(d, center, roi, endpoint='lrr', adjusted=True, landmark=0, feature='eitn_z'):
    g = d.loc[d.roi.eq(roi) & d[f'{endpoint}_time_months'].gt(landmark)].copy()
    g['time'] = g[f'{endpoint}_time_months'] - landmark
    g['event'] = g[f'{endpoint}_event']
    cov = [feature] + (ADJ if adjusted else [])
    x = g[['time', 'event'] + cov + (['center'] if center == 'Pooled external' else [])].replace([np.inf, -np.inf], np.nan).dropna()
    out = dict(center=center, roi=roi, endpoint=endpoint, adjusted=adjusted,
               landmark=landmark, feature=feature, n=len(x), events=int(x.event.sum()))
    if x.event.sum() < (len(cov) if adjusted else 3):
        return dict(out, hr=np.nan, low=np.nan, high=np.nan, p=np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = CoxPHFitter().fit(x, duration_col='time', event_col='event',
                                  strata=['center'] if center == 'Pooled external' else None)
    z = model.summary.loc[feature]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ph_p = float(proportional_hazard_test(model, x, time_transform='rank').summary.loc[feature, 'p'])
    except (ValueError, KeyError, np.linalg.LinAlgError):
        ph_p = np.nan
    return dict(out, hr=np.exp(z['coef']), low=np.exp(z['coef lower 95%']),
                high=np.exp(z['coef upper 95%']), p=z['p'], ph_p=ph_p)


def bh(values):
    v = np.asarray(values, float)
    order = np.argsort(v)
    out = np.empty(len(v))
    out[order] = np.minimum(1, np.minimum.accumulate((v[order] * len(v) / np.arange(1, len(v)+1))[::-1])[::-1])
    return out


def aj_at(d, grid):
    t = d.lrr_time_months.to_numpy(float)
    code = d.lrr_competing_code.to_numpy(int)
    unique = np.unique(t)
    risk = np.array([(t >= v).sum() for v in unique])
    target = np.array([((t == v) & (code == 1)).sum() for v in unique])
    events = np.array([((t == v) & (code != 0)).sum() for v in unique])
    prior = np.r_[1.0, np.cumprod(1 - events / risk)[:-1]]
    cif = np.cumsum(prior * target / risk)
    pos = np.searchsorted(unique, grid, side='right') - 1
    return np.where(pos >= 0, cif[np.maximum(0, pos)], 0)



def cif_interval(frame,grid,rng,replicates=500):
    """Pointwise patient bootstrap; center strata stay separate."""
    strata=[g for _,g in frame.groupby('center')]
    values=[]
    for _ in range(replicates):
        sample=pd.concat([g.iloc[rng.integers(0,len(g),len(g))] for g in strata])
        values.append(aj_at(sample,grid))
    bounds=np.percentile(values,[2.5,97.5],axis=0)
    return aj_at(frame,grid),bounds[0],bounds[1]

def paired_adjustment(frame,extra_covariates,*,external=False):
    """Core and expanded Cox fits in the same complete cases (Supplement S10).

    The input already supplies curated sex/pathology/therapy/EBV variables and
    EITN scaled by the full internal phenotype cohort SD. No prediction fitting
    or external expanded multivariable development occurs here.
    """
    if external and any(c!='sex_male' for c in extra_covariates):
        raise ValueError('The current study did not fit full expanded external Cox models')
    fields=['time','event','eitn_z','center']+ADJ+list(extra_covariates)
    sample=frame.loc[frame.time.gt(0),list(dict.fromkeys(fields))].replace([np.inf,-np.inf],np.nan).dropna()
    rows=[]
    for name,variables in [('core_same_patients',ADJ),('expanded',ADJ+list(extra_covariates))]:
        x=sample[['time','event','eitn_z']+variables+(['center'] if external else [])]
        model=CoxPHFitter().fit(x,duration_col='time',event_col='event',strata=['center'] if external else None)
        r=model.summary.loc['eitn_z']
        rows.append(dict(adjustment=name,n=len(x),events=int(x.event.sum()),hr=float(np.exp(r['coef'])),
            low=float(np.exp(r['coef lower 95%'])),high=float(np.exp(r['coef upper 95%'])),p=float(r['p'])))
    return pd.DataFrame(rows)


def select_association_population(frame):
    """Apply supplied quality eligibility, independent of clinical outcomes."""
    if 'association_eligible' not in frame:raise ValueError('Explicit association_eligible flags are required')
    flag=pd.to_numeric(frame.association_eligible,errors='coerce')
    if not flag.isin([0,1]).all():raise ValueError('Association eligibility must be 0 or 1 for every row')
    keys=['population','patient_id']
    if frame.assign(_flag=flag).groupby(keys)._flag.nunique().gt(1).any():
        raise ValueError('Eligibility must agree across all ROIs for a participant')
    return frame.loc[flag.eq(1)].copy()
