"""Explicit input contracts and destinations; no inferred cohort exclusions."""
from pathlib import Path
import csv, json
import numpy as np
import pandas as pd
from ._ridge import FIXED_EITN
PREFIXES=('ct__','mri__','mri_delta__','dose_dvh__','dose_rad__')

def output_dir(path):
    path=Path(path).resolve()
    root=Path(__file__).resolve().parents[2]
    if path==root or root in path.parents:
        raise ValueError('Choose an output directory outside the code package')
    path.mkdir(parents=True,exist_ok=False)
    return path

def load_frame(path, features, *, expected_n=None):
    path=Path(path)
    with path.open(newline='') as f:header=next(csv.reader(f))
    if len(header)!=len(set(header)):raise ValueError('Duplicate input column names')
    d=pd.read_csv(path,dtype={'patient_id':str})
    required=['patient_id','center','time_months','endpoint_code','age','t_stage','n_stage','gtv_volume_cm3',FIXED_EITN]+list(features)
    absent=set(required)-set(d)
    if absent:raise ValueError('Missing columns: '+str(sorted(absent)[:10]))
    if d.patient_id.isna().any() or d.patient_id.str.strip().eq('').any() or d.patient_id.duplicated().any():
        raise ValueError('Patient keys must be nonempty and unique within each cohort')
    if d.center.isna().any():raise ValueError('Center strata must be supplied')
    numeric=list(dict.fromkeys(c for c in required if c not in {'patient_id','center'}))
    values=d[numeric].apply(pd.to_numeric,errors='coerce').replace([np.inf,-np.inf],np.nan)
    d=pd.concat([d[['patient_id','center']],values],axis=1)
    d.loc[~d.age.between(0,120),'age']=np.nan
    if not d.time_months.between(0,60).all() or not d.endpoint_code.isin([0,1,2,3]).all():
        raise ValueError('Valid endpoint times in [0,60] and codes 0/1/2/3 are required')
    if expected_n is not None and len(d)!=expected_n:raise ValueError('Cohort size differs from the fixed input contract')
    if (d[FIXED_EITN].notna()&~d[FIXED_EITN].between(0,1)).any():raise ValueError('EITN must be a fraction')
    d['time_after_landmark']=d.time_months # Historical API alias; time zero remains radiotherapy completion.
    return d

def load_inputs(config):
    path=Path(config).resolve();cfg=json.loads(path.read_text())
    feature_spec=cfg['conventional_features']
    features=list(json.loads((path.parent/feature_spec).read_text()) if isinstance(feature_spec,str) else feature_spec)
    if len(features)!=len(set(features)) or len(features)<30:raise ValueError('At least 30 distinct conventional candidates required')
    if any(not c.startswith(PREFIXES) or any(w in c.lower() for w in ['eitn','q_hl','endpoint','recurrence']) for c in features):
        raise ValueError('Only the explicit conventional image family may enter the image block')
    expected=cfg.get('expected_counts',{})
    cohorts={k:load_frame(path.parent/v,features,expected_n=expected.get(k)) for k,v in cfg['cohorts'].items()}
    if 'internal' not in cohorts:raise ValueError('An internal development cohort is required')
    internal=set(cohorts['internal'].patient_id)
    if any(internal&set(d.patient_id) for k,d in cohorts.items() if k!='internal'):
        raise ValueError('Internal and external participant keys must be disjoint')
    return cohorts,features
