"""Join explicit, frozen cohort inputs; never select new patients from outcomes."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from ._ridge import FIXED_EITN

def prepare(config):
    path=Path(config).resolve();cfg=json.loads(path.read_text())
    features=cfg['conventional_features']
    if isinstance(features,str):features=json.loads((path.parent/features).read_text())
    corrections=pd.read_csv(path.parent/cfg['corrections'],dtype={'patient_id':str}) if cfg.get('corrections') else None
    output={}
    for name,spec in cfg['cohorts'].items():
        clinical=pd.read_csv(path.parent/spec['clinical'],dtype={'patient_id':str})
        image=pd.read_csv(path.parent/spec['features'],dtype={'patient_id':str})
        for d in [clinical,image]:
            if d.patient_id.isna().any() or d.patient_id.duplicated().any():raise ValueError('Nonunique participant keys')
        if len(clinical)!=spec['expected_n']:raise ValueError('Frozen cohort size changed')
        if not set(clinical.patient_id).issubset(set(image.patient_id)):raise ValueError('Missing feature records')
        if name!='internal' and set(clinical.patient_id)!=set(image.patient_id):raise ValueError('External membership mismatch')
        if corrections is not None:
            c=corrections.loc[corrections.population.eq('internal' if name=='internal' else 'external')]
            if c.patient_id.duplicated().any():raise ValueError('Duplicate correction keys')
            ids=clinical.patient_id.astype(str).str.replace(r'\.0$','',regex=True)
            for row in c.itertuples():
                for field in ['age','t_stage','n_stage']:
                    value=getattr(row,field,None)
                    if pd.notna(value):clinical.loc[ids.eq(row.patient_id),field]=value
        fields=['patient_id','endpoint_time_months','endpoint_code','age','t_stage','n_stage','gtv_volume_cm3']
        d=clinical[fields].rename(columns={'endpoint_time_months':'time_months'}).copy()
        d['center']='ZJ' if name=='internal' else np.where(clinical.center_B.eq(1),'B','A')
        images=image.set_index('patient_id').reindex(d.patient_id)[[FIXED_EITN]+features].reset_index()
        d=d.merge(images,on='patient_id',validate='one_to_one',sort=False)
        assert d.patient_id.equals(clinical.patient_id)
        output[name]=d
    return output,features


def prepare_associations(config):
    """Join approved main/sensitivity source rows and supplied quality flags.

    Source paths and any private excluded keys stay in the approved local config,
    outside the distributable package. No exclusion is inferred from an event.
    """
    path=Path(config).resolve();cfg=json.loads(path.read_text());spec=cfg['association']
    def read(key):return pd.read_csv(path.parent/spec[key],dtype={'patient_id':str})
    raw=read('internal_long');internal=raw.loc[raw.center.eq('A')].copy();internal['center']='Internal'
    internal['patient_id']=internal.patient_id.str.replace(r'\.0$','',regex=True)
    clinical=read('internal_clinical');clinical['patient_id']=clinical.patient_id.str.replace(r'\.0$','',regex=True)
    internal=internal.merge(clinical[['patient_id','sex']],on='patient_id',validate='many_to_one')
    manifest=read('external_manifest');features=read('external_features')
    primary=manifest.loc[manifest.eligible_set2.eq(1)].copy()
    external=features.merge(primary.drop(columns=[c for c in features if c in primary and c not in ['center','patient_id']]),
        on=['center','patient_id'],validate='many_to_one')
    corrections=pd.read_csv(path.parent/cfg['corrections'],dtype={'patient_id':str}) if cfg.get('corrections') else None
    parts=[]
    for name,d in [('internal',internal),('external',external)]:
        d=d.copy()
        if corrections is not None:
            c=corrections.loc[corrections.population.eq(name)]
            for row in c.itertuples():
                for field in ['age','t_stage','n_stage','sex']:
                    value=getattr(row,field,None)
                    if field in d and pd.notna(value):d.loc[d.patient_id.eq(row.patient_id),field]=value
        d.loc[~d.age.between(0,120),'age']=np.nan
        d['population']=name
        excluded=set(spec.get('excluded_external_keys',[])) if name=='external' else set()
        if excluded-set(d.patient_id):raise ValueError('An approved exclusion key is absent from the source population')
        d['association_eligible']=(~d.patient_id.isin(excluded)).astype(int)
        d['phenotype_eligible']=d['association_eligible']
        d['sex_male']=d.sex.astype(str).str.strip().map({'男':1,'女':0})
        parts.append(d)
    result=pd.concat(parts,ignore_index=True)
    if internal.patient_id.nunique()!=517 or external.patient_id.nunique()!=206:
        raise ValueError('The supplied main/sensitivity source populations must remain 517/206')
    count=result.loc[result.population.eq('external')&result.association_eligible.eq(1)].patient_id.nunique()
    if count!=203:raise ValueError('The current quality-restricted external association population must contain 203 participants')
    fields=['patient_id','population','center','roi','q_hl','age','t_stage','n_stage','gtv_volume_cm3','lrr_competing_code','phenotype_eligible','association_eligible','sex_male']
    fields += [f'{e}_{k}' for e in ['lrr','dm','pfs'] for k in ['time_months','event']]
    return result[fields]
