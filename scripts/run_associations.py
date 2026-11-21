"""Current main Cox/landmark and cumulative-incidence calculations."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse
import numpy as np
import pandas as pd
from npc_eitn.io import output_dir
from npc_eitn.associations import association,bh,aj_at,cif_interval,paired_adjustment,select_association_population

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=pd.read_csv(a.input,dtype={'patient_id':str})
    required={'patient_id','population','center','roi','q_hl','age','t_stage','n_stage','gtv_volume_cm3','lrr_competing_code','association_eligible'}
    required|={f'{e}_{k}' for e in ['lrr','dm','pfs'] for k in ['time_months','event']}
    if required-set(d):raise ValueError('Missing association fields: '+str(sorted(required-set(d))))
    if d.duplicated(['population','patient_id','roi']).any():raise ValueError('Duplicate participant/ROI observations')
    if not d.population.isin(['internal','external']).all():raise ValueError('Population must be internal or external')
    d.loc[~d.age.between(0,120),'age']=np.nan;d['log_gtv_volume']=np.log1p(d.gtv_volume_cm3)
    rois=['core','ring5','ring5_10','ring10_15','ring10','ring15']
    ref=d.loc[d.population.eq('internal')].groupby('roi').q_hl.agg(['mean','std'])
    if set(ref.index)!=set(rois) or ref['std'].le(0).any():raise ValueError('All six internal reference scales are required')
    d['eitn_z']=(d.q_hl-d.roi.map(ref['mean']))/d.roi.map(ref['std'])
    eligible=select_association_population(d)
    internal=eligible.loc[eligible.population.eq('internal')];external=eligible.loc[eligible.population.eq('external')]
    groups=[('Internal',internal),('Pooled external',external)]
    groups += [('External '+str(c),g) for c,g in external.groupby('center')]
    rows=[]
    for name,g in groups:
        for endpoint in ['lrr','dm','pfs']:
            for roi in rois:rows.append(association(g,name,roi,endpoint,adjusted=name!='External B'))
        for landmark in [3,6]:
            for roi in rois:rows.append(association(g,name,roi,'lrr',adjusted=name!='External B',landmark=landmark))
    result=pd.DataFrame(rows);result['q']=np.nan
    for _,g in result.loc[result.roi.isin(rois[1:4])].groupby(['center','endpoint','landmark','feature']):
        if g.p.notna().all():result.loc[g.index,'q']=bh(g.p)
    cif=[];grid=np.arange(0,61,6);rng=np.random.default_rng(20261017)
    for name,g in groups[:2]:
        for label in ['Low','High']:
            s=g.loc[g.roi.eq('ring15')&((g.q_hl>.1515) if label=='High' else (g.q_hl<=.1515))]
            if len(s):
                point,low,high=cif_interval(s,grid,rng,500)
                cif.append(pd.DataFrame(dict(center=name,group=label,time_months=grid,cif=point,low=low,high=high,n=len(s))))
    out=output_dir(a.output);result.to_csv(out/'associations.csv',index=False);ref.to_csv(out/'internal_reference_scales.csv')
    pd.concat(cif,ignore_index=True).to_csv(out/'cumulative_incidence.csv',index=False)
    broad=d.loc[d.population.eq('external')]
    if broad.patient_id.nunique()>external.patient_id.nunique():
        pd.DataFrame([association(broad,'Pooled external',roi) for roi in rois]).to_csv(out/'broader_association_sensitivity.csv',index=False)
    if 'sex_male' in d:
        sex=[]
        for name,g in groups[:2]:
            sample=g.loc[g.roi.eq('ring15')].rename(columns={'lrr_time_months':'time','lrr_event':'event'})
            fit=paired_adjustment(sample,['sex_male'],external=name=='Pooled external')
            row=fit.loc[fit.adjustment.eq('expanded')].iloc[0].to_dict();row['center']=name;sex.append(row)
        pd.DataFrame(sex).to_csv(out/'sex_adjusted.csv',index=False)

