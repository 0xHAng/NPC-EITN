"""Train current four models using explicit approved, analysis-ready inputs."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse,json,joblib
import numpy as np
import pandas as pd
from npc_eitn.io import load_inputs,output_dir
from npc_eitn.pipeline import Settings,nested,fit_frozen,predict,rows
from npc_eitn.evaluation import evaluate

def run(config,destination,*,demo=False,bootstrap=1000):
    cohorts,features=load_inputs(config)
    if demo:
        if not all(d.patient_id.str.startswith('SYN_').all() for d in cohorts.values()):
            raise ValueError('Reduced demo settings require explicitly synthetic participant keys')
        settings=Settings.demo()
    else:
        if len(features)!=1848 or len(cohorts['internal'])!=372 or len(cohorts.get('external',[]))!=197:
            raise ValueError('Published settings require 1848 conventional features and fixed 372/197 prediction cohorts')
        settings=Settings()
    out=output_dir(destination)
    internal=cohorts['internal']
    print('Developing on internal records only',flush=True)
    pred,outer,tuning=nested(internal,features,settings)
    models,grid=fit_frozen(internal,features,settings);tuning+=grid
    print('Internal development complete; applying frozen models externally',flush=True)
    parts=[pred]
    for name,frame in cohorts.items():
        if name=='internal':continue
        for h,fits in models.items():parts.append(rows(frame,predict(fits,frame),h,0,name,-1))
    pred=pd.concat(parts,ignore_index=True);metrics=[];comparisons=[]
    for (name,h,repeat),g in pred.groupby(['cohort','horizon_months','repeat']):
        f=cohorts[name]
        pp=g.pivot(index='patient_id',columns='model',values='prediction').reindex(f.patient_id)
        offset=0 if name=='internal' else 200000 if name=='external' else 100000
        results,pairs=evaluate(f,{m:pp[m].to_numpy() for m in pp},h,
            bootstrap=bootstrap if repeat==0 else 0,seed=20261107+h+offset)
        for d in (results,pairs):
            d['cohort']=name;d['horizon_months']=h;d['repeat']=repeat
        metrics.append(results);comparisons.append(pairs)
    pred.to_csv(out/'predictions.csv',index=False)
    pd.concat(metrics,ignore_index=True).to_csv(out/'metrics.csv',index=False)
    pd.concat(comparisons,ignore_index=True).to_csv(out/'paired_comparisons.csv',index=False)
    pd.DataFrame(tuning).to_csv(out/'tuning.csv',index=False)
    joblib.dump(dict(models=models,outer_models=outer),out/'fitted_models.joblib',compress=3)
    (out/'run.json').write_text(json.dumps(dict(mode='synthetic_demo' if demo else 'published_settings',
        internal_n=len(internal),external_counts={k:len(v) for k,v in cohorts.items() if k!='internal'},
        horizons=settings.horizons,outer_folds=settings.outer_folds,inner_folds=settings.inner_folds,
        repeats=settings.repeats,external_refitting=False,bootstrap_replicates=bootstrap),indent=2))
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--demo-settings',action='store_true');p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args()
    if a.bootstrap<0:p.error('--bootstrap must be nonnegative')
    run(a.config,a.output,demo=a.demo_settings,bootstrap=a.bootstrap)
