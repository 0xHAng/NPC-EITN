"""Evaluate approved fixed predictions without developing or updating prediction models."""
from pathlib import Path
import sys
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import argparse
import pandas as pd
from npc_eitn.io import output_dir
from npc_eitn.evaluation import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--horizon',type=int,choices=(36,60),required=True);p.add_argument('--bootstrap',type=int,default=1000)
    a=p.parse_args();d=pd.read_csv(a.input,dtype={'patient_id':str})
    required={'patient_id','center','time_months','endpoint_code','M1','M2','M3','M4'}
    if required-set(d) or d.patient_id.isna().any() or d.patient_id.duplicated().any():raise ValueError('Invalid paired prediction input')
    if a.bootstrap<0:raise ValueError('Bootstrap count must be nonnegative')
    d['time_after_landmark']=d.time_months
    metrics,pairs=evaluate(d,{k:d[k].to_numpy() for k in ['M1','M2','M3','M4']},a.horizon,
        bootstrap=a.bootstrap,seed=20261107+a.horizon+200000)
    out=output_dir(a.output);metrics.to_csv(out/'metrics.csv',index=False);pairs.to_csv(out/'paired_comparisons.csv',index=False)
