"""Invented inputs for testing software; no participant records are sampled."""
import numpy as np
import pandas as pd
from scipy.special import expit
from ._ridge import FIXED_EITN

def cohort(n,label,seed,p=64):
    rng=np.random.default_rng(seed);x=rng.normal(size=(n,p));age=rng.normal(50,10,n)
    t=rng.integers(1,5,n);node=rng.integers(0,4,n);volume=rng.lognormal(3.4,.5,n)
    eitn=expit(.7*x[:,0]+rng.normal(size=n))
    event=rng.uniform(size=n)<expit(-1+.7*x[:,1]+1.2*(eitn-.5)+.15*(t-2))
    code=np.where(event,1,np.where(rng.uniform(size=n)<.12,2,0))
    time=np.where(code>0,rng.uniform(5,59,n),np.where(rng.uniform(size=n)<.12,24.,60.))
    d=pd.DataFrame(dict(patient_id=[f'SYN_{label}_{i:04d}' for i in range(n)],center=label if label=='INT' else np.where(np.arange(n)%2,'B','A'),
        time_months=time,endpoint_code=code,age=age,t_stage=t,n_stage=node,gtv_volume_cm3=volume))
    d[FIXED_EITN]=eitn
    columns=[f'mri__T1__core__synthetic_{i:03d}' for i in range(p)]
    d=pd.concat([d,pd.DataFrame(x,columns=columns)],axis=1)
    return d,columns

def registered_images(seed=42):
    rng=np.random.default_rng(seed);shape=(32,34,24);gtv=np.zeros(shape,bool);gtv[13:18,13:18,9:14]=True
    base=rng.uniform(80,160,size=shape).astype(np.float32)
    d={k:(base+rng.normal(0,6,shape)).astype(np.float32) for k in ('pre_t1','post_t1','pre_t1ce','post_t1ce','pre_t2','post_t2')}
    d['post_t1ce'][gtv]+=15;d['post_t2'][gtv]-=10
    d['ct']=np.zeros(shape,np.float32);d['gtv']=gtv.astype(np.float32)
    return d,(1.,1.,3.)
