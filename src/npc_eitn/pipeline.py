"""Internal-only development followed by frozen external inference."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from . import _ridge as ridge, _elastic as elastic, _joint as joint
from .splits import SEED,make_splits

@dataclass(frozen=True)
class Settings:
    horizons:tuple=(36,60)
    repeats:int=6
    outer_folds:int=5
    inner_folds:int=4
    c_grid:tuple=ridge.DEFAULT_C_GRID
    alphas:tuple=elastic.ALPHAS
    ratios:tuple=elastic.RATIOS
    eitn_cs:tuple=elastic.E_CS

    @classmethod
    def demo(cls):
        return cls(horizons=(36,),repeats=1,outer_folds=2,inner_folds=2,
                   c_grid=(.01,1.),alphas=(.5,1.),ratios=(.5,.1),eitn_cs=(0.,.1,1.))

def train_four(internal,features,horizon,splits,settings):
    """This function cannot access an external dataset."""
    spec=ridge.ModelSpec('M1')
    c,grid=ridge.tune_model(internal,features,spec,horizon,settings.c_grid,splits)
    m1=ridge.fit_model(internal,features,spec,horizon,c)
    choice,jgrid=joint.tune(internal,features,horizon,splits,alpha_grid=settings.alphas,ratio_grid=settings.ratios)
    m2=joint.fitted(internal,features,horizon,choice)
    (alpha,ratio,e_c),egrid=elastic.tune(internal,features,horizon,splits,
        alpha_grid=settings.alphas,ratio_grid=settings.ratios,eitn_grid=settings.eitn_cs)
    prep=elastic.LegacyPrep(features,horizon).fit(internal)
    m3=elastic.estimate(prep,internal,alpha,ratio)
    m4=(elastic.estimate(prep,internal,alpha,ratio,include=True,e_C=e_c) if e_c
        else dict(m3,estimator=dict(m3['estimator'],eitn_C=0.)))
    assert m3['preprocessor'] is m4['preprocessor']
    assert m3['estimator']['lam']==m4['estimator']['lam']
    tuning=[dict(model='M1',**r) for r in grid]+[dict(model='M2',**r) for r in jgrid]+[dict(model='M3_M4',**r) for r in egrid]
    return dict(M1=m1,M2=m2,M3=m3,M4=m4),tuning

def predict(models,frame):
    return dict(M1=models['M1'].predict_probability(frame),
                **{k:elastic.predict(models[k],frame) for k in ('M2','M3','M4')})

def rows(frame,probability,horizon,repeat,cohort,folds):
    return pd.concat([pd.DataFrame(dict(patient_id=frame.patient_id.to_numpy(),center=frame.center.to_numpy(),
        model=m,prediction=p,horizon_months=horizon,repeat=repeat,cohort=cohort,outer_fold=folds))
        for m,p in probability.items()],ignore_index=True)

def nested(internal,features,settings):
    outputs=[];tuning=[];models={}
    for h in settings.horizons:
        for repeat in range(settings.repeats):
            seed=SEED+h+10000*repeat
            p={k:np.full(len(internal),np.nan) for k in ('M1','M2','M3','M4')};fold_ids=np.full(len(internal),-1)
            for fold,(ti,vi) in enumerate(make_splits(internal,h,settings.outer_folds,seed)):
                train=internal.iloc[ti].reset_index(drop=True);test=internal.iloc[vi].reset_index(drop=True)
                assert not set(train.patient_id)&set(test.patient_id)
                fits,grid=train_four(train,features,h,make_splits(train,h,settings.inner_folds,seed*10+fold),settings)
                for m,v in predict(fits,test).items():p[m][vi]=v
                fold_ids[vi]=fold;models[(h,repeat,fold)]=fits
                tuning.extend(dict(horizon_months=h,repeat=repeat,outer_fold=fold,**r) for r in grid)
            assert all(np.isfinite(v).all() for v in p.values())
            outputs.append(rows(internal,p,h,repeat,'internal',fold_ids))
    return pd.concat(outputs,ignore_index=True),models,tuning

def fit_frozen(internal,features,settings):
    models={};tuning=[]
    for h in settings.horizons:
        splits=make_splits(internal,h,settings.inner_folds,SEED+h+900000)
        fits,grid=train_four(internal,features,h,splits,settings)
        models[h]=fits
        tuning.extend(dict(horizon_months=h,outer_fold=-1,**r) for r in grid)
    return models,tuning
