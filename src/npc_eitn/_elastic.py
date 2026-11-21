"""Numerical kernels extracted from the current matched imaging/EITN analysis.

See provenance/source_provenance.json. The numerical function bodies retain the
source definitions; tuning grids are explicit arguments for the synthetic demo.
"""
from __future__ import annotations
import hashlib
import warnings
import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import log_loss
from ._ridge import FoldPreprocessor, ModelSpec, FIXED_EITN, horizon_labels_weights
ALPHAS=(.1,.3,.5,.7,.9,1.)
RATIOS=(1.,.5,.25,.1,.05,.025,.01)
E_CS=(0.,.01,.1,1.,10.)
def clinical_mle(x, y, w):
    known = np.isfinite(y)
    weight = w[known]*(known.sum()/w[known].sum())
    fit = LogisticRegression(penalty=None, solver='lbfgs', max_iter=5000, tol=1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        warnings.simplefilter('error', ConvergenceWarning)
        fit.fit(x[known], y[known].astype(int), sample_weight=weight)
    return np.r_[fit.intercept_[0], fit.coef_[0]]


class LegacyPrep:
    def __init__(self, columns, horizon):
        self.columns = list(columns); self.horizon = horizon

    def fit(self, frame):
        self.preprocessor = FoldPreprocessor(self.columns,
            ModelSpec('legacy_common_prep', include_eitn=True, image_mode='ridge')).fit(frame)
        x = self.preprocessor.transform(frame)
        assert self.preprocessor.summary['fixed_columns'][4] == FIXED_EITN
        y, w = horizon_labels_weights(frame, self.horizon, frame)
        ok = np.isfinite(y)
        self.clinical_initial = clinical_mle(x[:,:4], y, w)
        names = self.preprocessor.image_cols_
        order = sorted(range(len(names)), key=lambda i: names[i])
        frequency = [i for i in order if pd.Series(x[:,5+i]).value_counts(normalize=True).iloc[0] < .995]
        rank = stats.rankdata(x[:,np.array(frequency)+5], axis=0)
        rank -= rank.mean(axis=0); rank /= np.sqrt(np.sum(rank**2,axis=0))
        corr = np.clip(rank.T @ rank, -1, 1)
        kept = []
        for j in range(len(frequency)):
            if not kept or np.max(abs(corr[j,kept])) < .9:
                kept.append(j)
        decorrelated = np.array([frequency[j] for j in kept],dtype=int)
        weight = w[ok]/w[ok].sum()
        residual = weight*(special.expit(self.clinical_initial[0]+x[ok,:4]@self.clinical_initial[1:])-y[ok])
        gradient = x[ok][:,decorrelated+5].T @ residual
        selected = np.argsort(-abs(gradient),kind='stable')[:30]
        self.selected_indices = decorrelated[selected]
        self.selected_names = [names[i] for i in self.selected_indices]
        self.max_gradient = float(np.max(abs(gradient[selected])))
        self.summary = dict(raw_candidates=len(self.columns), missing_variance_retained=len(names),
            after_frequency=len(frequency), after_correlation=len(decorrelated),
            after_screen=len(self.selected_indices), selected_image_names='|'.join(self.selected_names),
            training_patient_hash=hashlib.sha256('\n'.join(frame.patient_id).encode()).hexdigest())
        assert len(self.selected_indices) == 30 and self.max_gradient > 0
        assert not any(c.startswith(('formal__','legacy__')) for c in self.selected_names)
        return self

    def transform(self, frame, include):
        x = self.preprocessor.transform(frame)
        return x[:,:5 if include else 4], x[:,self.selected_indices+5]


def objective(theta, fixed, images, y, w, lam, alpha, e_rho):
    q, p = fixed.shape[1], images.shape[1]
    beta = theta[1:1+q]; positive=theta[1+q:1+q+p]; negative=theta[1+q+p:]
    gamma = positive-negative
    eta = theta[0]+fixed@beta+images@gamma
    loss = np.dot(w,np.logaddexp(0,eta)-y*eta)/w.sum()
    value = loss+lam*(alpha*(positive.sum()+negative.sum())+.5*(1-alpha)*np.dot(gamma,gamma))
    gbeta_penalty = np.zeros(q)
    if q == 5:
        value += .5*e_rho*beta[4]**2
        gbeta_penalty[4] = e_rho*beta[4]
    residual = w*(special.expit(eta)-y)/w.sum()
    gf = fixed.T@residual+gbeta_penalty
    gi = images.T@residual+lam*(1-alpha)*gamma
    return float(value), np.r_[residual.sum(),gf,gi+lam*alpha,-gi+lam*alpha]


def kkt(coef, fixed, images, y, w, lam, alpha, e_rho):
    q = fixed.shape[1]
    beta, gamma = coef[1:1+q], coef[1+q:]
    residual = w*(special.expit(coef[0]+fixed@beta+images@gamma)-y)/w.sum()
    gf = fixed.T@residual
    if q == 5: gf[4] += e_rho*beta[4]
    gi = images.T@residual+lam*(1-alpha)*gamma
    active = abs(gamma)>1e-7
    violations = np.maximum(abs(gi)-lam*alpha,0)
    violations[active] = abs(gi[active]+lam*alpha*np.sign(gamma[active]))
    return float(max(abs(residual.sum()),np.max(abs(gf)),np.max(violations,initial=0)))


def fit_elastic(fixed, images, labels, weights, lam, alpha, e_C=None, initial=None):
    ok = np.isfinite(labels)
    x, z, y = fixed[ok], images[ok], labels[ok]
    w = weights[ok]*(ok.sum()/weights[ok].sum())
    q,p = x.shape[1],z.shape[1]
    e_rho = 0. if e_C is None else 1/(e_C*len(y))
    if initial is None:
        initial=np.zeros(1+q+2*p)
        initial[0]=special.logit(np.average(y,weights=w))
    result=optimize.minimize(objective,initial,args=(x,z,y,w,lam,alpha,e_rho),jac=True,
        method='L-BFGS-B',bounds=[(None,None)]*(1+q)+[(0.,None)]*(2*p),
        options=dict(maxiter=5000,ftol=1e-13,gtol=1e-9,maxls=50,maxcor=20))
    theta=result.x
    coef=np.r_[theta[:1+q],theta[1+q:1+q+p]-theta[1+q+p:]]
    error=kkt(coef,x,z,y,w,lam,alpha,e_rho)
    solver='L-BFGS-B';iterations=int(result.nit)
    if not result.success or error>3e-6:
        # Refine exactly the same convex objective when bound-constrained
        # relative-function convergence precedes the KKT criterion.
        gamma=coef[1+q:]
        start=np.r_[coef[:1+q],np.maximum(gamma,0),np.maximum(-gamma,0)]
        refined=optimize.minimize(objective,start,args=(x,z,y,w,lam,alpha,e_rho),jac=True,
            method='SLSQP',bounds=[(None,None)]*(1+q)+[(0.,None)]*(2*p),
            options=dict(maxiter=5000,ftol=1e-13))
        theta=refined.x
        coef=np.r_[theta[:1+q],theta[1+q:1+q+p]-theta[1+q+p:]]
        error=kkt(coef,x,z,y,w,lam,alpha,e_rho)
        result=refined;iterations+=int(refined.nit);solver='L-BFGS-B + same-objective SLSQP'
    if error>3e-6 or not np.isfinite(coef).all():
        raise RuntimeError(f'EN solver: {result.message}, KKT={error}, alpha={alpha}, lambda={lam}')
    return dict(coef=coef,split_theta=theta,alpha=float(alpha),lam=float(lam),
                eitn_C=e_C,max_kkt=error,iterations=iterations,objective=float(result.fun),solver=solver)


def estimate(prep, frame, alpha, ratio, include=False, e_C=None, initial=None):
    fixed, images = prep.transform(frame,include)
    y,w=horizon_labels_weights(frame,prep.horizon,frame)
    if initial is None:
        initial=np.zeros(1+fixed.shape[1]+2*images.shape[1])
        initial[:5]=prep.clinical_initial
    fit=fit_elastic(fixed,images,y,w,ratio*prep.max_gradient/alpha,alpha,e_C,initial)
    return dict(preprocessor=prep,estimator=fit,include_eitn=include,horizon=prep.horizon,ratio=float(ratio))


def predict(model, frame):
    fixed, images=model['preprocessor'].transform(frame,model['include_eitn'])
    coef=model['estimator']['coef'];q=fixed.shape[1]
    return special.expit(coef[0]+fixed@coef[1:1+q]+images@coef[1+q:])


def diagnostics(model):
    prep=model['preprocessor'];est=model['estimator'];q=5 if model['include_eitn'] else 4
    gamma=est['coef'][1+q:];active=abs(gamma)>1e-6
    return dict(prep.summary,alpha=est['alpha'],lambda_ratio=model['ratio'],lambda_absolute=est['lam'],
        eitn_C=est['eitn_C'],max_kkt=est['max_kkt'],solver_iterations=est['iterations'],solver=est['solver'],
        eitn_coefficient=float(est['coef'][5]) if model['include_eitn'] else 0.,
        nonzero_images=int(active.sum()),nonzero_image_names='|'.join(np.array(prep.selected_names)[active]))


def tune(train, columns, h, splits, *, alpha_grid=ALPHAS, ratio_grid=RATIOS, eitn_grid=E_CS):
    packs=[]
    for ti,vi in splits:
        t,v=train.iloc[ti].reset_index(drop=True),train.iloc[vi].reset_index(drop=True)
        prep=LegacyPrep(columns,h).fit(t)
        y,w=horizon_labels_weights(v,h,t);ok=np.isfinite(y)
        tc,tx=prep.transform(t,True);vc,vx=prep.transform(v,True)
        ty,tw=horizon_labels_weights(t,h,t)
        packs.append((prep,tc,tx,ty,tw,vc[ok],vx[ok],y[ok].astype(int),w[ok]))
    losses={(a,r):[] for a in alpha_grid for r in ratio_grid}
    for prep,tc,tx,ty,tw,vc,vx,y,w in packs:
        for a in alpha_grid:
            initial=None
            for r in ratio_grid:
                if initial is None:
                    initial=np.zeros(5+2*tx.shape[1]);initial[:5]=prep.clinical_initial
                fitted=fit_elastic(tc[:,:4],tx,ty,tw,r*prep.max_gradient/a,a,initial=initial)
                initial=fitted['split_theta'];coef=fitted['coef']
                p=special.expit(coef[0]+vc[:,:4]@coef[1:5]+vx@coef[5:])
                losses[(a,r)].append(float(log_loss(y,np.clip(p,1e-12,1-1e-12),sample_weight=w,labels=[0,1])))
    minimum=min(np.mean(v) for v in losses.values())
    alpha,ratio=max((key for key,val in losses.items() if np.mean(val)<=minimum+1e-4),key=lambda x:(x[0],x[1]))
    rows=[dict(stage='baseline_grid',alpha=a,lambda_ratio=r,mean_inner_log_loss=float(np.mean(val)),
               selected=int((a,r)==(alpha,ratio)),**{f'inner_loss_{j+1}':x for j,x in enumerate(val)})
          for (a,r),val in losses.items()]
    e_losses={c:[] for c in eitn_grid}
    e_losses[0.]=list(losses[(alpha,ratio)])
    for prep,tc,tx,ty,tw,vc,vx,y,w in packs:
        for C in eitn_grid[1:]:
            initial=np.zeros(6+2*tx.shape[1]);initial[:5]=prep.clinical_initial
            fitted=fit_elastic(tc,tx,ty,tw,ratio*prep.max_gradient/alpha,alpha,e_C=C,initial=initial)
            coef=fitted['coef'];p=special.expit(coef[0]+vc@coef[1:6]+vx@coef[6:])
            e_losses[C].append(float(log_loss(y,np.clip(p,1e-12,1-1e-12),
                                                 sample_weight=w,labels=[0,1])))
    minimum=min(np.mean(v) for v in e_losses.values())
    chosen_C=min(c for c,val in e_losses.items() if np.mean(val)<=minimum+1e-4)
    rows.extend(dict(stage='eitn_grid',alpha=alpha,lambda_ratio=ratio,eitn_C=c,
                     mean_inner_log_loss=float(np.mean(val)),selected=int(c==chosen_C),
                     **{f'inner_loss_{j+1}':x for j,x in enumerate(val)}) for c,val in e_losses.items())
    return (alpha,ratio,chosen_C),rows

