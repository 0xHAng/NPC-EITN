import unittest,tempfile,json,sys
from unittest.mock import patch
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.optimize import check_grad
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from npc_eitn import _ridge as r,_elastic as e
from npc_eitn.synthetic import cohort,registered_images
from npc_eitn.phenotype import extract_arrays,features
from npc_eitn.splits import fast_auc,make_splits
from npc_eitn.io import load_frame,output_dir
from npc_eitn.associations import aj_at,paired_adjustment,select_association_population
from npc_eitn.prepare import prepare_associations

class EndpointTests(unittest.TestCase):
    def frame(self):return pd.DataFrame(dict(time_after_landmark=[12.,24.,36.,36.,60.],endpoint_code=[0,1,2,0,0]))
    def test_exact_horizon_and_competing_events(self):
        y,w=r.horizon_labels_weights(self.frame(),36)
        np.testing.assert_allclose(y[1:],[1,0,0,0]);self.assertTrue(np.isnan(y[0]));np.testing.assert_allclose(w[1:],1.25)
    def test_training_censoring_source(self):
        d=self.frame();v=pd.DataFrame(dict(time_after_landmark=[36.],endpoint_code=[0]))
        _,w=r.horizon_labels_weights(v,36,d);self.assertAlmostEqual(w[0],1.25)
        _,own=r.horizon_labels_weights(v,36);self.assertEqual(own[0],1.)
    def test_invalid_times_and_codes(self):
        for key,value in [('time_after_landmark',-1),('endpoint_code',9)]:
            d=self.frame();d.loc[0,key]=value
            with self.assertRaises(ValueError):r.horizon_labels_weights(d,36)
    def test_brier_uses_full_cohort_denominator(self):
        d=self.frame();p=np.full(5,.2);result=r.weighted_metrics(d,p,36)
        self.assertAlmostEqual(result['ipcw_brier'],1.25*(.8**2+3*.2**2)/5)
    def test_auc_against_pairwise_definition(self):
        y=np.array([1,0,1,0]);p=np.array([.7,.3,.3,.3]);w=np.array([2.,1.,3.,4.])
        manual=sum(w[i]*w[j]*((p[i]>p[j])+.5*(p[i]==p[j])) for i in [0,2] for j in [1,3])/(w[y==1].sum()*w[y==0].sum())
        self.assertAlmostEqual(fast_auc(y,p,w),manual)
    def test_tied_competing_events_in_cif(self):
        d=pd.DataFrame(dict(lrr_time_months=[5.,5.,10.,10.],lrr_competing_code=[1,2,1,0]))
        np.testing.assert_allclose(aj_at(d,np.array([0,5,10])),[0,.25,.5])

class EligibilityTests(unittest.TestCase):
    def source_frames(self):
        def frame(n,center,prefix):
            d=pd.DataFrame({'patient_id':[f'SYN_{prefix}_{i}' for i in range(n)],'center':center,
                'roi':'ring15','q_hl':.15,'age':50,'t_stage':3,'n_stage':1,'gtv_volume_cm3':20,
                'lrr_competing_code':0,'sex':'男'})
            for endpoint in ['lrr','dm','pfs']:
                d[f'{endpoint}_time_months']=60.;d[f'{endpoint}_event']=0
            return d
        internal=frame(517,'A','I');external=frame(206,'A','E')
        manifest=external[['patient_id','center']].copy();manifest['eligible_set2']=1
        return {'internal':internal.drop(columns='sex'),'clinical':internal[['patient_id','sex']],
            'manifest':manifest,'features':external.drop(columns='sex')}
    def prepare_sources(self,excluded):
        frames=self.source_frames()
        # Sex originates in the external clinical manifest in the approved join.
        frames['manifest']['sex']='男'
        cfg={'association':{'internal_long':'internal','internal_clinical':'clinical',
            'external_manifest':'manifest','external_features':'features','excluded_external_keys':excluded}}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'config.json';p.write_text(json.dumps(cfg))
            with patch('npc_eitn.prepare.pd.read_csv',side_effect=lambda p,**kwargs:frames[p.name].copy()):
                return prepare_associations(p)
    def test_main_phenotype_flags_keep_broader_sensitivity_separate(self):
        d=self.prepare_sources(['SYN_E_0','SYN_E_1','SYN_E_2'])
        external=d.loc[d.population.eq('external')]
        self.assertEqual(len(external),206)
        self.assertEqual(int(external.phenotype_eligible.sum()),203)
        np.testing.assert_array_equal(d.phenotype_eligible,d.association_eligible)
        self.assertEqual(len(select_association_population(d)),517+203)
    def test_unknown_approved_exclusion_key_fails(self):
        with self.assertRaisesRegex(ValueError,'exclusion key'):
            self.prepare_sources(['SYN_E_0','SYN_E_1','SYN_UNKNOWN'])
    def test_quality_flag_selects_whole_participants(self):
        d=pd.DataFrame(dict(population=['external']*12,patient_id=['SYN_A']*6+['SYN_B']*6,association_eligible=[1]*6+[0]*6))
        result=select_association_population(d);self.assertEqual(len(result),6);self.assertEqual(set(result.patient_id),{'SYN_A'})
    def test_inconsistent_quality_flags_fail(self):
        d=pd.DataFrame(dict(population=['external']*2,patient_id=['SYN_A']*2,association_eligible=[0,1]))
        with self.assertRaises(ValueError):select_association_population(d)

class ImageTests(unittest.TestCase):
    def test_six_regions_and_shell_union(self):
        images,spacing=registered_images();d,q=extract_arrays(images,spacing);d=d.set_index('roi')
        self.assertEqual(len(d),6);self.assertTrue(q['automatic_qc_pass'])
        self.assertEqual(d.loc['ring15','geometric_nvox'],d.loc[['ring5','ring5_10','ring10_15'],'geometric_nvox'].sum())
        n=d.loc[['ring5','ring5_10','ring10_15'],'valid_nvox'];qhl=d.loc[n.index,'q_hl']
        self.assertAlmostEqual(d.loc['ring15','q_hl'],float((n*qhl).sum()/n.sum()))
    def test_common_mask_requires_all_six_raw_volumes(self):
        images,spacing=registered_images();a,_=extract_arrays(images,spacing)
        images['post_t2'][14,14,10]=0;b,_=extract_arrays(images,spacing)
        self.assertEqual(int(a.iloc[0].valid_nvox)-int(b.iloc[0].valid_nvox),1)
    def test_ct_window_does_not_limit_core(self):
        images,spacing=registered_images();images['ct'][images['gtv']>0]=500
        d,_=extract_arrays(images,spacing);self.assertEqual(int(d.iloc[0].geometric_nvox),125)
        images['ct'][:]=300;d,q=extract_arrays(images,spacing)
        self.assertEqual(int(d.iloc[-1].geometric_nvox),0);self.assertFalse(q['automatic_qc_pass'])
    def test_anisotropic_physical_distance(self):
        rng=np.random.default_rng(4);shape=(21,21,9);base=rng.uniform(10,30,shape).astype(np.float32)
        images={k:base.copy() for k in ['pre_t1','post_t1','pre_t1ce','post_t1ce','pre_t2','post_t2']}
        images['gtv']=np.zeros(shape,np.float32);images['gtv'][10,10,4]=1;images['ct']=np.zeros(shape,np.float32)
        d,_=extract_arrays(images,(1,1,6));self.assertEqual(int(d.set_index('roi').loc['ring5','geometric_nvox']),80)
    def test_eitn_uses_strict_enhancement_and_nonincreasing_t2(self):
        de=np.array([1.,0.,-1.,1.]*3).reshape(2,2,3);dt=np.array([0.,0.,0.,1.]*3).reshape(2,2,3)
        arrays=dict(delta_enh=de,delta_t2=dt,pre_enh=np.zeros_like(de),post_enh=de,pre_t2=np.zeros_like(dt),post_t2=dt)
        result=features(np.ones_like(de,bool),arrays,12,1.,1.)
        self.assertEqual(result['q_hl'],.25);self.assertEqual(result['q_hh'],.25)
        self.assertEqual(result['q_ll'],.5)

    def test_empty_or_invalid_geometry_fails(self):
        images,spacing=registered_images();images['gtv'][:]=0
        with self.assertRaises(ValueError):extract_arrays(images,spacing)
        images,spacing=registered_images()
        with self.assertRaises(ValueError):extract_arrays(images,(1,float('nan'),1))

class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d,cls.columns=cohort(240,'INT',20261115);cls.d['time_after_landmark']=cls.d.time_months
    def test_shared_folds_and_disjoint_training(self):
        for ti,vi in make_splits(self.d,36,5,20261051):
            self.assertFalse(set(ti)&set(vi));self.assertEqual(len(ti)+len(vi),240)
    def test_scaling_uses_training_only(self):
        train=self.d.iloc[:170];test=self.d.iloc[170:].copy();test['age']=10000.
        prep=r.FoldPreprocessor(self.columns,r.ModelSpec('M1')).fit(train)
        before=prep.output_scaler_.mean_.copy();prep.transform(test)
        np.testing.assert_array_equal(prep.output_scaler_.mean_,before)
        self.assertAlmostEqual(prep.fixed_medians_['age'],train.age.median())
    def test_m3_omits_eitn_while_sharing_imaging_with_m4(self):
        prep=e.LegacyPrep(self.columns,36).fit(self.d)
        a=e.estimate(prep,self.d,1.,.25);b=e.estimate(prep,self.d,1.,.25,include=True,e_C=.1)
        self.assertIs(a['preprocessor'],b['preprocessor']);self.assertEqual(a['estimator']['lam'],b['estimator']['lam'])
        altered=self.d.copy();altered[r.FIXED_EITN]=1-self.d[r.FIXED_EITN]
        np.testing.assert_array_equal(e.predict(a,self.d),e.predict(a,altered))
    def test_convex_objective_gradient(self):
        rng=np.random.default_rng(3);x=rng.normal(size=(60,5));z=rng.normal(size=(60,4));y=rng.binomial(1,.3,60);w=np.ones(60)
        theta=rng.uniform(.1,.4,14)
        error=check_grad(lambda v:e.objective(v,x,z,y,w,.1,.7,.2)[0],lambda v:e.objective(v,x,z,y,w,.1,.7,.2)[1],theta)
        self.assertLess(error,1e-6)
    def test_predictor_feature_list_has_no_eitn(self):
        cols=json.loads((Path(__file__).resolve().parents[1]/'configs/conventional_features.json').read_text())
        self.assertEqual(len(cols),1848);self.assertFalse(any('eitn' in c.lower() or 'q_hl' in c.lower() for c in cols))
    def test_external_expanded_cox_is_not_enabled(self):
        with self.assertRaises(ValueError):paired_adjustment(pd.DataFrame(),['log_ebv_pre'],external=True)

class FileTests(unittest.TestCase):
    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):output_dir(tmp)
    def test_invalid_age_is_missing_without_removing_participant(self):
        d,columns=cohort(70,'INT',1);d.loc[0,'age']=551
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';d.to_csv(p,index=False);result=load_frame(p,columns)
            self.assertEqual(len(result),70);self.assertTrue(np.isnan(result.loc[0,'age']))

    def test_duplicate_input_keys_fail(self):
        d,columns=cohort(70,'INT',1);d.loc[1,'patient_id']=d.loc[0,'patient_id']
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'data.csv';d.to_csv(p,index=False)
            with self.assertRaises(ValueError):load_frame(p,columns)

if __name__=='__main__':unittest.main()
