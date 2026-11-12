"""Synthetic and mocked contracts only. No acquired data fit or live submission."""
from __future__ import annotations
import ast
from collections import Counter
import io
import json
from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings
import zipfile

PACKAGE=Path(__file__).resolve().parents[1]
CANONICAL=Path('F:/ResearchRepositories/ieee9bus_research_canonical')
sys.path[:0]=[str(PACKAGE/'scripts'),str(CANONICAL/'Contributions/C1_Preprocessing_and_dataset/R07/scripts')]
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import c1e32_spec as s
import c1e32_common as c
import c1e32_dataset as dataset
import c1e32_worker as worker
import c1e32_statistics as stats
import c1e32_campaign as campaign
import c1e32_figures as figures
import c1e32_analysis as analysis
import c1e32_pdf as pdf_export
import c1e27_signal as signal

def rows_fixture(production=False):
    result=[]
    for family in range(39 if production else 5):
        for label in s.LABELS:
            count=(4 if label=='noFault' else 39 if family==0 else 3) if production else 1
            for number in range(count):
                i=len(result); result.append({'Id':i,'window_id':f'synthetic_{i}','acquisition_id':f'synthetic_{i}',
                    'family_id':f'F{family:03}','source_group_id':f'F{family:03}','fold':family%5+1,
                    'fault':label,'healthy_anchor':number==0 if label=='noFault' else None,
                    'factors':[1.,1.,1.],'nominal':family==0,'line_id':'synthetic_line','fraction':.5,
                    'density_control_included':number<3 or label=='noFault' or family!=0})
    return result

class FakeEstimator(ClassifierMixin,BaseEstimator):
    """Deliberately not a fitted scientific classifier."""
    def __init__(self,warning=False): self.warning=warning
    def fit(self,x,y):
        self.classes_=np.array(s.LABELS); self.fitted_columns=x.shape[1]
        if self.warning:
            from sklearn.exceptions import ConvergenceWarning
            warnings.warn('SYNTHETIC convergence warning',ConvergenceWarning)
        return self
    def predict(self,x): return np.array(['ag']*len(x))
    def predict_proba(self,x): return np.ones((len(x),12))/12

class TestScope(unittest.TestCase):
    def test_fit_budget_and_two_host_balance(self):
        scope=s.scope(); self.assertEqual(scope['fit_budget'],18930)
        self.assertEqual(scope['fit_budget_per_role'],{'PC1':9466,'PC2':9464})
        self.assertEqual(len(s.learning_plan()),205)
    def test_all_jobs_have_unique_complete_identities(self):
        ids=[j['job_id'] for u in s.learning_plan() for j in s.jobs(u['preprocessor'],u['fold'],u['classifier'])]
        self.assertEqual(len(ids),len(set(ids))); self.assertEqual(len(ids),18930)
    def test_scope_has_no_old_model_reuse_or_simulation(self):
        self.assertEqual(s.scope()['historical_fits_reused'],0); self.assertEqual(s.scope()['new_native_simulations'],0)
    def test_all_prospective_contrasts_exist(self):
        cases=stats.expected_cases(); comparisons=stats.comparison_plan()
        self.assertEqual(len(cases),11814); self.assertEqual(len(comparisons),728)
        self.assertTrue(all(s.digest(v[k]) in cases for v in comparisons for k in ('left','right')))
    def test_matrix_covers_every_main_axis(self):
        self.assertEqual(len(s.BASE_CONDITIONS),26); self.assertEqual(len(s.CONDITIONS),30)
        self.assertEqual(len(s.PREPS),5); self.assertEqual(len(s.MODELS),8)
    def test_raw_is_descriptors_and_feature_names_exclude_privileged_metadata(self):
        self.assertEqual(len(s.REPS['RAW12']),12); self.assertEqual(len(s.FEATURES),36)
        self.assertFalse(any(v in ' '.join(s.FEATURES).lower() for v in ('distance','fault_line','family_id','position')))
    def test_skewness_only_and_bundle_are_distinct(self):
        self.assertEqual(len(s.REPS['RAW_SKEW']),18); self.assertEqual(len(s.REPS['MULTI_SKEW']),24)
        self.assertEqual(len(s.REPS['RAW_PLUS']),30); self.assertEqual(len(s.REPS['MULTI_PLUS']),36)
        self.assertTrue(set(s.REPS['RAW_SKEW'])<set(s.REPS['RAW_PLUS']))
    def test_channel_controls_are_disjoint(self):
        for suffix in ('RMS','RAW','MULTI'):
            self.assertFalse(set(s.REPS['I_'+suffix])&set(s.REPS['V_'+suffix]))
    def test_extra_noise_repeats_are_bounded(self):
        repeats=[v for v in s.CONDITIONS if v['repeat']]
        self.assertEqual(len(repeats),4); self.assertEqual({v['repeat'] for v in repeats},{1,2})
    def test_changed_budget_or_scope_is_rejected(self):
        cfg={'experiment_id':s.EXPERIMENT,'stage':'B','scope_sha256':s.digest(s.scope()),'preprocessor':'savgol',
             'fold':1,'classifier':'svm','expected_fits':78}
        s.validate_configuration(cfg,78)
        with self.assertRaises(ValueError): s.validate_configuration(cfg,77)
        cfg['scope_sha256']='0'*64
        with self.assertRaises(ValueError): s.validate_configuration(cfg,78)

class TestFamilies(unittest.TestCase):
    def test_production_fixture_contract(self): s.validate_rows(rows_fixture(True))
    def test_small_synthetic_contract(self): s.validate_rows(rows_fixture(),production=False)
    def test_family_assignment_is_order_independent_and_balanced(self):
        names=[f'F{i:03}' for i in range(39)]
        a=dataset.family_folds(names); self.assertEqual(a,dataset.family_folds(names[::-1]))
        self.assertEqual(sorted(Counter(a.values()).values()),[7,8,8,8,8])
    def test_cross_fold_family_is_rejected(self):
        rows=rows_fixture(); rows[1]['fold']=2
        with self.assertRaises(ValueError): s.validate_rows(rows,False)
    def test_duplicate_window_is_rejected(self):
        rows=rows_fixture(); rows[1]['window_id']=rows[0]['window_id']
        with self.assertRaises(ValueError): s.validate_rows(rows,False)
    def test_duplicate_acquisition_is_rejected(self):
        rows=rows_fixture(); rows[1]['acquisition_id']=rows[0]['acquisition_id']
        with self.assertRaisesRegex(ValueError,'Repeated acquisition'): s.validate_rows(rows,False)

    def audit_fixture(self):
        rows=rows_fixture(True)
        for i,r in enumerate(rows):
            r['factors']=[.75+int(r['family_id'][1:])*.01,1,1]
            r['raw_sha256']=hashlib_sha(('raw'+str(i)).encode())
            r['clean_window_numeric_sha256']=hashlib_sha(('window'+str(i)).encode())
        return rows

    def test_direct_leakage_audit_accepts_unique_family_regimes(self):
        evidence=dataset.leakage_audit(self.audit_fixture())
        self.assertEqual(evidence['cross_family_exact_load_collisions'],0)

    def test_exact_regime_crossing_families_stops_before_training(self):
        rows=self.audit_fixture(); rows[-1]['factors']=rows[0]['factors']
        with self.assertRaisesRegex(ValueError,'Exact load regime'): dataset.leakage_audit(rows)

    def test_repeated_raw_or_numeric_window_stops_before_training(self):
        for name in ('raw_sha256','clean_window_numeric_sha256'):
            rows=self.audit_fixture(); rows[-1][name]=rows[0][name]
            with self.assertRaisesRegex(ValueError,'Repeated saved waveform'): dataset.leakage_audit(rows)
    def test_missing_test_class_is_rejected(self):
        rows=rows_fixture(); rows[0]['fault']='ab'
        with self.assertRaises(ValueError): s.validate_rows(rows,False)
    def test_old_858_cohort_is_rejected(self):
        with self.assertRaises(ValueError): s.validate_rows(rows_fixture(True)[:858])
    def test_training_support_control_does_not_change_test_rows(self):
        rows=rows_fixture(True); job={'fold':2,'anchor_only':True,'density_balanced':False}
        selected=worker.training_mask(rows,job)
        self.assertFalse(any(m and r['fold']==2 for r,m in zip(rows,selected)))
        self.assertTrue(all(r['fault']!='noFault' or r['healthy_anchor'] for r,m in zip(rows,selected) if m))
    def test_density_control_retains_healthy_and_reduces_nominal_faults(self):
        rows=rows_fixture(True); selected=worker.training_mask(rows,{'fold':2,'anchor_only':False,'density_balanced':True})
        self.assertEqual(sum(m and r['nominal'] and r['fault']=='ab' for r,m in zip(rows,selected)),3)
        self.assertEqual(sum(m and r['fault']=='noFault' for r,m in zip(rows,selected)),124)

class TestStatistics(unittest.TestCase):
    def setUp(self):
        self.rows=rows_fixture(); self.truth=np.array([s.LABELS.index(r['fault']) for r in self.rows]); self.ids=np.arange(len(self.rows))
    def test_family_class_cells_each_have_weight_one(self):
        rows=rows_fixture(True); w=stats.weights(rows)
        for family in {r['family_id'] for r in rows}:
            for label in s.LABELS:
                self.assertAlmostEqual(sum(v for r,v in zip(rows,w) if r['family_id']==family and r['fault']==label),1.)
    def test_perfect_scores(self):
        score,matrix=stats.score(self.rows,self.truth,self.truth)
        for name in ('macro_f1','accuracy','balanced_accuracy','mcc'): self.assertAlmostEqual(score[name],1.)
        self.assertAlmostEqual(score['healthy_false_alarm'],0.); self.assertEqual(matrix.shape,(12,12))
    def test_constant_prediction_baseline(self):
        score,_=stats.score(self.rows,self.truth,np.zeros(len(self.rows),dtype=int))
        self.assertAlmostEqual(score['balanced_accuracy'],1/12); self.assertAlmostEqual(score['macro_f1'],2/(13*12))
    def test_primary_score_invariant_to_duplicate_density(self):
        pred=self.truth.copy(); pred[1]=0
        score,_=stats.score(self.rows,self.truth,pred)
        rows=self.rows+[dict(self.rows[1]) for _ in range(20)]; truth=np.r_[self.truth,np.repeat(self.truth[1],20)]; p=np.r_[pred,np.zeros(20,int)]
        other,_=stats.score(rows,truth,p)
        self.assertAlmostEqual(score['macro_f1'],other['macro_f1'])
        pooled,_=stats.score(rows,truth,p,'pooled'); self.assertNotAlmostEqual(other['macro_f1'],pooled['macro_f1'])
    def test_metric_values_match_sklearn_weighted_reference(self):
        from sklearn.metrics import f1_score,balanced_accuracy_score,matthews_corrcoef
        pred=np.roll(self.truth,1); pred[:5]=self.truth[:5]
        score,_=stats.score(self.rows,self.truth,pred); w=stats.weights(self.rows)
        self.assertAlmostEqual(score['macro_f1'],f1_score(self.truth,pred,average='macro',sample_weight=w))
        self.assertAlmostEqual(score['balanced_accuracy'],balanced_accuracy_score(self.truth,pred,sample_weight=w))
        self.assertAlmostEqual(score['mcc'],matthews_corrcoef(self.truth,pred,sample_weight=w))
    def test_identical_predictions_have_zero_interval_and_p_one(self):
        data={'row_ids':self.ids,'truth':self.truth,'predicted':self.truth}
        result=stats.paired(data,data,self.rows,bootstrap=100,swaps=200)
        self.assertEqual(result['difference'],0); self.assertEqual(result['permutation_p'],1)
        self.assertEqual(result['lower_95'],0); self.assertEqual(result['families'],5)
    def test_paired_identity_mismatch_is_rejected(self):
        left={'row_ids':self.ids,'truth':self.truth,'predicted':self.truth}
        right={**left,'row_ids':self.ids[::-1]}
        with self.assertRaises(ValueError): stats.paired(left,right,self.rows,bootstrap=10,swaps=20)
    def test_cluster_bootstrap_and_permutation_are_reproducible(self):
        left={'row_ids':self.ids,'truth':self.truth,'predicted':self.truth}
        right={**left,'predicted':np.zeros(len(self.rows),dtype=int)}
        a=stats.paired(left,right,self.rows,bootstrap=100,swaps=200,seed=78)
        self.assertEqual(a,stats.paired(left,right,self.rows,bootstrap=100,swaps=200,seed=78))
        self.assertGreater(a['lower_95'],0); self.assertGreater(a['difference'],.9)
    def test_holm_correction(self):
        self.assertTrue(np.allclose(stats.holm([.01,.04,.03]),[.03,.06,.06]))
        self.assertEqual(stats.holm([1.,1.]).tolist(),[1.,1.])
    def test_all_class_support_required(self):
        mask=self.truth!=0
        with self.assertRaises(ValueError): stats.score(self.rows,self.truth,self.truth,mask=mask)

class TestSignals(unittest.TestCase):
    def setUp(self):
        time=np.arange(8000)/40000
        self.x=np.column_stack([(i+1)*np.sin(2*np.pi*60*time+i*.2) for i in range(6)])
        self.row=rows_fixture()[0]
    def test_clean_is_exact_and_repeat_identity_is_deterministic(self):
        actual,_=worker.observed(self.x,self.row,'clean'); self.assertTrue(np.array_equal(actual,self.x))
        a,_=worker.observed(self.x,self.row,'awgn__20__r1'); b,_=worker.observed(self.x,self.row,'awgn__20__r1')
        self.assertTrue(np.array_equal(a,b))
        other,_=worker.observed(self.x,self.row,'awgn__20__r2'); self.assertFalse(np.array_equal(a,other))
    def test_measured_awgn_snr(self):
        for level in s.base.SNR_LEVELS:
            actual,_=worker.observed(self.x,self.row,'awgn__'+str(level))
            measured=10*np.log10(np.mean(self.x**2,axis=0)/np.mean((actual-self.x)**2,axis=0))
            self.assertTrue(np.allclose(measured,level,atol=1e-7,rtol=0))
    def test_composite_snr_is_not_total_error_snr(self):
        _,report=worker.observed(self.x,self.row,'mixed_gain_additive__120')
        self.assertTrue(np.allclose(report['additive_snr_db'],120,atol=1e-7,rtol=0))
        self.assertLess(max(report['total_error_snr_db']),100)
    def test_all_filters_are_finite_and_same_shape(self):
        for name in s.PREPS:
            y=signal.preprocess(self.x,name); self.assertEqual(y.shape,self.x.shape); self.assertTrue(np.isfinite(y).all())
    def test_base_descriptor_order(self):
        values=signal.features(self.x)
        self.assertTrue(np.array_equal(values[:18],s.base.feature_vector(self.x))); self.assertEqual(values.shape,(36,))
    def test_tpi_flags_do_not_clip_amplification(self):
        rng=np.random.default_rng(3); x=rng.normal(size=(8000,6)); value=signal.tpi(x,x,2*x)
        self.assertTrue(np.allclose(value[:,:,3],4)); self.assertTrue(np.all(value[:,:,3]>1))
    def test_degenerate_channel_descriptor_rule_is_finite(self):
        x=self.x.copy(); x[:,0]=0; features=signal.features(x)
        self.assertTrue(np.isfinite(features).all()); self.assertEqual(features[18:21].tolist(),[0.,3.,0.])

class TestExecution(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='C1_E32_synthetic_'); self.root=Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    def test_safe_paths_and_immutable_writes(self):
        for path in ('../escape','C:/escape','folder\\escape'):
            with self.assertRaises(ValueError): c.child(self.root,path)
        path=self.root/'receipt.json'; c.write(path,{'a':1}); c.write(path,{'a':1})
        with self.assertRaises(ValueError): c.write(path,{'a':2})
    def test_pc2_workspace_must_stay_on_dedicated_h(self):
        self.assertEqual(campaign.workspace(r'H:\r07node_e32\workspace'),r'H:\r07node_e32\workspace')
        for value in ('H:\\',r'C:\r07node_e32',r'H:\..\escape'):
            with self.assertRaises(ValueError): campaign.workspace(value)
    def test_estimator_factories_are_unfitted_and_scalers_are_explicit(self):
        for model in s.MODELS:
            for scale in s.NORMALIZERS:
                estimator=worker.make_estimator(model,scale)
                self.assertFalse(hasattr(estimator,'classes_'))
                self.assertEqual(hasattr(estimator,'named_steps'),scale!='none')
    def test_native_predict_is_not_confused_with_probability_argmax(self):
        model=FakeEstimator().fit(np.ones((3,1)),['ag']*3)
        pred,prob,native=worker.predict(model,np.ones((2,1)))
        self.assertEqual(pred.tolist(),[0,0]); self.assertEqual(native.tolist(),[5,5])
    def test_mocked_fit_preserves_warning_and_budget(self):
        rows=rows_fixture(); meta=self.root/'metadata.json'; c.write(meta,rows); inputs={'metadata':meta}
        for entry in s.CONDITIONS:
            path=self.root/(entry['name']+'.npy'); np.save(path,np.ones((len(rows),36)),allow_pickle=False); inputs[entry['name']]=path
        job=next(j for j in s.jobs('none',1,'logistic_regression') if j['study']=='channels' and j['training_condition']=='clean')
        output=self.root/'output'; output.mkdir(); cfg={'preprocessor':'none','fold':1,'classifier':'logistic_regression','expected_fits':1}
        with patch.object(s,'validate_rows',side_effect=lambda value:value),patch.object(s,'jobs',return_value=[job]):
            count=worker.fit(cfg,inputs,output,estimator_factory=lambda *args:FakeEstimator(warning=True))
        self.assertEqual(count,1); records=c.read(output/'fit_records.json'); self.assertEqual(records['reused_fits'],0)
        self.assertFalse(records['records'][0]['eligible_for_inference']); catalogue=c.read(output/'prediction_catalogue.json')
        self.assertTrue(all(not r['eligible_for_inference'] for r in catalogue))
        with np.load(output/catalogue[0]['path']) as data:
            self.assertTrue(np.all(data['predicted']==0)); self.assertTrue(np.all(data['native_predicted']==5))

    def test_scaler_does_not_see_extreme_heldout_values(self):
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        rows=rows_fixture(); meta=self.root/'metadata.json'; c.write(meta,rows); inputs={'metadata':meta}
        values=np.full((len(rows),36),3.); test=np.array([r['fold']==1 for r in rows]); values[test]=1e9
        for entry in s.CONDITIONS:
            path=self.root/(entry['name']+'.npy'); np.save(path,values,allow_pickle=False); inputs[entry['name']]=path
        job=next(j for j in s.jobs('none',1,'logistic_regression') if j['study']=='channels' and j['training_condition']=='clean')
        models=[]
        def factory(*args):
            model=Pipeline([('scale',StandardScaler()),('classifier',FakeEstimator())]); models.append(model); return model
        output=self.root/'scaler_output'; output.mkdir()
        cfg={'preprocessor':'none','fold':1,'classifier':'logistic_regression','expected_fits':1}
        with patch.object(s,'validate_rows',side_effect=lambda v:v),patch.object(s,'jobs',return_value=[job]),patch('builtins.print'):
            worker.fit(cfg,inputs,output,estimator_factory=factory)
        self.assertTrue(np.all(models[0].named_steps['scale'].mean_==3.))
        self.assertEqual(models[0].named_steps['scale'].n_samples_seen_,int((~test).sum()))
    def test_task_archives_are_deterministic_and_runner_bound(self):
        source=self.root/'metadata.json'; c.write(source,{'synthetic':True}); paths={'metadata':source}
        cfg={'inputs':c.pins(paths),'stage':'A'}
        a,ra=campaign.task(PACKAGE,CANONICAL,self.root/'a',cfg,paths)
        b,rb=campaign.task(PACKAGE,CANONICAL,self.root/'b',cfg,paths)
        self.assertEqual(c.sha(a),c.sha(b)); self.assertEqual(ra,rb)
        with zipfile.ZipFile(a) as archive:
            self.assertEqual(hashlib_sha(archive.read('run_unit.py')),ra)
            self.assertIn('c1e32_worker.py',archive.namelist()); self.assertNotIn('c1e27_recovery.py',archive.namelist())
    def test_foreign_input_hash_is_rejected(self):
        source=self.root/'data'; source.mkdir(); path=source/'x.json'; c.write(path,{'a':1})
        config={'inputs':{'x':{'path':'data/x.json','sha256':'0'*64}}}
        with self.assertRaises(ValueError): c.inputs(config,self.root)
    def test_svg_primitive_and_forest_are_well_formed(self):
        figure=figures.SVG(self.root/'sample.svg','SYNTHETIC TEST ONLY',600,400)
        figure.line([(50,100),(500,200)]); figure.finish('Synthetic fixture, no study results')
        import xml.etree.ElementTree as ET
        self.assertTrue(ET.parse(self.root/'sample.svg').getroot().tag.endswith('svg'))
        with self.assertRaises(ValueError): figure.dot(float('nan'),1,'red')
    def test_python_ast_and_no_hidden_external_execution(self):
        for path in (PACKAGE/'scripts').glob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8'))
            self.assertFalse(any(isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr in ('Popen','run') for node in tree.body))
    def test_complete_learning_unit_catalogue_with_fake_estimators(self):
        rows=rows_fixture(); meta=self.root/'metadata.json'; c.write(meta,rows); inputs={'metadata':meta}
        for entry in s.CONDITIONS:
            path=self.root/(entry['name']+'.npy'); np.save(path,np.ones((len(rows),36)),allow_pickle=False); inputs[entry['name']]=path
        output=self.root/'full_mock_unit'; output.mkdir()
        cfg={'preprocessor':'none','fold':1,'classifier':'logistic_regression','expected_fits':len(s.jobs('none',1,'logistic_regression'))}
        with patch.object(s,'validate_rows',side_effect=lambda value:value):
            with patch('builtins.print'):
                count=worker.fit(cfg,inputs,output,estimator_factory=lambda *args:FakeEstimator())
        entries=c.read(output/'prediction_catalogue.json'); identities={s.digest(r['case']) for r in entries}
        expected={key for key,case in stats.expected_cases().items() if case['classifier']=='logistic_regression' and case['preprocessor']=='none'}
        self.assertEqual(identities,expected); self.assertEqual(len(entries),len(identities)); self.assertEqual(count,cfg['expected_fits'])
    def test_preprocessing_unit_receipt_passes_qualified_result_contract(self):
        rows=rows_fixture(True); folder=self.root/'task'; data=folder/'data'; data.mkdir(parents=True)
        meta=data/'metadata.json'; c.write(meta,rows)
        t=np.arange(8000)/40000; raw=np.column_stack([(i+1)*np.sin(2*np.pi*60*t+i*.3) for i in range(6)])
        clean=data/'clean.npy'; np.save(clean,raw[None],allow_pickle=False)
        wheel=CANONICAL/'Contributions/C1_Preprocessing_and_dataset/R07/deployment/scientific_dependencies/pywavelets-1.8.0-cp312-cp312-win_amd64.whl'
        import shutil
        shutil.copyfile(wheel,data/'pywavelets_wheel.whl')
        paths={'metadata':meta,'clean':clean,'pywavelets_wheel':data/'pywavelets_wheel.whl'}
        config=campaign.config('A',paths,chunk=0,row_ids=[0]); configuration=folder/'configuration.json'; c.write(configuration,config)
        output=self.root/'output'; output.mkdir()
        args=SimpleNamespace(configuration=str(configuration),output=str(output),configuration_sha256=c.sha(configuration),
            unit_id='1'*64,role='PC1',assignment_sha256='2'*64,expected_fits=0)
        with patch.object(worker,'activate_wavelet'),patch('builtins.print'):
            worker.execute(args)
        archive=Path(shutil.make_archive(str(self.root/'result'),'zip',root_dir=output))
        from r07.distributed.contracts import validate_result_archive
        result=validate_result_archive(archive,{'unit_id':args.unit_id,'role':args.role,
            'assignment_sha256':args.assignment_sha256,'configuration_sha256':args.configuration_sha256,'expected_fits':0})
        self.assertEqual(result['fits_completed'],0)
        with np.load(output/'features.npz') as values:
            self.assertEqual(values['features'].shape,(30,5,1,36)); self.assertTrue(np.isfinite(values['features']).all())
    def test_storage_headroom_contract(self):
        from c1e27_storage_guard import check_space
        with self.assertRaises(RuntimeError): check_space(self.root,40*1024**3,3*1024**3,disk_usage=lambda _:SimpleNamespace(free=1))
        free,_=check_space(self.root,40*1024**3,3*1024**3,disk_usage=lambda _:SimpleNamespace(free=100*1024**3))
        self.assertEqual(free,100*1024**3)

    def aggregate_fixture(self,folds=range(1,6),excluded=False,corrupt=False):
        rows=rows_fixture(); assignment={'units':[]}; archives={}
        case=stats.robust('logistic_regression','clean'); identity=s.digest(case)
        for fold in folds:
            unit={'unit_id':str(fold),'identity':{'preprocessor':'none','fold':fold,'classifier':'logistic_regression'},
                  'expected_fits':1,'role':'PC1','expected_host':'SYNTHETIC'}
            assignment['units'].append(unit); ids=np.array([r['Id'] for r in rows if r['fold']==fold])
            truth=np.array([s.LABELS.index(rows[i]['fault']) for i in ids]); probability=np.eye(12)[truth]
            stream=io.BytesIO(); np.savez_compressed(stream,row_ids=ids,truth=truth,predicted=truth,
                probabilities=probability,native_predicted=truth); payload=stream.getvalue()
            path=self.root/f'fold{fold}.zip'; archives[str(fold)]=path
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr('fit_records.json',json.dumps({'fits_completed':1,'reused_fits':0,
                    'records':[{'job':{'job_id':str(fold),'classifier':'logistic_regression'}}]}))
                archive.writestr('predictions.npz',payload)
                archive.writestr('prediction_catalogue.json',json.dumps([{'case':case,'path':'predictions.npz',
                    'sha256':'0'*64 if corrupt else hashlib_sha(payload),'eligible_for_inference':not(excluded and fold==1)}]))
        return rows,assignment,archives,{identity:case}

    def collect_fixture(self,rows,assignment,archives,expected):
        output=self.root/'aggregate'; output.mkdir()
        with patch('run_c1e25_pipeline.returned_archive',side_effect=lambda unit,*_:archives[unit['unit_id']]),\
             patch.object(stats,'expected_cases',return_value=expected),\
             patch.object(s,'jobs',side_effect=lambda prep,fold,model:[{'job_id':str(fold),'classifier':'logistic_regression'}]),\
             patch.object(analysis,'collection_units',side_effect=lambda value:[(u,value) for u in value['units']]),\
             patch.object(s,'scope',return_value={'fit_budget':len(assignment['units'])}):
            return analysis.collect(assignment,self.root,output,rows)

    def test_complete_five_fold_aggregation_preserves_exclusion_and_order(self):
        rows,assignment,archives,expected=self.aggregate_fixture(excluded=True)
        index,records=self.collect_fixture(rows,assignment,archives,expected)
        self.assertEqual(len(records),5); item=index[next(iter(expected))]
        self.assertFalse(item['eligible_for_inference'])
        data=analysis.pooled(self.root/'aggregate'/item['path'])
        self.assertEqual(data['row_ids'].tolist(),list(range(len(rows))))

    def test_missing_outer_fold_stops_aggregation(self):
        with self.assertRaisesRegex(ValueError,'Incomplete/duplicate outer folds'):
            self.collect_fixture(*self.aggregate_fixture(range(1,5)))

    def test_corrupted_prediction_stops_aggregation(self):
        with self.assertRaisesRegex(ValueError,'Prediction catalogue hash mismatch'):
            self.collect_fixture(*self.aggregate_fixture(corrupt=True))

    def test_pdf_converter_rejects_missing_runtime_without_installing(self):
        with self.assertRaises(ValueError): campaign.check_graphics(PACKAGE,self.root/'absent.exe',pdf_export.DEFAULT_FONT)

    def test_overleaf_panels_have_sufficient_logical_font_size(self):
        # PDF backend converts 720 SVG units to 504 pt; 12 units become 8.4 pt.
        from xml.etree import ElementTree as ET
        lookup={key:{'macro_f1':.6,'eligible_for_inference':True} for key in stats.expected_cases()}
        with patch.object(s.old,'NOISES',s.old.NOISES[:1]),patch.object(s,'PREPS',s.PREPS[:1]):
            figures.publication_curves(self.root,lookup)
        for source in self.root.glob('publication_*.svg'):
            tree=ET.parse(source); minimum=min(float(e.attrib['font-size']) for e in tree.iter() if e.tag.endswith('}text'))
            self.assertGreaterEqual(minimum*504/float(tree.getroot().attrib['width']),8)

def hashlib_sha(value):
    import hashlib
    return hashlib.sha256(value).hexdigest()

if __name__=='__main__': unittest.main()
