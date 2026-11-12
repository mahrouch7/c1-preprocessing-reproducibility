"""Synthetic/mocked recovery checks: no live API, acquired fits or task submission."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

PACKAGE = Path(__file__).resolve().parents[1]
CANONICAL = Path('F:/ResearchRepositories/ieee9bus_research_canonical')
sys.path[:0] = [str(PACKAGE/'scripts'), str(CANONICAL/'Contributions/C1_Preprocessing_and_dataset/R07/scripts'),
               str(CANONICAL/'research_record/tools')]
import c1e32_campaign as campaign
import c1e32_common as c
import c1e32_spec as spec
import c1e32_analysis as analysis
import c1e32_recovery as recovery
import c1e32_recovery_gate as gate


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root/'task.zip'; self.archive.write_bytes(b'SYNTHETIC TASK FIXTURE')
        self.units = []
        for definition in spec.learning_plan():
            cfg = {k: definition[k] for k in ('preprocessor','fold','classifier','expected_fits')}
            cfg['stage'] = 'B'
            self.units.append(campaign.unit(self.archive,'1'*64,cfg,definition['role'],'a'*40))
        self.assignment = campaign.freeze(self.units,'2'*64)
        self.kept = {u['unit_id'] for u in self.units if u['identity']['fold']==1 and
            u['identity']['preprocessor']=='none' and u['identity']['classifier'] in
            ('logistic_regression','random_forest','decision_tree','gradient_boosting')}
        (self.root/'original/stage_b').mkdir(parents=True)
        c.write(self.root/'original/stage_b/assignment.json', self.assignment)
        (self.root/'transport/markers').mkdir(parents=True)
        for uid in self.kept: c.write(self.root/'transport/markers'/(uid+'.json'), {'synthetic':True})

    def reconcile(self):
        path = self.root/'original/stage_b/assignment.json'
        with patch.object(recovery,'OLD_ASSIGNMENT_FILE',c.sha(path)), patch.object(recovery,'KEPT',self.kept), \
                patch('run_c1e25_pipeline.returned_archive',return_value=self.archive) as reader:
            result = recovery.reconcile(self.root/'original',self.root/'transport')
            self.assertEqual(reader.call_count,4)
        return result

    def test_exact_remaining_budget_and_roles(self):
        original, missing = self.reconcile()
        self.assertEqual(len(missing),201)
        self.assertEqual(sum(v['expected_fits'] for v in missing),18286)
        self.assertEqual(sum(v['expected_fits'] for v in missing if v['role']=='PC1'),9267)
        self.assertEqual(sum(v['expected_fits'] for v in missing if v['role']=='PC2'),9019)
        self.assertEqual(644 + 18286 + 97,19027)

    def test_extra_return_refuses_resubmission(self):
        extra = next(u for u in self.units if u['unit_id'] not in self.kept)
        c.write(self.root/'transport/markers'/(extra['unit_id']+'.json'), {'synthetic':True})
        with self.assertRaisesRegex(ValueError,'Original returns changed'): self.reconcile()

    def test_composite_aggregation_preserves_origin_and_full_coverage(self):
        kept_units = [u for u in self.units if u['unit_id'] in self.kept]
        missing_units = []
        for u in self.units:
            if u['unit_id'] in self.kept: continue
            cfg = {k: u['identity'][k] for k in ('stage','preprocessor','fold','classifier','expected_fits')}
            missing_units.append(campaign.unit(self.archive,'1'*64,cfg,u['role'],'b'*40))
        new = campaign.freeze(missing_units,'3'*64)
        result = analysis.collection_units([(self.assignment,self.kept),(new,None)])
        self.assertEqual(len(result),205)
        self.assertEqual(sum(u['expected_fits'] for u,_ in result),18930)
        self.assertTrue(all(origin is self.assignment for u,origin in result if u['unit_id'] in self.kept))
        with self.assertRaisesRegex(ValueError,'Duplicate logical scientific unit'):
            analysis.collection_units([(self.assignment,None),(new,None)])
        with self.assertRaisesRegex(ValueError,'coverage differs'):
            analysis.collection_units([(self.assignment,self.kept)])

    def test_tampered_full_assignment_cannot_be_used(self):
        self.assignment['units'][0]['expected_fits'] += 1
        with self.assertRaises(ValueError): analysis.collection_units(self.assignment)

    def test_gate_capsule_contains_exact_fix_and_original_diagnostic(self):
        definition, directory = recovery.prepare_gate(PACKAGE,CANONICAL,self.root,'c'*40,r'H:\r07node_e32\workspace')
        self.assertEqual(definition['expected_fits'],0)
        self.assertEqual(definition['identity']['stage'],'NUMERIC')
        with zipfile.ZipFile(directory/(definition['task_archive_sha256']+'.zip')) as archive:
            self.assertEqual(archive.read('run_unit.py'),(PACKAGE/'scripts/c1e32_recovery_gate.py').read_bytes())
            self.assertEqual(archive.read('c1e32_probabilities.py'),(PACKAGE/'scripts/c1e32_probabilities.py').read_bytes())
            self.assertEqual(archive.read('c1e32_prediction_diagnostic.py'),(CANONICAL/'research_record/tools/c1e32_prediction_diagnostic.py').read_bytes())

    def test_wrong_confirmation_stops_before_resume(self):
        args = ['recovery','--confirm-new-fits','18930','--acknowledge-prior-partial-fits','97']
        with patch.object(sys,'argv',args),patch.object(recovery,'resume',side_effect=AssertionError('NO LAUNCH')):
            with self.assertRaisesRegex(ValueError,'Explicit 18286'): recovery.main()


class GateTests(unittest.TestCase):
    def setUp(self):
        import gzip
        import pickle
        import numpy as np
        from test_c1e32_prediction_diagnostic import synthetic_model
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/gate.diagnostic.UNIT
        self.task = self.root/'execution/task'; self.output = self.root/'execution/output'
        (self.task/'data').mkdir(parents=True); self.output.mkdir()
        c.write(self.task/'data/metadata.json',[{'fold':1},{'fold':2},{'fold':1},{'fold':2}])
        inputs = {'metadata':{'path':'data/metadata.json'}}
        for condition in gate.diagnostic.CONDITIONS:
            np.save(self.task/'data'/(condition+'.npy'),np.ones((4,36)),allow_pickle=False)
            inputs[condition] = {'path':'data/'+condition+'.npy'}
        c.write(self.task/'configuration.json',{'inputs':inputs})
        self.path = self.output/('model_'+gate.diagnostic.JOB+'.pickle.gz')
        with gzip.open(self.path,'wb') as stream: pickle.dump(synthetic_model(),stream,protocol=5)
        self.digest = c.sha(self.path)
        self.original = {'status':'ORIGINAL_CONTRACT_REJECTION_REPRODUCED',
            'first_rejected_condition_in_original_order':'awgn__20',
            'artifacts':{'retained_model':{'sha256':self.digest}}}

    def test_same_model_three_views_no_fit_no_write(self):
        from sklearn.pipeline import Pipeline
        with patch.object(gate.diagnostic,'diagnose',return_value=self.original), \
                patch.object(Pipeline,'fit',side_effect=AssertionError('FIT FORBIDDEN')):
            result = gate.compare(self.root,self.digest)
        self.assertEqual(result['fits'],0)
        self.assertEqual(len(result['comparisons']),3)
        self.assertTrue(all(v['main_argmax_disagreements']==0 for v in result['comparisons']))
        self.assertEqual(c.sha(self.path),self.digest)

    def test_changed_model_stops_before_comparison(self):
        with patch.object(gate.diagnostic,'diagnose',return_value=self.original):
            with self.assertRaisesRegex(ValueError,'Retained model changed'):
                gate.compare(self.root,'0'*64)

    def test_changed_argmax_stops_campaign_gate(self):
        import numpy as np
        p=np.zeros((2,12)); p[:,1]=1
        with patch.object(gate.diagnostic,'diagnose',return_value=self.original), \
                patch.object(gate.posterior,'checked_prediction',return_value=(np.array(['ab','ab']),p)):
            with self.assertRaisesRegex(ValueError,'classification decisions changed'):
                gate.compare(self.root,self.digest)


if __name__ == '__main__': unittest.main()
