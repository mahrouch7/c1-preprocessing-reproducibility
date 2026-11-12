"""Pure helper tests; no figures, imports of plotting packages or scientific runs."""
import ast
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import reproduce_paper_figures as figures


class PaperFigureTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def test_unsafe_paths_refused(self):
        for value in ('../a','data/../../a','C:/outside','data/a:stream','/absolute','data//a'):
            with self.subTest(value=value),self.assertRaises(ValueError):figures.safe(self.root,value)

    def test_payload_hash_and_seven_numbered_figures(self):
        p=self.root/'data';p.mkdir();(p/'a.csv').write_bytes(b'x\n1\n')
        m={'schema':1,'figures':[{'number':i} for i in range(1,8)],'files':[{'path':'data/a.csv','sha256':figures.sha(p/'a.csv'),'bytes':4}]}
        figures.verify_payload(self.root,m)
        changed=copy.deepcopy(m);changed['figures'].pop()
        with self.assertRaises(ValueError):figures.verify_payload(self.root,changed)
        (p/'a.csv').write_bytes(b'changed')
        with self.assertRaises(ValueError):figures.verify_payload(self.root,m)

    def test_completed_comparison_and_table_identity_required(self):
        (self.root/'analysis_receipt.json').write_text(json.dumps({'status':'COMPLETE_DESCRIPTIVE_ANALYSIS_FOR_AUTHOR_REVIEW','new_training':False,'new_hypothesis_tests':False}))
        (self.root/'table.csv').write_bytes(b'x\n1\n')
        m={'dynamic_comparison_tables_sha256':{'table.csv':figures.sha(self.root/'table.csv')}}
        figures.verify_comparison(self.root,m)
        (self.root/'table.csv').write_bytes(b'different')
        with self.assertRaises(ValueError):figures.verify_comparison(self.root,m)

    def test_personal_configuration_is_redirected_not_whitelisted(self):
        inherited={'USERPROFILE':'old-profile','HOME':'old-home','RL_test':'custom',
                   'PYTHONPATH':'external-code','SystemRoot':'system-folder'}
        env=figures.configuration_environment(self.root,inherited)
        self.assertEqual(env['USERPROFILE'],str(self.root/'isolated_profile'))
        self.assertEqual(env['HOME'],env['USERPROFILE'])
        self.assertEqual(env['MPL_IGNORE_SYSTEM_FONTS'],'1')
        self.assertNotIn('RL_test',env);self.assertNotIn('PYTHONPATH',env)
        self.assertEqual(env['SystemRoot'],'system-folder')
        self.assertEqual(inherited['USERPROFILE'],'old-profile')

    def test_frozen_data_counts_and_real_example_contract(self):
        data=self.root/'data';(data/'waveforms').mkdir(parents=True)
        (data/'abc_abcg_audit_summary.json').write_text(json.dumps({'pairs':153,'exact_point_equal_pairs':0}))
        (data/'e42_figure_data.csv').write_text('value\n'+'1\n'*120)
        summary=data/'waveforms/summary.json';summary.write_text(json.dumps({'synthetic_fixture':False,'selection':{'Id':963}}))
        figures.verify_data(self.root)
        summary.write_text(json.dumps({'synthetic_fixture':True,'selection':{'Id':963}}))
        with self.assertRaises(ValueError):figures.verify_data(self.root)

    def test_generation_scope_excludes_network_and_acquisition(self):
        manifest={'figures':[{'number':i} for i in range(1,8)]}
        self.assertEqual([r['number'] for r in figures.chart_specs(manifest)],[3,4,5,6,7])
        manifest['figures'].pop()
        with self.assertRaises(ValueError):figures.chart_specs(manifest)

    def test_real_payload_inventory_and_driver_syntax_without_execution(self):
        bundle=Path(__file__).resolve().parent/'payload';manifest=figures.read(bundle/'manifest.json')
        figures.verify_payload(bundle,manifest)
        drivers=[r for r in manifest['files'] if r['kind']=='adapted_generator']
        self.assertEqual(len(drivers),11)
        self.assertEqual(set(manifest['dynamic_comparison_tables_sha256']),{'metrics_per_run.csv','metrics_summary.csv'})
        for row in drivers:ast.parse((bundle/row['path']).read_bytes(),filename=row['path'])
        # A reference-only copy cannot accidentally be presented as a generation driver.
        references=[r for r in manifest['files'] if r['kind']=='approved_reference_for_QA_only']
        self.assertEqual(len(references),7)
        self.assertTrue(all(r['path'].startswith('reference/') for r in references))


if __name__=='__main__':unittest.main()
