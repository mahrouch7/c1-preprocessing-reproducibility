"""Synthetic tests. No internet, environment installation or scientific imports."""
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import clean_environment as ce
from environment_probe import validate_layout


class CleanEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()

    def wheel_document(self):
        return {'info':{'name':'example','version':'1.2.3'},'urls':[
            {'filename':'example-1.2.3-py3-none-any.whl','packagetype':'bdist_wheel','yanked':False,
             'url':'https://files.pythonhosted.org/packages/example.whl','size':123,'digests':{'sha256':'a'*64}}]}

    def test_only_exact_package_version_pins(self):
        cfg={'schema':1,'profiles':{'comparison':{'numpy':'2.3.5'},'graphics':{'matplotlib':'3.11.2'}}}
        ce.validate_pins(cfg)
        bad=copy.deepcopy(cfg);bad['profiles']['comparison']['numpy']='>=2'
        with self.assertRaises(ValueError):ce.validate_pins(bad)
        bad=copy.deepcopy(cfg);bad['profiles']['comparison']['../code']='1.0'
        with self.assertRaises(ValueError):ce.validate_pins(bad)

    def test_only_supported_wheel_tags(self):
        self.assertEqual(ce.wheel_rank('example-1.2.3-cp312-cp312-win_amd64.whl'),0)
        self.assertEqual(ce.wheel_rank('example-1.2.3-py2.py3-none-any.whl'),1)
        self.assertEqual(ce.wheel_rank('example-1.2.3-py3-none-win_amd64.whl'),1)
        for filename in ('example-1.2.3.tar.gz','../example.whl','example-1-cp313-cp313-win_amd64.whl',
                         'example-1-cp312-cp312-win32.whl','example-1-cp312-cp312-manylinux_x86_64.whl'):
            with self.subTest(filename=filename):self.assertIsNone(ce.wheel_rank(filename))

    def test_compiled_wheel_preferred_and_yanked_refused(self):
        document=self.wheel_document();compiled=copy.deepcopy(document['urls'][0])
        compiled['filename']='example-1.2.3-cp312-cp312-win_amd64.whl';document['urls'].append(compiled)
        self.assertEqual(ce.choose_wheel(document,'example','1.2.3')['filename'],compiled['filename'])
        for item in document['urls']:item['yanked']=True
        with self.assertRaises(ValueError):ce.choose_wheel(document,'example','1.2.3')

    def test_download_contract_rejects_wrong_version_host_and_hash(self):
        for change in ('version','host','hash'):
            doc=self.wheel_document()
            if change=='version':doc['info']['version']='9.0'
            elif change=='host':doc['urls'][0]['url']='https://unrelated.invalid/file.whl'
            else:doc['urls'][0]['digests']['sha256']='bad'
            with self.subTest(change=change),self.assertRaises(ValueError):ce.choose_wheel(doc,'example','1.2.3')

    def test_stream_verifies_downloaded_bytes_before_finalizing(self):
        content=b'synthetic wheel bytes';digest=hashlib.sha256(content).hexdigest();target=self.root/'good.whl'
        ce.download_stream(io.BytesIO(content),target,digest,len(content))
        self.assertEqual(target.read_bytes(),content)
        self.assertFalse((self.root/'good.whl.partial').exists())
        bad=self.root/'bad.whl'
        with self.assertRaises(ValueError):ce.download_stream(io.BytesIO(content),bad,'0'*64,len(content))
        self.assertFalse(bad.exists());self.assertTrue((self.root/'bad.whl.partial').exists())

    def test_existing_wheel_is_preserved(self):
        target=self.root/'existing.whl';target.write_bytes(b'original');content=b'new'
        with self.assertRaises(ValueError):ce.download_stream(io.BytesIO(content),target,hashlib.sha256(content).hexdigest(),len(content))
        self.assertEqual(target.read_bytes(),b'original')

    def test_environment_layout_rejects_base_and_external_site_packages(self):
        prefix=self.root/'venv';base=self.root/'base';cfg='include-system-site-packages = false\n'
        validate_layout(prefix,base,False,[str(prefix/'Lib/site-packages'),str(base/'Lib')],cfg)
        invalid=[(prefix,prefix,False,[],cfg),(prefix,base,True,[],cfg),
                 (prefix,base,False,[str(base/'Lib/site-packages')],cfg),(prefix,base,False,[],'include-system-site-packages = true')]
        for args in invalid:
            with self.subTest(args=args),self.assertRaises(ValueError):validate_layout(*args)

    def test_three_figure_headers_and_source_tables_are_checked(self):
        folder=self.root/'plots';figures=folder/'figures';figures.mkdir(parents=True)
        table=folder/'table.csv';table.write_bytes(b'x\n1\n');digest=ce.sha(table)
        names=['fig_a','fig_b','fig_c']
        for name in names:
            for extension,header in [('pdf',b'%PDF-1.4\n'),('png',b'\x89PNG\r\n\x1a\n'),('svg',b'<?xml version="1.0"?><svg ' )]:
                (figures/(name+'.'+extension)).write_bytes(header+b'x'*120)
        manifest={'figures':[{'name':name} for name in names],
                  'scientific_predictions_or_metrics_recomputed':False,'source_tables_sha256':{'table.csv':digest}}
        (figures/'figures_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        result=ce.figure_checks(folder,[{'file':'table.csv','actual_sha256':digest}])
        self.assertEqual(len(result['files']),9);self.assertEqual(result['actual_PDF_visual_QA'],'PENDING')
        table.write_bytes(b'changed')
        with self.assertRaises(ValueError):ce.figure_checks(folder,[{'file':'table.csv','actual_sha256':digest}])


if __name__=='__main__':unittest.main()
