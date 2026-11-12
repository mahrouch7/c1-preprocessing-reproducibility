"""Synthetic tests only. The production comparator is never imported or run."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from guarded_compare import AccessGuard
import restore_validate as rv


class RestorationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.workspace=self.root/'workspace';self.workspace.mkdir()

    def archive(self,*,bad_hash=False,duplicate=False):
        payload={'data/example.json':b'{"test":true}\n','reference/table.csv':b'x,y\n1,2\n'}
        rows=[{'path':n,'sha256':hashlib.sha256(b).hexdigest(),'bytes':len(b)} for n,b in payload.items()]
        if bad_hash:rows[0]['sha256']='0'*64
        path=self.root/'fake.zip'
        with zipfile.ZipFile(path,'w') as z:
            for n,b in payload.items():z.writestr(n,b)
            if duplicate:z.writestr('DATA/example.json',b'collision')
            z.writestr('archive_info/test/MANIFEST.json',json.dumps({'schema':1,'files':rows}))
            z.writestr('archive_info/test/README.md','Synthetic fixture only.\n')
        return path,{'sha256':rv.sha_file(path),'group':'test','members':2,'source_bytes':sum(map(len,payload.values()))}

    def test_safe_paths_reject_traversal_drives_and_ads(self):
        for value in ('../x','data/../../x','C:/x','/x','data/x:stream','data//x','data/./x','\\\\host\\share'):
            with self.subTest(value=value),self.assertRaises(ValueError):rv.safe(self.workspace,value)
        self.assertEqual(rv.safe(self.workspace,'data/valid.json'),self.workspace/'data/valid.json')

    def test_extracts_and_hashes_only_into_new_workspace(self):
        archive,pin=self.archive();restored={}
        result=rv.restore_zip(archive,pin,self.workspace,restored)
        self.assertEqual(result['verified_data_members'],2)
        self.assertEqual(len(restored),2)
        self.assertEqual((self.workspace/'data/example.json').read_bytes(),b'{"test":true}\n')
        for name,digest in restored.items():self.assertEqual(rv.sha_file(self.workspace/name),digest)

    def test_wrong_zip_hash_stops_before_extract(self):
        archive,pin=self.archive();pin['sha256']='f'*64
        with self.assertRaises(ValueError):rv.restore_zip(archive,pin,self.workspace,{})
        self.assertEqual(list(self.workspace.iterdir()),[])

    def test_member_hash_mismatch_is_not_accepted(self):
        archive,pin=self.archive(bad_hash=True)
        with self.assertRaises(ValueError):rv.restore_zip(archive,pin,self.workspace,{})

    def test_case_collision_is_rejected(self):
        archive,pin=self.archive(duplicate=True)
        with self.assertRaises(ValueError):rv.restore_zip(archive,pin,self.workspace,{})
        self.assertEqual(list(self.workspace.iterdir()),[])

    def test_existing_file_is_never_overwritten(self):
        archive,pin=self.archive();p=self.workspace/'data/example.json';p.parent.mkdir();p.write_bytes(b'keep')
        with self.assertRaises(ValueError):rv.restore_zip(archive,pin,self.workspace,{})
        self.assertEqual(p.read_bytes(),b'keep')

    def test_guard_refuses_original_inputs_and_input_writes(self):
        library=self.root/'python';library.mkdir();g=AccessGuard(self.workspace,[library])
        g.check(self.workspace/'data/a.json');g.check(library/'module.py')
        g.check(self.workspace/'outputs/result.csv',write=True)
        for value,write in [(self.root/'original/data.npz',False),(library/'module.py',True),(self.workspace/'data/a.json',True)]:
            with self.subTest(path=value),self.assertRaises(PermissionError):g.check(value,write=write)
        self.assertEqual(len(g.blocked),3)

    def test_configuration_cannot_escape_restored_data(self):
        config={k:'data/example' for k in ['campaign_root','ml_analysis_root','resnet_run','resnet_audit','inceptiontime_run','inceptiontime_audit']}
        self.assertEqual(rv.local_paths(self.workspace,config)['output_root'],'outputs')
        config['campaign_root']='reference/example'
        with self.assertRaises(ValueError):rv.local_paths(self.workspace,config)

    def test_all_seventeen_tables_must_be_identical(self):
        result=self.workspace/'outputs';result.mkdir();ref=self.workspace/'reference/comparison';ref.mkdir(parents=True)
        restored={}
        for i in range(17):
            name=f'table{i:02}.csv';b=f'x\n{i}\n'.encode()
            (ref/name).write_bytes(b);(result/name).write_bytes(b)
            restored['reference/comparison/'+name]=hashlib.sha256(b).hexdigest()
        self.assertEqual(len(rv.table_checks(self.workspace,result,restored)),17)
        (result/'table03.csv').write_bytes(b'x\nwrong\n')
        with self.assertRaises(ValueError):rv.table_checks(self.workspace,result,restored)
        checks=json.loads((result/'restoration_table_checks.json').read_text())
        self.assertEqual(sum(not c['byte_identical'] for c in checks),1)

    def test_copy_selects_code_and_never_executes_or_copies_live_data(self):
        package=self.root/'package';source=package/'sources/test/never_run.py';source.parent.mkdir(parents=True)
        source.write_text('raise RuntimeError("Must never execute")\n',encoding='utf-8')
        (package/'live_data.csv').write_text('DO NOT COPY',encoding='utf-8')
        with (package/'selected_sources.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=['destination','sha256']);writer.writeheader()
            writer.writerow({'destination':'sources/test/never_run.py','sha256':rv.sha_file(source)})
            writer.writerow({'destination':'live_data.csv','sha256':rv.sha_file(package/'live_data.csv')})
        for name in ['run.py','Run-C1.ps1','README.md','DATA.md','entrypoints.json','data_requirements.json']:
            (package/name).write_text('synthetic',encoding='utf-8')
        copied=rv.copy_code(package,self.workspace)
        self.assertEqual(len(copied),7)
        self.assertFalse((self.workspace/'live_data.csv').exists())
        self.assertEqual(rv.sha_file(self.workspace/'sources/test/never_run.py'),rv.sha_file(source))


if __name__=='__main__':unittest.main()
