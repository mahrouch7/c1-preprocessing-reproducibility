"""New E43 overlay must preserve frozen E33 bytes and bind its own code/config."""
import importlib.util
from pathlib import Path
import json
import pytest
from c1inception.io import digest, sha256, write_json

spec = importlib.util.spec_from_file_location('e43_bootstrap',
    Path(__file__).resolve().parents[1]/'kaggle/bootstrap.py')
bootstrap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)


def reference(tmp_path):
    original = tmp_path/'e33'
    (original/'data/conditions').mkdir(parents=True)
    (original/'data/metadata.json').write_text('[{"synthetic":true}]')
    (original/'data/conditions/clean.npy').write_bytes(b'SYNTHETIC_TEST_ONLY_NOT_NUMERIC_DATA')
    condition = {'path':'data/conditions/clean.npy',
                 'sha256':sha256(original/'data/conditions/clean.npy')}
    manifest = {'experiment_id':'C1-E33','code_commit':'original-test-commit',
        'conditions':{'clean':condition},'files':{p.relative_to(original).as_posix():sha256(p)
        for p in original.rglob('*') if p.is_file()}}
    write_json(original/'handoff_manifest.json',manifest)
    config = {'profiles':{'pilot':['clean']},'frozen_input_reference':{
        'manifest_sha256':sha256(original/'handoff_manifest.json'),
        'code_commit':'original-test-commit','conditions':manifest['conditions']}}
    return original, config


def test_e43_handoff_preserves_e33_and_binds_new_source(tmp_path):
    original, config = reference(tmp_path)
    before = {p.relative_to(original).as_posix():p.read_bytes() for p in original.rglob('*') if p.is_file()}
    bundle = tmp_path/'new.bundle';bundle.write_bytes(b'SYNTHETIC_NEW_SOURCE')
    payload = tmp_path/'e43'
    handoff = bootstrap.stage_handoff(original,payload,config,'new-test-commit',bundle)
    assert handoff['experiment_id']=='C1-E43'
    assert handoff['code_commit']=='new-test-commit'
    assert handoff['config_sha256']==digest(config)
    for relative, data in before.items():
        assert (original/relative).read_bytes()==data
    assert (payload/'data/conditions/clean.npy').read_bytes()==before['data/conditions/clean.npy']
    assert handoff['files']['code.bundle']==sha256(bundle)
    with pytest.raises(FileExistsError):
        bootstrap.stage_handoff(original,payload,config,'new-test-commit',bundle)


def test_altered_frozen_input_is_rejected(tmp_path):
    original, config = reference(tmp_path)
    (original/'data/conditions/clean.npy').write_bytes(b'ALTERED')
    bundle = tmp_path/'new.bundle';bundle.write_bytes(b'SYNTHETIC_NEW_SOURCE')
    with pytest.raises(ValueError,match='checksum mismatch'):
        bootstrap.stage_handoff(original,tmp_path/'e43',config,'test',bundle)


def test_changed_original_manifest_is_rejected(tmp_path):
    original, config = reference(tmp_path)
    with (original/'handoff_manifest.json').open('a') as out:out.write(' ')
    with pytest.raises(ValueError,match='identity drift'):
        bootstrap.stage_handoff(original,tmp_path/'e43',config,'test',tmp_path/'absent.bundle')
