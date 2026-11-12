"""File-integrity and repeat-run checks; synthetic files only, no scientific fits."""
from pathlib import Path
import sys
import zipfile
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from import_kaggle_results import copy_verified, extract_preserving, sha


def test_extract_is_repeatable_and_preserves_changed_files(tmp_path):
    archive=tmp_path/'saved.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('phase/result.json','{"synthetic":true}')
    destination=tmp_path/'run'
    extract_preserving(archive,destination)
    first=(destination/'phase/result.json').read_bytes()
    extract_preserving(archive,destination)
    assert (destination/'phase/result.json').read_bytes()==first
    (destination/'phase/result.json').write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='refusing overwrite'):
        extract_preserving(archive,destination)
    assert (destination/'phase/result.json').read_bytes()==b'CHANGED'


def test_extract_rejects_path_escape(tmp_path):
    archive=tmp_path/'bad.zip'
    with zipfile.ZipFile(archive,'w') as z:z.writestr('../outside.txt','invalid')
    with pytest.raises(ValueError,match='Unsafe archive path'):
        extract_preserving(archive,tmp_path/'run')
    assert not (tmp_path/'outside.txt').exists()


def test_verified_copy_is_repeatable_and_preserves_different_target(tmp_path):
    source=tmp_path/'source.bin';source.write_bytes(b'SYNTHETIC')
    target=tmp_path/'result/archive.bin'
    copy_verified(source,target,sha(source));copy_verified(source,target,sha(source))
    target.write_bytes(b'CHANGED')
    with pytest.raises(ValueError,match='Existing destination differs'):
        copy_verified(source,target,sha(source))
    assert target.read_bytes()==b'CHANGED'


def test_wrong_source_checksum_is_rejected_before_copy(tmp_path):
    source=tmp_path/'source.bin';source.write_bytes(b'SYNTHETIC')
    target=tmp_path/'target.bin'
    with pytest.raises(ValueError,match='Source checksum mismatch'):
        copy_verified(source,target,'0'*64)
    assert not target.exists()
