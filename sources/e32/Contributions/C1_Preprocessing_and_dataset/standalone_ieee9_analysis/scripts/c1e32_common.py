"""Shared immutable I/O. No fits, subprocesses or network access on import."""
from __future__ import annotations
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import os
import shutil
import uuid
import c1e32_spec as spec

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream,'sha256').hexdigest()
def now(): return datetime.now(timezone.utc).isoformat()
def require(ok,message):
    if not ok: raise ValueError(message)
def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=spec.canonical(value)
    if path.exists():
        require(path.read_bytes()==data,'Conflicting immutable output: '+str(path)); return
    partial=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with partial.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    try: os.link(partial,path)
    except FileExistsError: require(path.read_bytes()==data,'Concurrent conflicting publication')
    partial.unlink()
def table(path,rows):
    rows=list(rows); require(bool(rows),'Refuse empty scientific table')
    with Path(path).open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)
def child(root,relative):
    root=Path(root).resolve(); relative=str(relative)
    require(':' not in relative and '\\' not in relative and all(v not in ('','.','..') for v in relative.split('/')),'Unsafe relative path')
    path=(root/relative).resolve(); require(root in path.parents,'Path escapes root'); return path
def capacity(path,gib):
    probe=Path(path)
    while not probe.exists(): probe=probe.parent
    free=shutil.disk_usage(probe).free
    require(free>=gib*1024**3,f'Insufficient storage: need {gib} GiB, observed {free/1024**3:.1f}; no deletion/retry')
    return free
def pins(paths): return {k:{'path':'data/'+k+Path(v).suffix,'sha256':sha(v)} for k,v in paths.items()}
def inputs(config,root):
    result={}
    for name,pin in config['inputs'].items():
        path=child(root,pin['path']); require(path.is_file() and not path.is_symlink() and sha(path)==pin['sha256'],'Input hash changed: '+name)
        result[name]=path
    return result
