"""Verify the local release inventory; standard library only, no science or network."""
import hashlib
import json
from pathlib import Path, PureWindowsPath

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'provenance/assembly_manifest.json').read_text(encoding='utf-8-sig'))
seen = set()
for row in manifest['files']:
    name = row['path']
    if (PureWindowsPath(name).drive or ':' in name or '\\' in name or name.startswith('/')
            or any(p in ('', '.', '..') for p in name.split('/')) or name.casefold() in seen):
        raise ValueError('Unsafe or duplicate manifest path')
    seen.add(name.casefold())
    path = root
    for part in name.split('/'):
        path = path / part
        if path.is_symlink() or (path.exists() and getattr(path.lstat(), 'st_file_attributes', 0) & 0x400):
            raise ValueError('Link refused: ' + name)
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != row['sha256'] or path.stat().st_size != row['bytes']:
        raise ValueError('Release file differs: ' + name)
print(json.dumps({'status': 'ASSEMBLY_INVENTORY_VERIFIED', 'files': len(seen),
                  'scientific_execution': False, 'publication': False}))
