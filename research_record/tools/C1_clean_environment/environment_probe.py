"""Executed by the author inside a new venv; verify isolation and module origins."""
import argparse
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import site
import sys

MODULES={'scikit-learn':'sklearn','pillow':'PIL','fonttools':'fontTools','python-dateutil':'dateutil'}


def inside(path,root):return path==root or root in path.parents
def canonical(name):return name.lower().replace('_','-').replace('.','-')


def validate_layout(prefix,base_prefix,user_site,paths,cfg):
    if prefix==base_prefix or user_site:raise ValueError('Interpreter is not an isolated venv')
    if 'include-system-site-packages = false' not in cfg.lower():raise ValueError('Base packages are visible')
    for value in paths:
        p=Path(value).resolve()
        if any(part.lower() in ('site-packages','dist-packages') for part in p.parts) and not inside(p,prefix):
            raise ValueError('External site-packages detected: '+str(p))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,required=True);p.add_argument('--profile',required=True)
    p.add_argument('--phase',choices=['before','after'],required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.config.read_text(encoding='utf-8-sig'))
    prefix=Path(sys.prefix).resolve();base=Path(sys.base_prefix).resolve()
    validate_layout(prefix,base,site.ENABLE_USER_SITE,sys.path,(prefix/'pyvenv.cfg').read_text())
    packages={canonical(d.metadata['Name']):{'version':d.version,'location':str(Path(d.locate_file('')).resolve())} for d in metadata.distributions()}
    expected=config['profiles'][a.profile];allowed=set(expected)|{'pip','setuptools','wheel'}
    if a.phase=='before' and set(packages)-{'pip','setuptools','wheel'}:raise ValueError('New venv already contains non-bootstrap dependencies')
    origins={}
    if a.phase=='after':
        if a.profile=='graphics':
            profile=a.output.resolve().parent/'graphics_probe_profile'
            profile.mkdir(parents=True,exist_ok=True)
            for key in list(os.environ):
                if key.upper().startswith('RL_'):os.environ.pop(key,None)
            os.environ.update(USERPROFILE=str(profile),HOME=str(profile),
                              APPDATA=str(profile/'AppData/Roaming'),LOCALAPPDATA=str(profile/'AppData/Local'),
                              MPLCONFIGDIR=str(profile/'mplconfig'),MPL_IGNORE_SYSTEM_FONTS='1')
        if set(packages)-allowed:raise ValueError('Unexpected dependencies installed')
        for name,version in expected.items():
            if packages.get(name,{}).get('version')!=version:raise ValueError('Dependency version differs: '+name)
            if not inside(Path(packages[name]['location']),prefix):raise ValueError('External distribution: '+name)
            module=importlib.import_module(MODULES.get(name,name.replace('-','_')))
            filename=Path(module.__file__).resolve()
            if not inside(filename,prefix):raise ValueError('Module imported from older environment: '+name)
            origins[name]=str(filename.relative_to(prefix))
    report={'status':'ISOLATED_ENVIRONMENT_'+a.phase.upper()+'_VERIFIED','profile':a.profile,'python':sys.version,
            'prefix':str(prefix),'base_prefix':str(base),'user_site_enabled':site.ENABLE_USER_SITE,
            'packages':packages,'module_origins_relative_to_venv':origins,'new_training':False}
    a.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':main()
