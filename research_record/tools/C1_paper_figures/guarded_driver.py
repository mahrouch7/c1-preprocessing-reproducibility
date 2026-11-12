"""Run one staged figure driver; reference PDFs cannot be generation inputs."""
import argparse
import json
import os
from pathlib import Path
import runpy
import sys
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--driver',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a,arguments=p.parse_known_args();work=a.workspace.resolve();driver=a.driver.resolve()
    if work not in driver.parents or driver.parent!=work/'drivers':raise ValueError('Driver must be staged')
    if work not in a.report.resolve().parents:raise ValueError('Report must be staged')
    roots=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve()]
    if os.environ.get('SystemRoot'):roots.append(Path(os.environ['SystemRoot']).resolve())
    inside=lambda path,root:path==root or root in path.parents
    sys.path[:]=[str(work)]+[s for s in sys.path if s and any(inside(Path(s).resolve(),r) for r in roots)]
    from guarded_io import AccessGuard
    guard=AccessGuard(work,roots)
    def hook(event,args):
        if event in ('open','os.listdir','os.scandir') and args and not isinstance(args[0],int):
            path=Path(os.fsdecode(args[0])).resolve() if args[0] is not None else Path.cwd()
            if inside(path,work/'reference'):
                guard.blocked.append({'event':event,'path':str(path),'write':False})
                raise PermissionError('Reference PDFs are QA material, never generator inputs')
        guard.hook(event,args)
    sys.addaudithook(hook)
    sys.argv=[str(driver)]+(arguments[1:] if arguments[:1]==['--'] else arguments)
    success=False
    try:
        runpy.run_path(str(driver),run_name='__main__');success=True
    except SystemExit as exc:
        success=exc.code in (None,0)
        if not success:traceback.print_exc()
    except Exception:traceback.print_exc()
    a.report.write_text(json.dumps({'status':'COMPLETE' if success and not guard.blocked else 'FAILED',
        'driver':driver.name,'blocked_operations':guard.blocked,'study_reads':sorted(guard.study_reads),
        'file_event_counts':dict(guard.counts),'scope':'Python audit events after interpreter startup; not OS/native isolation'},indent=2)+'\n',encoding='utf-8')
    return 0 if success and not guard.blocked else 1


if __name__=='__main__':sys.exit(main())
