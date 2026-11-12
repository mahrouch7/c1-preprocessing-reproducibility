"""Run frozen comparator under a Python audit-hook file-access guard."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import traceback


def inside(path,root):
    return path==root or root in path.parents


class AccessGuard:
    def __init__(self,workspace,library_roots):
        self.workspace=workspace.resolve()
        self.library_roots=[p.resolve() for p in library_roots]
        self.counts=Counter();self.blocked=[];self.study_reads=set()

    def check(self,value,write=False,event='open'):
        if value is None:value=os.getcwd()
        if isinstance(value,int):return  # inherited stdin/stdout descriptors, not path lookup
        p=Path(os.fsdecode(value)).resolve()
        if inside(p,self.workspace):
            if write and any(inside(p,self.workspace/n) for n in ('data','reference','sources')):
                self.blocked.append({'event':event,'path':str(p),'write':True})
                raise PermissionError('Restored scientific inputs/code are read-only: '+str(p))
            self.counts['workspace_writes' if write else 'workspace_reads']+=1
            if not write and (inside(p,self.workspace/'data') or inside(p,self.workspace/'reference')):
                self.study_reads.add(p.relative_to(self.workspace).as_posix())
            return
        if not write and any(inside(p,r) for r in self.library_roots):
            self.counts['library_or_system_reads']+=1
            return
        self.blocked.append({'event':event,'path':str(p),'write':write})
        raise PermissionError('Access outside restored workspace/libraries refused: '+str(p))

    def hook(self,event,args):
        if event=='open':
            mode=args[1] if len(args)>1 else None;flags=args[2] if len(args)>2 else 0
            write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool((flags or 0)&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            self.check(args[0],write,event)
        elif event in ('os.listdir','os.scandir'):
            self.check(args[0] if args else None,False,event)
        elif event in ('os.mkdir','os.remove','os.rmdir','os.chmod','os.utime','os.chdir'):
            self.check(args[0],True,event)
        elif event in ('os.rename','os.link'):
            self.check(args[0],True,event);self.check(args[1],True,event)
        elif event.startswith(('subprocess.','os.system','os.spawn')) or event in ('socket.connect','socket.getaddrinfo','os.symlink'):
            self.blocked.append({'event':event,'path':'[operation refused]','write':False})
            raise PermissionError('Operation not allowed during restored-data comparison: '+event)


def module_origins(workspace):
    found={}
    for name,module in list(sys.modules.items()):
        if name=='c1inception' or name.startswith('c1inception.'):
            filename=getattr(module,'__file__',None)
            if filename:
                p=Path(filename).resolve()
                if not inside(p,workspace):raise ValueError('Scientific module loaded outside restored code: '+name)
                found[name]=p.relative_to(workspace).as_posix()
    return found


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();workspace=a.workspace.resolve()
    for path in (a.policy,a.output,a.report):
        if not inside(path.resolve(),workspace):raise ValueError('Guard paths must be in restored workspace')
    roots=[Path(sys.prefix),Path(sys.base_prefix)]
    if os.name=='nt' and os.environ.get('SystemRoot'):roots.append(Path(os.environ['SystemRoot']))
    # Ignore externally injected source paths; retain the existing Python runtime/libraries.
    sys.path[:]=[str(workspace/'sources/inceptiontime/src')]+[s for s in sys.path if s and any(inside(Path(s).resolve(),r.resolve()) for r in roots)]
    if any(n=='c1inception' or n.startswith('c1inception.') for n in sys.modules):
        raise ValueError('Scientific package was unexpectedly preloaded before the guard')
    guard=AccessGuard(workspace,roots)
    sys.addaudithook(guard.hook)
    success=False;origins={}
    try:
        from c1inception.comparison import analyze
        module_origins(workspace)
        analyze(a.policy,a.output)
        origins=module_origins(workspace)
        success=True
    except Exception:
        traceback.print_exc()
    report={'status':'GUARDED_COMPARISON_COMPLETE' if success and not guard.blocked else 'GUARDED_COMPARISON_FAILED',
            'comparison_complete':success,'blocked_operations':guard.blocked,'file_event_counts':dict(guard.counts),
            'study_paths_read':sorted(guard.study_reads),'scientific_module_origins':origins,
            'guard_scope':'Python audit events after interpreter startup; existing libraries/system resources allowed; not an OS/native-extension sandbox',
            'new_training':False,'source_package_modified':False}
    a.report.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return 0 if success and not guard.blocked else 1


if __name__=='__main__':sys.exit(main())
