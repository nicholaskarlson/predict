"""Finite, resumable numerical checks for the standalone public companion."""
import argparse, contextlib, csv, hashlib, importlib, importlib.metadata, json, os, re, signal, subprocess, sys, time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git','.venv','__pycache__','build','.pytest_cache'}
FORBIDDEN = {'manuscript','templates','review','qa','config','audit','technical-notes','figures'}
BAD_NAMES = {'book.json','build_book_review.py','run_reader_stage.py','verify_book_review.py','review_math.py'}
SECRET_PATTERNS = [rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',rb'\bgh[pousr]_[A-Za-z0-9]{36,}\b',rb'\bgithub_pat_[A-Za-z0-9_]{50,}\b',rb'\bAKIA[A-Z0-9]{16}\b']

def need(ok,message):
    if not ok: raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def obj(p):return json.loads(p.read_text(encoding='utf-8'))
def files(root=ROOT):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not any(x in IGNORED for x in p.relative_to(root).parts))
def safe(root,name):
    rel=PurePosixPath(name);need(name and not rel.is_absolute() and '..' not in rel.parts and '\\' not in name,'unsafe inventory path')
    p=root/name;need(p.resolve().is_relative_to(root.resolve()),'path escapes export')
    need(not any(x.is_symlink() for x in [p,*p.parents] if x!=root.parent),'symlink forbidden '+name)
    return p
def save(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n',encoding='utf-8',newline='\n');tmp.replace(p)
def boundary(root=ROOT,quiet=False):
    inventory={}
    for line in (root/'SHA256SUMS').read_text(encoding='utf-8').splitlines():
        h,n=line.split('  ',1);need(len(h)==64 and n not in inventory,'invalid checksum inventory');inventory[n]=h
    actual=files(root);need(set(actual)==set(inventory)|{'SHA256SUMS'},'unlisted or missing export file')
    for n,h in inventory.items():
        p=safe(root,n);need(sha(p)==h,'export bytes changed '+n)
        rel=PurePosixPath(n);need(not any(x in FORBIDDEN for x in rel.parts) and p.name not in BAD_NAMES,'private book category '+n)
        need(p.suffix.lower() not in {'.pdf','.docx','.epub','.odt','.tex','.png','.svg','.jpg'},'book/media artifact forbidden '+n)
        need(not any(re.search(pattern,p.read_bytes()) for pattern in SECRET_PATTERNS),'credential-shaped value in '+n)
    export=obj(root/'EXPORT_MANIFEST.json')
    need(export['new_git_history_required'] and export['new_world_draws']==0 and not export['full_book_reconstruction_included'],'export policy')
    expected=set(export['copied_files'])|set(export['support_files'])|{'EXPORT_MANIFEST.json','SHA256SUMS'}
    need(set(actual)==expected,'positive export allowlist differs')
    for n,record in export['copied_files'].items():need(inventory[n]==record['sha256'],'original copied byte drift '+n)
    if not quiet: print('PREDICT_BOUNDARY_OK originals='+str(len(export['copied_files']))+' files='+str(len(actual))+' manuscript_and_renderer_absent=true',flush=True)
    return {n:sha(root/n) for n in actual}
def runtime(route):
    if route=='base_r':
        p=subprocess.run(['Rscript','--vanilla','-e','stopifnot(requireNamespace("methods",quietly=TRUE),requireNamespace("stats",quietly=TRUE));cat(as.character(getRversion()))'],capture_output=True,text=True,check=True,timeout=20)
        return {'R':p.stdout.strip()}
    need(importlib.metadata.version('numpy')=='1.26.4','NumPy 1.26.4 is required')
    return {'python':sys.version.split()[0],'numpy':importlib.metadata.version('numpy')}
def load_rows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def compare_outputs(case,folder,resource):
    module=importlib.import_module('contracts.'+case['comparison']); result={}
    supplementary=set(case.get('supplementary_csv',{}).get(folder.name,[]))
    need({p.name for p in folder.glob('*.csv')}=={k+'.csv' for k in case['files']}|supplementary,'output CSV inventory differs')
    for key,name in case['files'].items():
        actual=folder/(key+'.csv');expected=resource/name
        if case['command_kind']=='unit':
            fn=getattr(module,'compare',getattr(module,'compare_rows',None)); value=fn(load_rows(expected),load_rows(actual),folder.parent.name+'/'+key)
        elif case['comparison']=='envelope':value=module.compare(actual,expected)
        else:value=module.compare(actual,expected,case['tolerance'])
        result[key]=value
    if 'exact_summary.csv' in supplementary:
        from fractions import Fraction
        for exact_name,numeric_name,key in [('exact_summary.csv','summary.csv','name'),('exact_cutpoints.csv','cutpoints.csv','quantile')]:
            exact=load_rows(folder/exact_name);numeric=load_rows(folder/numeric_name)
            need(len(exact)==len(numeric),'supplementary rational row count')
            for x,y in zip(exact,numeric):
                need(x[key]==y[key] and abs(float(Fraction(x['fraction']))-float(y['value']))<=1e-12,'supplementary rational result disagrees')
    if 'exact_anchors.csv' in supplementary:
        from fractions import Fraction
        anchors=obj(resource/'expected-results/unit02_anchors.json')['anchors']
        for x in load_rows(folder/'exact_anchors.csv'):
            need(x['name'] in anchors,'unrecognized exact anchor')
            expected=anchors[x['name']]
            need(x['kind']==expected['kind'],'supplementary anchor kind differs')
            if x['kind']=='exact_fraction':need(Fraction(x['value'])==Fraction(expected['exact']),'supplementary exact fraction differs')
            else:need(abs(float(x['value'])-float(expected['value']))<=1e-12,'supplementary floating anchor differs')
    return result
def receipt_valid(receipt,proof,case,resource):
    try:
        r=obj(receipt);folder=ROOT/r['output_folder'];log=ROOT/r['log']
        need(r['status']=='success' and r['fingerprint']==proof and sha(log)==r['log_sha256'],'stale receipt')
        need(all(sha(folder/n)==h for n,h in r['output_sha256'].items()),'stale output')
        need(all(m in read_log(log) for m in case['markers'][r['route']]),'missing completion marker')
        compare_outputs(case,folder,resource)
        return True
    except (OSError,ValueError,KeyError):return False
def read_log(log):
    # Analyses write UTF-8 (PYTHONUTF8=1 below); replace anything else rather than fail on a log byte.
    return log.read_text(encoding='utf-8',errors='replace')
@contextlib.contextmanager
def case_lock(path):
    """One run per case at a time: fcntl on Linux and macOS, msvcrt on Windows.
    The file is opened for append, so a second run never truncates a file another run holds."""
    with path.open('a') as handle:
        try:
            import fcntl
        except ImportError:
            import msvcrt
            handle.seek(0)
            try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            except OSError:raise ValueError('this case is already running')
            try:yield
            finally:
                handle.seek(0);msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            return
        try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('this case is already running')
        yield
def stop(process):
    """End an analysis and anything it started (a process group on POSIX; the process on Windows)."""
    if hasattr(os,'killpg'):
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=5)
        except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)
    else:
        process.kill();process.wait(timeout=5)
def run_case(name,routes):
    inventory=boundary(quiet=True);cfg=obj(ROOT/'CASE_MANIFEST.json');case=cfg['cases'][name];resource=ROOT/cfg['resource'];records=[]
    parent=ROOT/'build'/(name+'-audit' if name.startswith('unit') else {'uncertainty':'score-uncertainty','planning':'sample-size-planning','envelope':'sealed-envelope'}[name]);parent.mkdir(parents=True,exist_ok=True)
    with case_lock(parent/'runner.lock'):
        for route in routes:
            versions=runtime(route);proof=hashlib.sha256(json.dumps({'files':inventory,'versions':versions,'route':route},sort_keys=True).encode()).hexdigest();receipt=parent/(route+'.json')
            if receipt.is_file() and receipt_valid(receipt,proof,case,resource):
                print('PREDICT_RESUME '+name+' '+route+' current_outputs=true',flush=True);records.append(obj(receipt));continue
            folder=parent/('attempt-'+str(time.time_ns()))/route;folder.mkdir(parents=True);log=folder.parent/(route+'.log')
            stem=case['stem'];source=resource/'source'/(stem+'_'+route+('.R' if route=='base_r' else '.py'))
            if route=='base_r':command=['Rscript','--vanilla',str(source),str(resource),str(folder)]
            elif case['command_kind']=='unit':command=[sys.executable,str(source),'--root',str(resource),'--outdir',str(folder)]
            else:command=[sys.executable,str(source),'--resource',str(resource),'--out',str(folder)]
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONUTF8='1')
            start=time.monotonic();status='failure';code=None;comparison=None
            print('PREDICT_START '+name+' '+route+' deadline_seconds='+str(case['deadline_seconds']),flush=True)
            with log.open('w',encoding='utf-8') as stream:
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env,cwd=ROOT,start_new_session=True)
                try:
                    while process.poll() is None:
                        elapsed=time.monotonic()-start;need(elapsed<case['deadline_seconds'],'case deadline reached; inspect '+str(log))
                        try:process.wait(timeout=min(20,case['deadline_seconds']-elapsed))
                        except subprocess.TimeoutExpired:print('PREDICT_RUNNING '+name+' '+route+' seconds='+str(int(time.monotonic()-start)),flush=True)
                    code=process.returncode;need(code==0,'analysis failed; inspect '+str(log))
                    need(all(m in read_log(log) for m in case['markers'][route]),'analysis completion marker missing; inspect '+str(log))
                    comparison=compare_outputs(case,folder,resource);status='success'
                finally:
                    if process.poll() is None:stop(process)
                    record={'schema_version':1,'case':name,'route':route,'status':status,'fingerprint':proof,'versions':versions,'command':command,'exit_code':process.returncode,'elapsed_seconds':round(time.monotonic()-start,3),'deadline_seconds':case['deadline_seconds'],'output_folder':folder.relative_to(ROOT).as_posix(),'log':log.relative_to(ROOT).as_posix(),'log_sha256':sha(log),'output_sha256':{p.name:sha(p) for p in folder.iterdir() if p.is_file()},'comparison_maxima_where_reported':comparison,'new_world_draws':0,'remote_ci_claimed':False}
                    save(receipt,record)
            records.append(record);print('PREDICT_OK '+name+' '+route,flush=True)
    save(parent/'current.json',{'schema_version':1,'status':'success','case':name,'routes':routes,'records':records,'new_world_draws':0,'remote_ci_claimed':False})
    print('PREDICT_CASE_OK '+name+' paths='+str(len(routes)),flush=True)
def execute(names,base_r=False):
    boundary();routes=['numpy','stdlib']+(['base_r'] if base_r else [])
    for name in names:run_case(name,routes)
    print('PREDICT_VERIFY_OK cases='+str(len(names))+' paths='+str(len(routes))+' World_draws=0',flush=True)
def guarded(fn):
    try:fn()
    except (OSError,ValueError,KeyError,subprocess.SubprocessError) as e:sys.exit('PREDICT_ERROR '+str(e))
def unit_cli(name):
    p=argparse.ArgumentParser();p.add_argument('--numeric',action='store_true');p.add_argument('--base-r',action='store_true');a=p.parse_args()
    guarded(lambda:execute([name],a.base_r) if a.numeric or a.base_r else boundary())
def method_cli():
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=['uncertainty','planning','all'],default='all');p.add_argument('--base-r',action='store_true');a=p.parse_args();guarded(lambda:execute(['uncertainty','planning'] if a.scope=='all' else [a.scope],a.base_r))
def stage_cli():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['verify','base-r','all'],default='all');a=p.parse_args();guarded(lambda:execute(list(obj(ROOT/'CASE_MANIFEST.json')['cases']),a.stage in {'base-r','all'}))
def main():
    p=argparse.ArgumentParser();p.add_argument('--case',default='all');p.add_argument('--base-r',action='store_true');p.add_argument('--boundary-only',action='store_true');a=p.parse_args()
    def action():
        if a.boundary_only:boundary();return
        cases=obj(ROOT/'CASE_MANIFEST.json')['cases'];need(a.case=='all' or a.case in cases,'unknown case');execute(list(cases) if a.case=='all' else [a.case],a.base_r)
    guarded(action)
if __name__=='__main__':main()
