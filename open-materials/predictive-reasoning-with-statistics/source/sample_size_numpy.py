"""Independent NumPy planning algebra and development likelihood diagnostics."""
import argparse,csv,json
from pathlib import Path
import numpy as np

def rows(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))

def plan(p,pi,r2,s,d,e,z,f):
    maximum=float(1-np.exp(2*(pi*np.log(pi)+(1-pi)*np.log(1-pi))))
    if not (p>0 and 0<pi<1 and 0<r2<min(s,maximum) and 0<s<1 and d>0 and e>0 and z>0 and 0<f<=1):raise ValueError('invalid assumptions')
    s2=r2/(r2+d*maximum)
    raw=np.asarray([p/((s-1)*np.log(1-r2/s)),p/((s2-1)*np.log(1-r2/s2)),z**2*pi*(1-pi)/e**2])
    n=np.ceil(raw).astype(int);required=int(n.max())
    return dict(parameters=p,prevalence=pi,r2_cs=r2,r2_max=maximum,shrinkage=s,optimism_delta=d,risk_margin=e,z=z,usable_fraction=f,optimism_shrinkage=s2,n_shrinkage_raw=float(raw[0]),n_optimism_raw=float(raw[1]),n_risk_raw=float(raw[2]),n_shrinkage=int(n[0]),n_optimism=int(n[1]),n_risk=int(n[2]),n_required=required,expected_events=required*pi,invitations=int(np.ceil(required/f)))

def calculate(resource):
    spec=json.loads((resource/'model-specs/sample_size_planning.json').read_text());ds=[];ss=[]
    for unit,model,case in [(7,'full_logistic','unit07_full'),(9,'update_refit','unit09_refit')]:
        data=rows(resource/f'data/unit{unit:02d}_rows.csv');key='model_id' if unit==7 else 'fit_id'
        if unit==7:y=np.asarray([int(r['y']) for r in data if r['role']=='development'])
        else:
            roles={r['row_id']:r['assignment'] for r in rows(resource/'splits/unit09_roles.csv')}
            y=np.asarray([int(r['binary_y']) for r in data if roles[r['row_id']]=='new_updating'])
        n=y.size;events=int(np.sum(y));pi=float(np.mean(y))
        params=[r for r in rows(resource/f'expected-results/unit{unit:02d}_coefficients.csv') if r[key]==model and r['term']!='intercept'];p=len(params)
        diagnostic=[r for r in rows(resource/f'expected-results/unit{unit:02d}_diagnostics.csv') if r[key]==model][0]
        if (int(diagnostic['n_fit']),int(diagnostic['events']),int(diagnostic['p']))!=(n,events,p+1):raise ValueError('fit boundary mismatch')
        nll=float(diagnostic['nll']);null=-float(np.sum(y*np.log(pi)+(1-y)*np.log(1-pi)))
        lr=2*(null-nll);apparent=float(1-np.exp(-lr/n));maximum=float(1-np.exp(-2*null/n))
        ds.append(dict(case=case,n=int(n),events=events,parameters=p,prevalence=pi,model_nll=nll,null_nll=null,likelihood_ratio=lr,heuristic_shrinkage=1-p/lr,r2_apparent=apparent,r2_max=maximum))
        for name,r2 in [('apparent_plugin_optimistic',apparent),('assumed_15pct_max',.15*maximum)]:ss.append(dict(scenario=case+'_'+name,origin='retrospective_assumption',available_n=int(n),**plan(p,pi,r2,.9,.05,.05,1.96,1.0)))
    for s in spec['prospective_scenarios']:ss.append(dict(scenario=s['id'],origin='prospective_assumption',available_n=0,**plan(s['parameters'],s['prevalence'],s['r2_cs'],s['shrinkage'],s['optimism_delta'],s['risk_margin'],s['z'],s['usable_fraction'])))
    return {'planning_diagnostics.csv':ds,'planning_scenarios.csv':ss}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resource',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for name,values in calculate(a.resource).items():
        with (a.out/name).open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(values[0]));w.writeheader()
            for row in values:w.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})
