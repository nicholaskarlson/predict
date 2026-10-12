"""Three Riley (2019) binary-development criteria, standard-library path.

Retrospective diagnostics use development/updating rows only. Prospective
scenarios are explicit assumptions, not estimates from final evaluation.
"""
import argparse,csv,json,math
from pathlib import Path

def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

def criteria(parameters,prevalence,r2,shrinkage,delta,margin,z,usable_fraction):
    maximum=-math.expm1(2*(prevalence*math.log(prevalence)+(1-prevalence)*math.log1p(-prevalence)))
    if not (parameters>0 and 0<prevalence<1 and 0<r2<min(shrinkage,maximum) and 0<shrinkage<1 and delta>0 and margin>0 and z>0 and 0<usable_fraction<=1):raise ValueError('invalid planning assumptions')
    optimism_shrinkage=r2/(r2+delta*maximum)
    n1=parameters/((shrinkage-1)*math.log1p(-r2/shrinkage))
    n2=parameters/((optimism_shrinkage-1)*math.log1p(-r2/optimism_shrinkage))
    n3=z*z*prevalence*(1-prevalence)/(margin*margin)
    required=max(math.ceil(n1),math.ceil(n2),math.ceil(n3))
    return dict(parameters=parameters,prevalence=prevalence,r2_cs=r2,r2_max=maximum,shrinkage=shrinkage,optimism_delta=delta,risk_margin=margin,z=z,usable_fraction=usable_fraction,optimism_shrinkage=optimism_shrinkage,n_shrinkage_raw=n1,n_optimism_raw=n2,n_risk_raw=n3,n_shrinkage=math.ceil(n1),n_optimism=math.ceil(n2),n_risk=math.ceil(n3),n_required=required,expected_events=required*prevalence,invitations=math.ceil(required/usable_fraction))

def calculate(resource):
    spec=json.loads((resource/'model-specs/sample_size_planning.json').read_text())
    diagnostics=[];scenarios=[]
    for unit,model,label in [(7,'full_logistic','unit07_full'),(9,'update_refit','unit09_refit')]:
        data=read(resource/f'data/unit{unit:02d}_rows.csv')
        if unit==7:
            fit=[r for r in data if r['role']=='development'];y=[int(r['y']) for r in fit]
        else:
            ids={r['row_id'] for r in read(resource/'splits/unit09_roles.csv') if r['assignment']=='new_updating'}
            fit=[r for r in data if r['row_id'] in ids];y=[int(r['binary_y']) for r in fit]
        n=len(fit);events=sum(y);prevalence=events/n
        coefficient=read(resource/f'expected-results/unit{unit:02d}_coefficients.csv')
        key='model_id' if unit==7 else 'fit_id'
        parameters=sum(r[key]==model and r['term']!='intercept' for r in coefficient)
        d=next(r for r in read(resource/f'expected-results/unit{unit:02d}_diagnostics.csv') if r[key]==model)
        if int(d['n_fit'])!=n or int(d['events'])!=events or int(d['p'])!=parameters+1:raise ValueError('development diagnostics disagree')
        nll=float(d['nll']);null_nll=-(events*math.log(prevalence)+(n-events)*math.log1p(-prevalence))
        lr=2*(null_nll-nll);apparent=-math.expm1(-lr/n);maximum=-math.expm1(-2*null_nll/n)
        diagnostics.append(dict(case=label,n=n,events=events,parameters=parameters,prevalence=prevalence,model_nll=nll,null_nll=null_nll,likelihood_ratio=lr,heuristic_shrinkage=1-parameters/lr,r2_apparent=apparent,r2_max=maximum))
        for name,r2 in [('apparent_plugin_optimistic',apparent),('assumed_15pct_max',.15*maximum)]:
            v=criteria(parameters,prevalence,r2,.9,.05,.05,1.96,1.0)
            scenarios.append(dict(scenario=label+'_'+name,origin='retrospective_assumption',available_n=n,**v))
    for s in spec['prospective_scenarios']:
        v=criteria(s['parameters'],s['prevalence'],s['r2_cs'],s['shrinkage'],s['optimism_delta'],s['risk_margin'],s['z'],s['usable_fraction'])
        scenarios.append(dict(scenario=s['id'],origin='prospective_assumption',available_n=0,**v))
    return {'planning_diagnostics.csv':diagnostics,'planning_scenarios.csv':scenarios}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resource',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for name,values in calculate(a.resource).items():
        with (a.out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader()
            w.writerows({k:format(v,'.17g') if isinstance(v,float) else v for k,v in r.items()} for r in values)
