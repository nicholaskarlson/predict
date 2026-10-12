#!/usr/bin/env python3
"""NumPy route: raw joins, repeated fit-slot transforms, QR, AIC and optimism."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
RESOURCE=Path(__file__).resolve().parents[1]
MODELS=('entry','baseline','rich')
TERMS={'entry':['intercept','z_prior_grade'],'baseline':['intercept','z_prior_grade','part_time','first_gen','z_age'],'rich':['intercept','z_prior_grade','part_time','first_gen','z_age','z_prior_grade_squared','z_age_squared','z_prior_grade_part_time','z_prior_grade_first_gen','z_prior_grade_z_age']}
def require(ok,message):
    if not ok:raise ValueError(message)
def read_rows(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def joined_rows(root):
    raw=root/'source/tonmick-v1.0.0/data/anchor';tables=[]
    for name in ('participants','baseline','registry'):
        a=read_rows(raw/(name+'.csv'));d={r['student_id']:r for r in a}
        require(len(a)==len(d),'duplicate raw key');tables.append(d)
    frame,base,outcomes=tables
    require(set(base)==set(outcomes)=={k for k,v in frame.items() if v['responded_baseline']=='1'},'respondent membership')
    rows=[]
    for key in sorted(base):
        f,b,o=frame[key],base[key],outcomes[key]
        require(f['college_id']==b['college_id'],'college join')
        if b['offer']!='0':continue
        require(f['part_time'] in ('0','1') and f['first_gen'] in ('0','1'),'binary predictor')
        prior,age=float(f['prior_grade']),float(f['age'])
        require(np.isfinite(prior) and 0<=prior<=4 and np.isfinite(age) and 16<=age<=80,'invalid predictor')
        y=None if not o['gpa12'] else float(o['gpa12'])
        require(y is None or (np.isfinite(y) and 0<=y<=4),'invalid grade')
        require(o['exit_type']!='consent_withdrawn' or y is None,'consent grade')
        role=('development' if f['region']=='West' else 'setting_check') if y is not None else 'excluded_missing_grade'
        rows.append(dict(row_id=key,college_id=f['college_id'],region=f['region'],part_time=int(f['part_time']),first_gen=int(f['first_gen']),age=age,prior_grade=prior,exit_type=o['exit_type'],gpa_at_exit=int(o['gpa_at_exit']),y=y,role=role))
    roles=read_rows(root/'splits/unit03_roles.csv')
    require(roles==[{k:r[k] for k in ('row_id','college_id','region','role')} for r in rows],'role membership mismatch')
    return rows
def bootstrap_samples(root,dev):
    indexed={r['row_id']:r for r in dev};draws=read_rows(root/'splits/unit03_bootstrap.csv');offset=0
    colleges=sorted({r['college_id'] for r in dev})
    for rep in range(1,201):
        name=f'B{rep:04d}';sample=[]
        for college in colleges:
            n=sum(r['college_id']==college for r in dev)
            for slot in range(1,n+1):
                require(offset<len(draws),'missing bootstrap slot');a=draws[offset];offset+=1
                require(a['replicate']==name and a['college_id']==college and a['slot']==f'{slot:03d}','bootstrap slot order')
                require(a['row_id'] in indexed and indexed[a['row_id']]['college_id']==college,'bootstrap fit-role or college mismatch')
                sample.append(dict(indexed[a['row_id']],slot=college+'-'+a['slot']))
        yield name,sample
    require(offset==len(draws),'extra bootstrap slots')
def transforms(train):
    out={}
    for column in ('prior_grade','age'):
        x=np.array([r[column] for r in train]);mu=float(np.mean(x));sd=float(np.std(x,ddof=0))
        require(np.isfinite(sd) and sd>1e-12,'zero SD');out[column]=(mu,sd)
    return out
def matrix(rows,model,params):
    g=np.array([(r['prior_grade']-params['prior_grade'][0])/params['prior_grade'][1] for r in rows])
    a=np.array([(r['age']-params['age'][0])/params['age'][1] for r in rows])
    pt=np.array([r['part_time'] for r in rows]);fg=np.array([r['first_gen'] for r in rows])
    cols=[np.ones(len(rows)),g]
    if model in ('baseline','rich'):cols.extend([pt,fg,a])
    if model=='rich':cols.extend([g*g,a*a,g*pt,g*fg,g*a])
    return np.column_stack(cols),TERMS[model]
def select(candidates):
    best=min(r['aic'] for r in candidates)
    return next(m for m in MODELS if next(r['aic'] for r in candidates if r['model_id']==m)<=best+1e-10)
def score(rows,pred):
    y=np.array([r['y'] for r in rows]);e=y-np.array(pred);var=float(np.mean((y-np.mean(y))**2))
    require(var>1e-12,'constant scored labels')
    mse=float(np.mean(e*e));return mse,1-mse/var
def qr_fit(x,y,spec):
    require(x.shape[0]>=x.shape[1],'insufficient fit rows')
    q,r=np.linalg.qr(x,mode='reduced');diagonal=np.abs(np.diag(r));p=x.shape[1]
    require(np.all(diagonal>spec['solver']['rank_relative_tolerance']*diagonal.max()),'rank-deficient QR design')
    inverse=np.linalg.solve(r,np.eye(p))
    condition=float(np.linalg.norm(r,np.inf)*np.linalg.norm(inverse,np.inf))
    require(np.isfinite(condition) and condition<=spec['solver']['maximum_condition'],'ill-conditioned QR design')
    beta=np.linalg.solve(r,q.T@y);residual=y-x@beta
    score=float(np.max(np.abs(x.T@residual))/len(y))
    require(score<=spec['solver']['maximum_normalized_score'],'OLS score audit failed')
    return beta,condition,score,float(np.mean(residual**2))

def calculate(root=RESOURCE):
    spec=json.loads((root/'model-specs/unit03.json').read_text())
    require(spec['model_ids']==list(MODELS) and spec['model_terms']==TERMS and spec['predictor_allowlist']==['prior_grade','part_time','first_gen','age'] and spec['selection']['criterion']=='n_log_MSE_plus_2_times_p_plus_1' and spec['selection']['refit_every_bootstrap'] and not spec['selection']['setting_check_labels_for_selection'] and not spec['final_test_labels_for_selection'] and spec['bootstrap']['replicates']==200,'model specification drift')
    rows=joined_rows(root);dev=[r for r in rows if r['role']=='development'];setting=[r for r in rows if r['role']=='setting_check']
    output={k:[] for k in ('coefficients','transforms','fit_rows','diagnostics','scores','selections','optimism','corrections','predictions','examples')}
    original_choice=None
    for rep,train in [('original',dev),*list(bootstrap_samples(root,dev))]:
        params=transforms(train)
        for column,(mu,sd) in params.items():output['transforms'].append(dict(replicate=rep,column=column,mean=mu,sd=sd))
        output['fit_rows'].extend(dict(replicate=rep,slot=r.get('slot',f'{i+1:03d}'),row_id=r['row_id'],college_id=r['college_id']) for i,r in enumerate(train))
        fits={};candidates=[]
        for model in MODELS:
            x,terms=matrix(train,model,params);beta,cond,gradient,mse=qr_fit(x,np.array([r['y'] for r in train]),spec)
            require(mse>0,'zero residual variance')
            aic=len(train)*float(np.log(mse))+2*(len(terms)+1)
            candidates.append(dict(model_id=model,aic=aic));fits[model]=(beta,params)
            output['diagnostics'].append(dict(replicate=rep,model_id=model,n_fit=len(train),p=len(terms),rank=len(terms),condition_r_inf=cond,score_inf=gradient,fit_mse=mse,aic=aic))
            output['coefficients'].extend(dict(replicate=rep,model_id=model,term=t,coefficient=float(b)) for t,b in zip(terms,beta))
        choice=select(candidates)
        if rep=='original':original_choice=choice
        output['selections'].append(dict(replicate=rep,selected_model=choice,selected_aic=next(v['aic'] for v in candidates if v['model_id']==choice)))
        metrics={};forecasts={}
        scopes=[('apparent',dev),('setting_check',setting)] if rep=='original' else [('bootstrap',train),('original_West',dev)]
        for model in MODELS:
            beta,processing=fits[model]
            for scope,test in scopes:
                x,_=matrix(test,model,processing);p=x@beta;mse,r2=score(test,p);metrics[model,scope]=(mse,r2);forecasts[model,scope]=p
                output['scores'].append(dict(replicate=rep,scope=scope,model_id=model,n=len(test),mse=mse,r2=r2))
        workflows=[('candidate_'+m,m) for m in MODELS] if rep=='original' else [('aic_selection',choice),('entry_fixed','entry')]
        for workflow,model in workflows:
            for scope,test in scopes:
                for i,(row,p) in enumerate(zip(test,forecasts[model,scope])):
                    e=row['y']-float(p)
                    output['predictions'].append(dict(replicate=rep,scope=scope,workflow=workflow,model_id=model,slot=row.get('slot',f'{i+1:03d}'),row_id=row['row_id'],y=row['y'],prediction=float(p),squared_error=e*e))
        if rep!='original':
            for workflow,model in [('aic_selection',choice),('entry_fixed','entry'),('frozen_choice_incomplete',original_choice)]:
                boot=metrics[model,'bootstrap'];orig=metrics[model,'original_West']
                output['optimism'].append(dict(replicate=rep,workflow=workflow,model_id=model,bootstrap_mse=boot[0],original_mse=orig[0],mse_gap=orig[0]-boot[0],bootstrap_r2=boot[1],original_r2=orig[1],r2_gap=boot[1]-orig[1]))
    for workflow,model in [('aic_selection',original_choice),('entry_fixed','entry'),('frozen_choice_incomplete',original_choice)]:
        apparent=next(s for s in output['scores'] if s['replicate']=='original' and s['scope']=='apparent' and s['model_id']==model)
        subset=[s for s in output['optimism'] if s['workflow']==workflow];mg=np.array([s['mse_gap'] for s in subset]);rg=np.array([s['r2_gap'] for s in subset])
        output['corrections'].append(dict(workflow=workflow,original_model=model,replicates=len(subset),apparent_mse=apparent['mse'],mean_mse_gap=float(np.mean(mg)),corrected_mse=apparent['mse']+float(np.mean(mg)),mse_gap_mcse=float(np.std(mg,ddof=1)/np.sqrt(len(mg))),apparent_r2=apparent['r2'],mean_r2_gap=float(np.mean(rg)),corrected_r2=apparent['r2']-float(np.mean(rg)),r2_gap_mcse=float(np.std(rg,ddof=1)/np.sqrt(len(rg)))))
    summary={}
    for label,subset in [('West',[r for r in rows if r['region']=='West']),('other',[r for r in rows if r['region']!='West'])]:
        valid=[r for r in subset if r['y'] is not None]
        for key,val in [('pool_n',len(subset)),('grade_n',len(valid)),('missing_n',len(subset)-len(valid)),('college_n',len({r['college_id'] for r in valid})),('exit_grade_n',sum(r['gpa_at_exit'] for r in valid))]:summary[label+'_'+key]=val
    for m in MODELS:summary['selected_'+m+'_n']=sum(s['replicate']!='original' and s['selected_model']==m for s in output['selections'])
    summary['replicates_n']=200;summary['fits_n']=len(output['diagnostics']);summary['fit_slots_n']=len(output['fit_rows']);summary['predictions_n']=len(output['predictions'])
    output['summary']=[dict(name=k,value=v) for k,v in sorted(summary.items())]
    output['examples']=[p for p in output['predictions'] if p['replicate']=='B0001' and p['workflow']=='aic_selection' and p['scope']=='original_West'][:3]
    return output
def emit(result,out):
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in result.items():
        if not rows:continue
        rows=sorted(rows,key=lambda r:tuple(str(r[k]) for k in r if k in ('replicate','scope','workflow','model_id','slot','row_id','column','term','name')))
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
            for row in rows:w.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
    try:
        result=calculate(a.root);emit(result,a.outdir);print('UNIT03_NUMPY_OK fits='+str(len(result['diagnostics']))+' predictions='+str(len(result['predictions']))+' numpy='+np.__version__)
    except (ValueError,KeyError,OSError,np.linalg.LinAlgError) as e:sys.exit('UNIT03_NUMPY_ERROR '+str(e))
