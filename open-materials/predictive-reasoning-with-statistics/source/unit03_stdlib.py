#!/usr/bin/env python3
"""Standard-library route with independent raw reconstruction, Householder QR
and scalar loss/AIC/bootstrap arithmetic. No scientific-library dependency.
Output schemas and fixed design are shared; no other analyzer is imported.
"""
import argparse,csv,json,math,sys
from pathlib import Path
RESOURCE=Path(__file__).resolve().parents[1]
MODELS=('entry','baseline','rich')
TERMS={'entry':['intercept','z_prior_grade'],'baseline':['intercept','z_prior_grade','part_time','first_gen','z_age'],'rich':['intercept','z_prior_grade','part_time','first_gen','z_age','z_prior_grade_squared','z_age_squared','z_prior_grade_part_time','z_prior_grade_first_gen','z_prior_grade_z_age']}
def require(ok,message):
    if not ok:raise ValueError(message)
def csv_table(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def keyed_table(p):
    rows=csv_table(p);keys=[r['student_id'] for r in rows]
    require(len(keys)==len(set(keys)),'duplicate raw key')
    return dict(zip(keys,rows))
def joined_rows(root):
    d=root/'source/tonmick-v1.0.0/data/anchor'
    people=keyed_table(d/'participants.csv');survey=keyed_table(d/'baseline.csv');grades=keyed_table(d/'registry.csv')
    require(set(survey)==set(grades) and set(survey)=={k for k,v in people.items() if v['responded_baseline']=='1'},'respondent membership')
    out=[]
    for sid in sorted(survey):
        a,b,g=people[sid],survey[sid],grades[sid]
        require(a['college_id']==b['college_id'],'college join')
        if b['offer']!='0':continue
        require(a['part_time'] in ['0','1'] and a['first_gen'] in ['0','1'],'binary predictor')
        age=float(a['age']);prior=float(a['prior_grade'])
        require(math.isfinite(age) and 16<=age<=80 and math.isfinite(prior) and 0<=prior<=4,'invalid predictor')
        label=float(g['gpa12']) if g['gpa12'] else None
        require(label is None or (math.isfinite(label) and 0<=label<=4),'invalid grade')
        require(g['exit_type']!='consent_withdrawn' or label is None,'consent grade')
        role='excluded_missing_grade' if label is None else 'development' if a['region']=='West' else 'setting_check'
        out.append(dict(row_id=sid,college_id=a['college_id'],region=a['region'],part_time=int(a['part_time']),first_gen=int(a['first_gen']),age=age,prior_grade=prior,exit_type=g['exit_type'],gpa_at_exit=int(g['gpa_at_exit']),y=label,role=role))
    expected=[dict(row_id=r['row_id'],college_id=r['college_id'],region=r['region'],role=r['role']) for r in out]
    require(csv_table(root/'splits/unit03_roles.csv')==expected,'role membership mismatch')
    return out
def bootstrap_samples(root,dev):
    manifest=csv_table(root/'splits/unit03_bootstrap.csv');lookup={r['row_id']:r for r in dev};index=0
    strata={c:[r for r in dev if r['college_id']==c] for c in sorted({r['college_id'] for r in dev})}
    for number in range(1,201):
        rep='B'+str(number).zfill(4);sample=[]
        for college,members in strata.items():
            for j in range(1,len(members)+1):
                require(index<len(manifest),'missing bootstrap slot');draw=manifest[index];index+=1
                require((draw['replicate'],draw['college_id'],draw['slot'])==(rep,college,str(j).zfill(3)),'bootstrap slot order')
                require(draw['row_id'] in lookup and lookup[draw['row_id']]['college_id']==college,'bootstrap fit-role or college mismatch')
                row=lookup[draw['row_id']].copy();row['slot']=college+'-'+draw['slot'];sample.append(row)
        yield rep,sample
    require(index==len(manifest),'extra bootstrap slots')
def average(x):return math.fsum(x)/len(x)
def mcse(x):
    mu=average(x);return math.sqrt(math.fsum((v-mu)**2 for v in x)/(len(x)-1)/len(x))
def transforms(train):
    d={}
    for k in ['prior_grade','age']:
        x=[r[k] for r in train];mu=average(x);sd=math.sqrt(average([(v-mu)**2 for v in x]))
        require(math.isfinite(sd) and sd>1e-12,'zero SD');d[k]=(mu,sd)
    return d
def matrix(rows,model,params):
    x=[]
    for r in rows:
        grade=(r['prior_grade']-params['prior_grade'][0])/params['prior_grade'][1]
        age=(r['age']-params['age'][0])/params['age'][1];v=[1.,grade]
        if model!='entry':v.extend([r['part_time'],r['first_gen'],age])
        if model=='rich':v.extend([grade**2,age**2,grade*r['part_time'],grade*r['first_gen'],grade*age])
        x.append(v)
    return x,TERMS[model]
def select(candidates):
    values={r['model_id']:r['aic'] for r in candidates};lowest=min(values.values())
    for m in MODELS:
        if values[m]-lowest<=1e-10:return m
def score(rows,pred):
    labels=[r['y'] for r in rows];mu=average(labels);variance=average([(y-mu)**2 for y in labels])
    require(variance>1e-12,'constant scored labels')
    mse=average([(y-p)**2 for y,p in zip(labels,pred)]);return mse,1-mse/variance
def triangular_solve(r,b):
    p=len(b);answer=[0.0]*p
    for i in range(p-1,-1,-1):
        require(abs(r[i][i])>0,'singular triangular solve')
        answer[i]=(b[i]-math.fsum(r[i][j]*answer[j] for j in range(i+1,p)))/r[i][i]
    return answer

def householder_fit(x,y,spec):
    n=len(x);p=len(x[0]);require(n>=p,'insufficient fit rows')
    a=[list(row) for row in x];qt_y=list(y)
    for j in range(p):
        v=[a[i][j] for i in range(j,n)];norm=math.sqrt(math.fsum(t*t for t in v))
        require(norm>0,'rank-deficient QR design')
        alpha=-math.copysign(norm,v[0]);v[0]-=alpha;denominator=math.fsum(t*t for t in v)
        for k in range(j,p):
            factor=2*math.fsum(v[i-j]*a[i][k] for i in range(j,n))/denominator
            for i in range(j,n):a[i][k]-=factor*v[i-j]
        factor=2*math.fsum(v[i-j]*qt_y[i] for i in range(j,n))/denominator
        for i in range(j,n):qt_y[i]-=factor*v[i-j]
    r=[[a[i][j] if j>=i else 0.0 for j in range(p)] for i in range(p)]
    diagonals=[abs(r[i][i]) for i in range(p)]
    require(min(diagonals)>spec['solver']['rank_relative_tolerance']*max(diagonals),'rank-deficient QR design')
    inverse_columns=[triangular_solve(r,[float(i==j) for i in range(p)]) for j in range(p)]
    norm_r=max(math.fsum(abs(v) for v in row) for row in r)
    norm_inverse=max(math.fsum(abs(col[i]) for col in inverse_columns) for i in range(p))
    condition=norm_r*norm_inverse
    require(math.isfinite(condition) and condition<=spec['solver']['maximum_condition'],'ill-conditioned QR design')
    beta=triangular_solve(r,qt_y[:p]);errors=[yi-math.fsum(t*b for t,b in zip(row,beta)) for row,yi in zip(x,y)]
    score=max(abs(math.fsum(row[j]*e for row,e in zip(x,errors)))/n for j in range(p))
    require(score<=spec['solver']['maximum_normalized_score'],'OLS score audit failed')
    return beta,condition,score,math.fsum(e*e for e in errors)/n

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
            x,terms=matrix(train,model,params);beta,cond,gradient,mse=householder_fit(x,[r['y'] for r in train],spec)
            require(mse>0,'zero residual variance')
            aic=len(train)*math.log(mse)+2*(len(terms)+1)
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
                x,_=matrix(test,model,processing);p=[math.fsum(v*b for v,b in zip(row,beta)) for row in x];mse,r2=score(test,p);metrics[model,scope]=(mse,r2);forecasts[model,scope]=p
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
        subset=[s for s in output['optimism'] if s['workflow']==workflow];mg=[s['mse_gap'] for s in subset];rg=[s['r2_gap'] for s in subset]
        output['corrections'].append(dict(workflow=workflow,original_model=model,replicates=len(subset),apparent_mse=apparent['mse'],mean_mse_gap=average(mg),corrected_mse=apparent['mse']+average(mg),mse_gap_mcse=mcse(mg),apparent_r2=apparent['r2'],mean_r2_gap=average(rg),corrected_r2=apparent['r2']-average(rg),r2_gap_mcse=mcse(rg)))
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
        result=calculate(a.root);emit(result,a.outdir);print('UNIT03_STDLIB_OK fits='+str(len(result['diagnostics']))+' predictions='+str(len(result['predictions'])))
    except (ValueError,KeyError,OSError,ArithmeticError) as e:sys.exit('UNIT03_STDLIB_ERROR '+str(e))
