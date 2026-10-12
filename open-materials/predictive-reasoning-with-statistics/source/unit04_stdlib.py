#!/usr/bin/env python3
"""Independent standard-library joins, Householder QR, predictions and metrics.

No NumPy and no imports from another analyzer. CSV schemas and the frozen
model record are the shared contract; numerical routines are separate.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

RESOURCE=Path(__file__).resolve().parents[1]

def ensure(ok,message):
    if not ok:raise ValueError(message)

def read_csv(path):
    with path.open(newline='') as handle:return list(csv.DictReader(handle))

def unique(path):
    records=read_csv(path);keys=[r['student_id'] for r in records]
    ensure(len(keys)==len(set(keys)),'duplicate raw student ID '+path.name)
    return {r['student_id']:r for r in records}

def load_records(root):
    records=[]
    for cohort in ['anchor','cohort2']:
        folder=root/'source/tonmick-v1.0.0/data'/cohort
        frame=unique(folder/'participants.csv');survey=unique(folder/'baseline.csv');registry=unique(folder/'registry.csv')
        respondents={key for key in frame if frame[key]['responded_baseline']=='1'}
        ensure(set(survey)==respondents and set(registry)==respondents,'respondent join membership')
        for key in sorted(respondents):
            student=frame[key];interview=survey[key];outcome=registry[key]
            ensure(student['college_id']==interview['college_id'],'college key mismatch')
            if cohort=='anchor' and interview['offer']=='1':continue
            ensure(cohort!='anchor' or interview['offer']=='0','invalid offer')
            if cohort=='cohort2' and student['college_group']!='continuing':continue
            y=float(outcome['gpa12']) if outcome['gpa12'] else None
            ensure(y is None or math.isfinite(y) and 0<=y<=4,'invalid recorded grade')
            ensure(outcome['exit_type']!='consent_withdrawn' or y is None,'observed consent-censored grade')
            entry=float(student['prior_grade']);age=float(student['age'])
            ensure(math.isfinite(entry) and 0<=entry<=4 and math.isfinite(age) and 16<=age<=80,'invalid enrollment predictor')
            ensure(student['part_time'] in ['0','1'] and student['first_gen'] in ['0','1'],'binary predictor coding')
            records.append(dict(row_id=key,cohort=cohort,college_id=student['college_id'],region=student['region'],
                part_time=int(student['part_time']),first_gen=int(student['first_gen']),age=age,prior_grade=entry,
                offer=0,exit_type=outcome['exit_type'],exit_month=float(outcome['exit_month']),
                gpa_at_exit=int(outcome['gpa_at_exit']),y=y,
                role='excluded_missing_grade' if y is None else 'development' if cohort=='anchor' else 'temporal_final'))
    records.sort(key=lambda r:r['row_id'])
    ensure(len(records)==len({r['row_id'] for r in records}),'overlapping student IDs')
    return records

def apply_assignments(root,rows):
    assigned=read_csv(root/'splits/unit04_assignments.csv')
    ensure([r['row_id'] for r in rows]==[a['row_id'] for a in assigned],'assignment membership/order')
    college_groups={}
    for r,a in zip(rows,assigned):
        for key in ('cohort','college_id','region','role'):ensure(str(r[key])==a[key],'assignment key or role mismatch')
        if r['role']=='development':
            ensure(a['student_fold'] in ['1','2','3','4','5'] and a['college_fold'] and a['region_fold']==r['region'],
                   'invalid development fold')
            college_groups.setdefault(r['college_id'],set()).add(a['college_fold'])
        else:ensure(not any(a[k] for k in ('student_fold','college_fold','region_fold')),'final/excluded internal folds')
        for key in ('student_fold','college_fold','region_fold'):r[key]=a[key]
    ensure(all(len(v)==1 for v in college_groups.values()),'college split across folds')
    ensure(len({next(iter(v)) for v in college_groups.values()})==len(college_groups),'college folds merged')
    return rows

def partitions(rows):
    training=[r for r in rows if r['role']=='development']
    for design in ['student','college','region']:
        key=design+'_fold'
        for fold in sorted(set(r[key] for r in training)):
            fit=[r for r in training if r[key]!=fold];held=[r for r in training if r[key]==fold]
            ensure(fit and held,'empty development fold')
            if design!='student':
                group='college_id' if design=='college' else 'region'
                ensure(set(r[group] for r in fit).isdisjoint(r[group] for r in held),'group overlap')
            yield design,fold,fit,held
    yield 'temporal','final',training,[r for r in rows if r['role']=='temporal_final']

def fitted_processing(rows):
    parameters={}
    for key in ['prior_grade','age']:
        values=[r[key] for r in rows];mean=math.fsum(values)/len(values)
        sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/len(values))
        ensure(math.isfinite(sd) and sd>1e-12,'zero SD in fitted processing')
        parameters[key]=(mean,sd)
    return parameters

def design_matrix(rows,model,processing,colleges):
    terms=['intercept']
    if model!='mean':terms+=['z_prior_grade']
    if model in ('baseline','college'):terms+=['part_time','first_gen','z_age']
    if model=='college':terms+=['effect_'+c for c in colleges[:-1]]
    matrix=[]
    for row in rows:
        v=[1.0]
        if model!='mean':v.append((row['prior_grade']-processing['prior_grade'][0])/processing['prior_grade'][1])
        if model in ('baseline','college'):
            v+=[float(row['part_time']),float(row['first_gen']),
                 (row['age']-processing['age'][0])/processing['age'][1]]
        if model=='college':
            v.extend(1.0 if row['college_id']==c else -1.0 if row['college_id']==colleges[-1] else 0.0 for c in colleges[:-1])
        matrix.append(v)
    return matrix,terms

def triangular_solve(r,b):
    p=len(b);answer=[0.0]*p
    for i in range(p-1,-1,-1):
        ensure(abs(r[i][i])>0,'singular triangular solve')
        answer[i]=(b[i]-math.fsum(r[i][j]*answer[j] for j in range(i+1,p)))/r[i][i]
    return answer

def householder_fit(x,y,spec):
    n=len(x);p=len(x[0]);ensure(n>=p,'insufficient fit rows')
    a=[list(row) for row in x];qt_y=list(y)
    for j in range(p):
        v=[a[i][j] for i in range(j,n)];norm=math.sqrt(math.fsum(t*t for t in v))
        ensure(norm>0,'rank-deficient QR design')
        alpha=-math.copysign(norm,v[0]);v[0]-=alpha;denominator=math.fsum(t*t for t in v)
        for k in range(j,p):
            factor=2*math.fsum(v[i-j]*a[i][k] for i in range(j,n))/denominator
            for i in range(j,n):a[i][k]-=factor*v[i-j]
        factor=2*math.fsum(v[i-j]*qt_y[i] for i in range(j,n))/denominator
        for i in range(j,n):qt_y[i]-=factor*v[i-j]
    r=[[a[i][j] if j>=i else 0.0 for j in range(p)] for i in range(p)]
    diagonals=[abs(r[i][i]) for i in range(p)]
    ensure(min(diagonals)>spec['solver']['rank_relative_tolerance']*max(diagonals),'rank-deficient QR design')
    inverse_columns=[triangular_solve(r,[float(i==j) for i in range(p)]) for j in range(p)]
    norm_r=max(math.fsum(abs(v) for v in row) for row in r)
    norm_inverse=max(math.fsum(abs(col[i]) for col in inverse_columns) for i in range(p))
    condition=norm_r*norm_inverse
    ensure(math.isfinite(condition) and condition<=spec['solver']['maximum_condition'],'ill-conditioned QR design')
    beta=triangular_solve(r,qt_y[:p]);errors=[yi-math.fsum(t*b for t,b in zip(row,beta)) for row,yi in zip(x,y)]
    score=max(abs(math.fsum(row[j]*e for row,e in zip(x,errors)))/n for j in range(p))
    ensure(score<=spec['solver']['maximum_normalized_score'],'OLS score audit failed')
    return beta,condition,score,math.fsum(e*e for e in errors)/n

def calculate(root=RESOURCE):
    spec=json.loads((root/'model-specs/unit04.json').read_text())
    models=['mean','entry','baseline','college']
    ensure(spec['model_ids']==models and spec['predictor_allowlist']==['prior_grade','part_time','first_gen','age','college_id']
           and not spec['final_test_labels_for_selection'] and spec['model_selection']=='none_all_four_models_prespecified',
           'model/predictor specification drift')
    rows=apply_assignments(root,load_records(root));pred=[];coefs=[];diagnostics=[];transforms=[];fitrows=[];scores=[]
    for design,fold,fit,held in partitions(rows):
        ensure(held,'empty evaluation population')
        processing=fitted_processing(fit);colleges=sorted(set(r['college_id'] for r in fit))
        fitrows.extend(dict(design=design,fold=fold,row_id=r['row_id']) for r in fit)
        transforms.extend(dict(design=design,fold=fold,column=c,mean=v[0],sd=v[1]) for c,v in processing.items())
        for model in models:
            x,terms=design_matrix(fit,model,processing,colleges)
            beta,condition,score,fitmse=householder_fit(x,[r['y'] for r in fit],spec)
            coefs.extend(dict(design=design,fold=fold,model_id=model,term=t,coefficient=b) for t,b in zip(terms,beta))
            diagnostics.append(dict(design=design,fold=fold,model_id=model,n_fit=len(fit),p=len(terms),rank=len(terms),
                                    condition_r_inf=condition,score_inf=score,fit_mse=fitmse))
            v,_=design_matrix(held,model,processing,colleges);errors=[]
            for row,features in zip(held,v):
                forecast=math.fsum(t*b for t,b in zip(features,beta));e=row['y']-forecast;errors.append(e)
                pred.append(dict(design=design,fold=fold,model_id=model,row_id=row['row_id'],y=row['y'],prediction=forecast,
                                 squared_error=e*e,absolute_error=abs(e),unseen_college=int(row['college_id'] not in colleges)))
            mse=math.fsum(e*e for e in errors)/len(errors)
            scores.append(dict(design=design,fold=fold,model_id=model,n=len(held),n_unseen=sum(r['college_id'] not in colleges for r in held),
                               mse=mse,mae=math.fsum(abs(e) for e in errors)/len(errors),rmse=math.sqrt(mse)))
    aggregate=[]
    for design in ['student','college','region','temporal']:
        for model in models:
            selected=[r for r in pred if r['design']==design and r['model_id']==model]
            folds=[r['mse'] for r in scores if r['design']==design and r['model_id']==model]
            mse=math.fsum(r['squared_error'] for r in selected)/len(selected)
            aggregate.append(dict(design=design,model_id=model,n=len(selected),folds=len(folds),mse=mse,
                mae=math.fsum(r['absolute_error'] for r in selected)/len(selected),rmse=math.sqrt(mse),
                fold_mean_mse=math.fsum(folds)/len(folds),fold_min_mse=min(folds),fold_max_mse=max(folds)))
    summary={}
    for cohort,label in [('anchor','anchor'),('cohort2','temporal')]:
        pool=[r for r in rows if r['cohort']==cohort];observed=[r for r in pool if r['y'] is not None]
        summary[label+'_pool_n']=len(pool);summary[label+'_grade_n']=len(observed);summary[label+'_missing_n']=len(pool)-len(observed)
        summary[label+'_consent_missing_n']=sum(r['exit_type']=='consent_withdrawn' for r in pool)
        summary[label+'_exit_grade_n']=sum(r['gpa_at_exit'] for r in observed)
        summary[label+'_college_n']=len({r['college_id'] for r in observed})
        summary[label+'_grade_mean']=math.fsum(r['y'] for r in observed)/len(observed)
    summary['wrong_zero_anchor_mean']=math.fsum(r['y'] for r in rows if r['role']=='development')/summary['anchor_pool_n']
    examples=sorted([r for r in pred if r['design']=='temporal' and r['model_id']=='baseline'],key=lambda r:r['row_id'])[:3]
    summary['example_mse']=math.fsum(r['squared_error'] for r in examples)/len(examples)
    summary['example_mae']=math.fsum(r['absolute_error'] for r in examples)/len(examples)
    summary['example_rmse']=math.sqrt(summary['example_mse'])
    return dict(rows=rows,predictions=pred,coefficients=coefs,diagnostics=diagnostics,transforms=transforms,
                fit_rows=fitrows,fold_scores=scores,aggregate=aggregate,summary=summary,examples=examples)

def emit(result,out):
    out.mkdir(parents=True,exist_ok=True)
    records={k:v for k,v in result.items() if k not in ('rows','summary')}
    records['summary']=[dict(name=k,value=v) for k,v in sorted(result['summary'].items())]
    numeric={'y','prediction','squared_error','absolute_error','coefficient','n_fit','p','rank','condition_r_inf',
             'score_inf','fit_mse','mean','sd','n','n_unseen','mse','mae','rmse','folds','fold_mean_mse','fold_min_mse','fold_max_mse','value'}
    for name,rows in records.items():
        fields=list(rows[0]);rows=sorted(rows,key=lambda r:tuple(str(r[k]) for k in fields if k not in numeric))
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
            for row in rows:w.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=RESOURCE)
    parser.add_argument('--outdir',type=Path,required=True);a=parser.parse_args()
    try:
        result=calculate(a.root);emit(result,a.outdir)
        print('UNIT04_STDLIB_OK fits='+str(len(result['diagnostics']))+' predictions='+str(len(result['predictions']))+' solver=Householder_QR')
    except (ValueError,KeyError,OSError) as e:sys.exit('UNIT04_STDLIB_ERROR '+str(e))
