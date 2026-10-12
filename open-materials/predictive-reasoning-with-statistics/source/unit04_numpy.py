#!/usr/bin/env python3
"""Primary path: independent joins, fold-specific transforms, QR and scoring."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

RESOURCE=Path(__file__).resolve().parents[1]
MODELS=('mean','entry','baseline','college')

def require(ok,message):
    if not ok:raise ValueError(message)

def read_rows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def joined_rows(root):
    result=[]
    for cohort in ('anchor','cohort2'):
        raw=root/'source/tonmick-v1.0.0/data'/cohort
        tables=[]
        for name in ('participants','baseline','registry'):
            records=read_rows(raw/(name+'.csv'));indexed={r['student_id']:r for r in records}
            require(len(indexed)==len(records),'duplicate raw student ID '+name);tables.append(indexed)
        frame,survey,registry=tables
        require(set(survey)==set(registry)=={k for k,v in frame.items() if v['responded_baseline']=='1'},
                'respondent one-to-one membership')
        for key in sorted(survey):
            p,b,y=frame[key],survey[key],registry[key]
            require(p['college_id']==b['college_id'],'college join mismatch')
            if cohort=='anchor' and b['offer']!='0':continue
            if cohort=='cohort2' and p['college_group']!='continuing':continue
            require(p['part_time'] in ('0','1') and p['first_gen'] in ('0','1'),'binary predictor coding')
            grade=None if y['gpa12']=='' else float(y['gpa12'])
            require(grade is None or (np.isfinite(grade) and 0<=grade<=4),'invalid observed grade')
            require(y['exit_type']!='consent_withdrawn' or grade is None,'consent grade cannot be observed')
            prior,age=float(p['prior_grade']),float(p['age'])
            require(np.isfinite(prior) and 0<=prior<=4 and np.isfinite(age) and 16<=age<=80,
                    'missing or invalid enrollment predictor')
            result.append(dict(row_id=key,cohort=cohort,college_id=p['college_id'],region=p['region'],
                part_time=int(p['part_time']),first_gen=int(p['first_gen']),age=age,prior_grade=prior,
                offer=0,exit_type=y['exit_type'],exit_month=float(y['exit_month']),
                gpa_at_exit=int(y['gpa_at_exit']),y=grade,
                role=('development' if cohort=='anchor' else 'temporal_final') if grade is not None
                     else 'excluded_missing_grade'))
    result.sort(key=lambda r:r['row_id'])
    require(len({r['row_id'] for r in result})==len(result),'overlapping cohort IDs')
    return result

def attach_assignments(root,rows):
    assignments=read_rows(root/'splits/unit04_assignments.csv')
    require([a['row_id'] for a in assignments]==[r['row_id'] for r in rows],'assignment membership/order')
    groups={}
    for row,a in zip(rows,assignments):
        require(all(str(row[k])==a[k] for k in ('row_id','cohort','college_id','region','role')),
                'assignment key or role mismatch')
        if row['role']=='development':
            require(a['student_fold'] in ('1','2','3','4','5') and a['region_fold']==row['region']
                    and a['college_fold'],'invalid development folds')
            groups.setdefault(row['college_id'],set()).add(a['college_fold'])
        else:require(all(a[k]=='' for k in ('student_fold','college_fold','region_fold')),'final/excluded internal folds')
        row.update({k:a[k] for k in ('student_fold','college_fold','region_fold')})
    require(all(len(g)==1 for g in groups.values()) and len({next(iter(g)) for g in groups.values()})==len(groups),
            'college held-out groups mixed or merged')
    return rows

def jobs(rows):
    development=[r for r in rows if r['role']=='development']
    for design in ('student','college','region'):
        key=design+'_fold'
        for fold in sorted({r[key] for r in development}):
            train=[r for r in development if r[key]!=fold];test=[r for r in development if r[key]==fold]
            require(train and test,'empty fold')
            if design in ('college','region'):
                group=design+'_id' if design=='college' else 'region'
                require(not {r[group] for r in train}&{r[group] for r in test},'held-out group overlap')
            yield design,fold,train,test
    yield 'temporal','final',development,[r for r in rows if r['role']=='temporal_final']

def transforms(train):
    parameters={}
    for column in ('prior_grade','age'):
        x=np.array([r[column] for r in train]);mean=float(np.mean(x));sd=float(np.sqrt(np.mean((x-mean)**2)))
        require(np.isfinite(sd) and sd>1e-12,'zero SD in training transform')
        parameters[column]=(mean,sd)
    return parameters

def matrix(rows,model,parameters,colleges):
    columns=[np.ones(len(rows))];names=['intercept']
    if model!='mean':
        mu,sd=parameters['prior_grade'];columns.append(np.array([(r['prior_grade']-mu)/sd for r in rows]));names.append('z_prior_grade')
    if model in ('baseline','college'):
        for c in ('part_time','first_gen'):
            columns.append(np.array([r[c] for r in rows],dtype=float));names.append(c)
        mu,sd=parameters['age'];columns.append(np.array([(r['age']-mu)/sd for r in rows]));names.append('z_age')
    if model=='college':
        for college in colleges[:-1]:
            columns.append(np.array([1.0 if r['college_id']==college else -1.0 if r['college_id']==colleges[-1] else 0.0 for r in rows]))
            names.append('effect_'+college)
    return np.column_stack(columns),names

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
    spec=json.loads((root/'model-specs/unit04.json').read_text())
    require(spec['model_ids']==list(MODELS) and spec['predictor_allowlist']==['prior_grade','part_time','first_gen','age','college_id']
            and spec['model_selection']=='none_all_four_models_prespecified' and spec['final_test_labels_for_selection'] is False,
            'model/predictor specification drift')
    rows=attach_assignments(root,joined_rows(root))
    pred,coefs,diagnostics,processing,fitrows,foldscores=[],[],[],[],[],[]
    for design,fold,train,test in jobs(rows):
        require(test,'empty evaluation population')
        params=transforms(train);colleges=sorted({r['college_id'] for r in train})
        fitrows.extend(dict(design=design,fold=fold,row_id=r['row_id']) for r in train)
        processing.extend(dict(design=design,fold=fold,column=c,mean=mu,sd=sd) for c,(mu,sd) in params.items())
        for model in MODELS:
            x,terms=matrix(train,model,params,colleges);v,_=matrix(test,model,params,colleges)
            beta,cond,score,fitmse=qr_fit(x,np.array([r['y'] for r in train]),spec)
            coefs.extend(dict(design=design,fold=fold,model_id=model,term=t,coefficient=float(b)) for t,b in zip(terms,beta))
            diagnostics.append(dict(design=design,fold=fold,model_id=model,n_fit=len(train),p=len(terms),rank=len(terms),
                                    condition_r_inf=cond,score_inf=score,fit_mse=fitmse))
            predictions=v@beta;labels=np.array([r['y'] for r in test]);error=labels-predictions
            for row,forecast,e in zip(test,predictions,error):
                pred.append(dict(design=design,fold=fold,model_id=model,row_id=row['row_id'],y=row['y'],
                    prediction=float(forecast),squared_error=float(e*e),absolute_error=float(abs(e)),
                    unseen_college=int(row['college_id'] not in colleges)))
            foldscores.append(dict(design=design,fold=fold,model_id=model,n=len(test),
                n_unseen=sum(r['college_id'] not in colleges for r in test),mse=float(np.mean(error**2)),
                mae=float(np.mean(np.abs(error))),rmse=float(np.sqrt(np.mean(error**2)))))
    aggregates=[]
    for design in ('student','college','region','temporal'):
        for model in MODELS:
            selected=[r for r in pred if r['design']==design and r['model_id']==model]
            folds=[r['mse'] for r in foldscores if r['design']==design and r['model_id']==model]
            mse=float(np.mean([r['squared_error'] for r in selected]))
            aggregates.append(dict(design=design,model_id=model,n=len(selected),folds=len(folds),mse=mse,
                mae=float(np.mean([r['absolute_error'] for r in selected])),rmse=float(np.sqrt(mse)),
                fold_mean_mse=float(np.mean(folds)),fold_min_mse=float(min(folds)),fold_max_mse=float(max(folds))))
    summary={}
    for cohort,label in [('anchor','anchor'),('cohort2','temporal')]:
        pool=[r for r in rows if r['cohort']==cohort];eligible=[r for r in pool if r['y'] is not None]
        summary.update({label+'_pool_n':len(pool),label+'_grade_n':len(eligible),label+'_missing_n':len(pool)-len(eligible),
            label+'_consent_missing_n':sum(r['exit_type']=='consent_withdrawn' for r in pool),
            label+'_exit_grade_n':sum(r['gpa_at_exit'] for r in eligible),
            label+'_college_n':len({r['college_id'] for r in eligible}),label+'_grade_mean':float(np.mean([r['y'] for r in eligible]))})
    summary['wrong_zero_anchor_mean']=sum(r['y'] for r in rows if r['role']=='development')/summary['anchor_pool_n']
    examples=sorted([r for r in pred if r['design']=='temporal' and r['model_id']=='baseline'],key=lambda r:r['row_id'])[:3]
    summary['example_mse']=float(np.mean([r['squared_error'] for r in examples]))
    summary['example_mae']=float(np.mean([r['absolute_error'] for r in examples]))
    summary['example_rmse']=float(np.sqrt(summary['example_mse']))
    return dict(rows=rows,predictions=pred,coefficients=coefs,diagnostics=diagnostics,transforms=processing,
                fit_rows=fitrows,fold_scores=foldscores,aggregate=aggregates,summary=summary,examples=examples)

def emit(result,out):
    out.mkdir(parents=True,exist_ok=True)
    records={k:v for k,v in result.items() if k not in ('rows','summary')}
    records['summary']=[dict(name=k,value=v) for k,v in sorted(result['summary'].items())]
    for name,rows in records.items():
        if not rows:continue
        fields=list(rows[0]);rows=sorted(rows,key=lambda r:tuple(str(r[k]) for k in fields if k not in
          ('y','prediction','squared_error','absolute_error','coefficient','n_fit','p','rank','condition_r_inf',
           'score_inf','fit_mse','mean','sd','n','n_unseen','mse','mae','rmse','folds','fold_mean_mse','fold_min_mse','fold_max_mse','value')))
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
            for row in rows:w.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=RESOURCE)
    p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
    try:
        result=calculate(a.root);emit(result,a.outdir)
        print('UNIT04_NUMPY_OK fits='+str(len(result['diagnostics']))+' predictions='+str(len(result['predictions']))+' numpy='+np.__version__)
    except (ValueError,KeyError,OSError,np.linalg.LinAlgError) as e:sys.exit('UNIT04_NUMPY_ERROR '+str(e))
