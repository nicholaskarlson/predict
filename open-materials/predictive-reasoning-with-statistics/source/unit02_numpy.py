#!/usr/bin/env python3
"""Primary NumPy path: independently join observed inputs, fit and score."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

RESOURCE=Path(__file__).resolve().parents[1]

def require(ok,message):
    if not ok:raise ValueError(message)

def csvrows(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def records(root=RESOURCE):
    output=[]
    for cohort in ['anchor','cohort2']:
        folder=root/'source/tonmick-v1.0.0/data'/cohort;indexed=[]
        for file in ['participants','baseline','registry']:
            table=csvrows(folder/(file+'.csv'));index={r['student_id']:r for r in table}
            require(len(index)==len(table),'duplicate key '+file);indexed.append(index)
        frame,survey,registry=indexed
        require(set(survey)==set(registry)=={k for k,v in frame.items() if v['responded_baseline']=='1'},'one-to-one respondent join')
        for key in sorted(survey):
            p,b,o=frame[key],survey[key],registry[key]
            require(p['college_id']==b['college_id'],'college conflict')
            if cohort=='anchor' and b['offer']!='0':continue
            if cohort=='cohort2' and p['college_group']!='continuing':continue
            require(p['part_time'] in ('0','1'),'part-time coding')
            x=float(p['prior_grade']);require(np.isfinite(x) and 0<=x<=4,'entry grade')
            kind=o['exit_type'];require(kind in ('none','withdrawal','transfer','consent_withdrawn'),'exit category')
            known=kind!='consent_withdrawn'
            if known:require(o['withdrawn_12']==str(int(kind=='withdrawal')) and o['transferred_12']==str(int(kind=='transfer')),'known binary label')
            grade=float(o['gpa12']) if o['gpa12'] else None
            require(grade is None or (known and np.isfinite(grade) and 0<=grade<=4),'grade observation')
            require(o['gpa_at_exit'] in ('0','1'),'exit-grade flag')
            if grade is not None and o['gpa_at_exit']=='1':require(kind in ('withdrawal','transfer') and float(o['exit_month'])>=4,'exit grade timing')
            eligible_role='train' if cohort=='anchor' else 'final'
            output.append(dict(row_id=key,cohort=cohort,college_id=p['college_id'],part_time=int(p['part_time']),prior_grade=x,
                exit_type=kind,gpa_at_exit=int(o['gpa_at_exit']),withdrawal=int(kind=='withdrawal') if known else None,
                first_exit=int(kind in ('withdrawal','transfer')) if known else None,grade=grade,
                binary_role=eligible_role if known else 'excluded_unknown',grade_role=eligible_role if grade is not None else 'excluded_missing_grade'))
    output.sort(key=lambda r:r['row_id']);require(len(output)==len({r['row_id'] for r in output}),'cross-cohort duplicate')
    require(csvrows(root/'splits/unit02_roles.csv')==[{k:r[k] for k in ['row_id','binary_role','grade_role']} for r in output],'role drift')
    return output

def fit(rows):
    binary=[r for r in rows if r['binary_role']=='train'];grade=[r for r in rows if r['grade_role']=='train']
    require(binary and grade,'empty training population')
    cuts=np.quantile(np.array([r['prior_grade'] for r in binary]),[.25,.5,.75],method='linear')
    gy=np.array([r['grade'] for r in grade]);params={'cuts':cuts,'mean':float(np.mean(gy)),'median':float(np.median(gy))}
    fits=[dict(target='grade',recipe=k,component='constant',value=params[k]) for k in ('mean','median')];cells=[]
    key=np.array([4*r['part_time']+np.searchsorted(cuts,r['prior_grade'],side='left') for r in binary])
    ns=np.bincount(key,minlength=8)
    for target in ('first_exit','withdrawal'):
        raw=np.array([r[target] for r in binary]);require(np.all(np.isin(raw,[0,1])),'binary fit labels')
        prevalence=float(np.mean(raw));events=np.bincount(key,weights=raw,minlength=8).astype(int)
        risks=np.full(8,prevalence);np.divide(events,ns,out=risks,where=ns>0)
        params[target]={'prevalence':prevalence,'table':risks}
        fits.append(dict(target=target,recipe='prevalence',component='constant',value=prevalence))
        fits.extend(dict(target=target,recipe='risk_table',component='cut_'+str(j),value=float(q)) for j,q in enumerate(cuts,1))
        cells.extend(dict(target=target,part_time=k//4,grade_bin=k%4+1,n=int(ns[k]),events=int(events[k]),prediction=float(risks[k]),fallback=int(ns[k]==0)) for k in range(8))
    return params,sorted(fits,key=lambda r:(r['target'],r['recipe'],r['component'])),cells

def predict(params,row,target,recipe):
    if target=='grade':return row['prior_grade'] if recipe=='prior_grade' else params[recipe]
    if recipe=='prevalence':return params[target]['prevalence']
    return float(params[target]['table'][4*row['part_time']+np.searchsorted(params['cuts'],row['prior_grade'],side='left')])

def binary_losses(y,p):
    require(np.all(np.isin(y,[0,1])),'binary scoring labels')
    require(np.all(np.isfinite(p)) and np.all((0<=p)&(p<=1)),'probability range')
    clipped=np.clip(p,1e-12,1-1e-12)
    return (p-y)**2,np.where(y==1,-np.log(clipped),-np.log1p(-clipped)),int(np.count_nonzero(p!=clipped))

def calculate(root=RESOURCE):
    rows=records(root);params,fits,cells=fit(rows);summary={};scores=[];predictions=[];examples=[];fit_rows=[]
    for cohort in ('anchor','cohort2'):
        group=[r for r in rows if r['cohort']==cohort]
        summary.update({cohort+'_pool':len(group),cohort+'_known':sum(r['binary_role']!='excluded_unknown' for r in group),
            cohort+'_unknown':sum(r['binary_role']=='excluded_unknown' for r in group),cohort+'_withdrawals':sum(r['withdrawal']==1 for r in group),
            cohort+'_transfers':sum(r['exit_type']=='transfer' for r in group),cohort+'_grade_observed':sum(r['grade'] is not None for r in group),
            cohort+'_grade_missing':sum(r['grade'] is None for r in group),cohort+'_exit_grades':sum(r['grade'] is not None and r['gpa_at_exit']==1 for r in group)})
    for target in ('first_exit','grade','withdrawal'):
        rolekey='grade_role' if target=='grade' else 'binary_role'
        fit_rows.extend(dict(target=target,row_id=r['row_id']) for r in rows if r[rolekey]=='train')
        final=[r for r in rows if r[rolekey]=='final'];require(final,'empty final population');summary[target+'_final_n']=len(final)
        y=np.array([r[target] for r in final],dtype=float)
        if target!='grade':summary[target+'_train_prevalence']=params[target]['prevalence'];summary[target+'_final_prevalence']=float(np.mean(y))
        for recipe in (('mean','median','prior_grade') if target=='grade' else ('prevalence','risk_table')):
            p=np.array([predict(params,r,target,recipe) for r in final]);squared=(p-y)**2;absolute=np.abs(p-y)
            if target=='grade':measures={'MAE':float(np.mean(absolute)),'MSE':float(np.mean(squared))};log=None;clipping=0
            else:squared,log,clipping=binary_losses(y,p);measures={'Brier':float(np.mean(squared)),'log_loss':float(np.mean(log))}
            for measure,value in sorted(measures.items()):scores.append(dict(target=target,recipe=recipe,n=len(final),measure=measure,value=value,clipped=clipping if measure=='log_loss' else 0))
            for j,r in enumerate(final):
                predictions.append(dict(target=target,recipe=recipe,row_id=r['row_id'],y=float(y[j]),prediction=float(p[j])))
                if j<3:examples.append(dict(target=target,recipe=recipe,row_id=r['row_id'],y=float(y[j]),prediction=float(p[j]),squared=float(squared[j]),absolute=float(absolute[j]),log_loss='' if log is None else float(log[j])))
    summary['grade_train_mean']=params['mean'];summary['grade_train_median']=params['median']
    summary['withdrawal_wrong_unknown_denominator']=summary['anchor_withdrawals']/summary['anchor_pool']
    summary['first_exit_wrong_unknown_denominator']=(summary['anchor_withdrawals']+summary['anchor_transfers'])/summary['anchor_pool']
    summary['grade_wrong_zero_mean']=params['mean']*summary['anchor_grade_observed']/summary['anchor_pool']
    return dict(rows=rows,summary=summary,fits=fits,cells=cells,scores=scores,predictions=predictions,examples=examples,fit_rows=fit_rows)

def write(path,fields,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})

def emit(result,out):
    out.mkdir(parents=True,exist_ok=True)
    write(out/'summary.csv',['name','value'],[dict(name=k,value=v) for k,v in sorted(result['summary'].items())])
    for key,fields in [('fits',['target','recipe','component','value']),('cells',['target','part_time','grade_bin','n','events','prediction','fallback']),
        ('scores',['target','recipe','n','measure','value','clipped']),('predictions',['target','recipe','row_id','y','prediction']),
        ('examples',['target','recipe','row_id','y','prediction','squared','absolute','log_loss']),('fit_rows',['target','row_id'])]:write(out/(key+'.csv'),fields,result[key])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
    try:
        result=calculate(a.root);emit(result,a.outdir);print('UNIT02_NUMPY_OK predictions='+str(len(result['predictions']))+' numpy='+np.__version__)
    except (ValueError,KeyError,OSError) as error:sys.exit('UNIT02_NUMPY_ERROR '+str(error))
