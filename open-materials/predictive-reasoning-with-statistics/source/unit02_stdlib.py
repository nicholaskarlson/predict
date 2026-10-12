#!/usr/bin/env python3
"""Independent rational fitting and loss calculations from observed raw CSVs."""
import argparse
import csv
from fractions import Fraction
import math
from pathlib import Path
import sys

RESOURCE=Path(__file__).resolve().parents[1]
TARGETS=('first_exit','grade','withdrawal')

def need(ok,message):
    if not ok:raise ValueError(message)

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def records(root=RESOURCE):
    result=[]
    for cohort in ('anchor','cohort2'):
        source=root/'source/tonmick-v1.0.0/data'/cohort
        tables=[]
        for name in ('participants','baseline','registry'):
            raw=read(source/(name+'.csv'));ids=[r['student_id'] for r in raw]
            need(len(ids)==len(set(ids)),'duplicate '+name)
            tables.append({r['student_id']:r for r in raw})
        p,b,o=tables
        need(set(b)==set(o)=={i for i,r in p.items() if r['responded_baseline']=='1'},'respondent join')
        for i in sorted(b):
            person,base,out=p[i],b[i],o[i]
            need(person['college_id']==base['college_id'],'college join')
            if cohort=='anchor' and base['offer']!='0':continue
            if cohort=='cohort2' and person['college_group']!='continuing':continue
            need(person['part_time'] in ('0','1'),'part-time coding')
            prior=Fraction(person['prior_grade']);need(0<=prior<=4,'prior grade')
            kind=out['exit_type'];need(kind in ('none','withdrawal','transfer','consent_withdrawn'),'exit type')
            unknown=kind=='consent_withdrawn';grade=Fraction(out['gpa12']) if out['gpa12'] else None
            if not unknown:
                need(out['withdrawn_12']==str(int(kind=='withdrawal')) and out['transferred_12']==str(int(kind=='transfer')),'known label')
            need(grade is None or (0<=grade<=4 and not unknown),'observed grade')
            need(out['gpa_at_exit'] in ('0','1'),'exit-grade flag')
            if grade is not None and out['gpa_at_exit']=='1':
                need(kind in ('withdrawal','transfer') and Fraction(out['exit_month'])>=4,'exit-grade timing')
            role='train' if cohort=='anchor' else 'final'
            result.append(dict(row_id=i,cohort=cohort,college_id=person['college_id'],part_time=int(person['part_time']),
                prior_grade=prior,exit_type=kind,gpa_at_exit=int(out['gpa_at_exit']),
                withdrawal=None if unknown else int(kind=='withdrawal'),first_exit=None if unknown else int(kind in ('withdrawal','transfer')),
                grade=grade,binary_role='excluded_unknown' if unknown else role,
                grade_role='excluded_missing_grade' if grade is None else role))
    result.sort(key=lambda r:r['row_id']);need(len(result)==len({r['row_id'] for r in result}),'cross-cohort key')
    roles=[dict(row_id=r['row_id'],binary_role=r['binary_role'],grade_role=r['grade_role']) for r in result]
    need(roles==read(root/'splits/unit02_roles.csv'),'role manifest')
    return result

def type7(values,q):
    values=sorted(values);need(bool(values),'empty quantile')
    position=(len(values)-1)*q;left=position.numerator//position.denominator;weight=position-left
    return values[left] if not weight else values[left]+weight*(values[left+1]-values[left])

def role(row,target):return row['grade_role'] if target=='grade' else row['binary_role']
def recipes(target):return ('mean','median','prior_grade') if target=='grade' else ('prevalence','risk_table')

def fit(rows):
    binary=[r for r in rows if r['binary_role']=='train'];grade=[r for r in rows if r['grade_role']=='train']
    need(bool(binary) and bool(grade),'empty fit')
    cuts=[type7([r['prior_grade'] for r in binary],q) for q in (Fraction(1,4),Fraction(1,2),Fraction(3,4))]
    params={'cuts':cuts,'mean':sum(r['grade'] for r in grade)/len(grade),'median':type7([r['grade'] for r in grade],Fraction(1,2))}
    cells=[];fits=[dict(target='grade',recipe=k,component='constant',value=params[k]) for k in ('mean','median')]
    for target in ('first_exit','withdrawal'):
        need(all(r[target] in (0,1) for r in binary),'binary fit labels')
        prevalence=Fraction(sum(r[target] for r in binary),len(binary));table={}
        params[target]={'prevalence':prevalence,'table':table}
        fits.append(dict(target=target,recipe='prevalence',component='constant',value=prevalence))
        fits.extend(dict(target=target,recipe='risk_table',component='cut_'+str(j),value=q) for j,q in enumerate(cuts,1))
        for part in (0,1):
            for binid in range(1,5):
                members=[r for r in binary if r['part_time']==part and 1+sum(r['prior_grade']>q for q in cuts)==binid]
                n=len(members);events=sum(r[target] for r in members);prob=Fraction(events,n) if n else prevalence
                table[part,binid]=prob;cells.append(dict(target=target,part_time=part,grade_bin=binid,n=n,events=events,prediction=prob,fallback=int(not n)))
    return params,sorted(fits,key=lambda r:(r['target'],r['recipe'],r['component'])),cells

def predict(params,row,target,recipe):
    if target=='grade':return row['prior_grade'] if recipe=='prior_grade' else params[recipe]
    if recipe=='prevalence':return params[target]['prevalence']
    return params[target]['table'][row['part_time'],1+sum(row['prior_grade']>q for q in params['cuts'])]

def losses(y,p,binary):
    square=(p-y)**2;absolute=abs(p-y)
    if not binary:return square,absolute,None,0
    need(y in (0,1),'binary scoring labels');need(0<=p<=1,'probability range')
    v=float(p);clipped=min(max(v,1e-12),1-1e-12)
    log=-math.log(clipped) if y==1 else -math.log1p(-clipped)
    return square,absolute,log,int(v!=clipped)

def calculate(root=RESOURCE):
    rows=records(root);params,fits,cells=fit(rows);predictions=[];scores=[];examples=[];fit_rows=[];summary={};anchors={}
    for cohort in ('anchor','cohort2'):
        group=[r for r in rows if r['cohort']==cohort]
        conditions={'pool':lambda r:True,'known':lambda r:r['binary_role']!='excluded_unknown','unknown':lambda r:r['binary_role']=='excluded_unknown',
            'withdrawals':lambda r:r['withdrawal']==1,'transfers':lambda r:r['exit_type']=='transfer',
            'grade_observed':lambda r:r['grade'] is not None,'grade_missing':lambda r:r['grade'] is None,
            'exit_grades':lambda r:r['grade'] is not None and r['gpa_at_exit']==1}
        for name,test in conditions.items():summary[cohort+'_'+name]=sum(test(r) for r in group)
    for target in TARGETS:
        fit_rows.extend(dict(target=target,row_id=r['row_id']) for r in rows if role(r,target)=='train')
        final=[r for r in rows if role(r,target)=='final'];need(bool(final),'empty evaluation')
        summary[target+'_final_n']=len(final)
        if target!='grade':summary[target+'_train_prevalence']=params[target]['prevalence'];summary[target+'_final_prevalence']=Fraction(sum(r[target] for r in final),len(final))
        for recipe in recipes(target):
            squared=[];absolute=[];logs=[];clipping=0
            for index,r in enumerate(final):
                y=r[target];p=predict(params,r,target,recipe);sq,ab,lo,cl=losses(y,p,target!='grade')
                squared.append(sq);absolute.append(ab);clipping+=cl
                if lo is not None:logs.append(lo)
                predictions.append(dict(target=target,recipe=recipe,row_id=r['row_id'],y=y,prediction=p))
                if index<3:
                    example=dict(target=target,recipe=recipe,row_id=r['row_id'],y=y,prediction=p,squared=sq,absolute=ab,log_loss='' if lo is None else lo)
                    examples.append(example)
                    for name,value in example.items():
                        if name not in ('target','recipe','row_id') and value!='':anchors['u2_example_'+target+'_'+recipe+'_'+str(index+1)+'_'+name]=value
            measures={'MAE':sum(absolute)/len(final),'MSE':sum(squared)/len(final)} if target=='grade' else {'Brier':sum(squared)/len(final),'log_loss':math.fsum(logs)/len(final)}
            for measure,value in sorted(measures.items()):
                scores.append(dict(target=target,recipe=recipe,n=len(final),measure=measure,value=value,clipped=clipping if measure=='log_loss' else 0))
                anchors['u2_'+target+'_'+recipe+'_'+measure]=value
    summary['grade_train_mean']=params['mean'];summary['grade_train_median']=params['median']
    summary['withdrawal_wrong_unknown_denominator']=Fraction(summary['anchor_withdrawals'],summary['anchor_pool'])
    summary['first_exit_wrong_unknown_denominator']=Fraction(summary['anchor_withdrawals']+summary['anchor_transfers'],summary['anchor_pool'])
    summary['grade_wrong_zero_mean']=params['mean']*Fraction(summary['anchor_grade_observed'],summary['anchor_pool'])
    anchors.update({'u2_'+k:v for k,v in summary.items()})
    anchors.update({'u2_cut_'+str(j):q for j,q in enumerate(params['cuts'],1)})
    toy=[Fraction(0),Fraction(0),Fraction(3)]
    for recipe,p in [('mean',Fraction(1)),('median',Fraction(0))]:
        anchors['u2_toy_'+recipe+'_MSE']=sum((p-y)**2 for y in toy)/len(toy)
        anchors['u2_toy_'+recipe+'_MAE']=sum(abs(p-y) for y in toy)/len(toy)
    return dict(rows=rows,summary=summary,fits=fits,cells=cells,scores=scores,predictions=predictions,examples=examples,fit_rows=fit_rows,anchors=anchors)

def write(path,fields,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:format(float(v),'.17g') if isinstance(v,(Fraction,float)) else v for k,v in row.items()})

def emit(result,out):
    out.mkdir(parents=True,exist_ok=True)
    write(out/'summary.csv',['name','value'],[dict(name=k,value=v) for k,v in sorted(result['summary'].items())])
    for key,fields in [('fits',['target','recipe','component','value']),('cells',['target','part_time','grade_bin','n','events','prediction','fallback']),
        ('scores',['target','recipe','n','measure','value','clipped']),('predictions',['target','recipe','row_id','y','prediction']),
        ('examples',['target','recipe','row_id','y','prediction','squared','absolute','log_loss']),('fit_rows',['target','row_id'])]:write(out/(key+'.csv'),fields,result[key])
    write(out/'exact_anchors.csv',['name','kind','value'],[dict(name=k,kind='exact_fraction' if isinstance(v,(Fraction,int)) else 'floating_log',value=str(v)) for k,v in sorted(result['anchors'].items())])

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
    try:
        result=calculate(a.root);emit(result,a.outdir);print('UNIT02_STDLIB_OK predictions='+str(len(result['predictions']))+' exact_rational_losses=true log_loss=floating')
    except (ValueError,KeyError,OSError,ZeroDivisionError) as error:sys.exit('UNIT02_STDLIB_ERROR '+str(error))
