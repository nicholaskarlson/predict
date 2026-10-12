#!/usr/bin/env python3
"""Independent standard-library path using exact rational quantiles and scores."""
import argparse
import csv
from fractions import Fraction
from pathlib import Path
import sys

RESOURCE=Path(__file__).resolve().parents[1]

def ensure(ok,message):
    if not ok:raise ValueError(message)

def load_records(root):
    records=[]
    for group in ('anchor','cohort2'):
        def read(kind):
            p=root/'source/tonmick-v1.0.0/data'/group/(kind+'.csv')
            with p.open() as f:items=list(csv.DictReader(f))
            ids=[v['student_id'] for v in items]
            ensure(len(ids)==len(set(ids)),'nonunique '+kind)
            return {v['student_id']:v for v in items}
        people, survey, follow=read('participants'),read('baseline'),read('registry')
        ensure(set(survey)==set(follow),'survey/registry membership differs')
        ensure(set(survey)==set(i for i,p in people.items() if p['responded_baseline']=='1'),'respondent membership differs')
        for i in survey:
            p,s,t=people[i],survey[i],follow[i]
            ensure(p['college_id']==s['college_id'],'college join differs')
            if group=='anchor':
                if s['offer']!='0':continue
            elif p['college_group']!='continuing':continue
            ensure(t['exit_type'] in {'none','withdrawal','transfer','consent_withdrawn'},'invalid exit')
            unknown=t['exit_type']=='consent_withdrawn'
            if not unknown:ensure(t['withdrawn_12']==str(int(t['exit_type']=='withdrawal')),'incoherent outcome')
            ensure(p['part_time'] in {'0','1'},'invalid enrollment status')
            grade=Fraction(p['prior_grade']);ensure(0<=grade<=4,'invalid entry grade')
            records.append(dict(row_id=i,role='excluded_unknown' if unknown else 'train' if group=='anchor' else 'test',
                college_id=p['college_id'],part_time=int(p['part_time']),prior_grade=grade,y=None if unknown else int(t['withdrawn_12']),
                exit_type=t['exit_type']))
    ensure(len(records)==len(set(r['row_id'] for r in records)),'cross-cohort duplicate')
    return sorted(records,key=lambda r:r['row_id'])

def type7(values, probability):
    ordered=sorted(values)
    ensure(ordered,'empty quantile input')
    position=(len(ordered)-1)*probability
    left=position.numerator//position.denominator
    fraction=position-left
    return ordered[left] if fraction==0 else ordered[left]+fraction*(ordered[left+1]-ordered[left])

def calculate(root=RESOURCE):
    allrows=load_records(root)
    fit=[r for r in allrows if r['role']=='train']
    test=[r for r in allrows if r['role']=='test']
    ensure(fit and test,'empty fit/evaluation')
    cuts=[type7([r['prior_grade'] for r in fit],q) for q in [Fraction(1,4),Fraction(1,2),Fraction(3,4)]]
    baseline=Fraction(sum(r['y'] for r in fit),len(fit))
    def binof(row):return 1+sum(row['prior_grade']>q for q in cuts)
    cells=[];probabilities={}
    for part in (0,1):
        for gradebin in range(1,5):
            members=[r for r in fit if r['part_time']==part and binof(r)==gradebin]
            n=len(members);events=sum(r['y'] for r in members)
            risk=Fraction(events,n) if n else baseline
            probabilities[part,gradebin]=risk
            cells.append(dict(part_time=part,grade_bin=gradebin,n=n,events=events,risk=float(risk),fallback=int(n==0)))
    predictions=[];exact=[]
    for r in allrows:
        if r['role']=='excluded_unknown':continue
        binid=binof(r);risk=probabilities[r['part_time'],binid]
        predictions.append(dict(row_id=r['row_id'],role=r['role'],y=r['y'],grade_bin=binid,part_time=r['part_time'],
                                table_probability=float(risk),baseline_probability=float(baseline)))
        exact.append((r['role'],r['y'],risk))
    result=dict(control_pool_n=len(fit)+sum(r['role']=='excluded_unknown' for r in allrows),
                unknown_n=sum(r['role']=='excluded_unknown' for r in allrows),
                development_n=len(fit),development_events=sum(r['y'] for r in fit),
                temporal_n=len(test),temporal_events=sum(r['y'] for r in test),
                training_prevalence=baseline,temporal_prevalence=Fraction(sum(r['y'] for r in test),len(test)),
                development_transfer_n=sum(r['exit_type']=='transfer' for r in fit),
                temporal_transfer_n=sum(r['exit_type']=='transfer' for r in test))
    for role,label in [('train','development'),('test','temporal')]:
        members=[(y,p) for rr,y,p in exact if rr==role]
        result[label+'_brier_table']=sum((p-y)**2 for y,p in members)/len(members)
        result[label+'_brier_baseline']=sum((baseline-y)**2 for y,p in members)/len(members)
    result['temporal_improvement']=result['temporal_brier_baseline']-result['temporal_brier_table']
    result['temporal_relative_improvement']=result['temporal_improvement']/result['temporal_brier_baseline']
    return dict(rows=allrows,cutpoints=[float(q) for q in cuts],exact_cutpoints=[str(q) for q in cuts],cells=cells,predictions=predictions,
                summary={k:float(v) if isinstance(v,Fraction) else v for k,v in result.items()},
                exact_summary={k:str(v) for k,v in result.items()},fit_rows=[r['row_id'] for r in fit])

def output_csv(p,columns,values):
    with p.open('w',newline='') as file:
        out=csv.DictWriter(file,fieldnames=columns,lineterminator='\n');out.writeheader()
        for item in values:out.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in item.items()})

def emit(result,out):
    out.mkdir(parents=True,exist_ok=True)
    output_csv(out/'summary.csv',['name','value'],[dict(name=k,value=v) for k,v in sorted(result['summary'].items())])
    output_csv(out/'exact_summary.csv',['name','fraction'],[dict(name=k,fraction=v) for k,v in sorted(result['exact_summary'].items())])
    output_csv(out/'cutpoints.csv',['quantile','value'],[dict(quantile=p,value=q) for p,q in zip([.25,.5,.75],result['cutpoints'])])
    output_csv(out/'exact_cutpoints.csv',['quantile','fraction'],[dict(quantile=p,fraction=q) for p,q in zip([.25,.5,.75],result['exact_cutpoints'])])
    output_csv(out/'cells.csv',['part_time','grade_bin','n','events','risk','fallback'],result['cells'])
    output_csv(out/'predictions.csv',['row_id','role','y','grade_bin','part_time','table_probability','baseline_probability'],result['predictions'])
    output_csv(out/'fit_rows.csv',['row_id'],[dict(row_id=i) for i in result['fit_rows']])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=RESOURCE);parser.add_argument('--outdir',type=Path,required=True)
    options=parser.parse_args()
    try:
        result=calculate(options.root);emit(result,options.outdir)
        print('UNIT01_STDLIB_OK train='+str(result['summary']['development_n'])+' test='+str(result['summary']['temporal_n'])+' arithmetic=exact_rational')
    except (ValueError,KeyError,OSError,ZeroDivisionError) as error:sys.exit('UNIT01_STDLIB_ERROR '+str(error))
