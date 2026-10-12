"""Paired whole-college Brier bootstrap; Python standard library only.

Inputs are frozen Unit 7 probabilities and observed row/college membership.
This analyzer neither refits a model nor generates resampling assignments.
"""
import argparse, csv, math
from pathlib import Path

def rows(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))

def quantile(values, probability):
    x=sorted(values); position=(len(x)-1)*probability
    i=math.floor(position); j=min(i+1,len(x)-1)
    return x[i]+(position-i)*(x[j]-x[i])

def calculate(resource):
    evaluation={r['row_id']:r for r in rows(resource/'data/unit07_rows.csv') if r['role']=='temporal_final'}
    predictions={}
    for r in rows(resource/'predictions/unit07_predictions.csv'):
        key=(r['model_id'],r['row_id'])
        if key in predictions: raise ValueError('duplicate frozen probability')
        p=float(r['probability'])
        if not math.isfinite(p) or not 0<=p<=1: raise ValueError('invalid probability')
        if r['row_id'] not in evaluation or r['y']!=evaluation[r['row_id']]['y']: raise ValueError('outcome or membership mismatch')
        predictions[key]=p
    colleges=sorted({r['college_id'] for r in evaluation.values()})
    if len(evaluation)!=1341 or len(colleges)!=16: raise ValueError('evaluation target changed')
    assignments=rows(resource/'splits/score_college_bootstrap.csv')
    output={'score_colleges.csv':[], 'score_draws.csv':[], 'score_summary.csv':[]}
    for comparator in ['control_mean','control_table']:
        by_college={c:[] for c in colleges}
        for rid,r in sorted(evaluation.items()):
            y=int(r['y']); a=predictions[(comparator,rid)]; b=predictions[('full_logistic',rid)]
            by_college[r['college_id']].append((y-a)**2-(y-b)**2)
        totals={c:math.fsum(by_college[c]) for c in colleges}
        sizes={c:len(by_college[c]) for c in colleges}
        for c in colleges:
            output['score_colleges.csv'].append(dict(comparator=comparator,college_id=c,n=sizes[c],difference_sum=totals[c]))
        differences=[]
        for a in assignments:
            selected=[a['slot_'+str(i).zfill(2)] for i in range(1,17)]
            denominator=sum(sizes[c] for c in selected)
            difference=math.fsum(totals[c] for c in selected)/denominator
            differences.append(difference)
            output['score_draws.csv'].append(dict(comparator=comparator,replicate=int(a['replicate']),n=denominator,difference=difference))
        point=math.fsum(totals.values())/sum(sizes.values())
        output['score_summary.csv'].append(dict(comparator=comparator,n=1341,colleges=16,replicates=len(assignments),difference=point,lower=quantile(differences,.025),upper=quantile(differences,.975)))
    return output

def write(output, folder):
    folder.mkdir(parents=True,exist_ok=True)
    for name,values in output.items():
        with (folder/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader()
            writer.writerows({k:format(v,'.17g') if isinstance(v,float) else v for k,v in r.items()} for r in values)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resource',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    write(calculate(a.resource),a.out)
