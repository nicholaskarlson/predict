#!/usr/bin/env python3
"""Primary NumPy path: independently join, fit, predict, and score Unit 1."""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np

RESOURCE = Path(__file__).resolve().parents[1]

def require(ok, message):
    if not ok: raise ValueError(message)

def joined_rows(root):
    rows = []
    for cohort in ('anchor', 'cohort2'):
        source = root / 'source/tonmick-v1.0.0/data' / cohort
        tables = []
        for name in ('participants', 'baseline', 'registry'):
            with (source / (name + '.csv')).open(newline='') as f: raw = list(csv.DictReader(f))
            table = {r['student_id']:r for r in raw}
            require(len(table) == len(raw), 'duplicate student ID: ' + name)
            tables.append(table)
        frame, base, registry = tables
        require(set(base) == set(registry) == {k for k,v in frame.items() if v['responded_baseline']=='1'},
                'respondent one-to-one join failed')
        for key in sorted(base):
            p, b, y = frame[key], base[key], registry[key]
            require(p['college_id'] == b['college_id'], 'college mismatch')
            if cohort == 'anchor' and b['offer'] != '0': continue
            if cohort == 'cohort2' and p['college_group'] != 'continuing': continue
            known = y['exit_type'] != 'consent_withdrawn'
            require(y['exit_type'] in ('none','withdrawal','transfer','consent_withdrawn'), 'unknown exit type')
            if known:
                require(int(y['withdrawn_12']) == int(y['exit_type']=='withdrawal'), 'outcome/exit mismatch')
            role = 'excluded_unknown' if not known else 'train' if cohort == 'anchor' else 'test'
            require(p['part_time'] in ('0','1'), 'invalid part-time predictor')
            grade = float(p['prior_grade'])
            require(np.isfinite(grade) and 0 <= grade <= 4, 'missing or invalid entry grade')
            rows.append(dict(row_id=key, role=role, college_id=p['college_id'], part_time=int(p['part_time']),
                             prior_grade=grade, y=int(y['withdrawn_12']) if known else None,
                             exit_type=y['exit_type']))
    rows.sort(key=lambda x:x['row_id'])
    require(len({r['row_id'] for r in rows}) == len(rows), 'cohort IDs overlap')
    return rows

def calculate(root=RESOURCE):
    rows = joined_rows(root)
    training = [r for r in rows if r['role']=='train']
    evaluation = [r for r in rows if r['role']=='test']
    require(training and evaluation, 'empty fit or evaluation population')
    x = np.array([r['prior_grade'] for r in training], dtype=np.float64)
    y = np.array([r['y'] for r in training], dtype=np.int64)
    part = np.array([r['part_time'] for r in training], dtype=np.int64)
    cuts = np.quantile(x, [.25,.5,.75], method='linear')
    baseline = float(np.mean(y))
    bins = np.searchsorted(cuts, x, side='left')
    key = 4 * part + bins
    ns = np.bincount(key, minlength=8)
    events = np.bincount(key, weights=y, minlength=8).astype(np.int64)
    risks = np.full(8, baseline)
    np.divide(events, ns, out=risks, where=ns>0)
    cells = [dict(part_time=int(k//4), grade_bin=int(k%4+1), n=int(ns[k]),
                  events=int(events[k]), risk=float(risks[k]), fallback=int(ns[k]==0)) for k in range(8)]
    predictions = []
    for r in rows:
        if r['role']=='excluded_unknown': continue
        g = int(np.searchsorted(cuts, r['prior_grade'], side='left'))
        predictions.append(dict(row_id=r['row_id'],role=r['role'],y=r['y'],grade_bin=g+1,
             part_time=r['part_time'],table_probability=float(risks[4*r['part_time']+g]),
             baseline_probability=baseline))
    summary = dict(control_pool_n=len(training)+sum(r['role']=='excluded_unknown' for r in rows),
                   unknown_n=sum(r['role']=='excluded_unknown' for r in rows),
                   development_n=len(training),development_events=int(y.sum()),
                   temporal_n=len(evaluation),temporal_events=sum(r['y'] for r in evaluation),
                   training_prevalence=baseline,
                   temporal_prevalence=sum(r['y'] for r in evaluation)/len(evaluation),
                   development_transfer_n=sum(r['exit_type']=='transfer' for r in training),
                   temporal_transfer_n=sum(r['exit_type']=='transfer' for r in evaluation))
    for role,label in [('train','development'),('test','temporal')]:
        selected = [r for r in predictions if r['role']==role]
        outcome = np.array([r['y'] for r in selected])
        for model in ('table','baseline'):
            p = np.array([r[model+'_probability'] for r in selected])
            summary[label+'_brier_'+model] = float(np.mean(np.square(outcome-p)))
    summary['temporal_improvement'] = summary['temporal_brier_baseline']-summary['temporal_brier_table']
    summary['temporal_relative_improvement'] = summary['temporal_improvement']/summary['temporal_brier_baseline']
    return dict(rows=rows,cutpoints=[float(q) for q in cuts],cells=cells,
                predictions=predictions,summary=summary,fit_rows=[r['row_id'] for r in training])

def write_csv(path, fields, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
        for row in rows:
            writer.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})

def emit(result, out):
    out.mkdir(parents=True,exist_ok=True)
    write_csv(out/'summary.csv',['name','value'],[dict(name=k,value=v) for k,v in sorted(result['summary'].items())])
    write_csv(out/'cutpoints.csv',['quantile','value'],[dict(quantile=q,value=v) for q,v in zip([.25,.5,.75],result['cutpoints'])])
    write_csv(out/'cells.csv',['part_time','grade_bin','n','events','risk','fallback'],result['cells'])
    write_csv(out/'predictions.csv',['row_id','role','y','grade_bin','part_time','table_probability','baseline_probability'],result['predictions'])
    write_csv(out/'fit_rows.csv',['row_id'],[dict(row_id=i) for i in result['fit_rows']])

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=RESOURCE)
    parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    try:
        result=calculate(args.root);emit(result,args.outdir)
        print('UNIT01_NUMPY_OK train='+str(result['summary']['development_n'])+' test='+str(result['summary']['temporal_n'])+' numpy='+np.__version__)
    except (ValueError,KeyError,OSError) as e:sys.exit('UNIT01_NUMPY_ERROR '+str(e))
