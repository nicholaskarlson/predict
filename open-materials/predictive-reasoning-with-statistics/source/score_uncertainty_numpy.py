"""Independent vectorized computation of the paired college bootstrap."""
import argparse, csv
from pathlib import Path
import numpy as np

def read(path):
    with path.open(newline='') as stream: return list(csv.DictReader(stream))

def calculate(resource):
    records=[r for r in read(resource/'data/unit07_rows.csv') if r['role']=='temporal_final']
    records.sort(key=lambda r:r['row_id']); ids=[r['row_id'] for r in records]
    if len(set(ids))!=len(ids): raise ValueError('duplicate evaluation ID')
    y=np.asarray([int(r['y']) for r in records],dtype=float)
    colleges=sorted({r['college_id'] for r in records})
    membership=np.asarray([[r['college_id']==c for r in records] for c in colleges],dtype=float)
    sizes=membership.sum(axis=1).astype(int)
    prediction_rows=read(resource/'predictions/unit07_predictions.csv')
    probabilities={}
    for model in ['control_mean','control_table','full_logistic']:
        subset=sorted([r for r in prediction_rows if r['model_id']==model],key=lambda r:r['row_id'])
        if [r['row_id'] for r in subset]!=ids or not np.array_equal(np.asarray([int(r['y']) for r in subset]),y): raise ValueError('pairing mismatch')
        probabilities[model]=np.asarray([float(r['probability']) for r in subset])
        if not np.all(np.isfinite(probabilities[model])) or not np.all((probabilities[model]>=0)&(probabilities[model]<=1)): raise ValueError('invalid probability')
    if (len(ids),len(colleges))!=(1341,16): raise ValueError('target changed')
    assignments=read(resource/'splits/score_college_bootstrap.csv')
    positions={c:i for i,c in enumerate(colleges)}
    selections=np.asarray([[positions[r['slot_'+str(j).zfill(2)]] for j in range(1,17)] for r in assignments])
    draw_n=sizes[selections].sum(axis=1)
    result={'score_colleges.csv':[], 'score_draws.csv':[], 'score_summary.csv':[]}
    for model in ['control_mean','control_table']:
        losses=np.square(y-probabilities[model])-np.square(y-probabilities['full_logistic'])
        sums=membership@losses
        draws=sums[selections].sum(axis=1)/draw_n
        bounds=np.quantile(draws,[.025,.975],method='linear')
        result['score_colleges.csv'] += [dict(comparator=model,college_id=c,n=int(sizes[i]),difference_sum=float(sums[i])) for i,c in enumerate(colleges)]
        result['score_draws.csv'] += [dict(comparator=model,replicate=int(a['replicate']),n=int(draw_n[i]),difference=float(draws[i])) for i,a in enumerate(assignments)]
        result['score_summary.csv'].append(dict(comparator=model,n=len(ids),colleges=len(colleges),replicates=len(assignments),difference=float(losses.mean()),lower=float(bounds[0]),upper=float(bounds[1])))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--resource',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    for name,rows in calculate(a.resource).items():
        with (a.out/name).open('w',newline='') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader()
            for row in rows:w.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})
if __name__=='__main__':main()
