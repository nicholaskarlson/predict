"""Frozen enrollment predictions and sealed scores; NumPy route, no fitting."""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

TERMS = ['intercept','part_time','first_gen','z_age','gender_male','gender_another',
         'program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade']
PROGRAMS = ['Arts and Humanities','Social Sciences','STEM','Health','Business']

def read(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))

def predict(rows, coefficients, transforms):
    """Only enrollment fields enter this function; labels and risks are ignored."""
    columns = [np.ones(len(rows)), np.array([int(x['part_time']) for x in rows]),
               np.array([int(x['first_gen']) for x in rows])]
    columns.append((np.array([float(x['age']) for x in rows])-transforms['age'][0])/transforms['age'][1])
    columns += [np.array([int(x['gender']==g) for x in rows]) for g in ['male','another']]
    columns += [np.array([int(x['program_area']==p) for x in rows]) for p in PROGRAMS[1:]]
    columns.append((np.array([float(x['prior_grade']) for x in rows])-transforms['prior_grade'][0])/transforms['prior_grade'][1])
    eta = np.column_stack(columns) @ np.asarray(coefficients)
    e = np.exp(-np.abs(eta))
    return np.where(eta >= 0, 1/(1+e), e/(1+e))

def validate(rows, plan, colleges):
    if len(rows)!=plan['expected_rows'] or len({r['student_id'] for r in rows})!=len(rows): raise ValueError('sealed row count or duplicate ID')
    for r in rows:
        if list(r)!=plan['raw_header'] or not r['student_id'].startswith('SD'): raise ValueError('sealed schema or identity')
        if r['college_id'] not in colleges or r['region']!=colleges[r['college_id']]: raise ValueError('college or region')
        if r['part_time'] not in ['0','1'] or r['first_gen'] not in ['0','1']: raise ValueError('frame binary')
        if r['gender'] not in ['female','male','another'] or r['program_area'] not in PROGRAMS: raise ValueError('category')
        for name,lo,hi in [('age',16,80),('prior_grade',0,4),('risk12_no_transfer',0,1),('exit_month',0,12)]:
            value=float(r[name])
            if not math.isfinite(value) or not lo<=value<=hi: raise ValueError('range '+name)
        if r['exit_type'] not in ['none','withdrawal','transfer']: raise ValueError('unknown or censored outcome')
        if r['withdrawn_12']!=str(int(r['exit_type']=='withdrawal')) or r['transferred_12']!=str(int(r['exit_type']=='transfer')): raise ValueError('outcome coherence')

def auc(p, y):
    events=int(y.sum()); negatives=len(y)-events; denominator=2*events*negatives
    if not denominator:return 0,0,'','empty' if not len(y) else 'one_class'
    order=np.argsort(p,kind='stable');ps=p[order];ys=y[order]
    _, starts, counts=np.unique(ps,return_index=True,return_counts=True)
    positives=np.add.reduceat(ys,starts);neg=counts-positives
    previous=np.cumsum(neg)-neg
    pairs=int(np.sum(positives*(2*previous+neg)))
    return pairs,denominator,pairs/denominator,''

def score(p,y,q):
    if len(p)!=len(y) or len(q)!=len(y) or np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)) or np.any(~np.isfinite(q)) or np.any((q<0)|(q>1)) or np.any((y!=0)&(y!=1)):raise ValueError('score inputs')
    pairs,denominator,area,reason=auc(p,y)
    if not len(y):return [0,0,0,'','','','',pairs,denominator,area,reason,0,'','','','','empty']
    clipped=np.clip(p,1e-12,1-1e-12);gap=p-q
    return [len(y),int(y.sum()),len(y)-int(y.sum()),float(y.mean()),float(p.mean()),float(np.mean((p-y)**2)),
            float(np.mean(-y*np.log(clipped)-(1-y)*np.log1p(-clipped))),pairs,denominator,area,reason,
            int(np.count_nonzero(p!=clipped)),float(q.mean()),float(gap.mean()),float(np.mean(np.abs(gap))),float(np.sqrt(np.mean(gap**2))),'different_target_no_transfer_reference']

def write(path, header, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    def cell(v):return format(float(v),'.17g') if isinstance(v,(float,np.floating)) else v
    with path.open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(header);writer.writerows([[cell(v) for v in row] for row in rows])

SCORE_HEADER=['population','college_id','method_id','n','events','nonevents','prevalence','mean_probability','brier','log_loss','auc_pairs2','auc_denominator','auc','auc_reason','clipped_n','mean_no_transfer_risk','mean_gap_to_no_transfer_risk','mae_to_no_transfer_risk','rmse_to_no_transfer_risk','risk_reference_reason']

def main(resource,out):
    plan=json.loads((resource/'model-specs/sealed_envelope.json').read_text())
    rows=read(resource/'source/sealed-envelope/deployment_c2.csv')
    colleges={x['college_id']:x['region'] for x in read(resource/'source/sealed-envelope/colleges.csv')}
    validate(rows,plan,colleges)
    cs=read(resource/'model-specs/unit12_coefficients.csv'); ts=read(resource/'model-specs/unit12_transforms.csv')
    if [x['term'] for x in cs]!=TERMS:raise ValueError('frozen model term order')
    coeff=[float(x['coefficient']) for x in cs];transforms={x['feature']:(float(x['mean']),float(x['sd'])) for x in ts}
    p=predict(rows,coeff,transforms);constant=plan['comparator']['events']/plan['comparator']['n']
    probabilities={'frame_only':p,'development_prevalence':np.full(len(rows),constant)}
    y=np.array([int(x['withdrawn_12']) for x in rows],dtype=np.int64);q=np.array([float(x['risk12_no_transfer']) for x in rows])
    groupmap={cid:g for g,ids in plan['college_groups'].items() for cid in ids};groups=np.array([groupmap[x['college_id']] for x in rows]);ids=np.array([x['college_id'] for x in rows])
    predictions=[[x['student_id'],x['college_id'],groups[i],format(p[i],'.17g'),format(constant,'.17g')] for i,x in enumerate(rows)]
    scores=[];bins=[]
    for population in plan['report_populations']:
        mask=np.ones(len(rows),dtype=bool) if population=='all' else groups==population
        for method,prob in probabilities.items():
            scores.append([population,'all',method,*score(prob[mask],y[mask],q[mask])])
            for b in range(5):
                lower=b/5;upper=(b+1)/5;selected=mask&(prob>=lower)&((prob<=upper) if b==4 else (prob<upper))
                count=int(selected.sum())
                bins.append([population,method,b+1,lower,upper,count,int(y[selected].sum()),float(prob[selected].mean()) if count else '',float(y[selected].mean()) if count else '',float(q[selected].mean()) if count else '', '' if count else 'empty_bin'])
    for cid in sorted(colleges):
        mask=ids==cid
        for method,prob in probabilities.items():scores.append([groupmap[cid],cid,method,*score(prob[mask],y[mask],q[mask])])
    prior=read(resource/'expected-results/unit11_scores.csv');comparisons=[]
    for group,role in [('continuing','temporal_continuing'),('previously_new','temporal_new')]:
        old=next(x for x in prior if x['method_id']=='frame_only' and x['pattern']=='all' and x['role']==role)
        current=next(x for x in scores if x[:3]==[group,'all','frame_only'])
        comparisons.append([group,role,int(old['n_scored']),float(old['brier']),current[3],current[8],current[8]-float(old['brier']),'descriptive_different_recruitment_populations'])
    write(out/'sealed_predictions.csv',['student_id','college_id','population','frame_probability','comparator_probability'],predictions)
    write(out/'sealed_scores.csv',SCORE_HEADER,scores)
    write(out/'sealed_bins.csv',['population','method_id','bin','lower','upper','n','events','mean_probability','observed_fraction','mean_no_transfer_risk','reason'],bins)
    write(out/'sealed_comparisons.csv',['population','prior_role','prior_n','prior_brier','sealed_n','sealed_brier','difference','reason'],comparisons)
    print('SEALED_NUMPY_OK predictions='+str(len(rows))+' no_refit=true World_draws=0')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resource',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();main(a.resource,a.out)
