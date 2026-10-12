"""Independent NumPy design matrices, vector losses and direct-pair AUC."""
import csv
from pathlib import Path
import numpy as np

ROLES=['temporal_continuing','temporal_new']
TERMS=['intercept','part_time','first_gen','z_age','gender_male','gender_another','program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade']
def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def matrix(records,t):
    for x in records:
        if x['gender'] not in ['female','male','another'] or x['program_area'] not in ['Arts and Humanities','Social Sciences','STEM','Health','Business']:raise ValueError('invalid frame category')
        if str(x['part_time']) not in ['0','1'] or str(x['first_gen']) not in ['0','1']:raise ValueError('invalid frame binary')
    age=np.array([float(x['age']) for x in records]);grade=np.array([float(x['prior_grade']) for x in records])
    if not np.isfinite(age).all() or not np.isfinite(grade).all() or not ((age>=16)&(age<=80)).all() or not ((grade>=0)&(grade<=4)).all():raise ValueError('invalid frame range')
    return np.column_stack([np.ones(len(records)),[float(x['part_time']) for x in records],[float(x['first_gen']) for x in records],(age-t['age'][0])/t['age'][1],[x['gender']=='male' for x in records],[x['gender']=='another' for x in records],*[ [x['program_area']==name for x in records] for name in ['Social Sciences','STEM','Health','Business']],(grade-t['prior_grade'][0])/t['prior_grade'][1]])
def logistic(eta):
    eta=np.asarray(eta);out=np.empty_like(eta,dtype=float);pos=eta>=0
    out[pos]=1/(1+np.exp(-eta[pos]));v=np.exp(eta[~pos]);out[~pos]=v/(1+v);return out
def score(y,p):
    y=np.asarray(y);p=np.asarray(p)
    if not len(y) or y.shape!=p.shape or y.ndim!=1 or not np.isin(y,[0,1]).all() or not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('invalid scoring inputs')
    positive=p[y==1];negative=p[y==0];den=2*len(positive)*len(negative)
    pairs=int((2*(positive[:,None]>negative[None,:])+(positive[:,None]==negative[None,:])).sum())
    return {'n':len(y),'events':int(y.sum()),'nonevents':int((y==0).sum()),'brier':float(np.mean((p-y)**2)),'mean_probability':float(p.mean()),'auc_pairs2':pairs,'auc_denominator':den,'auc':pairs/den if den else '','auc_reason':'' if den else 'one_class'}
def offer_tables(records):
    if len({x['row_id'] for x in records})!=len(records):raise ValueError('duplicate randomized ID')
    if any(x['offer'] not in ['0','1'] or x['y'] not in ['','0','1'] or x['outcome_known']!=str(int(x['y']!='')) for x in records):raise ValueError('randomized label contract')
    output=[];total=len(records)
    for college in sorted({x['college_id'] for x in records})+['pooled']:
        a=[x for x in records if college=='pooled' or x['college_id']==college]
        groups=[ [x for x in a if x['offer']==str(z)] for z in [0,1]]
        if not all(groups) or (college!='pooled' and abs(len(groups[1])-len(groups[0]))>1):raise ValueError('within-college half assignment')
        rec={'college_id':college,'randomized_n':len(a),'weight':len(a)/total if college!='pooled' else 1.}
        for z,g in enumerate(groups):
            observed=np.array([int(x['y']) for x in g if x['y']!='']);n=len(g);k=len(observed);e=int(observed.sum())
            if not k:raise ValueError('no known arm outcomes')
            rec.update({f'assigned{z}':n,f'known{z}':k,f'unknown{z}':n-k,f'events{z}':e,f'risk_known{z}':float(observed.mean()),f'risk_lower{z}':e/n,f'risk_upper{z}':(e+n-k)/n})
        rec.update(rd_known=rec['risk_known1']-rec['risk_known0'],rd_lower=rec['risk_lower1']-rec['risk_upper0'],rd_upper=rec['risk_upper1']-rec['risk_lower0']);output.append(rec)
    summaries=[]
    for name,groups in [('standardized',output[:-1]),('pooled',[output[-1]])]:
        summaries.append({'contrast_id':name,'randomized_n':total,'known_n':sum(x['y']!='' for x in records),'unknown_n':sum(x['y']=='' for x in records),**{field:float(np.dot([x['weight'] for x in groups],[x[field] for x in groups])) for field in ['rd_known','rd_lower','rd_upper']}})
    return output,summaries
def compute(resource):
    resource=Path(resource);data=read(resource/'data/unit12_rows.csv');causal=read(resource/'data/unit12_randomized.csv');maps=read(resource/'splits/unit12_permutations.csv')
    coeff=read(resource/'model-specs/unit12_coefficients.csv')
    if [x['term'] for x in coeff]!=TERMS:raise ValueError('term inventory')
    beta=np.array([float(x['coefficient']) for x in coeff]);t={x['feature']:np.array([float(x['mean']),float(x['sd'])]) for x in read(resource/'model-specs/unit12_transforms.csv')}
    if set(t)!={'age','prior_grade'} or not np.isfinite(beta).all() or any(not np.isfinite(v).all() or v[1]<=0 for v in t.values()):raise ValueError('frozen model parameters')
    if len({x['row_id'] for x in data})!=len(data) or any(x['role'] not in ['development',*ROLES] or x['y'] not in ['0','1'] for x in data):raise ValueError('predictive keys/roles/labels')
    dev=[x for x in data if x['role']=='development'];constant=float(np.array([int(x['y']) for x in dev]).mean())
    predictions=[];scores=[];draws=[];summaries=[];grouped={}
    for x in maps:grouped.setdefault((x['role'],x['draw_id']),[]).append(x)
    for role in ROLES:
        a=sorted([x for x in data if x['role']==role],key=lambda x:x['row_id']);ids=[x['row_id'] for x in a];index={sid:i for i,sid in enumerate(ids)};X=matrix(a,t);y=np.array([int(x['y']) for x in a]);frame=logistic(X@beta)
        for method,ps in [('frame_only',frame),('training_prevalence',np.full(len(a),constant))]:
            predictions.extend({'role':role,'method_id':method,'row_id':x['row_id'],'college_id':x['college_id'],'probability':float(v)} for x,v in zip(a,ps));scores.append({'role':role,'method_id':method,**score(y,ps)})
        base=float(np.mean((frame-y)**2));deltas=[]
        for draw in range(1,65):
            group=grouped.get((role,str(draw)),[])
            if len(group)!=len(a) or {x['row_id'] for x in group}!=set(ids) or {x['donor_id'] for x in group}!=set(ids):raise ValueError('permutation is not a complete bijection')
            donor={x['row_id']:x['donor_id'] for x in group}
            if any(x['college_id']!=a[index[x['row_id']]]['college_id'] or x['college_id']!=a[index[x['donor_id']]]['college_id'] for x in group):raise ValueError('permutation crossed college')
            changed=X.copy();changed[:,-1]=X[[index[donor[sid]] for sid in ids],-1]
            loss=float(np.mean((logistic(changed@beta)-y)**2));delta=loss-base;deltas.append(delta)
            draws.append({'role':role,'draw_id':draw,'n':len(a),'baseline_brier':base,'permuted_brier':loss,'difference':delta})
        d=np.asarray(deltas);q=np.quantile(d,[.025,.975],method='linear')
        summaries.append({'role':role,'draws':64,'mean_difference':float(d.mean()),'permutation_sd':float(d.std(ddof=1)),'q025':float(q[0]),'q975':float(q[1]),'min_difference':float(d.min()),'max_difference':float(d.max()),'positive_draws':int((d>0).sum())})
    if len(maps)!=64*sum(x['role'] in ROLES for x in data):raise ValueError('extra permutation maps')
    counts,contrasts=offer_tables(causal);new=read(resource/'model-specs/unit12_new_student.csv')[0];eta=float((matrix([new],t)@beta)[0])
    return {'predictions':predictions,'scores':scores,'permutation_draws':draws,'permutation_summary':summaries,'offer_counts':counts,'offer_contrasts':contrasts,'new_prediction':[{'record_id':new['record_id'],'method_id':'frame_only','eta':eta,'probability':float(logistic(np.array([eta]))[0])}],'training':[{'n':len(dev),'events':int(sum(int(x['y']) for x in dev)),'prevalence':constant}]}
def write_outputs(out,results):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,rows in results.items():
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows({k:format(v,'.17g') if isinstance(v,float) else v for k,v in x.items()} for x in rows)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();write_outputs(a.outdir,compute(a.root));print('UNIT12_NUMPY_OK frozen_replay=true numpy='+np.__version__)
