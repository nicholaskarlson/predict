"""Independent scalar replay, direct definitions and tied-score pair counts."""
import bisect,csv,json,math
from pathlib import Path

ROLES=['temporal_continuing','temporal_new']
TERMS=['intercept','part_time','first_gen','z_age','gender_male','gender_another','program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade']
def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def validate_frame(x):
    if x['gender'] not in ['female','male','another'] or x['program_area'] not in ['Arts and Humanities','Social Sciences','STEM','Health','Business']:raise ValueError('frame category')
    if str(x['part_time']) not in ['0','1'] or str(x['first_gen']) not in ['0','1']:raise ValueError('binary frame')
    if not math.isfinite(float(x['age'])) or not 16<=float(x['age'])<=80 or not math.isfinite(float(x['prior_grade'])) or not 0<=float(x['prior_grade'])<=4:raise ValueError('frame range')
def design(x,transforms):
    validate_frame(x)
    return [1.,float(x['part_time']),float(x['first_gen']),(float(x['age'])-transforms['age'][0])/transforms['age'][1],float(x['gender']=='male'),float(x['gender']=='another'),float(x['program_area']=='Social Sciences'),float(x['program_area']=='STEM'),float(x['program_area']=='Health'),float(x['program_area']=='Business'),(float(x['prior_grade'])-transforms['prior_grade'][0])/transforms['prior_grade'][1]]
def probability(eta):
    return 1/(1+math.exp(-eta)) if eta>=0 else math.exp(eta)/(1+math.exp(eta))
def score(y,p):
    n=len(y);positives=[v for k,v in zip(y,p) if k==1];negatives=sorted(v for k,v in zip(y,p) if k==0)
    if not n or len(p)!=n:raise ValueError('empty or mismatched score population')
    if any(k not in [0,1] for k in y) or any(not math.isfinite(v) or not 0<=v<=1 for v in p):raise ValueError('score input')
    pairs=sum(bisect.bisect_left(negatives,v)+bisect.bisect_right(negatives,v) for v in positives)
    den=2*len(positives)*len(negatives)
    return {'n':n,'events':sum(y),'nonevents':n-sum(y),'brier':math.fsum((v-k)**2 for k,v in zip(y,p))/n,'mean_probability':math.fsum(p)/n,'auc_pairs2':pairs,'auc_denominator':den,'auc':pairs/den if den else '', 'auc_reason':'' if den else 'one_class'}
def quantile(a,q):
    s=sorted(a);h=(len(s)-1)*q;i=int(h);return s[i]+(h-i)*(s[min(i+1,len(s)-1)]-s[i])
def offer_tables(causal):
    if len({x['row_id'] for x in causal})!=len(causal):raise ValueError('duplicate randomized ID')
    result=[];total=len(causal)
    for c in sorted({x['college_id'] for x in causal})+['pooled']:
        a=[x for x in causal if c=='pooled' or x['college_id']==c];arms={}
        for z in [0,1]:
            g=[x for x in a if x['offer']==str(z)]
            if not g or any(x['y'] not in ['','0','1'] or x['outcome_known']!=str(int(x['y']!='')) for x in g):raise ValueError('assigned group/unknown outcomes')
            known=sum(x['y']!='' for x in g);event=sum(x['y']=='1' for x in g)
            if not known:raise ValueError('empty known-outcome arm')
            arms[z]=(len(g),known,len(g)-known,event,event/known,event/len(g),(event+len(g)-known)/len(g))
        if any(x['offer'] not in ['0','1'] for x in a) or (c!='pooled' and abs(arms[1][0]-arms[0][0])>1):raise ValueError('assignment mechanism')
        rec={'college_id':c,'randomized_n':len(a),'weight':len(a)/total if c!='pooled' else 1}
        for z in [0,1]:
            rec.update({f'{key}{z}':val for key,val in zip(['assigned','known','unknown','events','risk_known','risk_lower','risk_upper'],arms[z])})
        rec.update(rd_known=arms[1][4]-arms[0][4],rd_lower=arms[1][5]-arms[0][6],rd_upper=arms[1][6]-arms[0][5]);result.append(rec)
    colleges=result[:-1];summary=[]
    for name in ['standardized','pooled']:
        groups=colleges if name=='standardized' else [result[-1]]
        summary.append({'contrast_id':name,'randomized_n':total,'known_n':sum(x['y']!='' for x in causal),'unknown_n':sum(x['y']=='' for x in causal),'rd_known':math.fsum(x['weight']*x['rd_known'] for x in groups),'rd_lower':math.fsum(x['weight']*x['rd_lower'] for x in groups),'rd_upper':math.fsum(x['weight']*x['rd_upper'] for x in groups)})
    return result,summary
def compute(resource):
    resource=Path(resource);data=read(resource/'data/unit12_rows.csv');causal=read(resource/'data/unit12_randomized.csv');maps=read(resource/'splits/unit12_permutations.csv')
    coefficients=read(resource/'model-specs/unit12_coefficients.csv')
    if [x['term'] for x in coefficients]!=TERMS:raise ValueError('coefficient term order')
    beta=[float(x['coefficient']) for x in coefficients];transforms={x['feature']:(float(x['mean']),float(x['sd'])) for x in read(resource/'model-specs/unit12_transforms.csv')}
    if not all(math.isfinite(v) for v in beta) or set(transforms)!={'age','prior_grade'} or any(not all(math.isfinite(k) for k in v) or v[1]<=0 for v in transforms.values()):raise ValueError('finite coefficients/transforms')
    if len({x['row_id'] for x in data})!=len(data):raise ValueError('duplicate predictive ID')
    if any(x['role'] not in ['development',*ROLES] or x['y'] not in ['0','1'] for x in data):raise ValueError('predictive roles/labels')
    development=[x for x in data if x['role']=='development'];constant=sum(int(x['y']) for x in development)/len(development)
    frame={x['row_id']:probability(math.fsum(b*v for b,v in zip(beta,design(x,transforms)))) for x in data if x['role'] in ROLES}
    predictions=[];scores=[];draws=[];summaries=[]
    for role in ROLES:
        a=sorted((x for x in data if x['role']==role),key=lambda x:x['row_id']);byid={x['row_id']:x for x in a};y=[int(x['y']) for x in a]
        for method in ['frame_only','training_prevalence']:
            ps=[frame[x['row_id']] if method=='frame_only' else constant for x in a]
            predictions.extend({'role':role,'method_id':method,'row_id':x['row_id'],'college_id':x['college_id'],'probability':v} for x,v in zip(a,ps))
            scores.append({'role':role,'method_id':method,**score(y,ps)})
        base=score(y,[frame[x['row_id']] for x in a])['brier'];differences=[]
        for draw in range(1,65):
            mapping=[x for x in maps if x['role']==role and x['draw_id']==str(draw)]
            if len(mapping)!=len(a) or {x['row_id'] for x in mapping}!=set(byid) or {x['donor_id'] for x in mapping}!=set(byid):raise ValueError('permutation not a bijection')
            donor={x['row_id']:x['donor_id'] for x in mapping};ps=[]
            for m in mapping:
                if m['college_id']!=byid[m['row_id']]['college_id'] or byid[m['donor_id']]['college_id']!=m['college_id']:raise ValueError('cross-college permutation')
            for x in a:
                altered=dict(x,prior_grade=byid[donor[x['row_id']]]['prior_grade'])
                ps.append(probability(math.fsum(b*v for b,v in zip(beta,design(altered,transforms)))))
            loss=math.fsum((v-k)**2 for v,k in zip(ps,y))/len(a);delta=loss-base;differences.append(delta)
            draws.append({'role':role,'draw_id':draw,'n':len(a),'baseline_brier':base,'permuted_brier':loss,'difference':delta})
        mean=math.fsum(differences)/64;sd=math.sqrt(math.fsum((d-mean)**2 for d in differences)/63)
        summaries.append({'role':role,'draws':64,'mean_difference':mean,'permutation_sd':sd,'q025':quantile(differences,.025),'q975':quantile(differences,.975),'min_difference':min(differences),'max_difference':max(differences),'positive_draws':sum(d>0 for d in differences)})
    if len(maps)!=64*len(frame):raise ValueError('extra permutation maps')
    causal_counts,contrasts=offer_tables(causal);profiles=read(resource/'model-specs/unit12_new_student.csv')
    if len(profiles)!=1:raise ValueError('one handoff record required')
    new=profiles[0]
    eta=math.fsum(b*v for b,v in zip(beta,design(new,transforms)))
    profile=[{'record_id':new['record_id'],'method_id':'frame_only','eta':eta,'probability':probability(eta)}]
    return {'predictions':predictions,'scores':scores,'permutation_draws':draws,'permutation_summary':summaries,'offer_counts':causal_counts,'offer_contrasts':contrasts,'new_prediction':profile,'training':[{'n':len(development),'events':sum(int(x['y']) for x in development),'prevalence':constant}]}
def write_outputs(out,results):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    for name,a in results.items():
        with (out/(name+'.csv')).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(a[0]),lineterminator='\n');w.writeheader()
            w.writerows({k:format(v,'.17g') if isinstance(v,float) else v for k,v in x.items()} for x in a)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();write_outputs(a.outdir,compute(a.root));print('UNIT12_STDLIB_OK frozen_replay=true no_world_draws=true')
