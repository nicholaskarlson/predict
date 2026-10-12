"""Independent list/math route. No NumPy, imports from other analyzers or fit."""
import argparse,csv,json,math
from pathlib import Path

TERMS=['intercept','part_time','first_gen','z_age','gender_male','gender_another','program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade']
PROGRAMS=['Arts and Humanities','Social Sciences','STEM','Health','Business']
HEADER=['population','college_id','method_id','n','events','nonevents','prevalence','mean_probability','brier','log_loss','auc_pairs2','auc_denominator','auc','auc_reason','clipped_n','mean_no_transfer_risk','mean_gap_to_no_transfer_risk','mae_to_no_transfer_risk','rmse_to_no_transfer_risk','risk_reference_reason']

def read(path):
    with path.open(newline='') as stream:return list(csv.DictReader(stream))

def predict(record,coefficients,transforms):
    z_age=(float(record['age'])-transforms['age'][0])/transforms['age'][1]
    z_grade=(float(record['prior_grade'])-transforms['prior_grade'][0])/transforms['prior_grade'][1]
    design=[1,int(record['part_time']),int(record['first_gen']),z_age,int(record['gender']=='male'),int(record['gender']=='another')]
    design.extend(int(record['program_area']==p) for p in PROGRAMS[1:]);design.append(z_grade)
    eta=math.fsum(a*b for a,b in zip(design,coefficients));e=math.exp(-abs(eta))
    return 1/(1+e) if eta>=0 else e/(1+e)

def validate(records,plan,colleges):
    if len(records)!=plan['expected_rows']:raise ValueError('sealed row count')
    seen=set()
    for record in records:
        if list(record)!=plan['raw_header'] or not record['student_id'].startswith('SD') or record['student_id'] in seen:raise ValueError('sealed schema or duplicate ID')
        seen.add(record['student_id'])
        if record['college_id'] not in colleges or record['region']!=colleges[record['college_id']]:raise ValueError('college membership')
        if record['gender'] not in ['female','male','another'] or record['program_area'] not in PROGRAMS:raise ValueError('unknown category')
        if any(record[k] not in ['0','1'] for k in ['part_time','first_gen']):raise ValueError('frame binary')
        for k,lo,hi in [('age',16,80),('prior_grade',0,4),('risk12_no_transfer',0,1),('exit_month',0,12)]:
            v=float(record[k])
            if not math.isfinite(v) or not lo<=v<=hi:raise ValueError('numeric range '+k)
        exit=record['exit_type']
        if exit not in ['withdrawal','transfer','none'] or record['withdrawn_12']!=str(int(exit=='withdrawal')) or record['transferred_12']!=str(int(exit=='transfer')):raise ValueError('outcome coherence or censoring')

def auc(probabilities,outcomes):
    positive=sum(outcomes);negative=len(outcomes)-positive;denominator=2*positive*negative
    if not denominator:return 0,0,'','empty' if not outcomes else 'one_class'
    ordered=sorted(zip(probabilities,outcomes));less_negative=0;pairs=0;start=0
    while start<len(ordered):
        end=start+1
        while end<len(ordered) and ordered[end][0]==ordered[start][0]:end+=1
        yes=sum(y for _,y in ordered[start:end]);no=end-start-yes
        pairs+=yes*(2*less_negative+no);less_negative+=no;start=end
    return pairs,denominator,pairs/denominator,''

def score(p,y,q):
    if len(p)!=len(y) or len(q)!=len(y) or any(not math.isfinite(v) or not 0<=v<=1 for v in p) or any(not math.isfinite(v) or not 0<=v<=1 for v in q) or any(v not in [0,1] for v in y):raise ValueError('score inputs')
    pairs,denominator,area,reason=auc(p,y);n=len(y);events=sum(y)
    if not n:return [0,0,0,'','','','',pairs,denominator,area,reason,0,'','','','','empty']
    clipped=[min(1-1e-12,max(1e-12,v)) for v in p];gaps=[a-b for a,b in zip(p,q)]
    return [n,events,n-events,events/n,math.fsum(p)/n,math.fsum((a-b)**2 for a,b in zip(p,y))/n,
            -math.fsum(b*math.log(a)+(1-b)*math.log1p(-a) for a,b in zip(clipped,y))/n,pairs,denominator,area,reason,
            sum(a!=b for a,b in zip(p,clipped)),math.fsum(q)/n,math.fsum(gaps)/n,math.fsum(abs(v) for v in gaps)/n,math.sqrt(math.fsum(v*v for v in gaps)/n),'different_target_no_transfer_reference']

def write(path,header,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(header)
        for row in rows:writer.writerow([format(v,'.17g') if isinstance(v,float) else v for v in row])

def main(resource,out):
    plan=json.loads((resource/'model-specs/sealed_envelope.json').read_text());records=read(resource/'source/sealed-envelope/deployment_c2.csv')
    colleges={r['college_id']:r['region'] for r in read(resource/'source/sealed-envelope/colleges.csv')};validate(records,plan,colleges)
    coefs=read(resource/'model-specs/unit12_coefficients.csv');trans=read(resource/'model-specs/unit12_transforms.csv')
    if [r['term'] for r in coefs]!=TERMS:raise ValueError('model terms')
    coefficients=[float(r['coefficient']) for r in coefs];transforms={r['feature']:(float(r['mean']),float(r['sd'])) for r in trans}
    fitted=[predict(r,coefficients,transforms) for r in records];constant=plan['comparator']['events']/plan['comparator']['n']
    methods={'frame_only':fitted,'development_prevalence':[constant]*len(records)}
    outcomes=[int(r['withdrawn_12']) for r in records];risk=[float(r['risk12_no_transfer']) for r in records]
    groupmap={cid:g for g,ids in plan['college_groups'].items() for cid in ids};groups=[groupmap[r['college_id']] for r in records]
    populations={g:[i for i,r in enumerate(records) if groups[i]==g] for g in plan['report_populations'] if g!='all'};populations['all']=list(range(len(records)))
    scores=[];bins=[]
    for population in plan['report_populations']:
        indices=populations[population];y=[outcomes[i] for i in indices];q=[risk[i] for i in indices]
        for method,values in methods.items():
            p=[values[i] for i in indices];scores.append([population,'all',method,*score(p,y,q)])
            for b in range(5):
                lo=b/5;hi=(b+1)/5;selected=[i for i in indices if values[i]>=lo and (values[i]<=hi if b==4 else values[i]<hi)]
                n=len(selected)
                bins.append([population,method,b+1,lo,hi,n,sum(outcomes[i] for i in selected),math.fsum(values[i] for i in selected)/n if n else '',math.fsum(outcomes[i] for i in selected)/n if n else '',math.fsum(risk[i] for i in selected)/n if n else '', '' if n else 'empty_bin'])
    for cid in sorted(colleges):
        indices=[i for i,r in enumerate(records) if r['college_id']==cid];y=[outcomes[i] for i in indices];q=[risk[i] for i in indices]
        for method,values in methods.items():scores.append([groupmap[cid],cid,method,*score([values[i] for i in indices],y,q)])
    prior=read(resource/'expected-results/unit11_scores.csv');comparisons=[]
    for group,role in [('continuing','temporal_continuing'),('previously_new','temporal_new')]:
        previous=next(r for r in prior if r['method_id']=='frame_only' and r['pattern']=='all' and r['role']==role)
        current=next(r for r in scores if r[:3]==[group,'all','frame_only'])
        comparisons.append([group,role,int(previous['n_scored']),float(previous['brier']),current[3],current[8],current[8]-float(previous['brier']),'descriptive_different_recruitment_populations'])
    write(out/'sealed_predictions.csv',['student_id','college_id','population','frame_probability','comparator_probability'],[[r['student_id'],r['college_id'],groups[i],fitted[i],constant] for i,r in enumerate(records)])
    write(out/'sealed_scores.csv',HEADER,scores)
    write(out/'sealed_bins.csv',['population','method_id','bin','lower','upper','n','events','mean_probability','observed_fraction','mean_no_transfer_risk','reason'],bins)
    write(out/'sealed_comparisons.csv',['population','prior_role','prior_n','prior_brier','sealed_n','sealed_brier','difference','reason'],comparisons)
    print('SEALED_STDLIB_OK predictions='+str(len(records))+' no_refit=true World_draws=0')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resource',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();main(a.resource,a.out)
