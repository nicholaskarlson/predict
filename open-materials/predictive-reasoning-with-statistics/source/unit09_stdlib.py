#!/usr/bin/env python3
"""Independent Unit 9 stdlib fits, frozen-model application and scoring."""
import argparse,csv,json,math
from pathlib import Path
from fractions import Fraction
RESOURCE=Path(__file__).resolve().parents[1]
def ensure(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def keyed(path):
    records = read(path)
    keys = [record['student_id'] for record in records]
    ensure(len(keys) == len(set(keys)), 'duplicate raw key ' + path.name)
    return dict(zip(keys, records))


def item_mean(record, stem):
    scored = []
    for number in range(1, 6):
        item = record[stem + str(number)]
        if item:
            ensure(item in ('1', '2', '3', '4', '5'), 'invalid ordinal item')
            value = int(item)
            scored.append(6-value if number == 4 else value)
    return (float(Fraction(sum(scored), len(scored))) if len(scored) >= 3 else None, len(scored))


def load_records(root):
    records = []
    for population in ['anchor', 'cohort2']:
        directory = root/'source/tonmick-v1.0.0/data'/population
        participants = keyed(directory/'participants.csv')
        interviews = keyed(directory/'baseline.csv')
        outcomes = keyed(directory/'registry.csv')
        included = {key for key, person in participants.items() if person['responded_baseline'] == '1'}
        ensure(included == set(interviews) and included == set(outcomes), 'respondent membership')
        for key in sorted(included):
            person, interview, observation = participants[key], interviews[key], outcomes[key]
            ensure(person['college_id'] == interview['college_id'], 'college mismatch')
            if population == 'anchor' and interview['offer'] != '0':
                continue
            wb, wb_n = item_mean(interview, 'wb')
            st, st_n = item_mean(interview, 'st')
            exit_type = observation['exit_type']
            ensure(exit_type in ['none', 'transfer', 'withdrawal', 'consent_withdrawn'], 'invalid exit type')
            target = None if exit_type == 'consent_withdrawn' else int(exit_type == 'withdrawal')
            ensure(target is None or str(target) == observation['withdrawn_12'], 'outcome mismatch')
            role = ('excluded_unknown' if target is None else 'excluded_unscorable_items'
                    if wb is None or st is None else 'development' if population == 'anchor' else 'temporal_final')
            ensure(person['part_time'] in ('0', '1') and person['first_gen'] in ('0', '1'), 'binary coding')
            ensure(person['gender'] in ('female', 'male', 'another'), 'gender coding')
            ensure(person['program_area'] in ('Arts and Humanities', 'Social Sciences', 'STEM', 'Health', 'Business'), 'program coding')
            age = float(person['age']); grade = float(person['prior_grade'])
            ensure(math.isfinite(age) and 16 <= age <= 80 and math.isfinite(grade) and 0 <= grade <= 4, 'invalid continuous predictor')
            offer = int(interview['offer']) if population == 'anchor' else 0
            ensure(offer in (0, 1), 'offer coding')
            records.append(dict(row_id=key, cohort=population, college_id=person['college_id'], region=person['region'],
                part_time=int(person['part_time']), first_gen=int(person['first_gen']), age=age,
                gender=person['gender'], program_area=person['program_area'], prior_grade=grade,
                offer=offer, wellbeing_score=wb, stress_score=st, wellbeing_observed=wb_n, stress_observed=st_n,
                y=target, role=role, exit_type=exit_type, college_group='anchor' if population=='anchor' else person['college_group'], grade=float(observation['gpa12']) if observation['gpa12'] else None, gpa_at_exit=int(observation['gpa_at_exit'])))
    records.sort(key=lambda r: r['row_id'])
    ensure(len(records) == len({r['row_id'] for r in records}), 'overlapping cohort IDs')
    return records

def invlogit(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def dot(a, b):
    return math.fsum(x*y for x,y in zip(a,b))


def backsolve(r, rhs):
    answer = [0.] * len(rhs)
    for j in reversed(range(len(rhs))):
        ensure(r[j][j] != 0, 'rank deficiency')
        answer[j] = (rhs[j] - math.fsum(r[j][k]*answer[k] for k in range(j+1,len(rhs))))/r[j][j]
    return answer


def householder_step(design, residual, weights, rules):
    ensure(all(math.isfinite(w) and w > 0 for w in weights), 'separation or nonpositive weight')
    n, p = len(design), len(design[0])
    a = [[v*math.sqrt(w) for v in row] for row,w in zip(design,weights)]
    rhs = [e/math.sqrt(w) for e,w in zip(residual,weights)]
    for column in range(p):
        vector = [a[i][column] for i in range(column,n)]
        length = math.sqrt(dot(vector,vector))
        ensure(length > 0, 'rank deficiency')
        vector[0] += math.copysign(length,vector[0])
        squared_length = dot(vector,vector)
        for j in range(column,p):
            scale = 2*math.fsum(v*a[i][j] for v,i in zip(vector,range(column,n)))/squared_length
            for v,i in zip(vector,range(column,n)):
                a[i][j] -= scale*v
        scale = 2*math.fsum(v*rhs[i] for v,i in zip(vector,range(column,n)))/squared_length
        for v,i in zip(vector,range(column,n)):
            rhs[i] -= scale*v
    triangle = [[a[i][j] if j >= i else 0. for j in range(p)] for i in range(p)]
    diagonal = [abs(triangle[j][j]) for j in range(p)]
    ensure(min(diagonal) > rules['rank_relative_tolerance']*max(diagonal), 'rank deficiency')
    inverse_columns = [backsolve(triangle,[float(i==j) for i in range(p)]) for j in range(p)]
    condition = max(math.fsum(abs(v) for v in row) for row in triangle) * max(
        math.fsum(abs(column[i]) for column in inverse_columns) for i in range(p))
    ensure(math.isfinite(condition) and condition <= rules['condition_r_inf_limit'], 'conditioning failure')
    return backsolve(triangle,rhs[:p]), condition


def logistic_fit(design, target, specification, offset=None):
    n, p = len(design), len(design[0])
    offset = [0.]*n if offset is None else list(offset)
    ensure(n > p and len(target) == n and all(len(row) == p for row in design), 'fit dimensions')
    ensure(set(target) == {0,1} and all(math.isfinite(v) for row in design for v in row), 'fit needs finite predictors and both classes')
    rules = specification['solver']
    beta = [math.log(sum(target)/(n-sum(target)))] + [0.]*(p-1)
    def objective(coefficients):
        loss = []
        for row,y,o in zip(design,target,offset):
            eta = dot(row,coefficients)+o
            sign_eta = -eta if y else eta
            loss.append(max(0.,sign_eta) + math.log1p(math.exp(-abs(sign_eta))))
        return math.fsum(loss)
    for iteration in range(rules['maximum_iterations']+1):
        ensure(max(abs(v) for v in beta) <= rules['maximum_abs_coefficient'], 'separation or coefficient boundary')
        predicted = [invlogit(dot(row,beta)+o) for row,o in zip(design,offset)]
        residual = [y-p for y,p in zip(target,predicted)]
        weights = [p*(1-p) for p in predicted]
        step, condition = householder_step(design,residual,weights,rules)
        gradient = max(abs(math.fsum(row[j]*e for row,e in zip(design,residual))) for j in range(p))
        step_size = max(abs(v) for v in step)
        if gradient <= rules['gradient_inf_tolerance'] and step_size <= rules['newton_step_inf_tolerance']:
            return beta, dict(n_fit=n,events=sum(target),p=p,rank=p,nll=objective(beta),score_inf=gradient,
                              newton_step_inf=step_size,condition_r_inf=condition,min_weight=min(weights),finite=1)
        current = objective(beta); fraction = 1.
        while fraction >= rules['minimum_step']:
            trial = [b+fraction*s for b,s in zip(beta,step)]
            if objective(trial) <= current:
                break
            fraction *= rules['backtracking_factor']
        ensure(fraction >= rules['minimum_step'], 'backtracking failed')
        beta = trial
    raise ValueError('separation or logistic nonconvergence')



def ols(x,y,spec):return householder_step(x,y,[1.]*len(y),spec['solver'])
def fit_cal(eta,y,kind,spec):
 design=[[1.] if kind=='offset' else [1.,v] for v in eta];return logistic_fit(design,y,spec,eta if kind=='offset' else None)
def average(a):return math.fsum(a)/len(a)
def frozen_predictions(rows,model):
 processing={q['feature']:(float(q['mean']),float(q['sd'])) for q in model['transforms']};coefs=[float(q['coefficient']) for q in model['coefficients']];p=[]
 for row in rows:
  z={name:(row[name]-mean)/scale for name,(mean,scale) in processing.items()}
  x=[1.,row['part_time'],row['first_gen'],z['age'],float(row['gender']=='male'),float(row['gender']=='another')]
  x.extend(float(row['program_area']==v) for v in ['Social Sciences','STEM','Health','Business']);x.extend([z['prior_grade'],z['wellbeing_score'],z['stress_score'],0.]);p.append(invlogit(dot(x,coefs)))
 return p
def losses(y,p):
 bounded=[min(1-1e-12,max(1e-12,v)) for v in p]
 return dict(brier=math.fsum((v-t)**2 for v,t in zip(p,y))/len(y),log_loss=-math.fsum(math.log(v) if t else math.log1p(-v) for t,v in zip(y,bounded))/len(y))

def read_csv(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def logit(p):
 ensure(math.isfinite(p) and 0<=p<=1,'invalid probability');q=min(1-1e-12,max(1e-12,p));return math.log(q)-math.log1p(-q)
def diagnose(eta,y,kind,spec):
 if len(set(y))<2:return None,'one_class'
 if kind=='joint' and max(eta)-min(eta)<=1e-12:return None,'constant_prediction'
 try:b,d=fit_cal(eta,y,kind,spec);return (b,d),''
 except ValueError as e:return None,'fit_failure:'+str(e)

def scale(a):
 mean=math.fsum(a)/len(a);return mean,math.sqrt(math.fsum((v-mean)**2 for v in a)/len(a))
def new_predict(x,b):return [invlogit(dot(row,b)) for row in x]
def auc_score(y,p):
 positive=sum(y);negative=len(y)-positive
 if not positive or not negative:return '', 'one_class'
 cells={}
 for label,value in zip(y,p):
  cells.setdefault(value,[0,0]);cells[value][label]+=1
 twice=0;neg=0
 for value,(n0,n1) in sorted(cells.items()):twice+=n1*(2*neg+n0);neg+=n0
 return float(Fraction(twice,2*positive*negative)),''

TERMS=['intercept','part_time','first_gen','z_age','gender_male','gender_another','program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade','z_wellbeing_score','z_stress_score']
FEATURES=['age','prior_grade','wellbeing_score','stress_score']
def new_design(rows,transforms):
 out=[]
 for row in rows:
  z={k:(row[k]-m)/sd for k,(m,sd) in transforms.items()}
  out.append([1.,row['part_time'],row['first_gen'],z['age'],int(row['gender']=='male'),int(row['gender']=='another'),*[int(row['program_area']==v) for v in ['Social Sciences','STEM','Health','Business']],z['prior_grade'],z['wellbeing_score'],z['stress_score']])
 return out
def type7(values,q):
 ensure(values and 0<=q<=1,'quantile input');a=sorted(values);h=(len(a)-1)*q;j=int(h);return a[j]+(h-j)*(a[min(j+1,len(a)-1)]-a[j])
def score(y,p,spec):
 ensure(y and len(y)==len(p) and all(t in [0,1] for t in y),'binary scoring labels')
 ensure(all(math.isfinite(v) and 0<=v<=1 for v in p),'probability range');e=[logit(v) for v in p];cal={}
 for kind in ['offset','joint']:cal[kind]=diagnose(e,y,kind,spec)
 auc,reason=auc_score(y,p)
 return dict(n=len(y),events=sum(y),auc=auc,auc_reason=reason,observed_fraction=average(y),mean_probability=average(p),**losses(y,p),clipped_n=sum(v<1e-12 or v>1-1e-12 for v in p),offset_intercept=cal['offset'][0][0][0] if cal['offset'][0] else '',offset_reason=cal['offset'][1],joint_intercept=cal['joint'][0][0][0] if cal['joint'][0] else '',joint_slope=cal['joint'][0][0][1] if cal['joint'][0] else '',joint_reason=cal['joint'][1]),cal
def composition(draw,sites,method,base):
 n=sum(sites[k][method]['n'] for k in draw)
 sums={m:math.fsum([sites[k][method][m]*sites[k][method]['n'] for k in draw])/n for m in ['brier','log_loss']}
 original={m:math.fsum([sites[k][base][m]*sites[k][base]['n'] for k in draw])/n for m in ['brier','log_loss']}
 return dict(n=n,**sums,delta_brier=sums['brier']-original['brier'],delta_log_loss=sums['log_loss']-original['log_loss'])
def calculate(root=RESOURCE):
 spec=json.loads((root/'model-specs/unit09.json').read_text());ensure(spec['predictor_allowlist']==['part_time','first_gen','age','gender','program_area','prior_grade','wellbeing_score','stress_score'],'predictor allowlist')
 ensure(spec['methods']==['unchanged','offset','joint','refit'] and spec['selection_between_methods'] is False,'fixed comparison')
 rows=load_records(root);manifest=read_csv(root/'splits/unit09_roles.csv');ensure(len(manifest)==len(rows) and len({x['row_id'] for x in manifest})==len(rows),'manifest uniqueness and coverage')
 for a,x in zip(manifest,rows):
  ensure(all(a[k]==x[k] for k in ['row_id','cohort','college_id']),'manifest keys')
  allowed=['anchor_reference'] if x['college_group']=='anchor' else ['new_updating','new_final'] if x['college_group']=='new' else ['continuing_final']
  ensure(a['assignment'] in allowed,'invalid population role');x['assignment']=a['assignment']
 old_roles=read_csv(root/'splits/unit08_roles.csv');ensure(len(old_roles)==len(manifest) and all(a['row_id']==b['row_id'] and b['assignment']==('anchor_reference' if a['cohort']=='anchor' else a['assignment']) for a,b in zip(old_roles,manifest)),'reused role assignment altered')
 risk=[x for x in rows if x['cohort']=='cohort2' and x['y'] is not None and x['wellbeing_score'] is not None and x['stress_score'] is not None];model=json.loads((root/'model-specs/unit09_frozen_risk.json').read_text())
 for x,p in zip(risk,frozen_predictions(risk,model)):x['p']=p;x['eta']=logit(p)
 update=[x for x in risk if x['assignment']=='new_updating'];final=[x for x in risk if x['assignment']=='new_final'];sites=sorted({x['college_id'] for x in final});ensure(len(sites)==4,'four sites')
 result={k:[] for k in ['predictions','scores','coefficients','diagnostics','fit_rows','transforms','resampling','resample_summary','profiles','summary']}
 def register(id,scope,members,terms,fit):
  b,d=fit;result['coefficients'].extend(dict(fit_id=id,scope=scope,term=t,coefficient=v) for t,v in zip(terms,b));result['diagnostics'].append(dict(fit_id=id,scope=scope,**d));result['fit_rows'].extend(dict(fit_id=id,scope=scope,row_id=x['row_id']) for x in members)
 fits={}
 for kind in ['offset','joint']:
  fit=fit_cal([x['eta'] for x in update],[x['y'] for x in update],kind,spec);fits[kind]=fit[0];register('update_'+kind,'updating_rule',update,['intercept'] if kind=='offset' else ['intercept','slope'],fit)
 transforms={k:scale([x[k] for x in update]) for k in FEATURES}
 for k,(mean,sd) in transforms.items():ensure(math.isfinite(sd) and sd>0,'zero SD');result['transforms'].append(dict(fit_id='update_refit',feature=k,n=len(update),mean=mean,sd=sd))
 b,d=logistic_fit(list(new_design(update,transforms)),list([x['y'] for x in update]),spec);fits['refit']=list(b);register('update_refit','updating_rule',update,TERMS,(b,d))
 newp=new_predict(new_design(final,transforms),b);site_scores={k:{} for k in sites}
 for role in ['continuing_final','new_final']:
  test=[x for x in risk if x['assignment']==role];methods=['unchanged'] if role=='continuing_final' else spec['methods']
  for method in methods:
   p=[x['p'] for x in test] if method=='unchanged' else list(newp) if method=='refit' else [invlogit(fits[method][0]+(1 if method=='offset' else fits[method][1])*x['eta']) for x in test]
   result['predictions'].extend(dict(role=role,method=method,college_id=x['college_id'],row_id=x['row_id'],y=x['y'],probability=v) for x,v in zip(test,p))
   for site in ['all']+(sites if role=='new_final' else []):
    ids=[i for i,x in enumerate(test) if site=='all' or x['college_id']==site];sub=[test[i] for i in ids];s,cal=score([x['y'] for x in sub],[p[i] for i in ids],spec)
    result['scores'].append(dict(role=role,method=method,college_id=site,**s))
    if site!='all':site_scores[site][method]=s
    for kind in ['offset','joint']:
     if cal[kind][0]:register('diagnostic_'+role+'_'+method+'_'+site+'_'+kind,'diagnostic_only',sub,['intercept'] if kind=='offset' else ['intercept','slope'],cal[kind][0])
 # Equal-college loss answers a different empirical question from pooled row loss.
 for method in spec['methods']:
  for metric in ['brier','log_loss']:result['summary'].append(dict(kind='equal_college_loss',role='new_final',method=method,metric=metric,value=average([site_scores[k][method][metric] for k in sites])))
 draws=read_csv(root/'splits/unit09_college_draws.csv');ensure(len(draws)==1024,'draw inventory')
 import itertools
 expected=[dict(draw_id=str(i),position=str(j),college_id=k) for i,d in enumerate(itertools.product(sites,repeat=4),1) for j,k in enumerate(d,1)];ensure(draws==expected,'whole-college committed enumeration')
 for i in range(256):
  draw=[x['college_id'] for x in draws[4*i:4*i+4]]
  for method in spec['methods']:result['resampling'].append(dict(draw_id=i+1,method=method,**composition(draw,site_scores,method,'unchanged')))
 for method in spec['methods']:
  for metric in ['brier','log_loss','delta_brier','delta_log_loss']:
   values=[x[metric] for x in result['resampling'] if x['method']==method];result['resample_summary'].append(dict(method=method,metric=metric,draws=256,mean=average(values),minimum=min(values),q025=type7(values,.025),median=type7(values,.5),q975=type7(values,.975),maximum=max(values),positive_draws=sum(v>0 for v in values)))
 # Descriptive observed-frame composition, never nonresponse-adjusted target weights.
 for cohort in ['anchor','cohort2']:
  frame=read_csv(root/'source/tonmick-v1.0.0/data'/cohort/'participants.csv')
  for group in (['anchor'] if cohort=='anchor' else ['continuing','new']):
   pool=[x for x in frame if cohort=='anchor' or x['college_group']==group]
   for response in ['invited','responded']:
    a=[x for x in pool if response=='invited' or x['responded_baseline']=='1']
    result['profiles'].append(dict(cohort=cohort,college_group=group,frame_role=response,n=len(a),colleges=len({x['college_id'] for x in a}),part_time_fraction=average([int(x['part_time']) for x in a]),first_gen_fraction=average([int(x['first_gen']) for x in a]),prior_grade_mean=average([float(x['prior_grade']) for x in a])))
 for role in ['anchor_reference','continuing_final','new_updating','new_final']:
  a=[x for x in rows if x['assignment']==role];eligible=[x for x in a if x['y'] is not None and x['wellbeing_score'] is not None and x['stress_score'] is not None]
  for metric,val in [('pool_n',len(a)),('known_scorable_n',len(eligible)),('events',sum(x['y'] for x in eligible)),('unknown_n',sum(x['y'] is None for x in a)),('colleges',len({x['college_id'] for x in a}))]:result['summary'].append(dict(kind='eligibility',role=role,method='',metric=metric,value=val))
 return result
def emit(result,out):
 out.mkdir(parents=True,exist_ok=True)
 for key,rows in result.items():
  with (out/(key+'.csv')).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows([{k:format(float(v),'.17g') if isinstance(v,(float,float)) else v for k,v in x.items()} for x in rows])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();res=calculate(a.root);emit(res,a.outdir);print('UNIT09_ROUTE_OK predictions='+str(len(res['predictions']))+' college_draws=256')
