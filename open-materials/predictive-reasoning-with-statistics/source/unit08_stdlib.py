#!/usr/bin/env python3
"""Independent Unit 8 stdlib fits, frozen-model application and scoring."""
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
def conformal_quantile(scores,numerator=1,denominator=10):
 ensure(scores and all(math.isfinite(v) and v>=0 for v in scores),'invalid residuals')
 ensure(0<numerator<denominator,'invalid alpha')
 n=len(scores);k=((n+1)*(denominator-numerator)+denominator-1)//denominator
 return k,(sorted(scores)[k-1] if k<=n else math.inf)
def logit(p):
 ensure(math.isfinite(p) and 0<=p<=1,'invalid probability');q=min(1-1e-12,max(1e-12,p));return math.log(q)-math.log1p(-q)
def diagnose(eta,y,kind,spec):
 if len(set(y))<2:return None,'one_class'
 if kind=='joint' and max(eta)-min(eta)<=1e-12:return None,'constant_prediction'
 try:b,d=fit_cal(eta,y,kind,spec);return (b,d),''
 except ValueError as e:return None,'fit_failure:'+str(e)
def calculate(root=RESOURCE):
 spec=json.loads((root/'model-specs/unit08.json').read_text());ensure(spec['predictor_allowlist']==['part_time','first_gen','age','gender','program_area','prior_grade','wellbeing_score','stress_score'],'predictor allowlist')
 ensure(spec['grade_model']=='OLS_intercept_prior_grade_only' and spec['alpha_numerator']==1 and spec['alpha_denominator']==10,'frozen grade design')
 rows=load_records(root);manifest=read_csv(root/'splits/unit08_roles.csv');ensure(len(manifest)==len(rows) and len({x['row_id'] for x in manifest})==len(rows),'manifest uniqueness and coverage')
 for a,x in zip(manifest,rows):
  ensure(a['row_id']==x['row_id'] and a['cohort']==x['cohort'] and a['college_id']==x['college_id'],'manifest keys')
  allowed=['grade_fit','grade_calibration'] if x['college_group']=='anchor' else ['new_updating','new_final'] if x['college_group']=='new' else ['continuing_final']
  ensure(a['assignment'] in allowed,'invalid population role');x['assignment']=a['assignment']
  ensure(x['grade'] is None or math.isfinite(x['grade']) and 0<=x['grade']<=4 and x['exit_type']!='consent_withdrawn','invalid grade')
 risk=[x for x in rows if x['cohort']=='cohort2' and x['y'] is not None and x['wellbeing_score'] is not None and x['stress_score'] is not None];model=json.loads((root/'model-specs/unit08_frozen_risk.json').read_text())
 probabilities=frozen_predictions(risk,model)
 for x,p in zip(risk,probabilities):x['p']=p;x['eta']=logit(p)
 update=[x for x in risk if x['assignment']=='new_updating'];result={k:[] for k in ['predictions','binary_scores','bins','coefficients','diagnostics','fit_rows','residuals','intervals','coverage','selection','summary']}
 def register(fit_id,scope,membership,kind,fit):
  b,d=fit;result['coefficients'].extend(dict(fit_id=fit_id,scope=scope,term='intercept' if j==0 else 'slope',coefficient=v) for j,v in enumerate(b));result['diagnostics'].append(dict(fit_id=fit_id,scope=scope,**d));result['fit_rows'].extend(dict(fit_id=fit_id,scope=scope,row_id=x['row_id']) for x in membership)
 fits={}
 for kind in ['offset','joint']:
  fit=fit_cal([x['eta'] for x in update],[x['y'] for x in update],kind,spec);fits[kind]=fit[0];register('update_'+kind,'updating_rule',update,kind,fit)
 for role in ['continuing_final','new_final']:
  test=[x for x in risk if x['assignment']==role];y=[x['y'] for x in test]
  for method in (['unchanged'] if role=='continuing_final' else ['unchanged','offset','joint']):
   p=[x['p'] if method=='unchanged' else (invlogit)(fits[method][0]+(1 if method=='offset' else fits[method][1])*x['eta']) for x in test];eta=[logit(v) for v in p];cal={}
   for kind in ['offset','joint']:
    fit,reason=diagnose(eta,y,kind,spec);cal[kind]=(fit,reason)
    if fit:register('diagnostic_'+role+'_'+method+'_'+kind,'diagnostic_only',test,kind,fit)
   result['predictions'].extend(dict(role=role,method=method,row_id=x['row_id'],y=x['y'],probability=v) for x,v in zip(test,p))
   result['binary_scores'].append(dict(role=role,method=method,n=len(y),events=sum(y),observed_fraction=average(y),mean_probability=average(p),observed_minus_predicted=average(y)-average(p),**losses(y,p),offset_intercept=cal['offset'][0][0][0] if cal['offset'][0] else '',offset_reason=cal['offset'][1],joint_intercept=cal['joint'][0][0][0] if cal['joint'][0] else '',joint_slope=cal['joint'][0][0][1] if cal['joint'][0] else '',joint_reason=cal['joint'][1]))
   for j in range(5):
    ids=[i for i,v in enumerate(p) if spec['bins'][j]<=v and (v<spec['bins'][j+1] or j==4 and v<=1)];result['bins'].append(dict(role=role,method=method,bin=j+1,n=len(ids),mean_probability=average([p[i] for i in ids]) if ids else '',observed_fraction=average([y[i] for i in ids]) if ids else '',reason='' if ids else 'empty_bin'))
 fit=[x for x in rows if x['assignment']=='grade_fit' and x['grade'] is not None];cal=[x for x in rows if x['assignment']=='grade_calibration' and x['grade'] is not None]
 beta,condition=ols([[1,x['prior_grade']] for x in fit],[x['grade'] for x in fit],spec)
 result['coefficients'].extend(dict(fit_id='grade_ols',scope='grade_rule',term=term,coefficient=v) for term,v in zip(['intercept','slope'],beta));result['fit_rows'].extend(dict(fit_id='grade_ols',scope='grade_rule',row_id=x['row_id']) for x in fit);result['fit_rows'].extend(dict(fit_id='grade_quantile',scope='interval_calibration',row_id=x['row_id']) for x in cal)
 pred=lambda x:beta[0]+beta[1]*x['prior_grade'];res=[dict(row_id=x['row_id'],prediction=pred(x),y=x['grade'],absolute_residual=abs(x['grade']-pred(x))) for x in cal];res.sort(key=lambda x:(x['absolute_residual'],x['row_id']))
 result['residuals']=[dict(rank=j+1,**x) for j,x in enumerate(res)];k,q=conformal_quantile([x['absolute_residual'] for x in res]);result['selection'].append(dict(n_fit=len(fit),n_calibration=len(cal),k=k,q=q,width=2*q,condition_r_inf=condition))
 for role in ['continuing_final','new_final']:
  test=[x for x in rows if x['assignment']==role and x['grade'] is not None];records=[dict(role=role,row_id=x['row_id'],part_time=x['part_time'],y=x['grade'],prediction=pred(x),lower=pred(x)-q,upper=pred(x)+q,covered=int(pred(x)-q<=x['grade']<=pred(x)+q),width=2*q,gpa_at_exit=x['gpa_at_exit']) for x in test];result['intervals'].extend(records)
  for group in ['all','part_time_0','part_time_1']:
   a=[x for x in records if group=='all' or x['part_time']==int(group[-1])];result['coverage'].append(dict(role=role,group=group,n=len(a),covered=sum(x['covered'] for x in a),coverage=average([x['covered'] for x in a]),mean_width=average([x['width'] for x in a]),exit_grades=sum(x['gpa_at_exit'] for x in a),bounds_outside_0_4=sum(x['lower']<0 or x['upper']>4 for x in a)))
 for role in ['grade_fit','grade_calibration','continuing_final','new_updating','new_final']:
  a=[x for x in rows if x['assignment']==role];b=[x for x in risk if x['assignment']==role];result['summary'].append(dict(role=role,pool_n=len(a),binary_n=len(b),unknown_n=sum(x['y'] is None for x in a),unscorable_known_n=sum(x['y'] is not None and (x['wellbeing_score'] is None or x['stress_score'] is None) for x in a),observed_grade_n=sum(x['grade'] is not None for x in a),missing_grade_n=sum(x['grade'] is None for x in a),colleges=len({x['college_id'] for x in a})))
 return result

def emit(result,out):
 out.mkdir(parents=True,exist_ok=True)
 for key,rows in result.items():
  with (out/(key+'.csv')).open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows([{k:format(float(v),'.17g') if isinstance(v,(float,float)) else v for k,v in x.items()} for x in rows])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();res=calculate(a.root);emit(res,a.outdir);print('UNIT08_ROUTE_OK predictions='+str(len(res['predictions']))+' intervals='+str(len(res['intervals'])))
