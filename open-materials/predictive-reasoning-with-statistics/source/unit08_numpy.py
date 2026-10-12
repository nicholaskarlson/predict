#!/usr/bin/env python3
"""Independent Unit 8 numpy fits, frozen-model application and scoring."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
RESOURCE=Path(__file__).resolve().parents[1]
def require(ok, message):
    if not ok:
        raise ValueError(message)


def joined_rows(root):
    rows = []
    for cohort in ('anchor', 'cohort2'):
        folder = root / 'source/tonmick-v1.0.0/data' / cohort
        indexed = []
        for name in ('participants', 'baseline', 'registry'):
            with (folder / (name + '.csv')).open(newline='') as stream:
                records = list(csv.DictReader(stream))
            table = {r['student_id']: r for r in records}
            require(len(table) == len(records), 'duplicate raw key ' + name)
            indexed.append(table)
        frame, survey, registry = indexed
        require(set(survey) == set(registry) == {
            k for k, p in frame.items() if p['responded_baseline'] == '1'}, 'respondent join')
        for key in sorted(survey):
            p, b, outcome = frame[key], survey[key], registry[key]
            require(p['college_id'] == b['college_id'], 'college join')
            if cohort == 'anchor' and b['offer'] != '0':
                continue
            require(outcome['exit_type'] in ('none', 'withdrawal', 'transfer', 'consent_withdrawn'), 'exit type')
            known = outcome['exit_type'] != 'consent_withdrawn'
            if known:
                require(outcome['withdrawn_12'] == str(int(outcome['exit_type'] == 'withdrawal')), 'outcome exit mismatch')
            scales, observed = {}, {}
            for prefix, feature in [('wb', 'wellbeing_score'), ('st', 'stress_score')]:
                values = np.array([int(b[prefix + str(i)]) if b[prefix + str(i)] else np.nan
                                   for i in range(1, 6)], dtype=float)
                require(np.all(np.isnan(values) | ((values >= 1) & (values <= 5))), 'ordinal item range')
                require(all(b[prefix+str(i)] == '' or str(int(b[prefix+str(i)])) == b[prefix+str(i)]
                            for i in range(1, 6)), 'ordinal item type')
                values[3] = 6 - values[3]
                observed[feature] = int(np.count_nonzero(~np.isnan(values)))
                scales[feature] = float(np.nanmean(values)) if observed[feature] >= 3 else None
            role = ('excluded_unknown' if not known else 'excluded_unscorable_items'
                    if any(v is None for v in scales.values()) else 'development'
                    if cohort == 'anchor' else 'temporal_final')
            require(p['part_time'] in ('0', '1') and p['first_gen'] in ('0', '1'), 'binary predictors')
            require(p['gender'] in ('female', 'male', 'another') and p['program_area'] in (
                'Arts and Humanities', 'Social Sciences', 'STEM', 'Health', 'Business'), 'categorical predictor')
            age, grade = float(p['age']), float(p['prior_grade'])
            require(np.isfinite(age) and 16 <= age <= 80 and np.isfinite(grade) and 0 <= grade <= 4, 'continuous predictor')
            offer = int(b['offer']) if cohort == 'anchor' else 0
            require(offer in (0, 1), 'offer')
            rows.append(dict(row_id=key, cohort=cohort, college_id=p['college_id'], region=p['region'],
                             part_time=int(p['part_time']), first_gen=int(p['first_gen']), age=age,
                             gender=p['gender'], program_area=p['program_area'], prior_grade=grade,
                             offer=offer, y=int(outcome['withdrawn_12']) if known else None,
                             role=role, exit_type=outcome['exit_type'], college_group='anchor' if cohort=='anchor' else p['college_group'], grade=float(outcome['gpa12']) if outcome['gpa12'] else None, gpa_at_exit=int(outcome['gpa_at_exit']), **scales,
                             wellbeing_observed=observed['wellbeing_score'], stress_observed=observed['stress_score']))
    rows.sort(key=lambda r: r['row_id'])
    require(len({r['row_id'] for r in rows}) == len(rows), 'cohort IDs overlap')
    return rows

def sigmoid(eta):
    eta = np.asarray(eta, dtype=float)
    out = np.empty_like(eta)
    positive = eta >= 0
    out[positive] = 1 / (1 + np.exp(-eta[positive]))
    e = np.exp(eta[~positive])
    out[~positive] = e / (1 + e)
    return out


def weighted_qr(x, residual, weights, limits):
    require(np.all(np.isfinite(weights)) and np.all(weights > 0), 'separation or nonpositive weight')
    a = x * np.sqrt(weights)[:, None]
    q, r = np.linalg.qr(a, mode='reduced')
    require(np.min(np.abs(np.diag(r))) > limits['rank_relative_tolerance'] * np.max(np.abs(np.diag(r))), 'rank deficiency')
    condition = float(np.linalg.norm(r, np.inf) * np.linalg.norm(np.linalg.solve(r, np.eye(r.shape[0])), np.inf))
    require(np.isfinite(condition) and condition <= limits['condition_r_inf_limit'], 'conditioning failure')
    step = np.linalg.solve(r, q.T @ (residual / np.sqrt(weights)))
    return step, condition


def logistic_fit(x, y, spec, offset=None):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    limits = spec['solver']
    offset = np.zeros(len(y)) if offset is None else np.asarray(offset,dtype=float)
    require(x.ndim == 2 and len(y) == len(x) and len(x) > x.shape[1], 'fit dimensions')
    require(np.all(np.isfinite(x)) and set(y) == {0., 1.}, 'fit needs finite predictors and both classes')
    beta = np.zeros(x.shape[1])
    beta[0] = math.log(float(np.mean(y)) / (1 - float(np.mean(y))))
    def objective(value):
        eta = x @ value + offset
        return math.fsum(np.logaddexp(0, np.where(y == 1, -eta, eta)).tolist())
    for iteration in range(limits['maximum_iterations'] + 1):
        require(np.max(np.abs(beta)) <= limits['maximum_abs_coefficient'], 'separation or coefficient boundary')
        eta = x @ beta + offset
        probability = sigmoid(eta)
        residual, weights = y - probability, probability * (1 - probability)
        step, condition = weighted_qr(x, residual, weights, limits)
        score = float(np.max(np.abs(x.T @ residual)))
        step_inf = float(np.max(np.abs(step)))
        if score <= limits['gradient_inf_tolerance'] and step_inf <= limits['newton_step_inf_tolerance']:
            return beta, dict(n_fit=len(y), events=int(y.sum()), p=x.shape[1], rank=x.shape[1],
                              nll=objective(beta), score_inf=score, newton_step_inf=step_inf,
                              condition_r_inf=condition, min_weight=float(weights.min()), finite=1)
        current, fraction = objective(beta), 1.
        while fraction >= limits['minimum_step'] and objective(beta + fraction * step) > current:
            fraction *= limits['backtracking_factor']
        require(fraction >= limits['minimum_step'], 'backtracking failed')
        beta += fraction * step
    raise ValueError('separation or logistic nonconvergence')



def ols(x,y,spec):
 x=np.asarray(x,float);y=np.asarray(y,float)
 b,cond=weighted_qr(x,y,np.ones(len(y)),spec['solver']);return list(b),cond
def fit_cal(eta,y,kind,spec):
 e=np.asarray(eta,float);x=np.ones((len(e),1)) if kind=='offset' else np.column_stack([np.ones(len(e)),e])
 b,d=logistic_fit(x,y,spec,e if kind=='offset' else None);return list(b),d
def average(a):return float(np.mean(a))
def frozen_predictions(rows,model):
 t={x['feature']:(float(x['mean']),float(x['sd'])) for x in model['transforms']};beta=np.array([float(x['coefficient']) for x in model['coefficients']]);x=[]
 for row in rows:
  z={k:(row[k]-m)/s for k,(m,s) in t.items()}
  x.append([1,row['part_time'],row['first_gen'],z['age'],int(row['gender']=='male'),int(row['gender']=='another'),*[int(row['program_area']==v) for v in ['Social Sciences','STEM','Health','Business']],z['prior_grade'],z['wellbeing_score'],z['stress_score'],0])
 return list(sigmoid(np.array(x)@beta))
def losses(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);q=np.clip(p,1e-12,1-1e-12)
 return dict(brier=float(np.mean((y-p)**2)),log_loss=float(np.mean(np.where(y==1,-np.log(q),-np.log1p(-q)))))

def read_csv(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def conformal_quantile(scores,numerator=1,denominator=10):
 require(scores and all(math.isfinite(v) and v>=0 for v in scores),'invalid residuals')
 require(0<numerator<denominator,'invalid alpha')
 n=len(scores);k=((n+1)*(denominator-numerator)+denominator-1)//denominator
 return k,(sorted(scores)[k-1] if k<=n else math.inf)
def logit(p):
 require(math.isfinite(p) and 0<=p<=1,'invalid probability');q=min(1-1e-12,max(1e-12,p));return math.log(q)-math.log1p(-q)
def diagnose(eta,y,kind,spec):
 if len(set(y))<2:return None,'one_class'
 if kind=='joint' and max(eta)-min(eta)<=1e-12:return None,'constant_prediction'
 try:b,d=fit_cal(eta,y,kind,spec);return (b,d),''
 except ValueError as e:return None,'fit_failure:'+str(e)
def calculate(root=RESOURCE):
 spec=json.loads((root/'model-specs/unit08.json').read_text());require(spec['predictor_allowlist']==['part_time','first_gen','age','gender','program_area','prior_grade','wellbeing_score','stress_score'],'predictor allowlist')
 require(spec['grade_model']=='OLS_intercept_prior_grade_only' and spec['alpha_numerator']==1 and spec['alpha_denominator']==10,'frozen grade design')
 rows=joined_rows(root);manifest=read_csv(root/'splits/unit08_roles.csv');require(len(manifest)==len(rows) and len({x['row_id'] for x in manifest})==len(rows),'manifest uniqueness and coverage')
 for a,x in zip(manifest,rows):
  require(a['row_id']==x['row_id'] and a['cohort']==x['cohort'] and a['college_id']==x['college_id'],'manifest keys')
  allowed=['grade_fit','grade_calibration'] if x['college_group']=='anchor' else ['new_updating','new_final'] if x['college_group']=='new' else ['continuing_final']
  require(a['assignment'] in allowed,'invalid population role');x['assignment']=a['assignment']
  require(x['grade'] is None or math.isfinite(x['grade']) and 0<=x['grade']<=4 and x['exit_type']!='consent_withdrawn','invalid grade')
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
   p=[x['p'] if method=='unchanged' else (lambda z:float(sigmoid([z])[0]))(fits[method][0]+(1 if method=='offset' else fits[method][1])*x['eta']) for x in test];eta=[logit(v) for v in p];cal={}
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
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows([{k:format(float(v),'.17g') if isinstance(v,(float,np.floating)) else v for k,v in x.items()} for x in rows])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();res=calculate(a.root);emit(res,a.outdir);print('UNIT08_ROUTE_OK predictions='+str(len(res['predictions']))+' intervals='+str(len(res['intervals'])))
