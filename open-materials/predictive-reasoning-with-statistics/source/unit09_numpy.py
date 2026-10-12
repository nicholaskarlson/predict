#!/usr/bin/env python3
"""Independent Unit 9 numpy fits, frozen-model application and scoring."""
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
def logit(p):
 require(math.isfinite(p) and 0<=p<=1,'invalid probability');q=min(1-1e-12,max(1e-12,p));return math.log(q)-math.log1p(-q)
def diagnose(eta,y,kind,spec):
 if len(set(y))<2:return None,'one_class'
 if kind=='joint' and max(eta)-min(eta)<=1e-12:return None,'constant_prediction'
 try:b,d=fit_cal(eta,y,kind,spec);return (b,d),''
 except ValueError as e:return None,'fit_failure:'+str(e)

def scale(a):
 x=np.asarray(a,float);return float(np.mean(x)),float(np.std(x,ddof=0))
def new_predict(x,b):return list(sigmoid(np.asarray(x,float)@b))
def auc_score(y,p):
 y=np.asarray(y,int);p=np.asarray(p,float);n1=int(np.sum(y));n0=len(y)-n1
 if not n1 or not n0:return '', 'one_class'
 order=np.argsort(p,kind='stable');values=p[order];labels=y[order];credit=0.;neg=0;i=0
 while i<len(y):
  j=i+1
  while j<len(y) and values[j]==values[i]:j+=1
  pos=int(np.sum(labels[i:j]));z=j-i-pos;credit+=pos*(neg+.5*z);neg+=z;i=j
 return credit/(n1*n0),''

def lambda_sigmoid(z):return float(sigmoid([z])[0])
def lambda_sum(a):return float(np.sum(a))

TERMS=['intercept','part_time','first_gen','z_age','gender_male','gender_another','program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade','z_wellbeing_score','z_stress_score']
FEATURES=['age','prior_grade','wellbeing_score','stress_score']
def new_design(rows,transforms):
 out=[]
 for row in rows:
  z={k:(row[k]-m)/sd for k,(m,sd) in transforms.items()}
  out.append([1.,row['part_time'],row['first_gen'],z['age'],int(row['gender']=='male'),int(row['gender']=='another'),*[int(row['program_area']==v) for v in ['Social Sciences','STEM','Health','Business']],z['prior_grade'],z['wellbeing_score'],z['stress_score']])
 return out
def type7(values,q):
 require(values and 0<=q<=1,'quantile input')
 return float(np.quantile(np.asarray(values,float),q,method='linear'))
def score(y,p,spec):
 require(y and len(y)==len(p) and all(t in [0,1] for t in y),'binary scoring labels')
 require(all(math.isfinite(v) and 0<=v<=1 for v in p),'probability range');e=[logit(v) for v in p];cal={}
 for kind in ['offset','joint']:cal[kind]=diagnose(e,y,kind,spec)
 auc,reason=auc_score(y,p)
 return dict(n=len(y),events=sum(y),auc=auc,auc_reason=reason,observed_fraction=average(y),mean_probability=average(p),**losses(y,p),clipped_n=sum(v<1e-12 or v>1-1e-12 for v in p),offset_intercept=cal['offset'][0][0][0] if cal['offset'][0] else '',offset_reason=cal['offset'][1],joint_intercept=cal['joint'][0][0][0] if cal['joint'][0] else '',joint_slope=cal['joint'][0][0][1] if cal['joint'][0] else '',joint_reason=cal['joint'][1]),cal
def composition(draw,sites,method,base):
 n=sum(sites[k][method]['n'] for k in draw)
 sums={m:lambda_sum([sites[k][method][m]*sites[k][method]['n'] for k in draw])/n for m in ['brier','log_loss']}
 original={m:lambda_sum([sites[k][base][m]*sites[k][base]['n'] for k in draw])/n for m in ['brier','log_loss']}
 return dict(n=n,**sums,delta_brier=sums['brier']-original['brier'],delta_log_loss=sums['log_loss']-original['log_loss'])
def calculate(root=RESOURCE):
 spec=json.loads((root/'model-specs/unit09.json').read_text());require(spec['predictor_allowlist']==['part_time','first_gen','age','gender','program_area','prior_grade','wellbeing_score','stress_score'],'predictor allowlist')
 require(spec['methods']==['unchanged','offset','joint','refit'] and spec['selection_between_methods'] is False,'fixed comparison')
 rows=joined_rows(root);manifest=read_csv(root/'splits/unit09_roles.csv');require(len(manifest)==len(rows) and len({x['row_id'] for x in manifest})==len(rows),'manifest uniqueness and coverage')
 for a,x in zip(manifest,rows):
  require(all(a[k]==x[k] for k in ['row_id','cohort','college_id']),'manifest keys')
  allowed=['anchor_reference'] if x['college_group']=='anchor' else ['new_updating','new_final'] if x['college_group']=='new' else ['continuing_final']
  require(a['assignment'] in allowed,'invalid population role');x['assignment']=a['assignment']
 old_roles=read_csv(root/'splits/unit08_roles.csv');require(len(old_roles)==len(manifest) and all(a['row_id']==b['row_id'] and b['assignment']==('anchor_reference' if a['cohort']=='anchor' else a['assignment']) for a,b in zip(old_roles,manifest)),'reused role assignment altered')
 risk=[x for x in rows if x['cohort']=='cohort2' and x['y'] is not None and x['wellbeing_score'] is not None and x['stress_score'] is not None];model=json.loads((root/'model-specs/unit09_frozen_risk.json').read_text())
 for x,p in zip(risk,frozen_predictions(risk,model)):x['p']=p;x['eta']=logit(p)
 update=[x for x in risk if x['assignment']=='new_updating'];final=[x for x in risk if x['assignment']=='new_final'];sites=sorted({x['college_id'] for x in final});require(len(sites)==4,'four sites')
 result={k:[] for k in ['predictions','scores','coefficients','diagnostics','fit_rows','transforms','resampling','resample_summary','profiles','summary']}
 def register(id,scope,members,terms,fit):
  b,d=fit;result['coefficients'].extend(dict(fit_id=id,scope=scope,term=t,coefficient=v) for t,v in zip(terms,b));result['diagnostics'].append(dict(fit_id=id,scope=scope,**d));result['fit_rows'].extend(dict(fit_id=id,scope=scope,row_id=x['row_id']) for x in members)
 fits={}
 for kind in ['offset','joint']:
  fit=fit_cal([x['eta'] for x in update],[x['y'] for x in update],kind,spec);fits[kind]=fit[0];register('update_'+kind,'updating_rule',update,['intercept'] if kind=='offset' else ['intercept','slope'],fit)
 transforms={k:scale([x[k] for x in update]) for k in FEATURES}
 for k,(mean,sd) in transforms.items():require(math.isfinite(sd) and sd>0,'zero SD');result['transforms'].append(dict(fit_id='update_refit',feature=k,n=len(update),mean=mean,sd=sd))
 b,d=logistic_fit(np.asarray(new_design(update,transforms)),np.asarray([x['y'] for x in update]),spec);fits['refit']=list(b);register('update_refit','updating_rule',update,TERMS,(b,d))
 newp=new_predict(new_design(final,transforms),b);site_scores={k:{} for k in sites}
 for role in ['continuing_final','new_final']:
  test=[x for x in risk if x['assignment']==role];methods=['unchanged'] if role=='continuing_final' else spec['methods']
  for method in methods:
   p=[x['p'] for x in test] if method=='unchanged' else list(newp) if method=='refit' else [lambda_sigmoid(fits[method][0]+(1 if method=='offset' else fits[method][1])*x['eta']) for x in test]
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
 draws=read_csv(root/'splits/unit09_college_draws.csv');require(len(draws)==1024,'draw inventory')
 import itertools
 expected=[dict(draw_id=str(i),position=str(j),college_id=k) for i,d in enumerate(itertools.product(sites,repeat=4),1) for j,k in enumerate(d,1)];require(draws==expected,'whole-college committed enumeration')
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
   writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows([{k:format(float(v),'.17g') if isinstance(v,(float,np.floating)) else v for k,v in x.items()} for x in rows])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();res=calculate(a.root);emit(res,a.outdir);print('UNIT09_ROUTE_OK predictions='+str(len(res['predictions']))+' college_draws=256')
