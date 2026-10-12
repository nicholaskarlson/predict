#!/usr/bin/env python3
"""Independent Unit 10 probability, subgroup, action and conditional-resampling computation."""
import argparse,csv,json,math
from collections import defaultdict
from pathlib import Path
def read(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def load(root):
 s=json.loads((root/'model-specs/unit10.json').read_text());m=json.loads((root/'model-specs/unit10_frozen_risk.json').read_text())
 if s['predictor_allowlist']!=['part_time','first_gen','age','gender','program_area','prior_grade','wellbeing_score','stress_score'] or s['fit_or_update'] or s['group_threshold_selection'] or s['final_labels_for_selection']:raise ValueError('information boundary')
 if not math.isfinite(s['action']['threshold']) or not 0<=s['action']['threshold']<=1 or any(type(v)!=int or v<0 for v in s['action']['capacity'].values()) or type(s['bootstrap']['draws'])!=int or s['bootstrap']['draws']<1:raise ValueError('invalid action or resampling specification')
 a=read(root/'data/unit10_rows.csv');roles=read(root/'splits/unit10_roles.csv');ids={x['row_id']:x for x in a};q={x['row_id']:x for x in roles}
 if len(ids)!=len(a) or len(q)!=len(roles) or set(ids)!=set(q):raise ValueError('duplicate or unmatched ID')
 draws=read(root/'splits/unit10_bootstrap_rows.csv');by=defaultdict(list)
 for x in a:
  z=q[x['row_id']]
  if z['college_id']!=x['college_id'] or z['assignment'] not in ['continuing_final','new_final','new_updating'] or x['offer']!='0':raise ValueError('role or policy')
  if x['binary_y'] not in ['0','1',''] or x['part_time'] not in ['0','1'] or x['first_gen'] not in ['0','1'] or x['gender'] not in ['female','male','another'] or x['program_area'] not in ['Arts and Humanities','Social Sciences','STEM','Health','Business']:raise ValueError('label or group')
  for k in ['age','prior_grade','wellbeing_score','stress_score']:
   if x[k] and not math.isfinite(float(x[k])):raise ValueError('nonfinite predictor')
  if z['assignment'] in s['evaluation_roles'] and x['binary_y']!='' and x['wellbeing_score']!='' and x['stress_score']!='':by[z['assignment']].append(x)
 populations={role:{x['row_id']:x for x in by[role]} for role in s['evaluation_roles']};assign=defaultdict(list);positions=defaultdict(set)
 for x in draws:
  role=x['role'];d=int(x['draw_id']);pos=int(x['position']);college=x['college_id'];key=(role,d,college)
  if role not in populations or d not in range(1,s['bootstrap']['draws']+1) or x['row_id'] not in populations[role] or populations[role][x['row_id']]['college_id']!=college or pos in positions[key]:raise ValueError('invalid resample role college or position')
  positions[key].add(pos);assign[role,d].append(x['row_id'])
 for role in s['evaluation_roles']:
  counts={g:sum(x['college_id']==g for x in by[role]) for g in {x['college_id'] for x in by[role]}}
  if s['action']['capacity'][role]>len(by[role]):raise ValueError('capacity exceeds role')
  for d in range(1,s['bootstrap']['draws']+1):
   if len(assign[role,d])!=len(by[role]):raise ValueError('resample size')
   for college,n in counts.items():
    if positions[role,d,college]!=set(range(1,n+1)):raise ValueError('resample college counts')
 return s,m,by,assign
def member(x,g):return g['field']=='all' or (x['part_time']+'|'+x['first_gen']==g['value'] if g['field']=='part_time_first_gen' else x[g['field']]==g['value'])
def emit(result,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 for key,rows in result.items():
  with (out/(key+'.csv')).open('w',newline='') as f:
   z=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');z.writeheader();z.writerows({k:(format(v,'.17g') if isinstance(v,float) else v) for k,v in row.items()} for row in rows)

import numpy as np
def forecasts(rows,m):
 if [x['term'] for x in m['coefficients']]!=['intercept','part_time','first_gen','z_age','gender_male','gender_another','program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade','z_wellbeing_score','z_stress_score','offer']:raise ValueError('feature order')
 t={x['feature']:(float(x['mean']),float(x['sd'])) for x in m['transforms']};b=np.array([float(x['coefficient']) for x in m['coefficients']]);X=[]
 if any(sd<=0 for mean,sd in t.values()):raise ValueError('invalid frozen scale')
 for x in rows:
  z={k:(float(x[k])-mean)/sd for k,(mean,sd) in t.items()}
  X.append([1,int(x['part_time']),int(x['first_gen']),z['age'],int(x['gender']=='male'),int(x['gender']=='another'),int(x['program_area']=='Social Sciences'),int(x['program_area']=='STEM'),int(x['program_area']=='Health'),int(x['program_area']=='Business'),z['prior_grade'],z['wellbeing_score'],z['stress_score'],0])
 eta=np.asarray(X)@b;p=np.empty_like(eta);positive=eta>=0;p[positive]=1/(1+np.exp(-eta[positive]));e=np.exp(eta[~positive]);p[~positive]=e/(1+e)
 return p
def auc_score(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);a=int(y.sum());b=len(y)-a
 if not a or not b:return '',0,0,'one_class'
 order=np.argsort(p,kind='stable');yy=y[order];pp=p[order];credit=0;lower_negative=0;i=0
 while i<len(y):
  j=i+1
  while j<len(y) and pp[j]==pp[i]:j+=1
  pos=int(yy[i:j].sum());neg=j-i-pos;credit+=2*pos*lower_negative+pos*neg;lower_negative+=neg;i=j
 return credit/(2*a*b),credit,2*a*b,''
def probability_metrics(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float)
 if np.any(~np.isfinite(p)) or np.any((p<0)|(p>1)):raise ValueError('invalid probability')
 n=len(y)
 if not n:return dict(n=0,events=0,nonevents=0,auc='',auc_pairs2=0,auc_denominator=0,auc_reason='empty_group',brier='',log_loss='',mean_probability='',observed_fraction='',calibration_bias='',ece='',clipped_n=0)
 auc,num,den,reason=auc_score(y,p);cl=np.clip(p,1e-12,1-1e-12);binid=np.minimum((p*5).astype(int),4);ece=0
 for j in range(5):
  ix=binid==j
  if ix.any():ece+=int(ix.sum())/n*abs(float(p[ix].mean()-y[ix].mean()))
 return dict(n=n,events=int(y.sum()),nonevents=n-int(y.sum()),auc=auc,auc_pairs2=num,auc_denominator=den,auc_reason=reason,brier=float(np.mean((p-y)**2)),log_loss=float(-np.mean(y*np.log(cl)+(1-y)*np.log1p(-cl))),mean_probability=float(p.mean()),observed_fraction=float(y.mean()),calibration_bias=float(p.mean()-y.mean()),ece=ece,clipped_n=int(np.count_nonzero(cl!=p)))
def decisions(rows,p,s,role):
 p=np.asarray(p);threshold=(p>=s['action']['threshold']).astype(int);order=sorted(range(len(rows)),key=lambda j:(-float(p[j]),rows[j]['row_id'],j));capacity=np.zeros(len(rows),dtype=int);capacity[order[:s['action']['capacity'][role]]]=1
 return dict(threshold_020=threshold,capacity=capacity)
def action_metrics(y,a):
 y=np.asarray(y,dtype=int);a=np.asarray(a,dtype=int);n=len(y);tp=int(np.sum((y==1)&(a==1)));fp=int(np.sum((y==0)&(a==1)));fn=int(np.sum((y==1)&(a==0)));tn=int(np.sum((y==0)&(a==0)));selected=tp+fp;ev=tp+fn;ne=fp+tn
 return dict(n=n,selected=selected,tp=tp,fp=fp,fn=fn,tn=tn,selection_rate=selected/n if n else '',tpr=tp/ev if ev else '',tpr_reason='' if ev else 'no_events',fpr=fp/ne if ne else '',fpr_reason='' if ne else 'no_nonevents',ppv=tp/selected if selected else '',ppv_reason='' if selected else 'no_selected',net_benefit=(tp-.25*fp)/n if n else '')
def percentile(values,q):return float(np.quantile(np.asarray(values),q,method='linear'))
def calculate(root):
 s,m,by,assign=load(root);out={k:[] for k in ['predictions','actions','scores','action_scores','bins','bootstrap','bootstrap_summary','toy']}
 for role in s['evaluation_roles']:
  rows=by[role];p=forecasts(rows,m);y=np.array([int(x['binary_y']) for x in rows]);rules=decisions(rows,p,s,role)
  for i,x in enumerate(rows):
   out['predictions'].append(dict(role=role,row_id=x['row_id'],college_id=x['college_id'],y=int(y[i]),probability=float(p[i])))
   for rule,a in rules.items():out['actions'].append(dict(role=role,row_id=x['row_id'],rule=rule,selected=int(a[i])))
  for g in s['groups']:
   ix=np.array([member(x,g) for x in rows]);out['scores'].append(dict(role=role,group=g['id'],**probability_metrics(y[ix],p[ix])))
   for rule,a in rules.items():out['action_scores'].append(dict(role=role,group=g['id'],rule=rule,**action_metrics(y[ix],a[ix])))
   for j in range(5):
    b=ix&(np.minimum((p*5).astype(int),4)==j);n=int(b.sum());out['bins'].append(dict(role=role,group=g['id'],bin=j,n=n,events=int(y[b].sum()),mean_probability=float(p[b].mean()) if n else '',observed_fraction=float(y[b].mean()) if n else '',reason='' if n else 'empty_bin'))
  index={x['row_id']:i for i,x in enumerate(rows)}
  for d in range(1,s['bootstrap']['draws']+1):
   take=np.array([index[k] for k in assign[role,d]]);sample=[rows[i] for i in take];sp=p[take];sy=y[take];srules=decisions(sample,sp,s,role)
   for g in s['groups']:
    ix=np.array([member(x,g) for x in sample]);n=int(ix.sum());out['bootstrap'].append(dict(role=role,draw_id=d,group=g['id'],rule='probability',n=n,brier=float(np.mean((sp[ix]-sy[ix])**2)) if n else '',calibration_bias=float(sp[ix].mean()-sy[ix].mean()) if n else '',selection_rate='',tpr='',fpr='',ppv=''))
    for rule,a in srules.items():
     z=action_metrics(sy[ix],a[ix]);out['bootstrap'].append(dict(role=role,draw_id=d,group=g['id'],rule=rule,n=n,brier='',calibration_bias='',**{k:z[k] for k in ['selection_rate','tpr','fpr','ppv']}))
 for role in s['evaluation_roles']:
  for g in s['groups']:
   for rule,metrics in [('probability',['brier','calibration_bias']),('threshold_020',['selection_rate','tpr','fpr','ppv']),('capacity',['selection_rate','tpr','fpr','ppv'])]:
    subset=[x for x in out['bootstrap'] if x['role']==role and x['group']==g['id'] and x['rule']==rule]
    for metric in metrics:
     v=[x[metric] for x in subset if x[metric]!=''];out['bootstrap_summary'].append(dict(role=role,group=g['id'],rule=rule,metric=metric,draws=len(subset),defined_draws=len(v),undefined_draws=len(subset)-len(v),lower=percentile(v,.025) if v else '',median=percentile(v,.5) if v else '',upper=percentile(v,.975) if v else '',reason='' if v else 'no_defined_replicates'))
 for group,ev,ne in [('A',10,90),('B',40,60)]:
  y=np.array([1]*ev+[0]*ne);a=np.array([1]*int(.8*ev)+[0]*(ev-int(.8*ev))+[1]*int(.1*ne)+[0]*(ne-int(.1*ne)));out['toy'].append(dict(group=group,prevalence=ev/100,**action_metrics(y,a)))
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();emit(calculate(a.root),a.outdir);print('UNIT10_NUMPY_OK')
