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

def forecasts(rows,m):
 coefficients={v['term']:float(v['coefficient']) for v in m['coefficients']};scales={v['feature']:(float(v['mean']),float(v['sd'])) for v in m['transforms']};answer=[]
 if any(sd<=0 for mean,sd in scales.values()):raise ValueError('invalid frozen scale')
 for row in rows:
  terms={'intercept':1.,'offer':0.,'part_time':float(row['part_time']),'first_gen':float(row['first_gen'])}
  for field,(mean,sd) in scales.items():terms['z_'+field]=(float(row[field])-mean)/sd
  for level in ['male','another']:terms['gender_'+level]=float(row['gender']==level)
  for level in ['Social Sciences','STEM','Health','Business']:terms['program_'+level.replace(' ','_')]=float(row['program_area']==level)
  eta=math.fsum(coefficients[k]*terms[k] for k in coefficients);e=math.exp(-abs(eta));answer.append(1/(1+e) if eta>=0 else e/(1+e))
 return answer
def auc_score(y,p):
 positive=[v for v,label in zip(p,y) if label==1];negative=[v for v,label in zip(p,y) if label==0]
 if not positive or not negative:return '',0,0,'one_class'
 numerator=sum(2 if a>b else 1 if a==b else 0 for a in positive for b in negative);denominator=2*len(positive)*len(negative)
 return numerator/denominator,numerator,denominator,''
def probability_metrics(y,p):
 if any(not math.isfinite(v) or not 0<=v<=1 for v in p):raise ValueError('invalid probability')
 n=len(y)
 if not n:return dict(n=0,events=0,nonevents=0,auc='',auc_pairs2=0,auc_denominator=0,auc_reason='empty_group',brier='',log_loss='',mean_probability='',observed_fraction='',calibration_bias='',ece='',clipped_n=0)
 events=sum(y);auc,num,den,reason=auc_score(y,p);mean=math.fsum(p)/n;freq=events/n;clipped=[min(max(v,1e-12),1-1e-12) for v in p];ece=0.
 for j in range(5):
  items=[i for i,v in enumerate(p) if min(int(v*5),4)==j]
  if items:ece+=abs(math.fsum(p[i]-y[i] for i in items))/n
 return dict(n=n,events=events,nonevents=n-events,auc=auc,auc_pairs2=num,auc_denominator=den,auc_reason=reason,brier=math.fsum((v-label)**2 for v,label in zip(p,y))/n,log_loss=-math.fsum(math.log(v) if label else math.log1p(-v) for v,label in zip(clipped,y))/n,mean_probability=mean,observed_fraction=freq,calibration_bias=mean-freq,ece=ece,clipped_n=sum(a!=b for a,b in zip(p,clipped)))
def decisions(rows,p,s,role):
 threshold=[int(v>=s['action']['threshold']) for v in p];chosen=set(sorted(range(len(rows)),key=lambda i:(-p[i],rows[i]['row_id'],i))[:s['action']['capacity'][role]])
 return dict(threshold_020=threshold,capacity=[int(i in chosen) for i in range(len(rows))])
def action_metrics(y,a):
 n=len(y);tp=fp=fn=tn=0
 for label,decision in zip(y,a):
  if label==1:
   if decision:tp+=1
   else:fn+=1
  elif decision:fp+=1
  else:tn+=1
 selected=tp+fp;events=tp+fn;nonevents=fp+tn
 return dict(n=n,selected=selected,tp=tp,fp=fp,fn=fn,tn=tn,selection_rate=selected/n if n else '',tpr=tp/events if events else '',tpr_reason='' if events else 'no_events',fpr=fp/nonevents if nonevents else '',fpr_reason='' if nonevents else 'no_nonevents',ppv=tp/selected if selected else '',ppv_reason='' if selected else 'no_selected',net_benefit=(tp-.25*fp)/n if n else '')
def percentile(values,q):
 a=sorted(values);h=(len(a)-1)*q;i=int(h);return a[i]+(h-i)*(a[min(i+1,len(a)-1)]-a[i])
def calculate(root):
 s,m,by,assign=load(root);result={k:[] for k in ['predictions','actions','scores','action_scores','bins','bootstrap','bootstrap_summary','toy']}
 for role in s['evaluation_roles']:
  rows=by[role];probability=forecasts(rows,m);labels=[int(row['binary_y']) for row in rows];actions=decisions(rows,probability,s,role)
  for row,label,risk,i in zip(rows,labels,probability,range(len(rows))):
   result['predictions'].append(dict(role=role,row_id=row['row_id'],college_id=row['college_id'],y=label,probability=risk))
   for rule,dec in actions.items():result['actions'].append(dict(role=role,row_id=row['row_id'],rule=rule,selected=dec[i]))
  for group in s['groups']:
   positions=[i for i,row in enumerate(rows) if member(row,group)];y=[labels[i] for i in positions];p=[probability[i] for i in positions];result['scores'].append(dict(role=role,group=group['id'],**probability_metrics(y,p)))
   for rule,dec in actions.items():result['action_scores'].append(dict(role=role,group=group['id'],rule=rule,**action_metrics(y,[dec[i] for i in positions])))
   for j in range(5):
    subset=[i for i in positions if min(int(probability[i]*5),4)==j];n=len(subset);result['bins'].append(dict(role=role,group=group['id'],bin=j,n=n,events=sum(labels[i] for i in subset),mean_probability=math.fsum(probability[i] for i in subset)/n if n else '',observed_fraction=sum(labels[i] for i in subset)/n if n else '',reason='' if n else 'empty_bin'))
  lookup={row['row_id']:i for i,row in enumerate(rows)}
  for draw in range(1,s['bootstrap']['draws']+1):
   take=[lookup[k] for k in assign[role,draw]];sample=[rows[i] for i in take];p=[probability[i] for i in take];y=[labels[i] for i in take];rules=decisions(sample,p,s,role)
   for group in s['groups']:
    ix=[i for i,row in enumerate(sample) if member(row,group)];n=len(ix);result['bootstrap'].append(dict(role=role,draw_id=draw,group=group['id'],rule='probability',n=n,brier=math.fsum((p[i]-y[i])**2 for i in ix)/n if n else '',calibration_bias=math.fsum(p[i] for i in ix)/n-sum(y[i] for i in ix)/n if n else '',selection_rate='',tpr='',fpr='',ppv=''))
    for rule,a in rules.items():
     z=action_metrics([y[i] for i in ix],[a[i] for i in ix]);result['bootstrap'].append(dict(role=role,draw_id=draw,group=group['id'],rule=rule,n=n,brier='',calibration_bias='',**{k:z[k] for k in ['selection_rate','tpr','fpr','ppv']}))
 buckets=defaultdict(list)
 for row in result['bootstrap']:buckets[row['role'],row['group'],row['rule']].append(row)
 for role in s['evaluation_roles']:
  for group in s['groups']:
   for rule,metrics in [('probability',['brier','calibration_bias']),('threshold_020',['selection_rate','tpr','fpr','ppv']),('capacity',['selection_rate','tpr','fpr','ppv'])]:
    subset=buckets[role,group['id'],rule]
    for metric in metrics:
     values=[row[metric] for row in subset if row[metric]!=''];result['bootstrap_summary'].append(dict(role=role,group=group['id'],rule=rule,metric=metric,draws=len(subset),defined_draws=len(values),undefined_draws=len(subset)-len(values),lower=percentile(values,.025) if values else '',median=percentile(values,.5) if values else '',upper=percentile(values,.975) if values else '',reason='' if values else 'no_defined_replicates'))
 for group,events,negative in [('A',10,90),('B',40,60)]:
  labels=[1]*events+[0]*negative;actions=[1]*int(.8*events)+[0]*(events-int(.8*events))+[1]*int(.1*negative)+[0]*(negative-int(.1*negative));result['toy'].append(dict(group=group,prevalence=events/100,**action_metrics(labels,actions)))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args();emit(calculate(a.root),a.outdir);print('UNIT10_STDLIB_OK')
