#!/usr/bin/env python3
"""Independent scalar joins, Householder ridge, cyclic lasso and loss sums."""
import argparse,csv,json,math,sys
from pathlib import Path
RESOURCE=Path(__file__).resolve().parents[1]
def ensure(ok,message):
 if not ok:raise ValueError(message)
def records(path):
 with path.open(newline='') as handle:return list(csv.DictReader(handle))
def unique(path):
 mapping={}
 for row in records(path):
  ensure(row['student_id'] not in mapping,'duplicate ID');mapping[row['student_id']]=row
 return mapping
def load_records(root):
 source=root/'source/tonmick-v1.0.0/data/anchor';people=unique(source/'participants.csv');survey=unique(source/'baseline.csv');registry=unique(source/'registry.csv');ensure(set(survey)==set(registry)=={k for k in people if people[k]['responded_baseline']=='1'},'respondent join');accepted=[];pool=0;missing=0
 for key in sorted(survey):
  row,outcome=people[key],registry[key];ensure(row['college_id']==survey[key]['college_id'],'college mismatch')
  if survey[key]['offer']!='0':continue
  pool+=1
  if not outcome['gpa12']:missing+=1;continue
  ensure(outcome['exit_type']!='consent_withdrawn','consent grade');y=float(outcome['gpa12']);ensure(math.isfinite(y) and 0<=y<=4,'invalid grade');ensure(outcome['gpa_at_exit'] in ['0','1'] and (outcome['gpa_at_exit']=='0' or float(outcome['exit_month'])>=4),'exit grade timing');ensure(row['part_time'] in ['0','1'] and row['first_gen'] in ['0','1'],'binary predictor');age=float(row['age']);grade=float(row['prior_grade']);ensure(math.isfinite(age) and 16<=age<=80 and math.isfinite(grade) and 0<=grade<=4,'entry predictor')
  accepted.append(dict(row_id=key,college_id=row['college_id'],region=row['region'],part_time=int(row['part_time']),first_gen=int(row['first_gen']),age=age,prior_grade=grade,y=y,gpa_at_exit=int(outcome['gpa_at_exit']),role='development' if row['region']=='West' else 'setting_check'))
 development={r['row_id']:r for r in accepted if r['role']=='development'};outer=records(root/'splits/unit06_outer.csv');assigned={}
 for a in outer:
  ensure(a['row_id'] in development and a['row_id'] not in assigned,'outer membership');ensure(a['college_id']==development[a['row_id']]['college_id'] and a['outer_fold'] in ['1','2','3','4','5'],'outer college/fold');assigned[a['row_id']]=a['outer_fold']
 ensure(set(assigned)==set(development) and set(assigned.values())=={'1','2','3','4','5'},'outer coverage');nested={};raw=records(root/'splits/unit06_inner.csv');contexts=['full','outer1','outer2','outer3','outer4','outer5']
 ensure(all(a['context'] in contexts for a in raw),'unexpected context')
 for context in contexts:
  expected={k for k in development if context=='full' or assigned[k]!=context[-1]};mapping={}
  for a in [x for x in raw if x['context']==context]:
   k=a['row_id'];ensure(k in expected and k not in mapping,'inner membership');ensure(a['college_id']==development[k]['college_id'] and a['inner_fold'] in ['1','2','3','4','5'],'inner college/fold');mapping[k]=a['inner_fold']
  ensure(set(mapping)==expected and set(mapping.values())=={'1','2','3','4','5'},'inner coverage');nested[context]=mapping
 return accepted,assigned,nested,pool,missing

def terms(row,model):
 g=float(row['prior_grade']);a=float(row['age']);p=float(row['part_time']);f=float(row['first_gen']);return [g] if model=='entry' else [g,p,f,a,g*g,a*a,g*p,g*f,g*a]
def fitted_processing(rows,model):
 values=[terms(r,model) for r in rows];columns=list(zip(*values));parameters=[]
 for column in columns:
  mean=math.fsum(column)/len(column);sd=math.sqrt(math.fsum((v-mean)**2 for v in column)/len(column));ensure(math.isfinite(sd) and sd>1e-12,'zero SD');parameters.append((mean,sd))
 return parameters
def matrix(rows,model,parameters):return [[1.0]+[(v-mean)/sd for v,(mean,sd) in zip(terms(r,model),parameters)] for r in rows]
def backsolve(r,b):
 answer=[0.0]*len(b)
 for i in range(len(b)-1,-1,-1):answer[i]=(b[i]-math.fsum(r[i][j]*answer[j] for j in range(i+1,len(b))))/r[i][i]
 return answer
def householder(x,y,spec):
 n=len(x);p=len(x[0]);ensure(n>=p,'QR rank');a=[list(row) for row in x];target=list(y)
 for j in range(p):
  v=[a[i][j] for i in range(j,n)];norm=math.sqrt(math.fsum(t*t for t in v));ensure(norm>0,'QR rank');alpha=-math.copysign(norm,v[0]);v[0]-=alpha;vv=math.fsum(t*t for t in v)
  for k in range(j,p):
   multiplier=2*math.fsum(v[i-j]*a[i][k] for i in range(j,n))/vv
   for i in range(j,n):a[i][k]-=multiplier*v[i-j]
  multiplier=2*math.fsum(v[i-j]*target[i] for i in range(j,n))/vv
  for i in range(j,n):target[i]-=multiplier*v[i-j]
 r=[[a[i][j] if j>=i else 0.0 for j in range(p)] for i in range(p)];diagonals=[abs(r[i][i]) for i in range(p)];ensure(min(diagonals)>spec['solver']['rank_relative_tolerance']*max(diagonals),'QR rank');inverse=[backsolve(r,[float(i==j) for i in range(p)]) for j in range(p)];condition=max(math.fsum(abs(v) for v in row) for row in r)*max(math.fsum(abs(col[i]) for col in inverse) for i in range(p));ensure(condition<=spec['solver']['maximum_condition'],'QR condition');return backsolve(r,target[:p]),condition

def fit(x,y,method,lam,spec):
 n=len(y);p=len(x[0]);beta,condition=householder(x,y,spec);iterations=0
 if method=='ridge' and lam>0:
  extra=[[0.0]+[math.sqrt(lam)*float(i==j) for j in range(p-1)] for i in range(p-1)];beta,_=householder(x+extra,y+[0.0]*(p-1),spec)
 elif method=='lasso' and lam>0:
  columns=list(zip(*x));means=[math.fsum(c)/n for c in columns[1:]];z=[[v-m for v,m in zip(row[1:],means)] for row in x];ym=math.fsum(y)/n;centered=[v-ym for v in y];gram=[[math.fsum(row[j]*row[k] for row in z) for k in range(p-1)] for j in range(p-1)];cross=[math.fsum(row[j]*v for row,v in zip(z,centered)) for j in range(p-1)];b=[0.0]*(p-1)
  for iterations in range(1,spec['solver']['maximum_sweeps']+1):
   change=0.0
   for j in range(p-1):
    old=b[j];partial=cross[j]-math.fsum(gram[j][k]*b[k] for k in range(p-1) if k!=j);b[j]=math.copysign(max(0.0,abs(partial)-lam/2),partial)/gram[j][j];change=max(change,abs(b[j]-old))
   gradient=[2*(math.fsum(gram[j][k]*b[k] for k in range(p-1))-cross[j]) for j in range(p-1)];violation=max(abs(g+lam*math.copysign(1,bj)) if bj!=0 else max(0,abs(g)-lam) for g,bj in zip(gradient,b))/n
   if change<=spec['solver']['maximum_coordinate_change'] and violation<=spec['solver']['maximum_normalized_kkt']:break
  else:raise ValueError('lasso did not converge')
  beta=[ym-math.fsum(m*v for m,v in zip(means,b))]+b
 residual=[math.fsum(v*b for v,b in zip(row,beta))-yi for row,yi in zip(x,y)];grad=[2*math.fsum(row[j]*e for row,e in zip(x,residual)) for j in range(p)];violations=[abs(grad[0])]
 for g,b in zip(grad[1:],beta[1:]):violations.append(abs(g+2*lam*b) if method=='ridge' else (abs(g+lam*math.copysign(1,b)) if b!=0 else max(0,abs(g)-lam)) if method=='lasso' else abs(g))
 kkt=max(violations)/n;ensure(kkt<=spec['solver']['maximum_normalized_kkt'],'fit KKT');rss=math.fsum(e*e for e in residual);penalty=lam*math.fsum(b*b for b in beta[1:]) if method=='ridge' else lam*math.fsum(abs(b) for b in beta[1:]) if method=='lasso' else 0.0
 return beta,dict(n=n,p=p,rank=p,condition_r_inf=condition,iterations=iterations,rss=rss,penalty=penalty,objective=rss+penalty,kkt=kkt,nonzero=sum(abs(b)>spec['nonzero_count_threshold'] for b in beta[1:]))
def choose(losses,tolerance):
 minimum=min(v for k,v in losses);key=max((float(k),k) for k,v in losses if v<=minimum+tolerance)[1];return key,minimum,next(v for k,v in losses if k==key)
def score(y,p):
 ensure(len(y)>0 and all(math.isfinite(v) for v in p),'score');n=len(y);py=math.fsum(p)/n;yy=math.fsum(y)/n;sst=math.fsum((v-py)**2 for v in p);slope=math.fsum((v-py)*(a-yy) for a,v in zip(y,p))/sst if sst>1e-12 else '';intercept=yy-slope*py if slope!='' else '';errors=[a-b for a,b in zip(y,p)]
 return dict(n=n,mse=math.fsum(e*e for e in errors)/n,mae=math.fsum(abs(e) for e in errors)/n,calibration_intercept=intercept,calibration_slope=slope,calibration_reason='' if slope!='' else 'constant_prediction',outside_0_4=sum(v<0 or v>4 for v in p))
def calculate(root=RESOURCE,only_context=None):
 spec=json.loads((root/'model-specs/unit06.json').read_text());ensure(spec['predictor_allowlist']==['prior_grade','part_time','first_gen','age'] and spec['model_ids']==['entry','rich','ridge','lasso'] and not spec['setting_labels_for_selection'],'model spec');rows,outer,inner,pool,missing=load_records(root);dev=[r for r in rows if r['role']=='development'];setting=[r for r in rows if r['role']=='setting_check'];out={k:[] for k in ['predictions','coefficients','diagnostics','transforms','fit_rows','tuning','selection','scores','summary']}
 def fitted(context,stage,model,lam,train,evaluations):
  fid='|'.join([context,stage,model,lam]);parameters=fitted_processing(train,model);x=matrix(train,model,parameters);beta,diagnostic=fit(x,[r['y'] for r in train],model,float(lam),spec);names=['prior_grade'] if model=='entry' else spec['rich_features'];out['diagnostics'].append(dict(fit_id=fid,context=context,stage=stage,model_id=model,lambda_value=lam,**diagnostic));out['coefficients'].extend(dict(fit_id=fid,term=t,coefficient=b) for t,b in zip(['intercept',*names],beta));out['transforms'].extend(dict(fit_id=fid,column=t,mean=m,sd=s) for t,(m,s) in zip(names,parameters));out['fit_rows'].extend(dict(fit_id=fid,row_id=r['row_id']) for r in train)
  for role,held in evaluations:
   for row,v in zip(held,matrix(held,model,parameters)):out['predictions'].append(dict(fit_id=fid,context=context,role=role,model_id=model,lambda_value=lam,row_id=row['row_id'],y=row['y'],prediction=math.fsum(a*b for a,b in zip(v,beta))))
 contexts=[only_context] if only_context else ['full','outer1','outer2','outer3','outer4','outer5']
 for context in contexts:
  train=[r for r in dev if context=='full' or outer[r['row_id']]!=context[-1]];selected={}
  for model in ['ridge','lasso']:
   losses=[]
   for lam in spec[model+'_grid']:
    for j in ['1','2','3','4','5']:fitted(context,'inner'+j,model,lam,[r for r in train if inner[context][r['row_id']]!=j],[('inner_validation',[r for r in train if inner[context][r['row_id']]==j])])
    pp=[p for p in out['predictions'] if p['context']==context and p['role']=='inner_validation' and p['model_id']==model and p['lambda_value']==lam];ensure(len(pp)==len(train),'inner prediction coverage');mse=math.fsum((p['y']-p['prediction'])**2 for p in pp)/len(pp);out['tuning'].append(dict(context=context,model_id=model,lambda_value=lam,n=len(pp),mse=mse));losses.append((lam,mse))
   lam,minimum,mse=choose(losses,spec['tie_absolute']);selected[model]=lam;out['selection'].append(dict(context=context,model_id=model,lambda_value=lam,minimum_mse=minimum,selected_mse=mse))
  evaluations=[('apparent',train),('setting_check',setting)] if context=='full' else [('outer_validation',[r for r in dev if outer[r['row_id']]==context[-1]])]
  for model in spec['model_ids']:fitted(context,'refit',model,selected.get(model,'0'),train,evaluations)
 scopes=[('full','apparent'),('full','setting_check')] if only_context=='full' else [(only_context,'outer_validation')] if only_context else [('full','apparent'),('full','setting_check'),*[(c,'outer_validation') for c in contexts if c!='full'],('pooled','outer_validation')]
 for context,role in scopes:
  for model in spec['model_ids']:
   pp=[p for p in out['predictions'] if p['role']==role and p['model_id']==model and (context=='pooled' or p['context']==context)];out['scores'].append(dict(context=context,role=role,model_id=model,**score([p['y'] for p in pp],[p['prediction'] for p in pp])))
 counts=dict(pool_n=pool,missing_n=missing,development_n=len(dev),setting_n=len(setting),development_colleges=len({r['college_id'] for r in dev}),setting_colleges=len({r['college_id'] for r in setting}),development_exit_grades=sum(r['gpa_at_exit'] for r in dev),setting_exit_grades=sum(r['gpa_at_exit'] for r in setting),inner_memberships=sum(len(v) for v in inner.values()),fits=len(out['diagnostics']),fit_memberships=len(out['fit_rows']),prediction_records=len(out['predictions']));out['summary']=[dict(name=k,value=v) for k,v in sorted(counts.items())];return out

def emit(out,path):
 path.mkdir(parents=True,exist_ok=True)
 for key,rows in out.items():
  with (path/(key+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows({k:format(v,'.17g') if isinstance(v,float) else v for k,v in r.items()} for r in rows)
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=RESOURCE);parser.add_argument('--outdir',type=Path,required=True);a=parser.parse_args()
 try:result=calculate(a.root);emit(result,a.outdir);print('UNIT06_STDLIB_OK predictions='+str(len(result['predictions'])))
 except (ValueError,KeyError,OSError) as e:sys.exit('UNIT06_STDLIB_ERROR '+str(e))
