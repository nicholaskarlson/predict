#!/usr/bin/env python3
"""Independent NumPy ridge/coordinate-descent lasso and nested evaluation."""
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np
RESOURCE=Path(__file__).resolve().parents[1]
def need(ok,message):
 if not ok:raise ValueError(message)
def read(p):
 with p.open(newline='') as f:return list(csv.DictReader(f))
def load(root):
 tables=[]
 for label in ['participants','baseline','registry']:
  raw=read(root/'source/tonmick-v1.0.0/data/anchor'/(label+'.csv'));t={x['student_id']:x for x in raw};need(len(t)==len(raw),'duplicate ID');tables.append(t)
 p,b,o=tables;need(set(b)==set(o)=={k for k,v in p.items() if v['responded_baseline']=='1'},'respondent join');rows=[];missing=0;pool=0
 for key in sorted(b):
  a,y=p[key],o[key];need(a['college_id']==b[key]['college_id'],'college mismatch')
  if b[key]['offer']!='0':continue
  pool+=1;grade=float(y['gpa12']) if y['gpa12'] else None
  need(y['exit_type']!='consent_withdrawn' or grade is None,'consent grade')
  if grade is None:missing+=1;continue
  need(np.isfinite(grade) and 0<=grade<=4,'invalid grade');need(y['gpa_at_exit'] in ['0','1'] and (y['gpa_at_exit']=='0' or float(y['exit_month'])>=4),'exit grade timing')
  need(a['part_time'] in ['0','1'] and a['first_gen'] in ['0','1'],'binary predictor');age=float(a['age']);prior=float(a['prior_grade']);need(np.isfinite(age) and np.isfinite(prior) and 16<=age<=80 and 0<=prior<=4,'entry predictor')
  rows.append(dict(row_id=key,college_id=a['college_id'],region=a['region'],part_time=int(a['part_time']),first_gen=int(a['first_gen']),age=age,prior_grade=prior,y=grade,gpa_at_exit=int(y['gpa_at_exit']),role='development' if a['region']=='West' else 'setting_check'))
 dev=[x for x in rows if x['role']=='development'];ids={x['row_id']:x for x in dev};outer=read(root/'splits/unit06_outer.csv');need(len(outer)==len(dev) and len({x['row_id'] for x in outer})==len(dev) and {x['row_id'] for x in outer}==set(ids),'outer membership')
 for x in outer:need(x['college_id']==ids[x['row_id']]['college_id'] and x['outer_fold'] in list(map(str,range(1,6))),'outer college/fold')
 folds={x['row_id']:x['outer_fold'] for x in outer};need(set(folds.values())==set(map(str,range(1,6))),'empty outer fold');nested={};inner=read(root/'splits/unit06_inner.csv')
 for context in ['full',*['outer'+str(i) for i in range(1,6)]]:
  a=[x for x in inner if x['context']==context];permitted={k for k in ids if context=='full' or folds[k]!=context[-1]};need(len(a)==len(permitted) and {x['row_id'] for x in a}==permitted,'inner membership')
  for x in a:need(x['college_id']==ids[x['row_id']]['college_id'] and x['inner_fold'] in list(map(str,range(1,6))),'inner college/fold')
  nested[context]={x['row_id']:x['inner_fold'] for x in a};need(set(nested[context].values())==set(map(str,range(1,6))),'empty inner fold')
 need(len(inner)==sum(len(v) for v in nested.values()),'unexpected context');return rows,folds,nested,pool,missing

def features(rows,model):
 a=np.array([[r['prior_grade'],r['part_time'],r['first_gen'],r['age']] for r in rows],dtype=float);g,pt,fg,age=a.T
 return a[:,:1] if model=='entry' else np.column_stack((g,pt,fg,age,g*g,age*age,g*pt,g*fg,g*age))
def transform(raw,mean=None,sd=None):
 if mean is None:mean=raw.mean(axis=0);sd=np.sqrt(np.mean((raw-mean)**2,axis=0))
 need(np.all(np.isfinite(sd)) and np.all(sd>1e-12),'zero SD');return np.column_stack((np.ones(len(raw)),(raw-mean)/sd)),mean,sd

def qrsolve(x,y,spec):
 q,r=np.linalg.qr(x,mode='reduced');d=np.abs(np.diag(r));need(len(d)==x.shape[1] and d.min()>spec['solver']['rank_relative_tolerance']*d.max(),'QR rank');condition=float(np.linalg.norm(r,np.inf)*np.linalg.norm(np.linalg.solve(r,np.eye(len(r))),np.inf));need(condition<=spec['solver']['maximum_condition'],'QR condition');return np.linalg.solve(r,q.T@y),condition

def fit(x,y,method,lam,spec):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);beta,condition=qrsolve(x,y,spec);iterations=0
 if method=='ridge' and lam>0:
  penalty=np.column_stack((np.zeros(x.shape[1]-1),np.sqrt(lam)*np.eye(x.shape[1]-1)));beta,_=qrsolve(np.vstack((x,penalty)),np.r_[y,np.zeros(len(penalty))],spec)
 elif method=='lasso' and lam>0:
  means=x[:,1:].mean(axis=0);z=x[:,1:]-means;ym=float(y.mean());gram=z.T@z;cross=z.T@(y-ym);b=np.zeros(z.shape[1]);threshold=lam/2
  for iterations in range(1,spec['solver']['maximum_sweeps']+1):
   change=0.0
   for j in range(len(b)):
    old=b[j];rho=float(cross[j]-gram[j]@b+gram[j,j]*old);b[j]=math.copysign(max(abs(rho)-threshold,0),rho)/gram[j,j];change=max(change,abs(b[j]-old))
   gradient=2*(gram@b-cross);violations=np.where(b!=0,np.abs(gradient+lam*np.sign(b)),np.maximum(np.abs(gradient)-lam,0));kkt=float(violations.max()/len(y))
   if change<=spec['solver']['maximum_coordinate_change'] and kkt<=spec['solver']['maximum_normalized_kkt']:break
  else:raise ValueError('lasso did not converge')
  beta=np.r_[ym-means@b,b]
 residual=x@beta-y;gradient=2*x.T@residual
 if method=='ridge':gradient[1:]+=2*lam*beta[1:]
 elif method=='lasso':gradient[1:]=np.where(beta[1:]!=0,np.abs(gradient[1:]+lam*np.sign(beta[1:])),np.maximum(np.abs(gradient[1:])-lam,0))
 kkt=float(np.max(np.abs(gradient))/len(y));need(kkt<=spec['solver']['maximum_normalized_kkt'],'fit KKT');rss=float(residual@residual);penalty=lam*float(beta[1:]@beta[1:]) if method=='ridge' else lam*float(np.abs(beta[1:]).sum()) if method=='lasso' else 0.0
 return beta,dict(n=len(y),p=x.shape[1],rank=x.shape[1],condition_r_inf=condition,iterations=iterations,rss=rss,penalty=penalty,objective=rss+penalty,kkt=kkt,nonzero=int(np.sum(np.abs(beta[1:])>spec['nonzero_count_threshold'])))

def choose(tuning,tolerance):
 minimum=min(v for k,v in tuning);eligible=[(k,v) for k,v in tuning if v<=minimum+tolerance];chosen=max(eligible,key=lambda z:float(z[0]));return chosen[0],minimum,chosen[1]
def score(y,p):
 y=np.asarray(y,dtype=float);p=np.asarray(p,dtype=float);need(len(y)>0 and np.all(np.isfinite(p)),'score');e=y-p;sst=float(np.sum((p-p.mean())**2))
 slope=float(np.sum((p-p.mean())*(y-y.mean()))/sst) if sst>1e-12 else '';intercept=float(y.mean()-slope*p.mean()) if slope!='' else ''
 return dict(n=len(y),mse=float(np.mean(e*e)),mae=float(np.mean(np.abs(e))),calibration_intercept=intercept,calibration_slope=slope,calibration_reason='' if slope!='' else 'constant_prediction',outside_0_4=int(np.sum((p<0)|(p>4))))

def calculate(root=RESOURCE,only_context=None):
 spec=json.loads((root/'model-specs/unit06.json').read_text());need(spec['predictor_allowlist']==['prior_grade','part_time','first_gen','age'] and spec['model_ids']==['entry','rich','ridge','lasso'] and not spec['setting_labels_for_selection'],'model spec');rows,outer,inner,pool,missing=load(root);dev=[r for r in rows if r['role']=='development'];setting=[r for r in rows if r['role']=='setting_check'];out={k:[] for k in ['predictions','coefficients','diagnostics','transforms','fit_rows','tuning','selection','scores','summary']}
 def fitted(context,stage,model,lam,train,evaluations):
  fid='|'.join([context,stage,model,lam]);raw=features(train,model);x,mu,sd=transform(raw);beta,d=fit(x,[r['y'] for r in train],model,float(lam),spec);names=['prior_grade'] if model=='entry' else spec['rich_features']
  out['diagnostics'].append(dict(fit_id=fid,context=context,stage=stage,model_id=model,lambda_value=lam,**d));out['coefficients'].extend(dict(fit_id=fid,term=t,coefficient=float(b)) for t,b in zip(['intercept',*names],beta));out['transforms'].extend(dict(fit_id=fid,column=t,mean=float(a),sd=float(b)) for t,a,b in zip(names,mu,sd));out['fit_rows'].extend(dict(fit_id=fid,row_id=r['row_id']) for r in train)
  for role,held in evaluations:
   v,_,_=transform(features(held,model),mu,sd);predictions=v@beta
   out['predictions'].extend(dict(fit_id=fid,context=context,role=role,model_id=model,lambda_value=lam,row_id=r['row_id'],y=r['y'],prediction=float(p)) for r,p in zip(held,predictions))
 contexts=[only_context] if only_context else ['full',*['outer'+str(i) for i in range(1,6)]]
 for context in contexts:
  train=[r for r in dev if context=='full' or outer[r['row_id']]!=context[-1]];selected={}
  for model in ['ridge','lasso']:
   grid=spec[model+'_grid'];losses=[]
   for lam in grid:
    for j in map(str,range(1,6)):
     fitrows=[r for r in train if inner[context][r['row_id']]!=j];held=[r for r in train if inner[context][r['row_id']]==j];fitted(context,'inner'+j,model,lam,fitrows,[('inner_validation',held)])
    pp=[p for p in out['predictions'] if p['context']==context and p['role']=='inner_validation' and p['model_id']==model and p['lambda_value']==lam];loss=float(np.mean([(p['y']-p['prediction'])**2 for p in pp]));need(len(pp)==len(train),'inner prediction coverage');out['tuning'].append(dict(context=context,model_id=model,lambda_value=lam,n=len(pp),mse=loss));losses.append((lam,loss))
   lam,minimum,loss=choose(losses,spec['tie_absolute']);selected[model]=lam;out['selection'].append(dict(context=context,model_id=model,lambda_value=lam,minimum_mse=minimum,selected_mse=loss))
  evaluations=[('apparent',train),('setting_check',setting)] if context=='full' else [('outer_validation',[r for r in dev if outer[r['row_id']]==context[-1]])]
  for model in spec['model_ids']:fitted(context,'refit',model,selected.get(model,'0'),train,evaluations)
 for context,role in ([('full','apparent'),('full','setting_check'),*[(f,'outer_validation') for f in contexts if f!='full'],('pooled','outer_validation')] if not only_context else [(only_context,'outer_validation')] if only_context!='full' else [('full','apparent'),('full','setting_check')]):
  for model in spec['model_ids']:
   pp=[p for p in out['predictions'] if p['role']==role and p['model_id']==model and (context=='pooled' or p['context']==context)];out['scores'].append(dict(context=context,role=role,model_id=model,**score([p['y'] for p in pp],[p['prediction'] for p in pp])))
 counts=dict(pool_n=pool,missing_n=missing,development_n=len(dev),setting_n=len(setting),development_colleges=len({r['college_id'] for r in dev}),setting_colleges=len({r['college_id'] for r in setting}),development_exit_grades=sum(r['gpa_at_exit'] for r in dev),setting_exit_grades=sum(r['gpa_at_exit'] for r in setting),inner_memberships=sum(len(v) for v in inner.values()),fits=len(out['diagnostics']),fit_memberships=len(out['fit_rows']),prediction_records=len(out['predictions']))
 out['summary']=[dict(name=k,value=v) for k,v in sorted(counts.items())];return out

def emit(out,path):
 path.mkdir(parents=True,exist_ok=True)
 for name,rows in out.items():
  with (path/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows({k:format(v,'.17g') if isinstance(v,float) else v for k,v in r.items()} for r in rows)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
 try:out=calculate(a.root);emit(out,a.outdir);print('UNIT06_NUMPY_OK predictions='+str(len(out['predictions']))+' numpy='+np.__version__)
 except (ValueError,KeyError,OSError) as e:sys.exit('UNIT06_NUMPY_ERROR '+str(e))
