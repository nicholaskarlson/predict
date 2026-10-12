#!/usr/bin/env python3
"""Independent NumPy joins, nested encoding, OLS probability fit and scores."""
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
 for name in ('participants','baseline','registry'):
  raw=read(root/'source/tonmick-v1.0.0/data/anchor'/(name+'.csv'));t={x['student_id']:x for x in raw};need(len(t)==len(raw),'duplicate ID');tables.append(t)
 p,b,y=tables;need(set(b)==set(y)=={k for k,v in p.items() if v['responded_baseline']=='1'},'respondent join');rows=[];unknown=0
 for k in sorted(b):
  need(b[k]['college_id']==p[k]['college_id'],'college mismatch')
  if b[k]['offer']!='0':continue
  a,z=p[k],y[k];need(z['exit_type'] in ('none','withdrawal','transfer','consent_withdrawn'),'exit type')
  if z['exit_type']=='consent_withdrawn':unknown+=1;continue
  need(z['withdrawn_12']==str(int(z['exit_type']=='withdrawal')),'label mismatch')
  need(a['part_time'] in ('0','1') and a['program_area'] in ('Arts and Humanities','Social Sciences','Business','STEM','Health'),'entry category')
  grade=float(a['prior_grade']);need(np.isfinite(grade) and 0<=grade<=4,'entry grade')
  rows.append(dict(row_id=k,college_id=a['college_id'],part_time=int(a['part_time']),grade=grade,category=a['college_id']+'|'+a['program_area']+'|'+a['part_time'],y=int(z['withdrawn_12']),exit_type=z['exit_type']))
 outer=read(root/'splits/unit05_outer.csv');need(len(outer)==len(rows) and len({x['row_id'] for x in outer})==len(rows),'outer IDs');ids={x['row_id']:x for x in rows};need({x['row_id'] for x in outer}==set(ids),'outer membership')
 for x in outer:need(x['college_id']==ids[x['row_id']]['college_id'] and x['outer_fold'] in ('1','2','3','4','5'),'outer college/fold')
 folds={x['row_id']:x['outer_fold'] for x in outer};inner=read(root/'splits/unit05_inner.csv');imap={}
 for f in map(str,range(1,6)):
  raw=[x for x in inner if x['outer_fold']==f];train={k for k in ids if folds[k]!=f};need(len(raw)==len(train) and {x['row_id'] for x in raw}==train,'inner membership')
  for x in raw:need(x['college_id']==ids[x['row_id']]['college_id'] and x['inner_fold'] in ('1','2','3','4','5'),'inner college/fold')
  imap[f]={x['row_id']:x['inner_fold'] for x in raw};need(set(imap[f].values())==set(map(str,range(1,6))),'empty inner fold')
 need(len(inner)==sum(len(v) for v in imap.values()),'unexpected inner outer fold');need(set(folds.values())==set(map(str,range(1,6))),'empty outer fold')
 return rows,folds,imap,unknown

def encoder(rows):
 mu=float(np.mean([x['y'] for x in rows]));keys=sorted({x['category'] for x in rows});table={}
 for k in keys:
  y=np.array([x['y'] for x in rows if x['category']==k]);n=len(y);s=int(y.sum());table[k]=(n,s,(s+5*mu)/(n+5))
 return mu,table
def encode(row,mu,table):return table[row['category']][2] if row['category'] in table else mu
def calibrate(x,y):
 x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float);sst=float(np.sum((x-x.mean())**2));need(sst>1e-12,'encoding rank')
 X=np.column_stack((np.ones(len(x)),x));q,r=np.linalg.qr(X,mode='reduced');beta=np.linalg.solve(r,q.T@y);need(np.all(np.isfinite(beta)),'nonfinite fit')
 return float(beta[0]),float(beta[1]),sst

def score(y,p):
 y=np.asarray(y,dtype=int);p=np.asarray(p,dtype=float);need(len(y)>0 and np.all(np.isfinite(p)) and np.all((p>=0)&(p<=1)),'score probabilities')
 positive=p[y==1];negative=p[y==0];pairs=len(positive)*len(negative);wins=int(sum(int(np.sum(a>negative))*2+int(np.sum(a==negative)) for a in positive));return dict(n=len(y),events=int(y.sum()),brier=float(np.mean((y-p)**2)),auc=wins/(2*pairs) if pairs else '',auc_pairs=pairs,auc_twice_wins=wins)

def calculate(root=RESOURCE,only_fold=None):
 spec=json.loads((root/'model-specs/unit05.json').read_text());need(spec['smoothing_weight']==5 and spec['predictor_allowlist']==['college_id','program_area','part_time','prior_grade'],'model spec');rows,folds,inner,unknown=load(root)
 out={k:[] for k in ('predictions','coefficients','encodings','training_features','fit_rows','cells','transforms','scores','summary')}
 def log_encoder(f,stage,j,fit):
  mu,t=encoder(fit)
  out['fit_rows'].extend(dict(fold=f,stage=stage,inner_fold=j,row_id=x['row_id']) for x in fit)
  out['encodings'].extend(dict(fold=f,stage=stage,inner_fold=j,category=k,n=v[0],events=v[1],prevalence=mu,encoding=v[2]) for k,v in sorted(t.items()));return mu,t
 for f in ([str(only_fold)] if only_fold is not None else map(str,range(1,6))):
  train=[x for x in rows if folds[x['row_id']]!=f];test=[x for x in rows if folds[x['row_id']]==f]
  mu,t=log_encoder(f,'outer_encoder','',train);pmu,pt=log_encoder(f,'invalid_pooled_encoder','',rows);oof={}
  out['fit_rows'].extend(dict(fold=f,stage='invalid_pooled_cutpoints',inner_fold='',row_id=x['row_id']) for x in rows)
  for j in map(str,range(1,6)):
   subset=[x for x in train if inner[f][x['row_id']]!=j];imu,it=log_encoder(f,'inner_encoder',j,subset)
   for x in train:
    if inner[f][x['row_id']]==j:oof[x['row_id']]=encode(x,imu,it)
  need(len(oof)==len(train),'crossfit coverage');fits={}
  for pipeline,feature in [('encoding_nested',lambda x:oof[x['row_id']]),('encoding_in_sample',lambda x:encode(x,mu,t)),('encoding_pooled',lambda x:encode(x,pmu,pt))]:
   x=[feature(a) for a in train];a,b,sst=calibrate(x,[a['y'] for a in train]);fits[pipeline]=(a,b)
   out['coefficients'].append(dict(fold=f,pipeline=pipeline,intercept=a,slope=b,n_fit=len(train),x_sst=sst))
   out['training_features'].extend(dict(fold=f,pipeline=pipeline,row_id=z['row_id'],inner_fold=inner[f][z['row_id']] if pipeline=='encoding_nested' else '',encoding=v,own_label_included=int(pipeline!='encoding_nested')) for z,v in zip(train,x))
   out['fit_rows'].extend(dict(fold=f,stage='calibration_'+pipeline,inner_fold='',row_id=z['row_id']) for z in train)
  tables={}
  for pipeline,cuts in [('risk_table',np.quantile([x['grade'] for x in train],[.25,.5,.75],method='linear')),('table_pooled_cutpoints',np.quantile([x['grade'] for x in rows],[.25,.5,.75],method='linear'))]:
   cells={}
   for k in range(8):
    subset=[x for x in train if 4*x['part_time']+int(np.searchsorted(cuts,x['grade'],side='left'))==k];n=len(subset);s=sum(x['y'] for x in subset);risk=s/n if n else mu;cells[k]=risk
    out['cells'].append(dict(fold=f,pipeline=pipeline,part_time=k//4,grade_bin=k%4+1,n=n,events=s,risk=risk,fallback=int(n==0)))
   tables[pipeline]=(cuts,cells);out['transforms'].extend(dict(fold=f,pipeline=pipeline,quantile=q,value=float(v)) for q,v in zip([.25,.5,.75],cuts))
  for pipeline,status in spec['pipeline_status'].items():
   for z in test:
    encoding='';fallback=0;clipped=0
    if pipeline=='prevalence':p=mu
    elif pipeline in tables:
     cuts,cells=tables[pipeline];p=cells[4*z['part_time']+int(np.searchsorted(cuts,z['grade'],side='left'))]
    elif pipeline=='future_registry':p=float(z['exit_type']=='withdrawal')
    else:
     emu,et=(pmu,pt) if pipeline=='encoding_pooled' else (mu,t);encoding=encode(z,emu,et);fallback=int(z['category'] not in et);a,b=fits[pipeline];raw=a+b*encoding;p=float(np.clip(raw,0,1));clipped=int(raw<0 or raw>1)
    out['predictions'].append(dict(fold=f,row_id=z['row_id'],pipeline=pipeline,status=status,y=z['y'],prediction=p,encoding=encoding,fallback=fallback,clipped=clipped))
 for f in ([str(only_fold)] if only_fold is not None else [*map(str,range(1,6)),'pooled']):
  for pipeline,status in spec['pipeline_status'].items():
   pp=[p for p in out['predictions'] if p['pipeline']==pipeline and (f=='pooled' or p['fold']==f)];out['scores'].append(dict(fold=f,scope='eligible',pipeline=pipeline,status=status,**score([p['y'] for p in pp],[p['prediction'] for p in pp])))
  pp=[p for p in out['predictions'] if p['pipeline']=='risk_table' and (f=='pooled' or p['fold']==f) and next(z['exit_type'] for z in rows if z['row_id']==p['row_id'])!='transfer'];out['scores'].append(dict(fold=f,scope='future_no_transfer',pipeline='risk_table',status='changed_target_invalid_enrollment_filter',**score([p['y'] for p in pp],[p['prediction'] for p in pp])))
 summary=dict(pool_n=len(rows)+unknown,eligible_n=len(rows),unknown_n=unknown,events=sum(x['y'] for x in rows),transfer_n=sum(x['exit_type']=='transfer' for x in rows),colleges=len({x['college_id'] for x in rows}),outer_folds=5,inner_memberships=sum(len(x) for x in inner.values()),prediction_records=len(out['predictions']),fit_memberships=len(out['fit_rows']),encoder_maps=len(out['encodings']),calibration_fits=len(out['coefficients']),fallback_nested=sum(x['fallback'] for x in out['predictions'] if x['pipeline']=='encoding_nested'),clipped_nested=sum(x['clipped'] for x in out['predictions'] if x['pipeline']=='encoding_nested'))
 out['summary']=[dict(name=k,value=v) for k,v in sorted(summary.items())];return out

def emit(out,path):
 path.mkdir(parents=True,exist_ok=True)
 for name,rows in out.items():
  with (path/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
   for row in rows:w.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=RESOURCE);p.add_argument('--outdir',type=Path,required=True);a=p.parse_args()
 try:out=calculate(a.root);emit(out,a.outdir);print('UNIT05_NUMPY_OK predictions='+str(len(out['predictions']))+' numpy='+np.__version__)
 except (ValueError,KeyError,OSError) as e:sys.exit('UNIT05_NUMPY_ERROR '+str(e))
