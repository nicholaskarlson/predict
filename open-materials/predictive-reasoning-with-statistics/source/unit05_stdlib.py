#!/usr/bin/env python3
"""Independent scalar/Fraction route for Unit 5 fitting and evaluation."""
import argparse,csv,json,math,sys
from fractions import Fraction
from pathlib import Path
RESOURCE=Path(__file__).resolve().parents[1]
def check(condition,reason):
 if not condition:raise ValueError(reason)
def records(path):
 with path.open(newline='') as handle:return list(csv.DictReader(handle))
def load_records(root):
 source=root/'source/tonmick-v1.0.0/data/anchor';maps={}
 for label in ['participants','baseline','registry']:
  entries=records(source/(label+'.csv'));mapping={}
  for item in entries:
   check(item['student_id'] not in mapping,'duplicate ID');mapping[item['student_id']]=item
  maps[label]=mapping
 frame,base,outcomes=(maps[x] for x in ['participants','baseline','registry'])
 check(set(base)==set(outcomes)=={key for key in frame if frame[key]['responded_baseline']=='1'},'respondent join');accepted=[];unknown=0
 for key in sorted(base):
  person,survey,outcome=frame[key],base[key],outcomes[key];check(person['college_id']==survey['college_id'],'college mismatch')
  if survey['offer']!='0':continue
  kind=outcome['exit_type'];check(kind in {'none','withdrawal','transfer','consent_withdrawn'},'exit type')
  if kind=='consent_withdrawn':unknown+=1;continue
  check(outcome['withdrawn_12']==str(int(kind=='withdrawal')),'label mismatch')
  check(person['part_time'] in ['0','1'] and person['program_area'] in ['Arts and Humanities','Social Sciences','Business','STEM','Health'],'entry category')
  grade=Fraction(person['prior_grade']);check(0<=grade<=4,'entry grade')
  accepted.append({'row_id':key,'college_id':person['college_id'],'part_time':int(person['part_time']),'grade':grade,'category':'|'.join([person['college_id'],person['program_area'],person['part_time']]),'y':int(outcome['withdrawn_12']),'exit_type':kind})
 by_id={x['row_id']:x for x in accepted};raw_outer=records(root/'splits/unit05_outer.csv');fold_map={}
 for x in raw_outer:
  key=x['row_id'];check(key in by_id and key not in fold_map,'outer IDs');check(x['college_id']==by_id[key]['college_id'] and x['outer_fold'] in list('12345'),'outer college/fold');fold_map[key]=x['outer_fold']
 check(set(fold_map)==set(by_id) and set(fold_map.values())==set('12345'),'outer membership');nested={f:{} for f in '12345'}
 for x in records(root/'splits/unit05_inner.csv'):
  f,key,j=x['outer_fold'],x['row_id'],x['inner_fold'];check(f in nested and key in by_id and fold_map[key]!=f and key not in nested[f],'inner membership');check(x['college_id']==by_id[key]['college_id'] and j in '12345','inner college/fold');nested[f][key]=j
 for f in '12345':check(set(nested[f])=={key for key in by_id if fold_map[key]!=f} and set(nested[f].values())==set('12345'),'inner coverage')
 return accepted,fold_map,nested,unknown

def fit_encoding(fitting):
 check(bool(fitting),'empty encoding fit');prior=sum(x['y'] for x in fitting)/len(fitting);counts={}
 for row in fitting:
  k=row['category'];n,s=counts.get(k,(0,0));counts[k]=(n+1,s+row['y'])
 return prior,{k:(n,s,(s+5*prior)/(n+5)) for k,(n,s) in counts.items()}
def value(row,prior,encoding):return encoding.get(row['category'],(0,0,prior))[2]
def linear_fit(values,labels):
 n=len(values);xm=math.fsum(values)/n;ym=math.fsum(labels)/n;xx=math.fsum((x-xm)**2 for x in values);check(xx>1e-12,'encoding rank');slope=math.fsum((x-xm)*(y-ym) for x,y in zip(values,labels))/xx;intercept=ym-slope*xm;check(math.isfinite(intercept) and math.isfinite(slope),'nonfinite fit');return intercept,slope,xx
def quantile(values,q):
 ordered=sorted(values);h=(len(ordered)-1)*Fraction(q);lo=h.numerator//h.denominator;return float(ordered[lo]+(h-lo)*(ordered[min(lo+1,len(ordered)-1)]-ordered[lo]))
def metrics(labels,probabilities):
 check(len(labels)>0 and all(math.isfinite(p) and 0<=p<=1 for p in probabilities),'score probabilities');positives=[p for y,p in zip(labels,probabilities) if y==1];negatives=[p for y,p in zip(labels,probabilities) if y==0];pairs=len(positives)*len(negatives);twice=sum(2 if a>b else 1 if a==b else 0 for a in positives for b in negatives)
 return dict(n=len(labels),events=sum(labels),brier=math.fsum((y-p)**2 for y,p in zip(labels,probabilities))/len(labels),auc=float(Fraction(twice,2*pairs)) if pairs else '',auc_pairs=pairs,auc_twice_wins=twice)

def calculate(root=RESOURCE,only_fold=None):
 specification=json.loads((root/'model-specs/unit05.json').read_text());check(specification['smoothing_weight']==5 and specification['predictor_allowlist']==['college_id','program_area','part_time','prior_grade'],'model spec');people,outer,nested,unknown=load_records(root);outputs={name:[] for name in ['predictions','coefficients','encodings','training_features','fit_rows','cells','transforms','scores','summary']}
 def recorded_encoder(f,stage,inner,fit):
  prior,table=fit_encoding(fit)
  for row in fit:outputs['fit_rows'].append(dict(fold=f,stage=stage,inner_fold=inner,row_id=row['row_id']))
  for key in sorted(table):
   n,s,q=table[key];outputs['encodings'].append(dict(fold=f,stage=stage,inner_fold=inner,category=key,n=n,events=s,prevalence=prior,encoding=q))
  return prior,table
 fold_list=[str(only_fold)] if only_fold is not None else list('12345')
 for f in fold_list:
  training=[x for x in people if outer[x['row_id']]!=f];validation=[x for x in people if outer[x['row_id']]==f];prior,outer_table=recorded_encoder(f,'outer_encoder','',training);bad_prior,bad_table=recorded_encoder(f,'invalid_pooled_encoder','',people);crossfit={}
  outputs['fit_rows'].extend(dict(fold=f,stage='invalid_pooled_cutpoints',inner_fold='',row_id=x['row_id']) for x in people)
  for j in '12345':
   permitted=[x for x in training if nested[f][x['row_id']]!=j];inner_prior,inner_table=recorded_encoder(f,'inner_encoder',j,permitted)
   for row in training:
    if nested[f][row['row_id']]==j:crossfit[row['row_id']]=value(row,inner_prior,inner_table)
  check(set(crossfit)=={x['row_id'] for x in training},'crossfit coverage');parameters={}
  for pipeline in ['encoding_nested','encoding_in_sample','encoding_pooled']:
   features=[crossfit[x['row_id']] if pipeline=='encoding_nested' else value(x,bad_prior,bad_table) if pipeline=='encoding_pooled' else value(x,prior,outer_table) for x in training];a,b,spread=linear_fit(features,[x['y'] for x in training]);parameters[pipeline]=(a,b);outputs['coefficients'].append(dict(fold=f,pipeline=pipeline,intercept=a,slope=b,n_fit=len(training),x_sst=spread))
   for row,q in zip(training,features):
    outputs['training_features'].append(dict(fold=f,pipeline=pipeline,row_id=row['row_id'],inner_fold=nested[f][row['row_id']] if pipeline=='encoding_nested' else '',encoding=q,own_label_included=int(pipeline!='encoding_nested')))
    outputs['fit_rows'].append(dict(fold=f,stage='calibration_'+pipeline,inner_fold='',row_id=row['row_id']))
  table_parameters={}
  for pipeline in ['risk_table','table_pooled_cutpoints']:
   source=training if pipeline=='risk_table' else people;cuts=[quantile([x['grade'] for x in source],q) for q in ['1/4','1/2','3/4']];cell_risks={}
   for k in range(8):
    cell=[x for x in training if 4*x['part_time']+sum(float(x['grade'])>c for c in cuts)==k];n=len(cell);events=sum(x['y'] for x in cell);risk=events/n if n else prior;cell_risks[k]=risk;outputs['cells'].append(dict(fold=f,pipeline=pipeline,part_time=k//4,grade_bin=k%4+1,n=n,events=events,risk=risk,fallback=int(n==0)))
   table_parameters[pipeline]=(cuts,cell_risks)
   for q,c in zip([.25,.5,.75],cuts):outputs['transforms'].append(dict(fold=f,pipeline=pipeline,quantile=q,value=c))
  for pipeline,status in specification['pipeline_status'].items():
   for row in validation:
    feature='';fallback=0;clipped=0
    if pipeline=='prevalence':probability=prior
    elif pipeline=='future_registry':probability=float(row['exit_type']=='withdrawal')
    elif pipeline in table_parameters:
     cuts,risks=table_parameters[pipeline];probability=risks[4*row['part_time']+sum(float(row['grade'])>c for c in cuts)]
    else:
     m,t=(bad_prior,bad_table) if pipeline=='encoding_pooled' else (prior,outer_table);feature=value(row,m,t);fallback=int(row['category'] not in t);intercept,slope=parameters[pipeline];raw=intercept+slope*feature;probability=min(1.,max(0.,raw));clipped=int(raw<0 or raw>1)
    outputs['predictions'].append(dict(fold=f,row_id=row['row_id'],pipeline=pipeline,status=status,y=row['y'],prediction=probability,encoding=feature,fallback=fallback,clipped=clipped))
 kinds={x['row_id']:x['exit_type'] for x in people}
 for f in fold_list+(['pooled'] if only_fold is None else []):
  for pipeline,status in specification['pipeline_status'].items():
   selected=[x for x in outputs['predictions'] if x['pipeline']==pipeline and (f=='pooled' or x['fold']==f)];outputs['scores'].append(dict(fold=f,scope='eligible',pipeline=pipeline,status=status,**metrics([x['y'] for x in selected],[x['prediction'] for x in selected])))
  filtered=[x for x in outputs['predictions'] if x['pipeline']=='risk_table' and (f=='pooled' or x['fold']==f) and kinds[x['row_id']]!='transfer'];outputs['scores'].append(dict(fold=f,scope='future_no_transfer',pipeline='risk_table',status='changed_target_invalid_enrollment_filter',**metrics([x['y'] for x in filtered],[x['prediction'] for x in filtered])))
 counts=dict(pool_n=len(people)+unknown,eligible_n=len(people),unknown_n=unknown,events=sum(x['y'] for x in people),transfer_n=sum(x['exit_type']=='transfer' for x in people),colleges=len({x['college_id'] for x in people}),outer_folds=5,inner_memberships=sum(len(t) for t in nested.values()),prediction_records=len(outputs['predictions']),fit_memberships=len(outputs['fit_rows']),encoder_maps=len(outputs['encodings']),calibration_fits=len(outputs['coefficients']),fallback_nested=sum(x['fallback'] for x in outputs['predictions'] if x['pipeline']=='encoding_nested'),clipped_nested=sum(x['clipped'] for x in outputs['predictions'] if x['pipeline']=='encoding_nested'))
 outputs['summary']=[dict(name=k,value=v) for k,v in sorted(counts.items())];return outputs

def emit(result,destination):
 destination.mkdir(parents=True,exist_ok=True)
 for key,rows in result.items():
  with (destination/(key+'.csv')).open('w',newline='') as handle:
   writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
   for row in rows:writer.writerow({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()})
if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,default=RESOURCE);parser.add_argument('--outdir',type=Path,required=True);args=parser.parse_args()
 try:result=calculate(args.root);emit(result,args.outdir);print('UNIT05_STDLIB_OK predictions='+str(len(result['predictions'])))
 except (ValueError,KeyError,OSError,ZeroDivisionError) as error:sys.exit('UNIT05_STDLIB_ERROR '+str(error))
