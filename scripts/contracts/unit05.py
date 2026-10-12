import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES={k:('predictions/unit05_predictions.csv' if k=='predictions' else 'splits/unit05_fit_rows.csv' if k=='fit_rows' else 'expected-results/unit05_'+k+'.csv') for k in ['predictions','coefficients','encodings','training_features','fit_rows','cells','transforms','scores','summary']}

TEXT={'fold','stage','inner_fold','row_id','category','pipeline','status','scope','name'}

INTEGER={'n','events','n_fit','fallback','clipped','y','own_label_included','part_time','grade_bin','auc_pairs','auc_twice_wins'}

def compare(a,b,label):
 need(len(a)==len(b),label+' row count')
 for i,(x,y) in enumerate(zip(a,b)):
  need(x.keys()==y.keys(),label+' columns')
  for k,v in x.items():
   if k in TEXT or v=='':need(str(v)==str(y[k]),label+' ID/order/blank '+k)
   elif k in INTEGER or (k=='value' and 'name' in x):need(Decimal(str(v))==Decimal(str(y[k])) and Decimal(str(v))%1==0,label+' integer '+k)
   else:
    tol=1e-9 if k in ('intercept','slope','x_sst') else 1e-10 if k=='prediction' else 1e-12
    need(math.isfinite(float(y[k])) and abs(float(v)-float(y[k]))<=tol,label+' numeric '+str(i)+' '+k)
