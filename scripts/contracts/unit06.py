import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES={k:('predictions/unit06_predictions.csv' if k=='predictions' else 'splits/unit06_fit_rows.csv' if k=='fit_rows' else 'expected-results/unit06_'+k+'.csv') for k in ['predictions','coefficients','diagnostics','transforms','fit_rows','tuning','selection','scores','summary']}

TEXT={'fit_id','context','stage','row_id','column','term','model_id','role','name','lambda_value','calibration_reason'}

INTEGER={'n','p','rank','nonzero','y_unused','outside_0_4','value'}

def compare(a,b,label):
 need(len(a)==len(b),label+' row count')
 for i,(x,y) in enumerate(zip(a,b)):
  need(x.keys()==y.keys(),label+' columns')
  for k,v in x.items():
   if k in TEXT or v=='':need(str(v)==str(y[k]),label+' ID/order/blank '+k)
   elif k=='iterations':need(0<=int(y[k])<=100000 and Decimal(str(y[k]))%1==0,label+' iteration limit')
   elif k=='kkt':need(math.isfinite(float(y[k])) and 0<=float(y[k])<=1e-10,label+' KKT threshold')
   elif k in INTEGER:need(Decimal(str(v))==Decimal(str(y[k])) and Decimal(str(v))%1==0,label+' integer '+k)
   else:
    tol=1e-8 if k in ('coefficient','calibration_intercept','calibration_slope','rss','penalty','objective','condition_r_inf') else 1e-9 if k=='prediction' else 1e-10
    need(math.isfinite(float(y[k])) and abs(float(v)-float(y[k]))<=tol,label+' numeric '+str(i)+' '+k)
