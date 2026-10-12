import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES={k:('predictions/unit08_predictions.csv' if k=='predictions' else 'splits/unit08_fit_rows.csv' if k=='fit_rows' else 'expected-results/unit08_'+k+'.csv') for k in ['predictions','binary_scores','bins','coefficients','diagnostics','fit_rows','residuals','intervals','coverage','selection','summary']}

TEXT={'role','method','row_id','group','fit_id','scope','term','offset_reason','joint_reason','reason'}

INTEGER={'n','events','p','rank','finite','bin','part_time','y','covered','gpa_at_exit','exit_grades','bounds_outside_0_4','n_fit','n_calibration','k','pool_n','binary_n','unknown_n','unscorable_known_n','observed_grade_n','missing_grade_n','colleges'}

def compare(a,b,label):
 need(len(a)==len(b),label+' row count')
 for i,(x,y) in enumerate(zip(a,b)):
  need(x.keys()==y.keys(),label+' columns')
  for k,v in x.items():
   if k in TEXT or v=='':need(v==y[k],label+' identifier/order/undefined '+k)
   elif k in ('score_inf','newton_step_inf'):need(0<=float(y[k])<=1e-10,label+' convergence '+k)
   elif k in INTEGER and (k!='y' or label.split('/')[-1]=='predictions'):need(Decimal(v)==Decimal(y[k]) and Decimal(y[k])%1==0,label+' exact count '+k)
   else:
    tol=1e-8 if k in ['coefficient','offset_intercept','joint_intercept','joint_slope','condition_r_inf'] else 1e-10
    need(math.isfinite(float(y[k])) and abs(float(v)-float(y[k]))<=tol,label+' numeric '+str(i)+' '+k)
