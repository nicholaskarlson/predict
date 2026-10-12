import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES={'predictions': 'predictions/unit09_predictions.csv', 'scores': 'expected-results/unit09_scores.csv', 'coefficients': 'expected-results/unit09_coefficients.csv', 'diagnostics': 'expected-results/unit09_diagnostics.csv', 'fit_rows': 'splits/unit09_fit_rows.csv', 'transforms': 'expected-results/unit09_transforms.csv', 'resampling': 'expected-results/unit09_resampling.csv', 'resample_summary': 'expected-results/unit09_resample_summary.csv', 'profiles': 'expected-results/unit09_profiles.csv', 'summary': 'expected-results/unit09_summary.csv'}

TEXT={'metric', 'auc_reason', 'feature', 'cohort', 'scope', 'row_id', 'frame_role', 'joint_reason', 'college_id', 'group', 'term', 'offset_reason', 'reason', 'college_group', 'role', 'method', 'fit_id', 'kind'}

INTEGER={'events', 'draw_id', 'n', 'positive_draws', 'n_fit', 'draws', 'p', 'finite', 'rank', 'clipped_n', 'colleges'}

def compare(a,b,label):
 need(len(a)==len(b),label+' row count')
 for i,(x,y) in enumerate(zip(a,b)):
  need(x.keys()==y.keys(),label+' columns')
  for k,v in x.items():
   if k in TEXT or v=='':need(v==y[k],label+' identifier/order/undefined '+k)
   elif k in ('score_inf','newton_step_inf'):need(0<=float(y[k])<=1e-10,label+' convergence '+k)
   elif k in INTEGER or k=='y':need(Decimal(v)==Decimal(y[k]) and Decimal(y[k])%1==0,label+' exact count '+k)
   else:
    tol=1e-8 if k in ['coefficient','offset_intercept','joint_intercept','joint_slope','condition_r_inf'] else 1e-10
    need(math.isfinite(float(y[k])) and abs(float(v)-float(y[k]))<=tol,label+' numeric '+str(i)+' '+k)
