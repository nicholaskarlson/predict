import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES={'predictions': 'predictions/unit10_predictions.csv', 'actions': 'expected-results/unit10_actions.csv', 'scores': 'expected-results/unit10_scores.csv', 'action_scores': 'expected-results/unit10_action_scores.csv', 'bins': 'expected-results/unit10_bins.csv', 'bootstrap': 'expected-results/unit10_bootstrap.csv', 'bootstrap_summary': 'expected-results/unit10_bootstrap_summary.csv', 'toy': 'expected-results/unit10_toy.csv'}

TEXT={'role','row_id','college_id','group','rule','auc_reason','tpr_reason','fpr_reason','ppv_reason','reason','metric'}

INTEGER={'n','events','nonevents','auc_pairs2','auc_denominator','clipped_n','selected','tp','fp','fn','tn','bin','draw_id','draws','defined_draws','undefined_draws','y'}

def compare(a,b,label):
 need(len(a)==len(b),label+' row count')
 for i,(x,y) in enumerate(zip(a,b)):
  need(x.keys()==y.keys(),label+' columns')
  for k,v in x.items():
   if k in TEXT or v=='':need(v==y[k],label+' identifier/order/undefined '+k)
   elif k in INTEGER:need(Decimal(v)==Decimal(y[k]) and Decimal(y[k])%1==0,label+' exact count '+k)
   else:need(math.isfinite(float(y[k])) and abs(float(v)-float(y[k]))<=1e-10,label+' numeric '+str(i)+' '+k)
