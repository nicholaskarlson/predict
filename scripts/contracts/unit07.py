import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES={n:('predictions/' if n=='predictions' else 'splits/' if n in {'fit_rows','action_ids'} else 'expected-results/')+'unit07_'+n+'.csv'
       for n in ['predictions','coefficients','diagnostics','transforms','fit_rows','scores','decisions','action_ids','table_cells','cutpoints','examples','summary']}

INTEGER={'n','n_fit','events','nonevents','p','rank','finite','offer_for_prediction','y','auc_twice_wins','auc_denominator',
 'clipped_n','selected','tp','fp','tn','fn','utility_total','part_time','grade_bin','fallback','threshold_selected','capacity_selected'}

NUMERIC={'probability','eta','brier_loss','log_loss','coefficient','nll','score_inf','newton_step_inf','condition_r_inf','min_weight',
 'mean','sd','auc','brier','mean_probability','net_benefit','quantile','value','wellbeing_score','stress_score'}

def tolerance(key,value):
    if key=='condition_r_inf':return 1e-8*max(1,abs(float(value)))
    if key=='coefficient':return 1e-8
    if key in {'probability','eta'}:return 1e-9
    if key in {'mean','sd'}:return 1e-12
    return 1e-10

def compare(expected,actual,label):
    need(len(expected)==len(actual),label+' row count')
    for i,(a,b) in enumerate(zip(expected,actual)):
        need(a.keys()==b.keys(),label+' columns')
        for key,v in a.items():
            other=b[key]
            integer=key in INTEGER or (key=='value' and (a.get('name','').endswith('_n') or a.get('name','').endswith('_events')))
            if v=='':need(other=='',label+' undefined '+key)
            elif integer:
                need(Decimal(str(v))==Decimal(str(other)) and Decimal(str(v))%1==0,label+' integer '+key)
            elif key in NUMERIC:
                need(math.isfinite(float(v)) and math.isfinite(float(other)) and abs(float(v)-float(other))<=tolerance(key,v),
                     label+' numeric row '+str(i)+' '+key)
            else:need(str(v)==str(other),label+' ID/order '+key)
    if label.endswith('diagnostics'):
        for row in actual:
            need(float(row['score_inf'])<=1e-9 and float(row['newton_step_inf'])<=1e-9,'logistic score/step convergence')
            need(int(row['rank'])==int(row['p']) and float(row['condition_r_inf'])<=1e5 and float(row['min_weight'])>0,'logistic rank/conditioning')
