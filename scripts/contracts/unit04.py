import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES = {'predictions':'predictions/unit04_predictions.csv',
 'coefficients':'expected-results/unit04_coefficients.csv',
 'diagnostics':'expected-results/unit04_diagnostics.csv',
 'transforms':'expected-results/unit04_transforms.csv',
 'fit_rows':'splits/unit04_fit_rows.csv',
 'fold_scores':'expected-results/unit04_fold_scores.csv',
 'aggregate':'expected-results/unit04_aggregate.csv',
 'summary':'expected-results/unit04_summary.csv',
 'examples':'expected-results/unit04_examples.csv'}

INTEGER={'n','folds','n_fit','p','rank','n_unseen','unseen_college'}

NUMERIC={'y','prediction','squared_error','absolute_error','coefficient','condition_r_inf',
 'score_inf','fit_mse','mean','sd','mse','mae','rmse','fold_mean_mse','fold_min_mse','fold_max_mse','value'}

def tolerance(key,value):
    if key=='condition_r_inf':return 1e-8*max(1,abs(float(value)))
    if key in {'coefficient','prediction'}:return 1e-9
    if key in {'mean','sd'}:return 1e-12
    return 1e-10

def compare(expected,actual,label):
    need(len(expected)==len(actual),label+' row count')
    for i,(a,b) in enumerate(zip(expected,actual)):
        need(a.keys()==b.keys(),label+' columns')
        for k,v in a.items():
            integer=k in INTEGER or (k=='value' and a.get('name','').endswith('_n'))
            if integer:
                need(Decimal(str(v))==Decimal(str(b[k])) and Decimal(str(v))%1==0,label+' integer '+k)
            elif k in NUMERIC:
                need(abs(float(v)-float(b[k]))<=tolerance(k,v),label+' numeric row '+str(i)+' '+k)
            else:need(str(v)==str(b[k]),label+' ID/order '+k)
