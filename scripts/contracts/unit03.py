import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES={k:('predictions/unit03_predictions.csv' if k=='predictions' else 'splits/unit03_fit_rows.csv' if k=='fit_rows' else 'expected-results/unit03_'+k+'.csv') for k in ['coefficients','transforms','fit_rows','diagnostics','scores','selections','optimism','corrections','predictions','examples','summary']}

INTEGER={'n','n_fit','p','rank','replicates','value'}

TEXT={'replicate','scope','workflow','model_id','slot','row_id','column','term','name','college_id','selected_model','original_model'}

def tolerance(k,v):
    if k=='condition_r_inf':return 1e-8*max(1,abs(float(v)))
    if k in ('coefficient','prediction'):return 1e-9
    if k in ('mean','sd'):return 1e-12
    return 1e-10

def compare(a,b,label):
    need(len(a)==len(b),label+' row count')
    for i,(x,y) in enumerate(zip(a,b)):
        need(x.keys()==y.keys(),label+' columns')
        for k,v in x.items():
            if k in TEXT:need(str(v)==str(y[k]),label+' ID/order '+k)
            elif k in INTEGER:need(Decimal(str(v))==Decimal(str(y[k])) and Decimal(str(v))%1==0,label+' integer '+k)
            else:need(math.isfinite(float(y[k])) and abs(float(v)-float(y[k]))<=tolerance(k,v),label+' numeric '+str(i)+' '+k)
