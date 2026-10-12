import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

TABLES=['predictions','scores','permutation_draws','permutation_summary','offer_counts','offer_contrasts','new_prediction','training']

TEXT={'role','method_id','row_id','college_id','auc_reason','contrast_id','record_id'}

INTEGER={'n','events','nonevents','auc_pairs2','auc_denominator','draw_id','draws','positive_draws','randomized_n','known_n','unknown_n',*[f'{k}{z}' for k in ['assigned','known','unknown','events'] for z in [0,1]]}

def compare(a,b,label):
    need(len(a)==len(b),label+' row count');worst=0.
    for i,(x,y) in enumerate(zip(a,b)):
        need(x.keys()==y.keys(),label+' columns')
        for k,v in x.items():
            w=y[k]
            if k in TEXT or v=='':need(v==w,label+' exact identifier/order/undefined '+k)
            elif k in INTEGER:need(Decimal(v)==Decimal(w) and Decimal(w)%1==0,label+' exact integer '+k)
            else:
                d=abs(float(v)-float(w));need(math.isfinite(float(w)) and d<=1e-10,label+' numeric '+str(i)+' '+k);worst=max(worst,d)
    return worst
