import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

TABLES=['coefficients','transforms','diagnostics','fit_rows','completed','predictions','scores','coverage','new_predictions','summary']

TEXT={'component','term','feature','row_id','role','pattern','method_id','record_id','route','legacy_role','college_id','support_reason','auc_reason','name'}

INTEGER={'n_fit','events','parameters','rank','wb_imputed','st_imputed','responded_baseline','y','outcome_known','training_pattern_n','n_predict','n_scored','unknown_n','nonevents','auc_pairs2','auc_denominator','invited','predictions','labels_known','labels_unknown','survey_route_n','frame_fallback_n','nonrespondents','partial_item_respondents','value'}

def compare(a,b,label):
    need(len(a)==len(b),label+' row count')
    worst=0.
    for i,(x,y) in enumerate(zip(a,b)):
        need(x.keys()==y.keys(),label+' columns')
        for key,v in x.items():
            w=y[key]
            if key in TEXT or v=='':need(v==w,label+' identifier/order/undefined '+key)
            elif key in INTEGER:need(Decimal(v)==Decimal(w) and Decimal(w)%1==0,label+' exact count '+key)
            else:
                d=abs(float(v)-float(w));need(math.isfinite(float(w)) and d<=1e-10,label+' numeric '+str(i)+' '+key);worst=max(worst,d)
    return worst
