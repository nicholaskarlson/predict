import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

FILES={n:('predictions/' if n=='predictions' else 'splits/' if n=='fit_rows' else 'expected-results/')+'unit02_'+n+'.csv'
       for n in ['summary','fits','cells','scores','predictions','examples','fit_rows']}

NUMERIC={'value','prediction','y','squared','absolute','log_loss'}

INTEGER={'n','events','part_time','grade_bin','fallback','clipped'}

def compare(expected,actual,label):
    need(len(expected)==len(actual),label+' row count')
    for index,(a,b) in enumerate(zip(expected,actual)):
        need(a.keys()==b.keys(),label+' columns')
        for key,v in a.items():
            other=b[key]
            if v=='':need(other=='',label+' undefined value')
            elif key in INTEGER or (key=='value' and (a.get('name','').startswith(('anchor_','cohort2_')) or a.get('name','').endswith('_final_n'))):
                need(Decimal(str(v))==Decimal(str(other)) and Decimal(str(v))%1==0,label+' integer '+key)
            elif key in NUMERIC:
                need(math.isfinite(float(v)) and math.isfinite(float(other)) and abs(float(v)-float(other))<=1e-12,
                     label+' numeric row '+str(index)+' '+key)
            else:need(str(v)==str(other),label+' ID/order '+key)
