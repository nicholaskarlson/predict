import math
from decimal import Decimal
from fractions import Fraction

def need(ok,message):
    if not ok: raise ValueError(message)

NUMERIC_COLUMNS={'value','n','events','risk','fallback','quantile','y','grade_bin','part_time',
                 'table_probability','baseline_probability'}

INTEGER_COLUMNS={'n','events','fallback','y','grade_bin','part_time'}

def compare_rows(expected,actual,label):
    need(len(expected)==len(actual),label+' row count')
    for index,(want,got) in enumerate(zip(expected,actual)):
        need(want.keys()==got.keys(),label+' column inventory')
        for key,value in want.items():
            integer = key in INTEGER_COLUMNS or (key=='value' and
                       (want.get('name','').endswith('_n') or want.get('name','').endswith('_events')))
            if integer:
                need(Fraction(str(value))==Fraction(str(got[key])) and Fraction(str(value)).denominator==1,
                     label+' integer drift row '+str(index)+' '+key)
            elif key in NUMERIC_COLUMNS:
                need(abs(float(value)-float(got[key]))<=1e-12,label+' numeric drift row '+str(index)+' '+key)
            else:need(str(value)==str(got[key]),label+' ID/order/value drift '+key)
