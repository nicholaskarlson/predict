import csv, math
from decimal import Decimal

def need(ok,message):
    if not ok: raise ValueError(message)

def rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
read = rows

INTEGER_FIELDS={'n','events','nonevents','auc_pairs2','auc_denominator','clipped_n','bin','prior_n','sealed_n'}

def compare(first,second):
 a=rows(first);b=rows(second);need(len(a)==len(b),'sealed output row inventory '+first.name)
 maximum=0.0
 for x,y in zip(a,b):
  need(list(x)==list(y),'sealed column inventory '+first.name)
  for key in x:
   if key in INTEGER_FIELDS:
    need(Decimal(x[key])==Decimal(y[key]) and Decimal(x[key])==int(Decimal(x[key])),'sealed exact count '+key);continue
   try:u=float(x[key]);v=float(y[key])
   except ValueError:need(x[key]==y[key],'sealed identity/reason '+key);continue
   need(math.isfinite(u) and math.isfinite(v),'sealed nonfinite output')
   tolerance=2e-12 if 'probability' in key and first.name=='sealed_predictions.csv' else 2e-10
   maximum=max(maximum,abs(u-v));need(abs(u-v)<=tolerance,'sealed independent disagreement '+first.name+' '+key)
 return maximum
