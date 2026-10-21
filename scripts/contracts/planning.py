import csv, math
from decimal import Decimal

def need(ok,message):
    if not ok: raise ValueError(message)

def rows(path):
    with path.open(newline="", encoding="utf-8") as f: return list(csv.DictReader(f))
read = rows

def compare(a,b,tolerance):
 x=read(a);y=read(b);need(len(x)==len(y),'numerical row inventory '+a.name)
 maximum=0.0
 for first,second in zip(x,y):
  need(first.keys()==second.keys(),'numerical column inventory '+a.name)
  for col in first:
   try:u=float(first[col]);v=float(second[col])
   except ValueError:need(first[col]==second[col],'numerical label or pairing '+col);continue
   need(math.isfinite(u) and math.isfinite(v),'nonfinite result')
   maximum=max(maximum,abs(u-v));need(math.isclose(u,v,rel_tol=tolerance,abs_tol=tolerance),'three-path disagreement '+a.name+' '+col)
 return maximum
