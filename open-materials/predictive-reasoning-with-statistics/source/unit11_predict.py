#!/usr/bin/env python3
"""Predict a single new record using a frozen bundle; no training data or outcome."""
import argparse,json,math
from pathlib import Path
from unit11_stdlib import METHODS,FRAME,ITEMS,SCALES,INDICATORS,TERMS,PROGRAMS,profile,predict_profiles

def predict(record,bundle):
    required=set(FRAME+['record_id','responded_baseline'])
    allowed=required|set(ITEMS)
    if not isinstance(record,dict) or not required<=record.keys() or not record.keys()<=allowed:
        raise ValueError('record requires frame, record_id and response flag; extra keys, including outcomes, are forbidden')
    if not isinstance(record['record_id'],str) or not record['record_id'].strip():raise ValueError('record_id required')
    if bundle.get('schema_version')!=1 or bundle.get('unit')!=11 or bundle.get('methods')!=METHODS or bundle.get('outcomes_required_for_prediction') is not False:
        raise ValueError('invalid outcome-free bundle contract')
    for family,names in [('imputers',SCALES),('models',METHODS)]:
        for name in names:
            m=bundle[family][name]
            expected=TERMS+(['z_'+s for s in SCALES] if family=='models' and name!='frame_only' else [])+(INDICATORS if name=='imputed_indicators' else [])
            if m['terms']!=expected or len(m['coefficients'])!=len(expected) or not all(math.isfinite(v) for v in m['coefficients']):raise ValueError('invalid coefficients/terms')
            features=['age','prior_grade']+(SCALES if family=='models' and name!='frame_only' else [])
            if set(m['transforms'])!=set(features) or not all(len(v)==2 and all(math.isfinite(z) for z in v) and v[1]>0 for v in m['transforms'].values()):raise ValueError('invalid transforms')
    return predict_profiles([record],bundle)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();result=predict(json.loads(a.record.read_text()),json.loads(a.bundle.read_text()));body=json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(body)
    else:print(body,end='')
if __name__=='__main__':main()
