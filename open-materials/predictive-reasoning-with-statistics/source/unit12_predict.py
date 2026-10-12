"""Three-file outcome-free frozen frame handoff. No training-data access."""
import argparse,json,math
from pathlib import Path
FIELDS={'record_id','part_time','first_gen','age','gender','program_area','prior_grade'}
TERMS=['intercept','part_time','first_gen','z_age','gender_male','gender_another','program_Social_Sciences','program_STEM','program_Health','program_Business','z_prior_grade']
def predict(record,bundle):
    if not isinstance(record,dict) or record.keys()!=FIELDS:raise ValueError('exact frame fields required; outcomes, future inputs and extra keys forbidden')
    if not isinstance(record['record_id'],str) or not record['record_id']:raise ValueError('record_id required')
    if record['gender'] not in ['female','male','another'] or record['program_area'] not in ['Arts and Humanities','Social Sciences','STEM','Health','Business']:raise ValueError('unsupported category')
    for key in ['part_time','first_gen']:
        if type(record[key]) is not int or record[key] not in [0,1]:raise ValueError('frame binary required')
    for key,lo,hi in [('age',16,80),('prior_grade',0,4)]:
        if type(record[key]) not in [float,int] or not math.isfinite(record[key]) or not lo<=record[key]<=hi:raise ValueError('frame range')
    if bundle['schema_version']!=1 or bundle['method_id']!='frozen_unit11_frame_only' or bundle['prediction_time']!='enrollment_snapshot' or bundle['terms']!=TERMS or bundle['outcome_required'] is not False or len(bundle['coefficients'])!=11 or not all(math.isfinite(v) for v in bundle['coefficients']):raise ValueError('frozen bundle contract')
    t=bundle['transforms']
    if set(t)!={'age','prior_grade'} or any(len(v)!=2 or not all(math.isfinite(k) for k in v) or v[1]<=0 for v in t.values()):raise ValueError('frozen transforms')
    z=lambda name:(record[name]-t[name][0])/t[name][1]
    x=[1,record['part_time'],record['first_gen'],z('age'),int(record['gender']=='male'),int(record['gender']=='another'),*[int(record['program_area']==v) for v in ['Social Sciences','STEM','Health','Business']],z('prior_grade')]
    eta=math.fsum(a*b for a,b in zip(x,bundle['coefficients']));e=math.exp(-abs(eta));p=1/(1+e) if eta>=0 else e/(1+e)
    return {'record_id':record['record_id'],'method_id':'frame_only','eta':eta,'probability':p}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=predict(json.loads(a.record.read_text()),json.loads(a.bundle.read_text()));a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
