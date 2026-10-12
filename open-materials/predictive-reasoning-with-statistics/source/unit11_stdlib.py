#!/usr/bin/env python3
"""Independent scalar Python route: Householder QR, Newton fitting and rank AUC."""
import argparse
import csv
import json
import math
from pathlib import Path

METHODS = ['frame_only', 'survey_fallback', 'imputed', 'imputed_indicators']
SCALES = ['wellbeing_score', 'stress_score']
INDICATORS = ['survey_absent', 'wb_item_missing', 'st_item_missing']
FRAME = ['part_time', 'first_gen', 'age', 'gender', 'program_area', 'prior_grade']
ITEMS = [prefix + str(i) for prefix in ['wb', 'st'] for i in range(1, 6)]
GENDERS = ['female', 'male', 'another']
PROGRAMS = ['Arts and Humanities', 'Social Sciences', 'STEM', 'Health', 'Business']
PATTERNS = ['survey_absent', 'scores_complete_items', 'scores_partial_wb', 'scores_partial_st',
            'scores_partial_both', 'wb_unscorable', 'st_unscorable', 'both_unscorable']
TERMS = ['intercept', 'part_time', 'first_gen', 'z_age', 'gender_male', 'gender_another',
         'program_Social_Sciences', 'program_STEM', 'program_Health', 'program_Business', 'z_prior_grade']


def require(value, message):
    if not value:
        raise ValueError(message)


def read_table(path):
    with Path(path).open(encoding='utf-8', newline='') as source:
        reader = csv.DictReader(source)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), 'CSV headers')
        data = list(reader)
    require(all(None not in r for r in data), 'CSV extra cells')
    return data


def index(path, key):
    result = {}
    for row in read_table(path):
        require(row[key] and row[key] not in result, 'duplicate/empty ID')
        result[row[key]] = row
    return result


def profile(record):
    result = dict(record)
    require(result['gender'] in GENDERS and result['program_area'] in PROGRAMS, 'unsupported frame category')
    for name in ['part_time', 'first_gen', 'responded_baseline']:
        require(str(result[name]) in ['0', '1'], 'invalid binary value')
        result[name] = int(result[name])
    for key, minimum, maximum in [('age', 16, 80), ('prior_grade', 0, 4)]:
        x = float(result[key]); require(math.isfinite(x) and minimum <= x <= maximum, 'invalid frame value ' + key)
        result[key] = x
    for prefix, key in [('wb', 'wellbeing_score'), ('st', 'stress_score')]:
        count, total = 0, 0
        for j in range(1, 6):
            item = prefix + str(j); x = result.get(item)
            if x is None or x == '':
                result[item] = None
            else:
                require(str(x) in ['1', '2', '3', '4', '5'], 'invalid scale item')
                x = int(x); result[item] = x
                count += 1; total += 6-x if j == 4 else x
        result[prefix + '_nobs'] = count
        result[key] = total/count if count >= 3 else None
    require(result['responded_baseline'] or result['wb_nobs'] + result['st_nobs'] == 0, 'nonresponse/items conflict')
    result['survey_absent'] = int(not result['responded_baseline'])
    result['wb_item_missing'] = int(result['wb_nobs'] != 5)
    result['st_item_missing'] = int(result['st_nobs'] != 5)
    if result['survey_absent']:
        name = 'survey_absent'
    elif result['wellbeing_score'] is None:
        name = 'both_unscorable' if result['stress_score'] is None else 'wb_unscorable'
    elif result['stress_score'] is None:
        name = 'st_unscorable'
    elif result['wb_item_missing'] and result['st_item_missing']:
        name = 'scores_partial_both'
    elif result['wb_item_missing']:
        name = 'scores_partial_wb'
    elif result['st_item_missing']:
        name = 'scores_partial_st'
    else:
        name = 'scores_complete_items'
    result['pattern'] = name
    return result


def load_records(root):
    root = Path(root); legacy = index(root/'splits/unit09_roles.csv', 'row_id'); output = []
    for cohort in ['anchor', 'cohort2']:
        folder = root/'source/tonmick-v1.0.0/data'/cohort
        frames = index(folder/'participants.csv', 'student_id'); surveys = index(folder/'baseline.csv', 'student_id')
        registry = index(root/'source/tonmick-observed-v1.1.0/data'/(cohort+'_observed_registry.csv'), 'student_id')
        require(set(registry) == set(frames), 'registry invitation join')
        require(set(surveys) == {k for k in frames if frames[k]['responded_baseline']=='1'}, 'survey response join')
        for key in sorted(frames):
            f = frames[key]; s = surveys.get(key); o = registry[key]
            require(not s or s['college_id']==f['college_id'], 'survey college join')
            require(o['cohort']==cohort and o['responded_baseline']==f['responded_baseline'], 'registry response join')
            row = {'row_id':key, 'cohort':cohort, 'college_id':f['college_id'], 'region':f['region'],
                   'college_group':f.get('college_group','anchor')}
            row.update({name:f[name] for name in FRAME});row['responded_baseline']=f['responded_baseline']
            row.update({item:s[item] if s else None for item in ITEMS});row=profile(row)
            require(o['offer'] in ['0','1'] and o['outcome_known'] in ['0','1'], 'observation flags')
            known=int(o['outcome_known']); offer=int(o['offer'])
            require((o['withdrawal_label']=='')==(known==0), 'unknown label must be blank')
            y = int(o['withdrawal_label']) if known else None
            require(not known or y in [0,1] and y==int(o['exit_type']=='withdrawal'), 'label/event')
            require(known or o['exit_type']=='consent_withdrawn', 'unknown event')
            require(not row['survey_absent'] or not offer and known, 'nonresponse observation policy')
            require(cohort=='anchor' or not offer, 'later offer policy')
            if cohort == 'anchor':
                role = 'excluded_offered' if offer else 'development' if known else 'excluded_unknown'
            else:
                require(row['college_group'] in ['continuing','new'], 'later college group')
                role = 'temporal_'+row['college_group']
            row.update(offer=offer,outcome_known=known,y=y,exit_type=o['exit_type'],observation_source=o['observation_source'],
                       role=role,legacy_role=legacy[key]['assignment'] if key in legacy else 'not_in_prior_respondent_analysis')
            output.append(row)
    output.sort(key=lambda r:r['row_id']);require(len({r['row_id'] for r in output})==len(output),'cohort ID collision')
    return output


def transforms_for(rows, keys):
    out={}
    for key in keys:
        values=[float(r[key]) for r in rows];mean=math.fsum(values)/len(values)
        sd=math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))
        require(all(math.isfinite(v) for v in values) and math.isfinite(sd) and sd>0,'invalid/zero SD')
        out[key]=(mean,sd)
    return out


def design(rows, transforms, scales=False, indicators=False):
    terms=list(TERMS)
    if scales:terms+=['z_'+key for key in SCALES]
    if indicators:terms+=INDICATORS
    output=[]
    for r in rows:
        z=lambda key:(r[key]-transforms[key][0])/transforms[key][1]
        vector=[1.,float(r['part_time']),float(r['first_gen']),z('age'),
                float(r['gender']=='male'),float(r['gender']=='another')]
        vector.extend(float(r['program_area']==name) for name in PROGRAMS[1:])
        vector.append(z('prior_grade'))
        if scales:vector.extend(z(key) for key in SCALES)
        if indicators:vector.extend(float(r[key]) for key in INDICATORS)
        require(all(math.isfinite(v) for v in vector),'invalid design')
        output.append(vector)
    return output,terms


def dot(a,b):return math.fsum(x*y for x,y in zip(a,b))


def backsolve(r, z):
    result=[0.]*len(z)
    for i in range(len(z)-1,-1,-1):result[i]=(z[i]-math.fsum(r[i][j]*result[j] for j in range(i+1,len(z))))/r[i][i]
    return result


def qr_fit(x, target):
    n=len(x);p=len(x[0]);require(n>p and len(target)==n,'fit dimensions')
    a=[row[:] for row in x];b=list(target)
    for k in range(p):
        v=[a[i][k] for i in range(k,n)]
        length=math.sqrt(dot(v,v));require(length>0,'rank failure')
        v[0]+=math.copysign(length,v[0])
        norm=math.sqrt(dot(v,v));v=[q/norm for q in v]
        for j in range(k,p):
            projection=2*math.fsum(v[i-k]*a[i][j] for i in range(k,n))
            for i in range(k,n):a[i][j]-=projection*v[i-k]
        projection=2*math.fsum(v[i-k]*b[i] for i in range(k,n))
        for i in range(k,n):b[i]-=projection*v[i-k]
    r=[[a[i][j] if j>=i else 0. for j in range(p)] for i in range(p)]
    diagonals=[abs(r[i][i]) for i in range(p)]
    require(min(diagonals)>1e-10*max(diagonals),'rank failure')
    inverse=[backsolve(r,[float(i==j) for i in range(p)]) for j in range(p)]
    condition=max(math.fsum(abs(v) for v in row) for row in r)*max(math.fsum(abs(inverse[j][i]) for j in range(p)) for i in range(p))
    require(math.isfinite(condition) and condition<=1e5,'conditioning failure')
    return backsolve(r,b[:p]),condition


def expit(eta):
    if eta>=0:return 1/(1+math.exp(-eta))
    e=math.exp(eta);return e/(1+e)


def objective(x,y,beta):
    total=[]
    for row,label in zip(x,y):
        eta=dot(row,beta);signed=-eta if label else eta
        total.append(max(0.,signed)+math.log1p(math.exp(-abs(signed))))
    return math.fsum(total)+.5*math.fsum(v*v for v in beta[1:])


def fit_logistic(x,y):
    require(set(y)=={0,1},'training classes')
    beta=[math.log(sum(y)/(len(y)-sum(y)))]+[0.]*(len(x[0])-1)
    for iteration in range(101):
        prob=[expit(dot(row,beta)) for row in x];weight=[p*(1-p) for p in prob]
        require(min(weight)>0 and all(math.isfinite(w) for w in weight),'separation/invalid weight')
        a=[[v*math.sqrt(w) for v in row] for row,w in zip(x,weight)]
        target=[(v-p)/math.sqrt(w) for v,p,w in zip(y,prob,weight)]
        for j in range(1,len(beta)):
            a.append([float(k==j) for k in range(len(beta))]);target.append(-beta[j])
        step,condition=qr_fit(a,target)
        gradient=[math.fsum(row[j]*(v-p) for row,v,p in zip(x,y,prob))-(beta[j] if j else 0.) for j in range(len(beta))]
        score=max(abs(v) for v in gradient);step_inf=max(abs(v) for v in step);nll=objective(x,y,beta)
        if score<=1e-9 and step_inf<=1e-9:
            require(all(math.isfinite(v) for v in beta) and max(abs(v) for v in beta)<=30,'excess/nonfinite coefficient')
            return beta,dict(n_fit=len(y),events=sum(y),parameters=len(beta),rank=len(beta),objective=nll,
                score_inf=score,step_inf=step_inf,condition_r_inf=condition,minimum_weight=min(weight))
        require(iteration<100,'logistic nonconvergence')
        alpha=1.
        for attempt in range(40):
            candidate=[b+alpha*d for b,d in zip(beta,step)]
            if objective(x,y,candidate)<=nll-1e-4*alpha*dot(gradient,step)+1e-12:
                beta=candidate;break
            alpha/=2
        else:raise ValueError('Newton backtracking failed')
    raise ValueError('logistic nonconvergence')


def complete(rows,imputers):
    output=[]
    for record in rows:
        row=dict(record)
        for prefix,key in [('wb','wellbeing_score'),('st','stress_score')]:
            absent=row[key] is None;row[prefix+'_imputed']=int(absent)
            if absent:
                m=imputers[key];x,terms=design([row],m['transforms'])
                require(terms==m['terms'],'imputer term order')
                row[key]=max(1.,min(5.,dot(x[0],m['coefficients'])))
        output.append(row)
    return output


def fit_pipelines(rows):
    training=[r for r in rows if r['role']=='development']
    require(len(training)==984 and all(r['cohort']=='anchor' and r['offer']==0 and r['y'] in [0,1] for r in training),'training membership')
    result=dict(coefficients=[],transforms=[],diagnostics=[],fit_rows=[])
    def store(name,tr,terms,beta,records,diag):
        result['coefficients'] += [dict(component=name,term=t,coefficient=v) for t,v in zip(terms,beta)]
        result['transforms'] += [dict(component=name,feature=k,n_fit=len(records),mean=v[0],sd=v[1]) for k,v in tr.items()]
        result['diagnostics'].append(dict(component=name,**diag))
        result['fit_rows'] += [dict(component=name,row_id=r['row_id']) for r in records]
    imputers={}
    for key in SCALES:
        records=[r for r in training if r[key] is not None];tr=transforms_for(records,['age','prior_grade'])
        x,terms=design(records,tr);target=[r[key] for r in records];beta,cond=qr_fit(x,target)
        residual=[v-dot(row,beta) for row,v in zip(x,target)]
        diag=dict(n_fit=len(records),events='',parameters=len(beta),rank=len(beta),objective=dot(residual,residual),
           score_inf=max(abs(math.fsum(row[j]*v for row,v in zip(x,residual))) for j in range(len(beta))),
           step_inf='',condition_r_inf=cond,minimum_weight='')
        store('imputer_'+key,tr,terms,beta,records,diag)
        imputers[key]={'transforms':tr,'terms':terms,'coefficients':beta}
    filled=complete(training,imputers);models={}
    for method in METHODS:
        scales=method!='frame_only';indicators=method=='imputed_indicators'
        if method=='survey_fallback':records=[r for r in training if all(r[k] is not None for k in SCALES)]
        elif method.startswith('imputed'):records=filled
        else:records=training
        tr=transforms_for(records,['age','prior_grade']+(SCALES if scales else []));x,terms=design(records,tr,scales,indicators)
        beta,diag=fit_logistic(x,[r['y'] for r in records]);store(method,tr,terms,beta,records,diag)
        models[method]={'transforms':tr,'terms':terms,'coefficients':beta}
    bundle={'schema_version':1,'unit':11,'methods':METHODS,'imputers':imputers,'models':models,
            'training_pattern_counts':{p:sum(r['pattern']==p for r in training) for p in PATTERNS},
            'prediction_time':'enrollment','offer_policy':'no_offer','outcomes_required_for_prediction':False}
    return bundle,result


def predict_profiles(profiles,bundle):
    result=[]
    for original in profiles:
        record=profile(original);filled=complete([record],bundle['imputers'])[0]
        for method in METHODS:
            route=method
            if method=='survey_fallback' and (record[SCALES[0]] is None or record[SCALES[1]] is None):route='frame_only'
            r=filled if method.startswith('imputed') else record;m=bundle['models'][route]
            x,terms=design([r],m['transforms'],route!='frame_only',route=='imputed_indicators')
            require(terms==m['terms'],'prediction term order');eta=dot(x[0],m['coefficients'])
            n=bundle['training_pattern_counts'][record['pattern']]
            result.append(dict(method_id=method,record_id=original.get('record_id',original.get('row_id','')),route=route,
                pattern=record['pattern'],training_pattern_n=n,support_reason='' if n else 'unseen_missingness_pattern',
                wellbeing_score=filled[SCALES[0]],stress_score=filled[SCALES[1]],wb_imputed=filled['wb_imputed'],
                st_imputed=filled['st_imputed'],eta=eta,probability=expit(eta)))
    return result


def score(labels,probabilities,n_predict):
    require(len(labels)==len(probabilities) and all(y in [0,1] for y in labels),'score labels')
    require(all(math.isfinite(p) and 0<=p<=1 for p in probabilities),'score probabilities')
    pairs=sorted(zip(probabilities,labels));positive=sum(labels);negative=len(labels)-positive;twice=0;prior_negatives=0;i=0
    while i<len(pairs):
        j=i
        while j<len(pairs) and pairs[j][0]==pairs[i][0]:j+=1
        positives=sum(y for p,y in pairs[i:j]);negatives=j-i-positives
        twice+=positives*(2*prior_negatives+negatives);prior_negatives+=negatives;i=j
    denominator=2*positive*negative;n=len(labels)
    return dict(n_predict=n_predict,n_scored=n,unknown_n=n_predict-n,events=positive,nonevents=negative,
      auc_pairs2=twice if denominator else '',auc_denominator=denominator,auc=twice/denominator if denominator else '',
      auc_reason='' if denominator else 'empty' if not n else 'one_class',
      brier=math.fsum((p-y)**2 for p,y in zip(probabilities,labels))/n if n else '',
      mean_probability=math.fsum(probabilities)/n if n else '')


def calculate(root):
    root=Path(root);rows=load_records(root);bundle,out=fit_pipelines(rows)
    later=[r for r in rows if r['cohort']=='cohort2'];p=predict_profiles(later,bundle)
    lookup={(r['method_id'],r['record_id']):r for r in p};out['completed']=[]
    for r in complete(rows,bundle['imputers']):
        out['completed'].append(dict(row_id=r['row_id'],role=r['role'],pattern=r['pattern'],wellbeing_score=r[SCALES[0]],
          stress_score=r[SCALES[1]],wb_imputed=r['wb_imputed'],st_imputed=r['st_imputed']))
    predictions=[];scores=[];coverage=[]
    for method in METHODS:
        for r in later:
            pr=lookup[method,r['row_id']]
            predictions.append(dict(method_id=method,row_id=r['row_id'],role=r['role'],college_id=r['college_id'],
              legacy_role=r['legacy_role'],responded_baseline=r['responded_baseline'],pattern=r['pattern'],y='' if r['y'] is None else r['y'],
              outcome_known=r['outcome_known'],route=pr['route'],probability=pr['probability'],eta=pr['eta']))
        for role in ['temporal_continuing','temporal_new']:
            population=[r for r in later if r['role']==role]
            for pattern in ['all']+PATTERNS:
                selected=[r for r in population if pattern=='all' or r['pattern']==pattern]
                known=[r for r in selected if r['outcome_known']]
                scores.append(dict(method_id=method,role=role,pattern=pattern,
                    **score([r['y'] for r in known],[lookup[method,r['row_id']]['probability'] for r in known],len(selected))))
            coverage.append(dict(method_id=method,role=role,invited=len(population),predictions=len(population),
              labels_known=sum(r['outcome_known'] for r in population),labels_unknown=sum(not r['outcome_known'] for r in population),
              survey_route_n=sum(lookup[method,r['row_id']]['route']=='survey_fallback' for r in population),
              frame_fallback_n=sum(method=='survey_fallback' and lookup[method,r['row_id']]['route']=='frame_only' for r in population),
              nonrespondents=sum(r['survey_absent'] for r in population),
              partial_item_respondents=sum(not r['survey_absent'] and (r['wb_item_missing'] or r['st_item_missing']) for r in population)))
    out.update(predictions=predictions,scores=scores,coverage=coverage)
    out['new_predictions']=predict_profiles(json.loads((root/'deployment/unit11_new_profiles.json').read_text()),bundle)
    out['bundle']=bundle
    stats={'anchor_invited':1600,'anchor_offered_excluded':sum(r['role']=='excluded_offered' for r in rows),
      'anchor_unknown_no_offer_excluded':sum(r['role']=='excluded_unknown' for r in rows),
      'development_n':sum(r['role']=='development' for r in rows),
      'development_respondents':sum(r['role']=='development' and r['responded_baseline']==1 for r in rows),
      'development_nonrespondents':sum(r['role']=='development' and r['survey_absent']==1 for r in rows),
      'later_invited':len(later),'later_nonrespondents':sum(r['survey_absent'] for r in later),
      'later_labels_known':sum(r['outcome_known'] for r in later),
      'training_unscorable_respondents':sum(r['role']=='development' and not r['survey_absent'] and any(r[k] is None for k in SCALES) for r in rows),
      'new_world_random_draws':0}
    out['summary']=[dict(name=k,value=stats[k]) for k in sorted(stats)]
    return out


def emit(out,folder):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    for name,rows in out.items():
        if name=='bundle':
            (folder/'bundle.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');continue
        with (folder/(name+'.csv')).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader()
            for row in rows:writer.writerow({k:format(v,'.17g') if isinstance(v,float) else '' if v is None else v for k,v in row.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args();emit(calculate(args.root),args.outdir)
    print('UNIT11_STDLIB_OK methods=4 imputers=2 later_predictions=14368 outcome_free_new_profiles=3 solver=Householder_QR')
