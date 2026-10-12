#!/usr/bin/env python3
"""Independent scalar Python joins, Householder IRLS, ranking and losses."""
import argparse
import csv
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

RESOURCE = Path(__file__).resolve().parents[1]
MODEL_ORDER = ('control_mean', 'control_table', 'entry_logistic', 'full_logistic')


def ensure(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def keyed(path):
    records = read(path)
    keys = [record['student_id'] for record in records]
    ensure(len(keys) == len(set(keys)), 'duplicate raw key ' + path.name)
    return dict(zip(keys, records))


def item_mean(record, stem):
    scored = []
    for number in range(1, 6):
        item = record[stem + str(number)]
        if item:
            ensure(item in ('1', '2', '3', '4', '5'), 'invalid ordinal item')
            value = int(item)
            scored.append(6-value if number == 4 else value)
    return (float(Fraction(sum(scored), len(scored))) if len(scored) >= 3 else None, len(scored))


def load_records(root):
    records = []
    for population in ['anchor', 'cohort2']:
        directory = root/'source/tonmick-v1.0.0/data'/population
        participants = keyed(directory/'participants.csv')
        interviews = keyed(directory/'baseline.csv')
        outcomes = keyed(directory/'registry.csv')
        included = {key for key, person in participants.items() if person['responded_baseline'] == '1'}
        ensure(included == set(interviews) and included == set(outcomes), 'respondent membership')
        for key in sorted(included):
            person, interview, observation = participants[key], interviews[key], outcomes[key]
            ensure(person['college_id'] == interview['college_id'], 'college mismatch')
            if population == 'cohort2' and person['college_group'] != 'continuing':
                continue
            wb, wb_n = item_mean(interview, 'wb')
            st, st_n = item_mean(interview, 'st')
            exit_type = observation['exit_type']
            ensure(exit_type in ['none', 'transfer', 'withdrawal', 'consent_withdrawn'], 'invalid exit type')
            target = None if exit_type == 'consent_withdrawn' else int(exit_type == 'withdrawal')
            ensure(target is None or str(target) == observation['withdrawn_12'], 'outcome mismatch')
            role = ('excluded_unknown' if target is None else 'excluded_unscorable_items'
                    if wb is None or st is None else 'development' if population == 'anchor' else 'temporal_final')
            ensure(person['part_time'] in ('0', '1') and person['first_gen'] in ('0', '1'), 'binary coding')
            ensure(person['gender'] in ('female', 'male', 'another'), 'gender coding')
            ensure(person['program_area'] in ('Arts and Humanities', 'Social Sciences', 'STEM', 'Health', 'Business'), 'program coding')
            age = float(person['age']); grade = float(person['prior_grade'])
            ensure(math.isfinite(age) and 16 <= age <= 80 and math.isfinite(grade) and 0 <= grade <= 4, 'invalid continuous predictor')
            offer = int(interview['offer']) if population == 'anchor' else 0
            ensure(offer in (0, 1), 'offer coding')
            records.append(dict(row_id=key, cohort=population, college_id=person['college_id'], region=person['region'],
                part_time=int(person['part_time']), first_gen=int(person['first_gen']), age=age,
                gender=person['gender'], program_area=person['program_area'], prior_grade=grade,
                offer=offer, wellbeing_score=wb, stress_score=st, wellbeing_observed=wb_n, stress_observed=st_n,
                y=target, role=role, exit_type=exit_type))
    records.sort(key=lambda r: r['row_id'])
    ensure(len(records) == len({r['row_id'] for r in records}), 'overlapping cohort IDs')
    assigned = read(root/'splits/unit07_roles.csv')
    ensure(len(assigned) == len(records), 'assignment count')
    for row, assignment in zip(records, assigned):
        ensure(all(str(row[k]) == assignment[k] for k in ['row_id', 'cohort', 'college_id', 'role']), 'assignment identity or role')
    return records


def invlogit(value):
    if value >= 0:
        return 1 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1 + exponential)


def dot(a, b):
    return math.fsum(x*y for x,y in zip(a,b))


def backsolve(r, rhs):
    answer = [0.] * len(rhs)
    for j in reversed(range(len(rhs))):
        ensure(r[j][j] != 0, 'rank deficiency')
        answer[j] = (rhs[j] - math.fsum(r[j][k]*answer[k] for k in range(j+1,len(rhs))))/r[j][j]
    return answer


def householder_step(design, residual, weights, rules):
    ensure(all(math.isfinite(w) and w > 0 for w in weights), 'separation or nonpositive weight')
    n, p = len(design), len(design[0])
    a = [[v*math.sqrt(w) for v in row] for row,w in zip(design,weights)]
    rhs = [e/math.sqrt(w) for e,w in zip(residual,weights)]
    for column in range(p):
        vector = [a[i][column] for i in range(column,n)]
        length = math.sqrt(dot(vector,vector))
        ensure(length > 0, 'rank deficiency')
        vector[0] += math.copysign(length,vector[0])
        squared_length = dot(vector,vector)
        for j in range(column,p):
            scale = 2*math.fsum(v*a[i][j] for v,i in zip(vector,range(column,n)))/squared_length
            for v,i in zip(vector,range(column,n)):
                a[i][j] -= scale*v
        scale = 2*math.fsum(v*rhs[i] for v,i in zip(vector,range(column,n)))/squared_length
        for v,i in zip(vector,range(column,n)):
            rhs[i] -= scale*v
    triangle = [[a[i][j] if j >= i else 0. for j in range(p)] for i in range(p)]
    diagonal = [abs(triangle[j][j]) for j in range(p)]
    ensure(min(diagonal) > rules['rank_relative_tolerance']*max(diagonal), 'rank deficiency')
    inverse_columns = [backsolve(triangle,[float(i==j) for i in range(p)]) for j in range(p)]
    condition = max(math.fsum(abs(v) for v in row) for row in triangle) * max(
        math.fsum(abs(column[i]) for column in inverse_columns) for i in range(p))
    ensure(math.isfinite(condition) and condition <= rules['condition_r_inf_limit'], 'conditioning failure')
    return backsolve(triangle,rhs[:p]), condition


def logistic_fit(design, target, specification):
    n, p = len(design), len(design[0])
    ensure(n > p and len(target) == n and all(len(row) == p for row in design), 'fit dimensions')
    ensure(set(target) == {0,1} and all(math.isfinite(v) for row in design for v in row), 'fit needs finite predictors and both classes')
    rules = specification['solver']
    beta = [math.log(sum(target)/(n-sum(target)))] + [0.]*(p-1)
    def objective(coefficients):
        loss = []
        for row,y in zip(design,target):
            eta = dot(row,coefficients)
            sign_eta = -eta if y else eta
            loss.append(max(0.,sign_eta) + math.log1p(math.exp(-abs(sign_eta))))
        return math.fsum(loss)
    for iteration in range(rules['maximum_iterations']+1):
        ensure(max(abs(v) for v in beta) <= rules['maximum_abs_coefficient'], 'separation or coefficient boundary')
        predicted = [invlogit(dot(row,beta)) for row in design]
        residual = [y-p for y,p in zip(target,predicted)]
        weights = [p*(1-p) for p in predicted]
        step, condition = householder_step(design,residual,weights,rules)
        gradient = max(abs(math.fsum(row[j]*e for row,e in zip(design,residual))) for j in range(p))
        step_size = max(abs(v) for v in step)
        if gradient <= rules['gradient_inf_tolerance'] and step_size <= rules['newton_step_inf_tolerance']:
            return beta, dict(n_fit=n,events=sum(target),p=p,rank=p,nll=objective(beta),score_inf=gradient,
                              newton_step_inf=step_size,condition_r_inf=condition,min_weight=min(weights),finite=1)
        current = objective(beta); fraction = 1.
        while fraction >= rules['minimum_step']:
            trial = [b+fraction*s for b,s in zip(beta,step)]
            if objective(trial) <= current:
                break
            fraction *= rules['backtracking_factor']
        ensure(fraction >= rules['minimum_step'], 'backtracking failed')
        beta = trial
    raise ValueError('separation or logistic nonconvergence')


def design(rows, model, processing, spec, zero_offer=False):
    result = []
    for row in rows:
        standardized = {k:(row[k]-m)/s for k,(m,s) in processing.items()}
        if model == 'entry_logistic':
            values = [1.,standardized['prior_grade']]
        else:
            values = [1.,float(row['part_time']),float(row['first_gen']),standardized['age']]
            values.extend(float(row['gender']==level) for level in spec['gender_levels'][1:])
            values.extend(float(row['program_area']==level) for level in spec['program_levels'][1:])
            values.extend(standardized[k] for k in ['prior_grade','wellbeing_score','stress_score'])
        values.append(0. if zero_offer else float(row['offer']))
        result.append(values)
    return result


def score_probabilities(target, probabilities):
    ensure(len(target) == len(probabilities) and len(target) and set(target) <= {0,1}, 'score population')
    ensure(all(math.isfinite(p) and 0 <= p <= 1 for p in probabilities), 'invalid probability')
    positive = [p for y,p in zip(target,probabilities) if y == 1]
    negative = [p for y,p in zip(target,probabilities) if y == 0]
    twice = sum(2 if a > b else 1 if a == b else 0 for a in positive for b in negative)
    denominator = 2*len(positive)*len(negative)
    bounded = [min(1-1e-12,max(1e-12,p)) for p in probabilities]
    n = len(target)
    return dict(n=n,events=len(positive),nonevents=len(negative),auc_twice_wins=twice if denominator else '',
        auc_denominator=denominator,auc=float(Fraction(twice,denominator)) if denominator else '',
        auc_reason='' if denominator else 'one_class',brier=math.fsum((p-y)**2 for y,p in zip(target,probabilities))/n,
        log_loss=-math.fsum(math.log(p) if y else math.log1p(-p) for y,p in zip(target,bounded))/n,
        clipped_n=sum(p!=q for p,q in zip(probabilities,bounded)),mean_probability=math.fsum(probabilities)/n)


def action_records(predictions, model):
    ordered = sorted(predictions,key=lambda r:(-r['probability'],r['row_id']))
    ranks = {r['row_id']:j+1 for j,r in enumerate(ordered)}
    n = len(predictions); events = sum(r['y'] for r in predictions)
    summaries, manifest = [], []
    for rule in ['threshold_020','capacity_300','all','none']:
        selected = {r['row_id'] for r in predictions if
                    (rule == 'threshold_020' and r['probability'] >= .20) or
                    (rule == 'capacity_300' and ranks[r['row_id']] <= 300) or rule == 'all'}
        tp = sum(r['y'] for r in predictions if r['row_id'] in selected)
        fp = len(selected)-tp; fn = events-tp; tn = n-events-fp
        summaries.append(dict(model_id=model,rule=rule,n=n,selected=len(selected),tp=tp,fp=fp,tn=tn,fn=fn,
                              net_benefit=float(Fraction(4*tp-fp,4*n)),utility_total=4*tp-fp))
        if rule in ['threshold_020','capacity_300']:
            manifest.extend(dict(model_id=model,rule=rule,row_id=r['row_id'],rank=ranks[r['row_id']],
                                 selected=int(r['row_id'] in selected)) for r in predictions)
    return summaries,manifest,ranks


def calculate(root=RESOURCE):
    spec = json.loads((root/'model-specs/unit07.json').read_text())
    ensure(spec['predictor_allowlist'] == ['part_time','first_gen','age','gender','program_area','prior_grade','wellbeing_score','stress_score','offer'], 'predictor allowlist')
    rows = load_records(root)
    training = [r for r in rows if r['role']=='development']; later = [r for r in rows if r['role']=='temporal_final']
    controls = [r for r in training if r['offer']==0]
    ensure(training and controls and len(later)>=300, 'population or capacity')
    prevalence = float(Fraction(sum(r['y'] for r in controls),len(controls)))
    grades = sorted(r['prior_grade'] for r in controls)
    cuts = []
    for q in [.25,.5,.75]:
        index = (len(grades)-1)*q; low = math.floor(index); high = math.ceil(index)
        cuts.append(grades[low]+(index-low)*(grades[high]-grades[low]))
    def grade_bin(value):
        return sum(value > cut for cut in cuts)
    cells = []
    for part in [0,1]:
        for index in range(4):
            group = [r for r in controls if r['part_time']==part and grade_bin(r['prior_grade'])==index]
            count = len(group); events = sum(r['y'] for r in group)
            cells.append(dict(part_time=part,grade_bin=index+1,n=count,events=events,
                              probability=float(Fraction(events,count)) if count else prevalence,fallback=int(count==0)))
    result = dict(rows=rows,predictions=[],coefficients=[],diagnostics=[],transforms=[],fit_rows=[],scores=[],
                  decisions=[],action_ids=[],table_cells=cells,cutpoints=[dict(quantile=q,value=v) for q,v in zip([.25,.5,.75],cuts)],examples=[])
    for model in MODEL_ORDER:
        membership = controls if model in ('control_mean','control_table') else training
        result['fit_rows'].extend(dict(model_id=model,row_id=r['row_id']) for r in membership)
        if model in ('control_mean','control_table'):
            p = [prevalence if model=='control_mean' else cells[4*r['part_time']+grade_bin(r['prior_grade'])]['probability'] for r in later]
            eta = [math.log(v/(1-v)) for v in p]
        else:
            processing = {}
            for feature in spec['standardize'][model]:
                values = [r[feature] for r in training];mean = math.fsum(values)/len(values)
                sd = math.sqrt(math.fsum((v-mean)**2 for v in values)/len(values))
                ensure(math.isfinite(sd) and sd>0, 'zero or invalid training SD')
                processing[feature] = (mean,sd)
                result['transforms'].append(dict(model_id=model,feature=feature,n_fit=len(training),mean=mean,sd=sd))
            beta,diagnostic = logistic_fit(design(training,model,processing,spec),[r['y'] for r in training],spec)
            result['diagnostics'].append(dict(model_id=model,**diagnostic))
            result['coefficients'].extend(dict(model_id=model,term=k,coefficient=v) for k,v in zip(spec['feature_order'][model],beta))
            eta = [dot(row,beta) for row in design(later,model,processing,spec,zero_offer=True)]
            p = [invlogit(v) for v in eta]
        predictions = []
        for row,probability,linear in zip(later,p,eta):
            q = min(1-1e-12,max(1e-12,probability))
            predictions.append(dict(model_id=model,row_id=row['row_id'],y=row['y'],offer_for_prediction=0,
                probability=probability,eta=linear,brier_loss=(probability-row['y'])**2,
                log_loss=-math.log(q) if row['y'] else -math.log1p(-q)))
        result['predictions'].extend(predictions)
        result['scores'].append(dict(model_id=model,**score_probabilities([r['y'] for r in later],p)))
        summaries,manifest,ranks = action_records(predictions,model)
        result['decisions'].extend(summaries);result['action_ids'].extend(manifest)
        if model=='full_logistic':
            for row,prediction in zip(later[:3],predictions[:3]):
                result['examples'].append(dict(row_id=row['row_id'],y=row['y'],wellbeing_score=row['wellbeing_score'],
                    stress_score=row['stress_score'],eta=prediction['eta'],probability=prediction['probability'],
                    brier_loss=prediction['brier_loss'],log_loss=prediction['log_loss'],rank=ranks[row['row_id']],
                    threshold_selected=int(prediction['probability']>=.2),capacity_selected=int(ranks[row['row_id']]<=300)))
    summary = {
        'anchor_pool_n':sum(r['cohort']=='anchor' for r in rows),'anchor_unknown_n':sum(r['cohort']=='anchor' and r['role']=='excluded_unknown' for r in rows),
        'anchor_unscorable_n':sum(r['cohort']=='anchor' and r['role']=='excluded_unscorable_items' for r in rows),
        'development_n':len(training),'development_events':sum(r['y'] for r in training),
        'development_offer0_n':len(controls),'development_offer0_events':sum(r['y'] for r in controls),
        'development_offer1_n':sum(r['offer']==1 for r in training),'development_offer1_events':sum(r['y'] for r in training if r['offer']==1),
        'temporal_n':len(later),'temporal_events':sum(r['y'] for r in later),
        'temporal_unscorable_n':sum(r['cohort']=='cohort2' and r['role']=='excluded_unscorable_items' for r in rows),
        'development_transfer_n':sum(r['exit_type']=='transfer' for r in training),'temporal_transfer_n':sum(r['exit_type']=='transfer' for r in later),
        'candidate_parameters_n':13,'full_coefficients_n':14,'events_per_candidate_parameter':sum(r['y'] for r in training)/13,
        'control_prevalence':prevalence,'temporal_prevalence':sum(r['y'] for r in later)/len(later),
        'development_partial_wb_n':sum(r['wellbeing_observed']<5 for r in training),'development_partial_st_n':sum(r['stress_observed']<5 for r in training),
        'temporal_partial_wb_n':sum(r['wellbeing_observed']<5 for r in later),'temporal_partial_st_n':sum(r['stress_observed']<5 for r in later)}
    result['summary'] = [dict(name=k,value=v) for k,v in sorted(summary.items())]
    return result


def emit(result,out):
    out.mkdir(parents=True,exist_ok=True)
    for name,records in result.items():
        if name=='rows':continue
        with (out/(name+'.csv')).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]),lineterminator='\n');writer.writeheader()
            writer.writerows({k:format(v,'.17g') if isinstance(v,float) else v for k,v in row.items()} for row in records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=RESOURCE);parser.add_argument('--outdir',type=Path,required=True)
    args=parser.parse_args()
    try:
        output=calculate(args.root);emit(output,args.outdir)
        print('UNIT07_STDLIB_OK logistic_fits=2 predictions='+str(len(output['predictions']))+' solver=Householder_IRLS')
    except (ValueError,KeyError,OSError) as error:sys.exit('UNIT07_STDLIB_ERROR '+str(error))
