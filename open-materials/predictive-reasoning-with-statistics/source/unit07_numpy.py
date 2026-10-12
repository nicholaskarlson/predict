#!/usr/bin/env python3
"""NumPy route: independent raw joins, weighted QR logistic fits and scores."""
import argparse
import csv
import json
import math
from pathlib import Path
import sys
import numpy as np

RESOURCE = Path(__file__).resolve().parents[1]
MODELS = ['control_mean', 'control_table', 'entry_logistic', 'full_logistic']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def joined_rows(root):
    rows = []
    for cohort in ('anchor', 'cohort2'):
        folder = root / 'source/tonmick-v1.0.0/data' / cohort
        indexed = []
        for name in ('participants', 'baseline', 'registry'):
            with (folder / (name + '.csv')).open(newline='') as stream:
                records = list(csv.DictReader(stream))
            table = {r['student_id']: r for r in records}
            require(len(table) == len(records), 'duplicate raw key ' + name)
            indexed.append(table)
        frame, survey, registry = indexed
        require(set(survey) == set(registry) == {
            k for k, p in frame.items() if p['responded_baseline'] == '1'}, 'respondent join')
        for key in sorted(survey):
            p, b, outcome = frame[key], survey[key], registry[key]
            require(p['college_id'] == b['college_id'], 'college join')
            if cohort == 'cohort2' and p['college_group'] != 'continuing':
                continue
            require(outcome['exit_type'] in ('none', 'withdrawal', 'transfer', 'consent_withdrawn'), 'exit type')
            known = outcome['exit_type'] != 'consent_withdrawn'
            if known:
                require(outcome['withdrawn_12'] == str(int(outcome['exit_type'] == 'withdrawal')), 'outcome exit mismatch')
            scales, observed = {}, {}
            for prefix, feature in [('wb', 'wellbeing_score'), ('st', 'stress_score')]:
                values = np.array([int(b[prefix + str(i)]) if b[prefix + str(i)] else np.nan
                                   for i in range(1, 6)], dtype=float)
                require(np.all(np.isnan(values) | ((values >= 1) & (values <= 5))), 'ordinal item range')
                require(all(b[prefix+str(i)] == '' or str(int(b[prefix+str(i)])) == b[prefix+str(i)]
                            for i in range(1, 6)), 'ordinal item type')
                values[3] = 6 - values[3]
                observed[feature] = int(np.count_nonzero(~np.isnan(values)))
                scales[feature] = float(np.nanmean(values)) if observed[feature] >= 3 else None
            role = ('excluded_unknown' if not known else 'excluded_unscorable_items'
                    if any(v is None for v in scales.values()) else 'development'
                    if cohort == 'anchor' else 'temporal_final')
            require(p['part_time'] in ('0', '1') and p['first_gen'] in ('0', '1'), 'binary predictors')
            require(p['gender'] in ('female', 'male', 'another') and p['program_area'] in (
                'Arts and Humanities', 'Social Sciences', 'STEM', 'Health', 'Business'), 'categorical predictor')
            age, grade = float(p['age']), float(p['prior_grade'])
            require(np.isfinite(age) and 16 <= age <= 80 and np.isfinite(grade) and 0 <= grade <= 4, 'continuous predictor')
            offer = int(b['offer']) if cohort == 'anchor' else 0
            require(offer in (0, 1), 'offer')
            rows.append(dict(row_id=key, cohort=cohort, college_id=p['college_id'], region=p['region'],
                             part_time=int(p['part_time']), first_gen=int(p['first_gen']), age=age,
                             gender=p['gender'], program_area=p['program_area'], prior_grade=grade,
                             offer=offer, y=int(outcome['withdrawn_12']) if known else None,
                             role=role, exit_type=outcome['exit_type'], **scales,
                             wellbeing_observed=observed['wellbeing_score'], stress_observed=observed['stress_score']))
    rows.sort(key=lambda r: r['row_id'])
    require(len({r['row_id'] for r in rows}) == len(rows), 'cohort IDs overlap')
    with (root / 'splits/unit07_roles.csv').open(newline='') as stream:
        roles = list(csv.DictReader(stream))
    require(roles == [{k: str(r[k]) for k in ('row_id', 'cohort', 'college_id', 'role')} for r in rows], 'committed roles differ')
    return rows


def sigmoid(eta):
    eta = np.asarray(eta, dtype=float)
    out = np.empty_like(eta)
    positive = eta >= 0
    out[positive] = 1 / (1 + np.exp(-eta[positive]))
    e = np.exp(eta[~positive])
    out[~positive] = e / (1 + e)
    return out


def weighted_qr(x, residual, weights, limits):
    require(np.all(np.isfinite(weights)) and np.all(weights > 0), 'separation or nonpositive weight')
    a = x * np.sqrt(weights)[:, None]
    q, r = np.linalg.qr(a, mode='reduced')
    require(np.min(np.abs(np.diag(r))) > limits['rank_relative_tolerance'] * np.max(np.abs(np.diag(r))), 'rank deficiency')
    condition = float(np.linalg.norm(r, np.inf) * np.linalg.norm(np.linalg.solve(r, np.eye(r.shape[0])), np.inf))
    require(np.isfinite(condition) and condition <= limits['condition_r_inf_limit'], 'conditioning failure')
    step = np.linalg.solve(r, q.T @ (residual / np.sqrt(weights)))
    return step, condition


def logistic_fit(x, y, spec):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    limits = spec['solver']
    require(x.ndim == 2 and len(y) == len(x) and len(x) > x.shape[1], 'fit dimensions')
    require(np.all(np.isfinite(x)) and set(y) == {0., 1.}, 'fit needs finite predictors and both classes')
    beta = np.zeros(x.shape[1])
    beta[0] = math.log(float(np.mean(y)) / (1 - float(np.mean(y))))
    def objective(value):
        eta = x @ value
        return math.fsum(np.logaddexp(0, np.where(y == 1, -eta, eta)).tolist())
    for iteration in range(limits['maximum_iterations'] + 1):
        require(np.max(np.abs(beta)) <= limits['maximum_abs_coefficient'], 'separation or coefficient boundary')
        eta = x @ beta
        probability = sigmoid(eta)
        residual, weights = y - probability, probability * (1 - probability)
        step, condition = weighted_qr(x, residual, weights, limits)
        score = float(np.max(np.abs(x.T @ residual)))
        step_inf = float(np.max(np.abs(step)))
        if score <= limits['gradient_inf_tolerance'] and step_inf <= limits['newton_step_inf_tolerance']:
            return beta, dict(n_fit=len(y), events=int(y.sum()), p=x.shape[1], rank=x.shape[1],
                              nll=objective(beta), score_inf=score, newton_step_inf=step_inf,
                              condition_r_inf=condition, min_weight=float(weights.min()), finite=1)
        current, fraction = objective(beta), 1.
        while fraction >= limits['minimum_step'] and objective(beta + fraction * step) > current:
            fraction *= limits['backtracking_factor']
        require(fraction >= limits['minimum_step'], 'backtracking failed')
        beta += fraction * step
    raise ValueError('separation or logistic nonconvergence')


def matrix(rows, model, transforms, spec, evaluation=False):
    def z(row, feature):
        mean, sd = transforms[feature]
        return (row[feature] - mean) / sd
    output = []
    for row in rows:
        offer = 0 if evaluation else row['offer']
        if model == 'entry_logistic':
            values = [1, z(row, 'prior_grade'), offer]
        else:
            values = [1, row['part_time'], row['first_gen'], z(row, 'age')]
            values += [int(row['gender'] == level) for level in spec['gender_levels'][1:]]
            values += [int(row['program_area'] == level) for level in spec['program_levels'][1:]]
            values += [z(row, name) for name in ('prior_grade', 'wellbeing_score', 'stress_score')]
            values.append(offer)
        output.append(values)
    return np.array(output, dtype=float)


def probability_scores(y, probabilities):
    y = np.asarray(y, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    require(len(y) == len(p) and len(y) and np.all(np.isfinite(p)) and np.all((0 <= p) & (p <= 1)), 'invalid probability')
    require(set(y) <= {0, 1}, 'invalid binary score label')
    clipped = np.clip(p, 1e-12, 1 - 1e-12)
    pos, neg = p[y == 1], p[y == 0]
    twice = int(2 * np.count_nonzero(pos[:, None] > neg) + np.count_nonzero(pos[:, None] == neg))
    denominator = int(2 * len(pos) * len(neg))
    return dict(n=len(y), events=len(pos), nonevents=len(neg), auc_twice_wins=twice if denominator else '',
                auc_denominator=denominator, auc=twice / denominator if denominator else '',
                auc_reason='' if denominator else 'one_class', brier=float(np.mean((p-y)**2)),
                log_loss=float(np.mean(np.where(y == 1, -np.log(clipped), -np.log1p(-clipped)))),
                clipped_n=int(np.count_nonzero(clipped != p)), mean_probability=float(np.mean(p)))


def actions(records, model):
    ordered = sorted(records, key=lambda r: (-r['probability'], r['row_id']))
    rank = {r['row_id']: i+1 for i, r in enumerate(ordered)}
    decisions, ids = [], []
    n = len(records)
    for rule in ('threshold_020', 'capacity_300', 'all', 'none'):
        chosen = {r['row_id'] for r in records if rule == 'all' or
                  (rule == 'threshold_020' and r['probability'] >= .2) or
                  (rule == 'capacity_300' and rank[r['row_id']] <= 300)}
        tp = sum(r['y'] == 1 and r['row_id'] in chosen for r in records)
        fp = len(chosen) - tp
        fn = sum(r['y'] for r in records) - tp
        tn = n - tp - fp - fn
        decisions.append(dict(model_id=model, rule=rule, n=n, selected=len(chosen), tp=tp, fp=fp, tn=tn, fn=fn,
                              net_benefit=(tp - .25*fp)/n, utility_total=4*tp-fp))
        if rule in ('threshold_020', 'capacity_300'):
            ids.extend(dict(model_id=model, rule=rule, row_id=r['row_id'], rank=rank[r['row_id']],
                            selected=int(r['row_id'] in chosen)) for r in records)
    return decisions, ids, rank


def calculate(root=RESOURCE):
    spec = json.loads((root/'model-specs/unit07.json').read_text())
    require(spec['predictor_allowlist'] == ['part_time', 'first_gen', 'age', 'gender', 'program_area',
                                         'prior_grade', 'wellbeing_score', 'stress_score', 'offer'], 'predictor allowlist')
    rows = joined_rows(root)
    fit = [r for r in rows if r['role'] == 'development']
    test = [r for r in rows if r['role'] == 'temporal_final']
    controls = [r for r in fit if r['offer'] == 0]
    require(fit and test and controls and len(test) >= 300, 'population or capacity')
    base = sum(r['y'] for r in controls)/len(controls)
    cuts = np.quantile([r['prior_grade'] for r in controls], [.25, .5, .75], method='linear')
    cells = []
    for part in (0, 1):
        for index in range(4):
            group = [r for r in controls if r['part_time'] == part and int(np.searchsorted(cuts, r['prior_grade'], side='left')) == index]
            events = sum(r['y'] for r in group)
            cells.append(dict(part_time=part, grade_bin=index+1, n=len(group), events=events,
                              probability=events/len(group) if group else base, fallback=int(not group)))
    result = dict(rows=rows, predictions=[], coefficients=[], diagnostics=[], transforms=[], fit_rows=[],
                  scores=[], decisions=[], action_ids=[], table_cells=cells,
                  cutpoints=[dict(quantile=q, value=float(v)) for q,v in zip([.25,.5,.75],cuts)], examples=[])
    for model in MODELS:
        membership = controls if model.startswith('control_') else fit
        result['fit_rows'].extend(dict(model_id=model, row_id=r['row_id']) for r in membership)
        if model == 'control_mean':
            probabilities = np.full(len(test), base)
            etas = np.full(len(test), math.log(base/(1-base)))
        elif model == 'control_table':
            probabilities = np.array([cells[4*r['part_time']+int(np.searchsorted(cuts,r['prior_grade'],side='left'))]['probability'] for r in test])
            etas = np.log(probabilities/(1-probabilities))
        else:
            processing = {}
            for feature in spec['standardize'][model]:
                values = np.array([r[feature] for r in fit])
                mean, sd = float(np.mean(values)), float(np.std(values, ddof=0))
                require(np.isfinite(sd) and sd > 0, 'zero or invalid training SD')
                processing[feature] = (mean, sd)
                result['transforms'].append(dict(model_id=model, feature=feature, n_fit=len(fit), mean=mean, sd=sd))
            x, y = matrix(fit, model, processing, spec), np.array([r['y'] for r in fit])
            beta, diagnostic = logistic_fit(x, y, spec)
            result['diagnostics'].append(dict(model_id=model, **diagnostic))
            result['coefficients'].extend(dict(model_id=model, term=term, coefficient=float(value))
                                          for term, value in zip(spec['feature_order'][model], beta))
            etas = matrix(test, model, processing, spec, evaluation=True) @ beta
            probabilities = sigmoid(etas)
        records = [dict(model_id=model, row_id=r['row_id'], y=r['y'], offer_for_prediction=0,
                        probability=float(p), eta=float(eta), brier_loss=float((p-r['y'])**2),
                        log_loss=float(-math.log(min(1-1e-12,max(1e-12,p)) if r['y'] else 1-min(1-1e-12,max(1e-12,p)))))
                   for r,p,eta in zip(test,probabilities,etas)]
        result['predictions'].extend(records)
        result['scores'].append(dict(model_id=model, **probability_scores([r['y'] for r in test],probabilities)))
        decisions, ids, ranks = actions(records, model)
        result['decisions'].extend(decisions); result['action_ids'].extend(ids)
        if model == 'full_logistic':
            for row, record in zip(test[:3], records[:3]):
                result['examples'].append(dict(row_id=row['row_id'], y=row['y'], wellbeing_score=row['wellbeing_score'],
                    stress_score=row['stress_score'], eta=record['eta'], probability=record['probability'],
                    brier_loss=record['brier_loss'], log_loss=record['log_loss'],
                    rank=ranks[row['row_id']], threshold_selected=int(record['probability'] >= .2),
                    capacity_selected=int(ranks[row['row_id']] <= 300)))
    summary = dict(anchor_pool_n=sum(r['cohort']=='anchor' for r in rows),
        anchor_unknown_n=sum(r['cohort']=='anchor' and r['role']=='excluded_unknown' for r in rows),
        anchor_unscorable_n=sum(r['cohort']=='anchor' and r['role']=='excluded_unscorable_items' for r in rows),
        development_n=len(fit), development_events=sum(r['y'] for r in fit),
        development_offer0_n=len(controls), development_offer0_events=sum(r['y'] for r in controls),
        development_offer1_n=sum(r['offer']==1 for r in fit), development_offer1_events=sum(r['y'] for r in fit if r['offer']==1),
        temporal_n=len(test), temporal_events=sum(r['y'] for r in test),
        temporal_unscorable_n=sum(r['cohort']=='cohort2' and r['role']=='excluded_unscorable_items' for r in rows),
        development_transfer_n=sum(r['exit_type']=='transfer' for r in fit),
        temporal_transfer_n=sum(r['exit_type']=='transfer' for r in test),
        candidate_parameters_n=13, full_coefficients_n=14, events_per_candidate_parameter=sum(r['y'] for r in fit)/13,
        control_prevalence=base, temporal_prevalence=sum(r['y'] for r in test)/len(test),
        development_partial_wb_n=sum(r['wellbeing_observed']<5 for r in fit),
        development_partial_st_n=sum(r['stress_observed']<5 for r in fit),
        temporal_partial_wb_n=sum(r['wellbeing_observed']<5 for r in test),
        temporal_partial_st_n=sum(r['stress_observed']<5 for r in test))
    result['summary'] = [dict(name=k, value=v) for k,v in sorted(summary.items())]
    return result


def emit(result, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, records in result.items():
        if name == 'rows':
            continue
        with (out/(name+'.csv')).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
            writer.writeheader()
            for record in records:
                writer.writerow({k: format(v,'.17g') if isinstance(v,float) else v for k,v in record.items()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=RESOURCE)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = calculate(args.root)
        emit(result, args.outdir)
        print('UNIT07_NUMPY_OK logistic_fits=2 predictions=' + str(len(result['predictions'])) + ' numpy=' + np.__version__)
    except (ValueError, KeyError, OSError, np.linalg.LinAlgError) as error:
        sys.exit('UNIT07_NUMPY_ERROR ' + str(error))
