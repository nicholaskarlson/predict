#!/usr/bin/env python3
"""Independent NumPy fit, imputation, prediction and score route for Unit 11."""
import argparse
import csv
import json
import math
from pathlib import Path
import numpy as np

METHODS = ['frame_only', 'survey_fallback', 'imputed', 'imputed_indicators']
GENDERS = ['female', 'male', 'another']
PROGRAMS = ['Arts and Humanities', 'Social Sciences', 'STEM', 'Health', 'Business']
ITEMS = [z + str(i) for z in ['wb', 'st'] for i in range(1, 6)]
FRAME = ['part_time', 'first_gen', 'age', 'gender', 'program_area', 'prior_grade']
SCALES = ['wellbeing_score', 'stress_score']
INDICATORS = ['survey_absent', 'wb_item_missing', 'st_item_missing']
PATTERNS = ['survey_absent', 'scores_complete_items', 'scores_partial_wb', 'scores_partial_st',
            'scores_partial_both', 'wb_unscorable', 'st_unscorable', 'both_unscorable']
TERMS = ['intercept', 'part_time', 'first_gen', 'z_age', 'gender_male', 'gender_another',
         'program_Social_Sciences', 'program_STEM', 'program_Health', 'program_Business', 'z_prior_grade']


def need(ok, message):
    if not ok:
        raise ValueError(message)


def chars(path):
    with Path(path).open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        need(reader.fieldnames and len(reader.fieldnames) == len(set(reader.fieldnames)), 'CSV headers')
        rows = list(reader)
    need(all(None not in r for r in rows), 'CSV extra columns')
    return rows


def keyed(path, field='student_id'):
    rows = chars(path)
    need(all(r[field] for r in rows) and len({r[field] for r in rows}) == len(rows), 'duplicate/missing key')
    return {r[field]: r for r in rows}


def profile(record):
    p = dict(record)
    need(p['gender'] in GENDERS and p['program_area'] in PROGRAMS, 'unsupported frame category')
    for key in ['part_time', 'first_gen', 'responded_baseline']:
        need(str(p[key]) in {'0', '1'}, 'binary frame/response flag')
        p[key] = int(p[key])
    for key, lo, hi in [('age', 16, 80), ('prior_grade', 0, 4)]:
        p[key] = float(p[key])
        need(math.isfinite(p[key]) and lo <= p[key] <= hi, 'invalid frame value ' + key)
    for prefix, score in zip(['wb', 'st'], SCALES):
        values = []
        for i in range(1, 6):
            key = prefix + str(i)
            raw = p.get(key)
            if raw is None or raw == '':
                p[key] = None
                continue
            need(str(raw) in {'1', '2', '3', '4', '5'}, 'invalid item')
            p[key] = int(raw)
            values.append(6 - p[key] if i == 4 else p[key])
        p[prefix + '_nobs'] = len(values)
        p[score] = sum(values) / len(values) if len(values) >= 3 else None
    need(p['responded_baseline'] == 1 or p['wb_nobs'] + p['st_nobs'] == 0,
         'nonrespondent cannot supply baseline items')
    p['survey_absent'] = 1 - p['responded_baseline']
    p['wb_item_missing'] = int(p['wb_nobs'] < 5)
    p['st_item_missing'] = int(p['st_nobs'] < 5)
    if p['survey_absent']:
        pat = 'survey_absent'
    elif p[SCALES[0]] is None and p[SCALES[1]] is None:
        pat = 'both_unscorable'
    elif p[SCALES[0]] is None:
        pat = 'wb_unscorable'
    elif p[SCALES[1]] is None:
        pat = 'st_unscorable'
    else:
        pat = ['scores_complete_items', 'scores_partial_st', 'scores_partial_wb',
               'scores_partial_both'][2 * p['wb_item_missing'] + p['st_item_missing']]
    p['pattern'] = pat
    return p


def load_records(root):
    root = Path(root)
    old = keyed(root / 'splits/unit09_roles.csv', 'row_id')
    result = []
    for cohort in ['anchor', 'cohort2']:
        folder = root / 'source/tonmick-v1.0.0/data' / cohort
        frame = keyed(folder / 'participants.csv')
        survey = keyed(folder / 'baseline.csv')
        obs = keyed(root / 'source/tonmick-observed-v1.1.0/data' / (cohort + '_observed_registry.csv'))
        need(set(obs) == set(frame) and set(survey) == {k for k, r in frame.items() if r['responded_baseline'] == '1'}, 'observed invitee/survey join')
        for key in sorted(frame):
            f, o = frame[key], obs[key]
            need(o['cohort'] == cohort and o['responded_baseline'] == f['responded_baseline'], 'release cohort/response join')
            s = survey.get(key, {})
            need(not s or s['college_id'] == f['college_id'], 'survey college mismatch')
            row = {'row_id': key, 'cohort': cohort, 'college_id': f['college_id'], 'region': f['region'],
                   'college_group': f.get('college_group', 'anchor'), **{k: f[k] for k in FRAME},
                   'responded_baseline': f['responded_baseline'], **{k: s.get(k, '') for k in ITEMS}}
            row = profile(row)
            need(o['offer'] in {'0', '1'} and o['outcome_known'] in {'0', '1'}, 'observation flags')
            offer, known = int(o['offer']), int(o['outcome_known'])
            need((known == 0) == (o['withdrawal_label'] == ''), 'unknown label must be blank')
            y = int(o['withdrawal_label']) if known else None
            need(not known or y in {0, 1} and y == int(o['exit_type'] == 'withdrawal'), 'label/event mismatch')
            need(known or o['exit_type'] == 'consent_withdrawn', 'unknown event rule')
            need(not row['survey_absent'] or offer == 0 and known == 1, 'nonresponse observation policy')
            need(cohort == 'anchor' or offer == 0, 'later offer policy')
            if cohort == 'anchor':
                role = 'excluded_offered' if offer else 'excluded_unknown' if not known else 'development'
            else:
                need(row['college_group'] in {'continuing', 'new'}, 'later college group')
                role = 'temporal_' + row['college_group']
            row.update(offer=offer, outcome_known=known, y=y, exit_type=o['exit_type'],
                       observation_source=o['observation_source'], role=role,
                       legacy_role=old.get(key, {}).get('assignment', 'not_in_prior_respondent_analysis'))
            result.append(row)
    result.sort(key=lambda r: r['row_id'])
    need(len({r['row_id'] for r in result}) == len(result), 'cohort key overlap')
    return result


def processing(rows, keys):
    values = np.array([[r[k] for k in keys] for r in rows], dtype=float)
    mean = values.mean(axis=0)
    sd = np.sqrt(np.mean((values - mean) ** 2, axis=0))
    need(np.isfinite(values).all() and np.isfinite(sd).all() and (sd > 0).all(), 'invalid/zero SD')
    return {key: (float(mean[j]), float(sd[j])) for j, key in enumerate(keys)}


def matrix(rows, tr, scales=False, indicators=False):
    cols = [np.ones(len(rows)), np.array([r['part_time'] for r in rows]), np.array([r['first_gen'] for r in rows])]
    def z(key):
        return (np.array([r[key] for r in rows], dtype=float) - tr[key][0]) / tr[key][1]
    cols.append(z('age'))
    for gender in GENDERS[1:]:
        cols.append(np.array([r['gender'] == gender for r in rows], dtype=float))
    for program in PROGRAMS[1:]:
        cols.append(np.array([r['program_area'] == program for r in rows], dtype=float))
    cols.append(z('prior_grade'))
    terms = list(TERMS)
    if scales:
        for key in SCALES:
            cols.append(z(key)); terms.append('z_' + key)
    if indicators:
        for key in INDICATORS:
            cols.append(np.array([r[key] for r in rows], dtype=float)); terms.append(key)
    x = np.column_stack(cols)
    need(np.isfinite(x).all(), 'nonfinite design')
    return x, terms


def solve_qr(x, target):
    need(x.shape[0] > x.shape[1], 'insufficient fit rows')
    q, r = np.linalg.qr(x, mode='reduced')
    diagonal = np.abs(np.diag(r))
    need(diagonal.min() > 1e-10 * diagonal.max(), 'rank failure')
    condition = float(np.linalg.norm(r, np.inf) * np.linalg.norm(np.linalg.solve(r, np.eye(len(r))), np.inf))
    need(math.isfinite(condition) and condition <= 1e5, 'conditioning failure')
    return np.linalg.solve(r, q.T @ target), condition


def logistic(x, y):
    y = np.array(y, dtype=float)
    need(set(y) == {0., 1.}, 'two training classes required')
    beta = np.zeros(x.shape[1]); beta[0] = math.log(float(y.sum() / (len(y) - y.sum())))
    for iteration in range(101):
        eta = x @ beta
        p = np.exp(-np.logaddexp(0, -eta))
        weights = p * (1 - p)
        need((weights > 0).all() and np.isfinite(weights).all(), 'separation/nonpositive weight')
        penalty = np.eye(x.shape[1])[1:]  # lambda=1; intercept unpenalized
        a = np.vstack((x * np.sqrt(weights[:, None]), penalty))
        target = np.concatenate(((y - p) / np.sqrt(weights), -beta[1:]))
        step, condition = solve_qr(a, target)
        g = x.T @ (y - p) - np.r_[0., beta[1:]]
        score = float(np.abs(g).max()); step_inf = float(np.abs(step).max())
        nll = float(np.logaddexp(0, np.where(y == 1, -eta, eta)).sum() + .5 * (beta[1:] @ beta[1:]))
        if score <= 1e-9 and step_inf <= 1e-9:
            need(np.isfinite(beta).all() and float(np.abs(beta).max()) <= 30, 'nonfinite/excess coefficient')
            return beta, {'n_fit': len(y), 'events': int(y.sum()), 'parameters': len(beta), 'rank': len(beta),
                          'objective': nll, 'score_inf': score, 'step_inf': step_inf,
                          'condition_r_inf': condition, 'minimum_weight': float(weights.min())}
        need(iteration < 100, 'logistic nonconvergence')
        alpha = 1.
        for _ in range(40):
            candidate = beta + alpha * step
            signed = np.where(y == 1, -(x @ candidate), x @ candidate)
            value = float(np.logaddexp(0, signed).sum() + .5 * (candidate[1:] @ candidate[1:]))
            if value <= nll - 1e-4 * alpha * float(g @ step) + 1e-12:
                beta = candidate; break
            alpha *= .5
        else:
            raise ValueError('logistic backtracking failed')
    raise ValueError('logistic nonconvergence')


def fit_pipelines(rows):
    train = [r for r in rows if r['role'] == 'development']
    need(len(train) == 984 and all(r['cohort'] == 'anchor' and r['offer'] == 0 and r['y'] in {0, 1} for r in train), 'development membership')
    coefficients, transforms, diagnostics, fit_rows = [], [], [], []
    def store_tr(name, tr, n):
        for key, (mean, sd) in tr.items():
            transforms.append(dict(component=name, feature=key, n_fit=n, mean=mean, sd=sd))
    def membership(name, records):
        fit_rows.extend(dict(component=name, row_id=r['row_id']) for r in records)
    imputers = {}
    for key in SCALES:
        selected = [r for r in train if r[key] is not None]
        tr = processing(selected, ['age', 'prior_grade']); x, terms = matrix(selected, tr)
        target = np.array([r[key] for r in selected]); beta, condition = solve_qr(x, target)
        resid = target - x @ beta
        diagnostics.append(dict(component='imputer_' + key, n_fit=len(selected), events='', parameters=len(beta),
                                rank=len(beta), objective=float(resid @ resid), score_inf=float(np.abs(x.T @ resid).max()),
                                step_inf='', condition_r_inf=condition, minimum_weight=''))
        name = 'imputer_' + key
        store_tr(name, tr, len(selected)); membership(name, selected)
        coefficients.extend(dict(component=name, term=t, coefficient=float(v)) for t, v in zip(terms, beta))
        imputers[key] = {'transforms': tr, 'terms': terms, 'coefficients': beta.tolist()}
    completed = complete(train, imputers)
    models = {}
    for method in METHODS:
        use_scales, indicators = method != 'frame_only', method == 'imputed_indicators'
        selected = [r for r in train if all(r[k] is not None for k in SCALES)] if method == 'survey_fallback' else completed if method.startswith('imputed') else train
        keys = ['age', 'prior_grade'] + (SCALES if use_scales else [])
        tr = processing(selected, keys); x, terms = matrix(selected, tr, use_scales, indicators)
        beta, diag = logistic(x, [r['y'] for r in selected])
        diagnostics.append(dict(component=method, **diag)); store_tr(method, tr, len(selected)); membership(method, selected)
        coefficients.extend(dict(component=method, term=t, coefficient=float(v)) for t, v in zip(terms, beta))
        models[method] = {'transforms': tr, 'terms': terms, 'coefficients': beta.tolist()}
    counts = {pat: sum(r['pattern'] == pat for r in train) for pat in PATTERNS}
    bundle = {'schema_version': 1, 'unit': 11, 'methods': METHODS, 'imputers': imputers, 'models': models,
              'training_pattern_counts': counts, 'prediction_time': 'enrollment', 'offer_policy': 'no_offer',
              'outcomes_required_for_prediction': False}
    return bundle, dict(coefficients=coefficients, transforms=transforms, diagnostics=diagnostics, fit_rows=fit_rows)


def complete(rows, imputers):
    result = [dict(r) for r in rows]
    for key in SCALES:
        model = imputers[key]
        missing = [i for i, r in enumerate(result) if r[key] is None]
        if missing:
            x, terms = matrix([result[i] for i in missing], model['transforms'])
            need(terms == model['terms'], 'imputer term order')
            values = np.clip(x @ np.array(model['coefficients']), 1, 5)
            for i, value in zip(missing, values):
                result[i][key] = float(value)
    for old, new in zip(rows, result):
        for prefix, key in zip(['wb', 'st'], SCALES):
            new[prefix + '_imputed'] = int(old[key] is None)
    return result


def predict_profiles(profiles, bundle):
    predictions = []
    for record in profiles:
        raw = profile(record)
        finished = complete([raw], bundle['imputers'])[0]
        for method in METHODS:
            route = 'frame_only' if method == 'survey_fallback' and any(raw[k] is None for k in SCALES) else method
            model = bundle['models'][route]
            using = finished if method.startswith('imputed') else raw
            x, terms = matrix([using], model['transforms'], route != 'frame_only', route == 'imputed_indicators')
            need(terms == model['terms'], 'prediction term order')
            eta = float((x @ np.array(model['coefficients']))[0])
            probability = float(np.exp(-np.logaddexp(0, -eta)))
            count = bundle['training_pattern_counts'][raw['pattern']]
            predictions.append(dict(method_id=method, record_id=record.get('record_id', record.get('row_id', '')),
                 route=route, pattern=raw['pattern'], training_pattern_n=count,
                 support_reason='' if count else 'unseen_missingness_pattern',
                 wellbeing_score=finished['wellbeing_score'], stress_score=finished['stress_score'],
                 wb_imputed=finished['wb_imputed'], st_imputed=finished['st_imputed'], eta=eta, probability=probability))
    return predictions


def score(y, p, n_predict):
    need(all(v in {0, 1} for v in y), "score labels must be observed binary values")
    y, p = np.array(y, dtype=int), np.array(p, dtype=float)
    need(len(y) == len(p) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'invalid scoring inputs')
    n = len(y); positive = p[y == 1]; negative = p[y == 0]
    den = 2 * len(positive) * len(negative)
    pairs = int(2 * (positive[:, None] > negative).sum() + (positive[:, None] == negative).sum()) if den else ''
    return dict(n_predict=n_predict, n_scored=n, unknown_n=n_predict-n, events=len(positive), nonevents=len(negative),
                auc_pairs2=pairs, auc_denominator=den, auc=pairs/den if den else '',
                auc_reason='' if den else 'empty' if not n else 'one_class',
                brier=float(np.mean((p-y)**2)) if n else '',
                mean_probability=float(p.mean()) if n else '')


def calculate(root):
    root = Path(root); rows = load_records(root)
    bundle, outputs = fit_pipelines(rows)
    later = [r for r in rows if r['cohort'] == 'cohort2']
    profiles = predict_profiles(later, bundle)
    by = {(x['method_id'], x['record_id']): x for x in profiles}
    predictions, scores, coverage = [], [], []
    completed = complete(rows, bundle['imputers'])
    outputs['completed'] = [dict(row_id=r['row_id'], role=r['role'], pattern=r['pattern'],
        wellbeing_score=r['wellbeing_score'], stress_score=r['stress_score'],
        wb_imputed=r['wb_imputed'], st_imputed=r['st_imputed']) for r in completed]
    for method in METHODS:
        for row in later:
            p = by[method, row['row_id']]
            predictions.append(dict(method_id=method, row_id=row['row_id'], role=row['role'], college_id=row['college_id'],
                legacy_role=row['legacy_role'], responded_baseline=row['responded_baseline'], pattern=row['pattern'],
                y='' if row['y'] is None else row['y'], outcome_known=row['outcome_known'], route=p['route'], probability=p['probability'], eta=p['eta']))
        for role in ['temporal_continuing', 'temporal_new']:
            selected = [r for r in later if r['role'] == role]
            for pat in ['all'] + PATTERNS:
                population = [r for r in selected if pat == 'all' or r['pattern'] == pat]
                known = [r for r in population if r['outcome_known']]
                values = [by[method, r['row_id']]['probability'] for r in known]
                scores.append(dict(method_id=method, role=role, pattern=pat, **score([r['y'] for r in known], values, len(population))))
            coverage.append(dict(method_id=method, role=role, invited=len(selected), predictions=len(selected),
                labels_known=sum(r['outcome_known'] for r in selected), labels_unknown=sum(not r['outcome_known'] for r in selected),
                survey_route_n=sum(by[method,r['row_id']]['route']=='survey_fallback' for r in selected),
                frame_fallback_n=sum(method=='survey_fallback' and by[method,r['row_id']]['route']=='frame_only' for r in selected),
                nonrespondents=sum(r['survey_absent'] for r in selected),
                partial_item_respondents=sum(not r['survey_absent'] and (r['wb_item_missing'] or r['st_item_missing']) for r in selected)))
    outputs['predictions'] = predictions; outputs['scores'] = scores; outputs['coverage'] = coverage
    outputs['new_predictions'] = predict_profiles(json.loads((root/'deployment/unit11_new_profiles.json').read_text()), bundle)
    outputs['bundle'] = bundle
    outputs['summary'] = [dict(name=k, value=v) for k, v in sorted({
        'anchor_invited':1600, 'anchor_offered_excluded':sum(r['role']=='excluded_offered' for r in rows),
        'anchor_unknown_no_offer_excluded':sum(r['role']=='excluded_unknown' for r in rows),
        'development_n':sum(r['role']=='development' for r in rows),
        'development_respondents':sum(r['role']=='development' and r['responded_baseline']==1 for r in rows),
        'development_nonrespondents':sum(r['role']=='development' and r['survey_absent']==1 for r in rows),
        'later_invited':len(later), 'later_nonrespondents':sum(r['survey_absent'] for r in later),
        'later_labels_known':sum(r['outcome_known'] for r in later),
        'training_unscorable_respondents':sum(r['role']=='development' and not r['survey_absent'] and any(r[k] is None for k in SCALES) for r in rows),
        'new_world_random_draws':0}.items())]
    return outputs


def emit(outputs, out):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        if name == 'bundle':
            (out/'bundle.json').write_text(json.dumps(rows, indent=2, sort_keys=True)+'\n'); continue
        with (out/(name+'.csv')).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); writer.writeheader()
            for row in rows:
                writer.writerow({k: format(v, '.17g') if isinstance(v,float) else '' if v is None else v for k,v in row.items()})
    (out/'environment.txt').write_text('NumPy '+np.__version__+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', type=Path, required=True); p.add_argument('--outdir', type=Path, required=True)
    args = p.parse_args(); emit(calculate(args.root), args.outdir)
    print('UNIT11_NUMPY_OK methods=4 imputers=2 later_predictions=14368 outcome_free_new_profiles=3')
