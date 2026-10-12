# Predictive Reasoning with Statistics: numerical companion

This is the data and code companion for Nicholas Elliott Karlson's
*Predictive Reasoning with Statistics: Validation, Calibration, and Reproducible
Prediction in Psychology and the Social Sciences* (NEKpress Research:
Proof-First Guides, Book 7).

All records, colleges and findings are fictional. The files let you reproduce
the book's statistics and run its supplied prediction rules. The book's
manuscript, problems, worked solutions, figures, typesetting and PDF-building
tools belong to the separate private authoring project.

## Start with one case

Download this repository's ZIP, or clone the companion, and enter its folder.
Use Python 3.10–3.12, NumPy 1.26.4 and, for the third route, base R with its
standard `methods` and `stats` packages. No R add-on packages, document renderer
or Tonmick World clone is needed.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-unit01.txt
.venv/bin/python scripts/verify_companion.py --boundary-only
.venv/bin/python scripts/check_unit01.py --numeric
.venv/bin/python scripts/check_unit01.py --base-r
```

The first numerical command runs NumPy and a separate standard-library Python
implementation. The second reuses their current verified outputs and adds an
independent base-R calculation. Each route reads the original inputs and model
specifications; it does not read another route's answers. The supplied reference
tables preserve full precision behind the book's rounded displays.

Generated tables and their current receipt are inside `build/unit01-audit/`.
Open the path named by `output_folder` in `numpy.json`, `stdlib.json` or
`base_r.json`. `current.json` records the completed routes. Keep the output,
log and edition identity together when reporting a reproduction.

## Follow the book

The chapter commands `scripts/check_unit02.py` through `check_unit12.py` work
the same way. The underlying files remain in
`open-materials/predictive-reasoning-with-statistics/`:

| Folder | Contents |
| --- | --- |
| `data`, `dictionaries` | Frozen observed extracts and variable definitions |
| `splits` | Original fitting, evaluation, resampling and permutation assignments |
| `model-specs` | Targets, permitted predictors, procedures, transformations and coefficients |
| `predictions`, `expected-results` | Individual forecasts, losses, counts and full-precision reference values |
| `source` | Independent NumPy, scalar Python and base-R analyzers; declared source snapshots |
| `deployment` | Frozen model bundles and outcome-free example records |
| `provenance` | Original scientific source identities and observation contracts |

The twelve cases cover a simple forecast, targets and loss, optimism, validation
design, leakage, regularization, binary risk, calibration and outcome intervals,
changing populations, subgroup performance, missing predictors, and prediction
versus action evidence. Additional commands reproduce the two worked extensions
and the sealed-envelope addendum:

```bash
.venv/bin/python scripts/check_method_additions.py --scope uncertainty --base-r
.venv/bin/python scripts/check_method_additions.py --scope planning --base-r
.venv/bin/python scripts/check_sealed_envelope.py --base-r
```

For all fifteen cases:

```bash
make PYTHON=.venv/bin/python verify
make PYTHON=.venv/bin/python r-audit
```

These are bounded, resumable checks. Progress appears every twenty seconds while
an analyzer is running. Unit 6 allows 600 seconds per route; other cases allow
180 seconds. Rerun the same command after an interruption. Only successful
receipts with matching sources, runtime, logs and output hashes can resume;
modified, incomplete or failed results are recomputed. The original comparison
functions and numerical thresholds are retained. Solver iterations may differ;
their original convergence conditions remain checked. No tolerance is widened
to obtain agreement.

To print the current table directories for one case:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
for record in json.loads(Path('build/unit07-audit/current.json').read_text())['records']:
    print(record['route'], record['output_folder'])
PY
```

## Predict one new record

The Unit 11 and 12 predictor scripts can be copied with their specified model
bundle and outcome-free record to a separate folder. They never need a label
for that record. For example, from this repository root:

```bash
.venv/bin/python open-materials/predictive-reasoning-with-statistics/source/unit12_predict.py \
  --bundle open-materials/predictive-reasoning-with-statistics/deployment/unit12_frame_bundle.json \
  --record open-materials/predictive-reasoning-with-statistics/deployment/unit12_new_student.json \
  --output build/unit12-new-prediction.json
```

Use the declared predictors and inspect support warnings. An unknown outcome
remains unknown; a missing predictor follows its specified deployment rule.

## Frozen World and the envelope

The numerical inputs are copied byte for byte from the verified Book 7 source
and original Tonmick World. The World baseline is
`b59adb629b5a5a9ff261017ddb301bc4b2e0f9a3`; its observed registry release is
`derived/v1.1.0`. The companion is self-contained: its ordinary unit analyzers
read only their declared observed snapshots. No World generator or unit
preparation script is included or run. Existing assignments are retained.

The separately labeled envelope reads 100,000 original deployment records
under `source/sealed-envelope/`. Opening those files reveals its outcomes and
simulator reference. They are public teaching material, not an access-controlled
test set. Stop before the reveal in the book if you want to record expectations.
The enrollment frame model is compatible; the survey model is not. Its fixed
comparator is the original development prevalence, 177/984.
`risk12_no_transfer` removes transfer and is a different target from withdrawal
before transfer. Its reported gaps are descriptive simulator diagnostics, not
primary-target calibration error or expected-Brier excess. The original local
freeze acknowledges previously known results; it is not blind preregistration.

The shared World project is `https://github.com/nicholaskarlson/tonmick-world`.
Its public visibility is managed separately; running this companion does not
depend on that repository being accessible. If the World is public, `hidden/`
is a teaching boundary, not secrecy or evidence about real people.

## Identity, rights and limits

`EXPORT_MANIFEST.json` identifies the exact export from private book baseline
`353494d7699708979af07b56707b97d59518a8ca`; it carries no private Git history.
`SHA256SUMS` covers every distributed file. The public repository should begin
with new history. Cite its actual companion commit when available and retain
these files rather than substituting a later development version.

Data, dictionaries, numerical specifications and provenance: CC BY 4.0.
Original code: MIT. See `LICENSES.md` and `LICENSE-CODE`. The manuscript and
branding have separate rights. Numerical agreement checks specified arithmetic;
it cannot establish transport, measurement validity, fairness, causation or
performance in real colleges. Report errors with the case, command, runtime,
commit and complete error log; do not post credentials or personal data.
