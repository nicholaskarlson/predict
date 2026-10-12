# Provenance

- **World:** Tonmick college system, World v1.0.0, frozen October 4, 2026.
- **Made by:** `python3 generator/generate_world.py .` then `python3 generator/dictionary.py DICTIONARY.csv` and `python3 generator/checks.py .`
- **Software:** Python 3.13.16, NumPy 2.5.3, pandas 3.0.5, on Linux.
- **Randomness:** root seed 20261003; one PCG64 stream per block, `SeedSequence(20261003, spawn_key=(block_id,))`. Block IDs are listed in `generator/spec.py` and `MANIFEST.json` and never change. Reserved for later versions: esm_substudy (101), literature_layer (102).
- **Calibrated intercepts** (solved by bisection so targets hold in the generated population; also in `hidden/calibrated_intercepts.json`): withdrawal log-hazard intercept -2.551685; baseline response 1.481473 (cohort 1) and 1.564604 (cohort 2); uptake 0.693437; follow-up response 12 months 1.386732, 3 months 2.415246, 6 months 1.800678.
- **Checks before freezing:** every check in `CHECKS.md` passed. A second run gave byte-identical files, and a run with an extra block of 1,000,000 draws inserted after the colleges block also gave byte-identical files, which confirms the append-only rule.
- **Not evidence:** files in `hidden/` are simulator truth. A matching hash confirms file identity only; it cannot validate recruitment, measurement, causation or transport.
