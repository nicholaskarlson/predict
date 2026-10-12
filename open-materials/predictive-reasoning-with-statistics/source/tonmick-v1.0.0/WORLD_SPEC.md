# Tonmick World v1.0: design values

The Tonmick college system and the Tonmick Student Pathways Study are fictional. Every value below is a design choice for teaching, set in ranges typical of research on first-year college students. `generator/spec.py` holds the same values in machine-readable form and is the authority if the two ever differ.

## Structure

| Layer | Design |
| --- | --- |
| Colleges | 40 colleges, 10 in each of 4 regions (North, East, South, West). Sizes lognormal (median about 1,275 entrants, range 380 to 3,000), plus one large college in North (30% of its region) and South (28%) that must be taken with certainty |
| Cohort 1 population | All entering students in the anchor year (about 63,000): part-time share about 24% (varies by college), first in family about 35%, program area in 5 groups, age, gender, entry grade average |
| Anchor sample, stage 1 | 4 colleges per region by Sampford's probability-proportional-to-size method, certainty colleges first; exact joint inclusion probabilities by enumeration |
| Anchor sample, stage 2 | 40 part-time and 60 full-time students invited per college (part-time oversampled); about 75% join the study, giving about 1,200 participants |
| Study | Peer mentoring offered at random to half the participants in each college (offer Z); about 60% of those offered attend at least 3 of 6 sessions (uptake D); no access without an offer |
| Waves | Survey at entry (time zero) and at 3, 6 and 12 months; registry follow-up to 12 months |
| Cohort 2 | Next year's entrants (sizes up about 2%, part-time share up 5 points, entry grades 0.08 lower, withdrawal log-odds up 0.10 plus college drift); about 1,350 participants at the 16 anchor colleges and about 1,300 at 4 new colleges, one per region; nobody offered mentoring |
| Deployment sample | 100,000 further cohort-2 students across all 40 colleges, hidden, for checking Book 7's predictions |

## Measures

| Measure | Design |
| --- | --- |
| Wellbeing and stress scale | 10 items with 5 response categories, two factors (wellbeing wb1 to wb5; stress st1 to st5), loadings 0.50 to 0.80, latent trait correlation -0.45. wb4 and st4 are reverse-keyed. st2 scores about 0.45 latent SD higher on paper than online at the same stress level; every other item is invariant across mode and over time |
| Survey mode | Paper share varies by college (mean 30% at entry; half that at follow-ups); mode is unrelated to wellbeing or stress |
| Withdrawal | Registry record of withdrawal by 12 months (Weibull time to withdrawal, shape 0.8); 15% risk without an offer; higher for part-time, first in family, lower entry grades, higher stress, lower wellbeing and some program areas; college log-odds SD 0.35 |
| Transfer and consent | Transfer out of the system (about 3% a year, more likely for strong entrants) competes with withdrawal. In the study, about 1% withdraw consent, which stops registry follow-up |
| Grade average | Registry grade average at 12 months (or at exit after month 4); college ICC about 0.08 |
| Intention to leave (Y*) | Survey item at 3 months: about 0.70 of eventual leavers say yes (0.62 if offered), and about 10% of stayers (8% if offered) |
| Support visits | Count over months enrolled, overdispersed (gamma-Poisson, shape 1.2), mean about 2.4 |

## Effects (truth, hidden from analysts)

| Effect | Design |
| --- | --- |
| Uptake on wellbeing | +0.30, +0.25 and +0.20 latent SD at 3, 6 and 12 months; +0.10 more for students first in family; college variation SD 0.05 |
| Uptake on stress | -0.25, -0.20 and -0.15 latent SD; -0.05 more for students first in family |
| Uptake on withdrawal | Hazard ratio about 0.78 (0.64 for students first in family) |
| Uptake on grade average | +0.08 grade points (+0.04 more for students first in family); college variation SD 0.04 |
| Uptake on support visits | Rate ratio about 1.35 |

## Missing data

| Source | Design |
| --- | --- |
| Baseline nonresponse | About 25%; less likely to respond: part-time, first in family, lower entry grades, higher stress (stress is unobserved for nonrespondents) |
| Follow-up attrition | Response about 85%, 77% and 68% at 3, 6 and 12 months; lower for part-time students, lower entry grades and higher current stress (missing not at random), higher if offered mentoring; about a third still respond after leaving college |
| Item nonresponse | About 3% per item, rising with stress for stress items |
| Planned missingness | Three-form design at 6 months: each form omits one pair of items |
| Censoring | Time to withdrawal censored at 12 months, at transfer, or at consent withdrawal |

## Rules

- Each block of draws has its own seeded stream (root seed 20261003, block IDs fixed in `spec.py`), so a later version can add blocks without changing a single existing value. A rerun with an extra block was checked to give byte-identical files.
- Reserved for later versions: an experience-sampling sub-study (Book 15) and a literature layer of fictional studies (Book 13). Both may depend on v1.0 values; nothing in v1.0 depends on them.
- Unit files for each book are extracts of these frozen files, carrying the world version in their provenance notes.
