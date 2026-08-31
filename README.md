# Premier League Probabilistic Forecast & Monte Carlo Season Simulator

A probabilistic machine-learning system for forecasting Premier League match
outcomes and simulating full seasons: dynamic team-strength ratings (Elo),
statistical and ML goal/outcome models (Poisson, Dixon-Coles, logistic
regression, XGBoost), calibration-focused evaluation with time-based
backtesting, and a vectorized Monte Carlo season simulator producing
title/top-4/relegation probabilities.

This is a portfolio/educational project. Every model output is a
probabilistic forecast, not a prediction of what *will* happen — football is
highly stochastic, and the system is built and evaluated on that premise.

## Project status

🚧 **Phase 8 (Monte Carlo simulation engine) complete.** See the roadmap below.

- [x] Phase 1 — Data ingestion & cleaning
- [x] Phase 2 — Feature engineering (Elo, rolling form, league context)
- [x] Phase 3 — Baselines & evaluation infrastructure (log loss/Brier/RPS, expanding-window backtest)
- [x] Phase 4 — Poisson / Dixon-Coles goal model
- [x] Phase 5 — Logistic regression (Model 2, curated feature set)
- [x] Phase 6 — XGBoost (Model 4) + feature-importance explainability
- [x] Phase 7 — Calibration analysis (reliability diagrams)
- [x] Phase 8 — Monte Carlo simulation engine
- [ ] Phase 9 — Current-season forecast (live data refresh)
- [ ] Phase 10 — API (FastAPI)
- [ ] Phase 11 — Frontend dashboard (React)

## Data

### Source

The natural primary source for this kind of project is
[football-data.co.uk](https://www.football-data.co.uk), which publishes one
CSV per English football division per season (Premier League = "E0") back to
1993-94, including match odds and, from 2000-01 onward, shots/corners/cards.

**That source is currently unreachable from this project's development
environment.** Connection attempts fail at the TLS handshake stage across
four independent client stacks (Windows schannel/curl, .NET/PowerShell,
Python+OpenSSL with relaxed cipher settings, and a server-side fetch
service), while other HTTPS hosts succeed without issue. That pattern points
to the site itself being down or blocking automated clients right now, not a
local network problem — but it means the pipeline can't be built and tested
against it today.

**Operational source:** a Kaggle dataset,
[`irkaal/english-premier-league-results`](https://www.kaggle.com/datasets/irkaal/english-premier-league-results),
which mirrors football-data.co.uk's own E0 season files (same columns, same
values, verified column-for-column below). If football-data.co.uk becomes
reachable again, only `src/data/ingest.py` and `configs/data.yaml` need to
change — everything from `src/data/validate.py` onward consumes whatever CSVs
land in `data/raw/`, regardless of which source produced them.

### Seasons and coverage

**29 seasons, 1993-94 through 2021-22, 11,113 matches.** Match counts per
season exactly match known Premier League history: 462 matches/season for
the 22-team era (1993-94, 1994-95), 380 matches/season for every 20-team
season since (1995-96 through 2020-21), and 309 matches for 2021-22 — that
last figure is a real gap, not a bug: the Kaggle mirror's last recorded match
is dated 2022-04-10, well short of a season that actually runs to late May.

**Known limitation:** this dataset has not been updated since the 2021-22
season. There is a real gap between where the historical data ends and the
present. **2022-23 through the current season must be backfilled** — via
football-data.org's API (see "Live updates" below) or another source —
before Phase 7 (current-season forecasting) can run against up-to-date team
strength. This is deliberately treated as a Phase 7 concern, not solved here,
to avoid pulling live-data-source scope into the historical pipeline.

### Variables available

Every match has: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`/`FTAG` (full-time
goals), `FTR` (H/D/A). From 1993-94: half-time goals/result (`HTHG`/`HTAG`/
`HTR`) for most matches (924 rows across the earliest seasons lack it).
**From 2000-01 onward only:** referee, shots/shots-on-target, corners, fouls,
and cards (`HS`,`AS`,`HST`,`AST`,`HC`,`AC`,`HF`,`AF`,`HY`,`AY`,`HR`,`AR`) —
entirely absent (null) before 2000-01, matching football-data.co.uk's own
known history. These extended columns are preserved in the processed dataset
but are not part of the core model input set per the project's Step 1 scope
decision (see project plan discussion) — they're a possible future
extension, not core scope.

### Licensing

football-data.co.uk's terms permit personal, non-commercial, educational use
with attribution to the site. This project is portfolio/educational use
only, not a commercial product, and stays inside those terms; the Kaggle
mirror inherits the same restriction.

### Download, storage, and reproducibility

- `data/raw/` holds the untouched downloaded files (gitignored — regenerated
  by the ingestion script, never hand-edited, never committed).
- `data/processed/matches.parquet` holds the cleaned, canonicalized,
  chronologically-sorted dataset every downstream module reads. Also
  gitignored and regeneratable — the pipeline that produces it is what's
  committed, not its output.
- `data/external/team_name_map.csv` is a small, versioned seed list mapping
  known club-name aliases (e.g. "Man Utd" → "Manchester United") to a
  canonical name, since team naming isn't always consistent.

To reproduce from a clean clone:

```bash
pip install -r requirements.txt

# One-time: create a Kaggle API token at kaggle.com -> Settings -> API,
# and place kaggle.json at ~/.kaggle/kaggle.json (see src/data/ingest.py
# docstring for full instructions, including what to do if Kaggle's UI
# gives you a bare token string instead of a kaggle.json download).

python -m src.data.ingest      # downloads data/raw/kaggle_epl/results.csv
python -m src.data.validate    # writes data/processed/matches.parquet
pytest tests/                  # runs the full test suite
```

### Live updates (Phase 7, not yet implemented)

The current in-progress season will be refreshed via
[football-data.org](https://www.football-data.org)'s free-tier API, kept
entirely separate from the historical Kaggle ingestion path (see
`configs/data.yaml` → `live_source`). Historical training data always comes
from the same frozen historical source for consistency; only the "what's
happened so far this season" layer refreshes live.

## Data quality issues found and fixed during Phase 1

Documented here deliberately, since catching and explaining exactly this
kind of issue is a core goal of this project (see "Avoid Data Leakage" /
data-validity principles in the project plan):

1. **Season-boundary bug (caught by an automated sanity check, not by
   manual inspection).** The initial season-assignment rule used "month ≥ 7"
   as the cutover into a new season. The COVID-disrupted 2019-20 season
   actually finished in late July 2020 (matches played behind closed doors
   to complete the fixture list). The July cutover misfiled those ~66
   trailing matches into "2020-21", inflating that season to 446 matches /
   23 implied teams instead of the correct 380/20. Fixed by moving the
   cutover to August — the Premier League has never started a season before
   August, so this has no equivalent failure mode. A regression test
   (`test_covid_delayed_season_stragglers_stay_in_prior_season`) locks this
   in. See `src/data/validate.py::assign_season`.

2. **Date-parsing correctness depended on batch composition.** Letting
   pandas auto-guess the date format (`pd.to_datetime(..., dayfirst=True)`,
   with or without `format="mixed"`) resolves day/month ambiguity in ISO
   8601 strings by scanning the array for *some* row unambiguous enough to
   lock in a format, then applying that format to the whole batch. A batch
   containing only ambiguous rows (every date component ≤ 12) silently
   misparses e.g. `"1993-09-01T00:00:00Z"` as 9 Jan instead of 1 Sep — and
   because correctness depended on which other rows happened to share the
   batch, this is exactly the kind of bug that would resurface on a small
   incremental live-data update (Phase 7) even though it "worked" on the
   full historical file. Fixed by detecting ISO input explicitly (the "T"
   separator) and parsing each format with an explicit rule rather than
   guessing. See `src/data/validate.py::parse_dates` and the
   `TestParseDates` tests.

Both were caught by the pipeline's own sanity checks (per-season match-count
validation) and by cross-checking the derived `Season` column against the
source's own `Season` column (0 mismatches across a spot-check sample) —
not by assuming the data or the code was correct.

## Feature engineering (Phase 2)

`src/features/build_dataset.py` assembles `data/processed/features.parquet`
(11,113 rows × 76 columns) from `matches.parquet`, combining three
independent feature sources - each leakage-tested on its own:

- **Elo ratings** (`src/features/elo.py`) - one overall rating per team,
  updated sequentially match-by-match, with a home-advantage constant, an
  optional goal-margin multiplier, season-boundary regression toward the
  mean, and a mean-minus-penalty starting rating for teams with no prior
  history. See the module docstring for the full mathematical explanation
  and design rationale (including why this uses one rating track per team
  rather than separate home/away tracks).
- **Rolling form** (`src/features/rolling.py`) - 3/5/10-match rolling
  goals-for/against, points-per-game, and win/draw rate per team, plus
  rest-days-since-last-match and relative attack-vs-defence "matchup"
  features, all computed with a rolling-then-shift pattern so a match's
  features reflect only strictly prior matches.
- **League table state** (`src/features/league_table.py`) - pre-match
  points, goal difference, games played, and table position, reset every
  season. Position is intentionally computed with a same-date-batched
  sequential pass rather than a vectorized cumsum: this dataset only
  records a match's date (not kickoff time), so two matches sharing a date
  must see an identical pre-match snapshot regardless of row order - a
  row-by-row update would let one same-day match's result leak into
  another same-day match's position feature. See the module docstring and
  `test_league_table.py::test_same_date_matches_see_identical_snapshot`.

**Sanity check against real data:** the final rows of the feature matrix
(matches from April 2022, the end of this dataset) show Manchester City on
73 points in 1st and Liverpool on 72 points in 2nd - recovering, purely
from raw match results, the historically famous single-point-margin
2021-22 title race.

**A second real bug was found and fixed the same way as Phase 1** - by a
unit test, not by inspection: in `EloRatingSystem`, the very first match of
the entire dataset gave the away team a "newly promoted" rating penalty
instead of the initial rating. The home team's lookup inserted into the
ratings dict first, so by the time the away team was checked a moment
later in the same match, the dict no longer looked empty. Fixed by
snapshotting "does the league have any ratings yet" once per match, before
either side is looked up - see `src/features/elo.py::EloRatingSystem.
process_match` and the regression test in `test_elo.py`.

161 tests pass across the full suite (`pytest tests/`), all against
synthetic data - no test depends on the downloaded dataset being present.
(This count was 68 as of Phase 2; Phases 3-8 added the metrics, baseline,
Elo-outcome, backtest, Poisson/Dixon-Coles, logistic-regression, XGBoost,
calibration, and simulation-engine/summary tests.)

## Model evaluation (Phase 3)

**Why not select models by accuracy?** Accuracy only asks "was the top
pick right" - it's blind to *how confident* a model was, which is exactly
what the Monte Carlo simulator (Phase 7) will need to sample from
honestly. Concretely: "always predict home win" scores **45.8% accuracy**
on this dataset - deceptively close to far more sophisticated models -
purely because home win is the single most common outcome in football
(~46% of matches historically). This project selects and compares models
on **log loss, Brier score, and RPS** (see `src/evaluation/metrics.py` for
the full mathematical explanation of each), with accuracy reported only
for interpretability.

**Backtesting methodology:** `src/evaluation/backtest.py` implements
expanding-window, season-by-season backtesting - train on the first 10
seasons (1993-94 → 2002-03), predict 2003-04, retrain including 2003-04,
predict 2004-05, and so on through 2021-22 (19 held-out seasons). Every
metric below is a **match-count-weighted average across all 19 seasons**,
never a single lucky/unlucky split.

### Model comparison (Phases 3-5)

| Model | Log Loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| Baseline (historical frequency) | 1.0651 | 0.6434 | 0.2281 | 45.70% |
| **Elo** (`elo_diff` → multinomial logit) | **0.9825** | **0.5850** | **0.1999** | **53.07%** |
| Poisson (independent) | 0.9891 | 0.5894 | 0.2021 | 52.57% |
| Poisson (Dixon-Coles) | 0.9890 | 0.5893 | 0.2020 | 52.54% |
| Logistic Regression (11 features) | 0.9848 | 0.5855 | 0.2000 | 53.03% |
| XGBoost (~50 features) | 0.9907 | 0.5898 | 0.2017 | 52.22% |

Both Elo and Poisson clear the baseline by a wide, consistent margin (every
metric, nearly every individual season - see `experiments/results.csv`).
Between Elo and Poisson, the picture is closer and worth reporting
honestly rather than declaring a winner:

- **Elo wins on every aggregate metric**, but only **narrowly** - and only
  **12 of the 19** individual held-out seasons; Poisson (Dixon-Coles) wins
  outright in the other **7**, including several recent seasons
  (2018-19, 2020-21, 2021-22). This is a real, close competition between
  two very different modeling approaches, not a rout.
- **The Dixon-Coles correlation adjustment barely moves the needle** here
  (log loss 0.98910 → 0.98904, a ~0.006% relative change), even though the
  fitted `rho` (~ -0.04) is in a sensible, literature-consistent range.
  The low-score correlation effect Dixon & Coles documented appears to be
  real but small enough, once averaged over a full season of matches
  varied in scoreline, to barely register in season-level log loss - worth
  knowing before assuming the added complexity is "obviously" earning its
  keep.
- Neither model's hyperparameters have been tuned yet (Elo's `k_factor`/
  `home_advantage`/`season_shrinkage`; Poisson's `xi`/`promoted_penalty`
  time-decay and promoted-team settings) - both are running on
  literature-typical defaults, not values fit to this data. Closing that
  gap is future work, not a claim that the current ranking is final.
- **Log loss alone doesn't settle which model belongs in this project.**
  Poisson gives something Elo structurally cannot: actual expected-goals
  numbers and a full scoreline probability grid (`predict_expected_goals`,
  `predict_score_grid`) - needed for the "expected goals: 1.72 / 1.31" and
  "most likely scorelines" match-prediction output the project plan calls
  for, and useful groundwork for sampling realistic scorelines in the
  Monte Carlo simulator, not just a W/D/L outcome. Every model here exists
  for a reason beyond "wins on log loss" - this is why Poisson stays in
  the comparison even where Elo currently edges it out.

Reproduce with `python -m experiments.run_phase6_xgboost` (supersedes
earlier phase experiment scripts with the same result files, now including
XGBoost and its feature-importance report).

**Model 1 (Elo) design note:** rather than hand-picking a fixed formula to
split Elo's two-outcome "expected score" into three W/D/L probabilities,
`src/models/elo_model.py` fits a multinomial logistic regression with
`elo_diff` as its only input - effectively just calibrating a rating gap
into a probability triple (~4 free parameters), refit fresh on each
backtest fold's training window. See the module docstring for the full
reasoning, including why this is kept deliberately simpler than Model 2
(logistic regression with many features, Phase 5).

**Model 3 (Poisson/Dixon-Coles) design notes:**
`src/models/poisson_model.py` fits per-team attack/defense strength
parameters (Maher, 1982) via maximum likelihood, with an optional
Dixon-Coles (1997) low-score correlation correction. Two implementation
details worth knowing about:
- **Two-stage fitting for speed.** A first pass tried optimizing all ~100
  parameters (attack/defense/home-advantage/rho) jointly via generic
  numerical-gradient optimization - correct, but **60-114 seconds per
  fit** at this dataset's largest training window, which would have made
  a 19-fold backtest impractically slow. Deriving and supplying an
  **analytical gradient** for the (dominant) independent-Poisson
  likelihood, then fitting `rho` separately in a second, fast 1-D search
  with attack/defense/home-advantage held fixed, cut that to **~0.4
  seconds** (~285x) with the fitted parameters essentially unchanged (rho
  -0.0392 → -0.0389, home-advantage 0.1938 → 0.1937) - confirming the
  two-stage simplification isn't costing meaningful accuracy.
- **Identifiability.** Attack/defense parameters are only determined up to
  a joint additive shift (raise every team's attack and defense together
  and every predicted score stays identical), so the model fixes one
  reference team's attack rating to 0 - see the module docstring for the
  full explanation and why this doesn't bias predictions.

**Model 2 (logistic regression) result - and why it matters more than the
table alone suggests:** `src/models/logistic_model.py` fits a multinomial
logistic regression over 11 curated features - `elo_diff`, recent-form and
attack/defense-matchup differentials (window=5), rest-days difference, and
pre-match table points/position/goal-difference for both teams (feature
list and the reasoning behind each choice are in the module docstring).
Despite eleven inputs against Elo's one, **it barely beats Elo alone**
(log loss 0.9848 vs. 0.9825 - Elo is still marginally ahead) and doesn't
beat it on accuracy at all. The standardized coefficients (also fit on the
full dataset and printed by `run_phase5_logistic.py`) explain why plainly:
`elo_diff`'s coefficient magnitude (~0.46-0.49) is roughly **2-40x larger**
than every other feature's (table points: 0.11-0.36; everything else,
including all three recent-form features, under 0.05). This is a genuine,
useful finding, not a disappointing one: it says Elo alone is already
capturing nearly all of the linearly-usable signal in this feature set,
and the richer feature engineering from Phase 2 mostly *isn't* adding
independent predictive value on top of it, at least not in a form a linear
model can exploit. Two things this does NOT rule out, and that Phase 6
(XGBoost) is specifically positioned to test: non-linear interactions
between features (e.g. "recent form matters more when Elo ratings are
close"), and interactions a purely additive linear model structurally
can't represent.

**Phase 6 (XGBoost) result - the hypothesis above did NOT hold up, and
that's the most important finding in this comparison table.**
`src/models/xgboost_model.py` gave gradient boosting the richest feature
set of any model here (~50 columns, all three rolling windows, NaN values
passed through natively rather than imputed - see the module docstring
for why a tree ensemble can use a broader, less-curated set than the
logistic model could). If the closeness between Elo/Poisson/logistic
regression really were "these three linear-ish models have hit a ceiling
that a model capable of non-linear interactions could break through,"
XGBoost should have been the clear winner. **It's actually the worst
performer among every non-baseline model** - log loss 0.9907, behind even
independent Poisson, and its accuracy (52.2%) is the lowest of the four
real models. Its own feature importances still rank `elo_diff` far above
everything else (gain 0.186 vs. 0.017-0.024 for the next dozen features -
see `run_phase6_xgboost.py` output / README "Explainability"), so it isn't
finding and exploiting some hidden interaction the linear models miss -
if anything, it's diluting Elo's strong, clean signal across a large
feature set on a comparatively small dataset (~7,000-10,000 training rows
per fold against ~50 features) with default, untuned hyperparameters,
which is precisely the overfitting failure mode gradient boosting is known
for at this data scale. This is exactly the outcome the project's stated
principle ("if a simpler approach performs better, tell you") exists to
surface rather than to paper over with a bigger model. It does NOT mean
XGBoost is a dead end - hyperparameter tuning (shallower/fewer trees,
stronger regularization) is untried and could change this - but as
currently built, it does not earn a place ahead of Elo in this project,
and that result is reported plainly rather than quietly dropped from the
comparison.

**Explainability (Phase 6):** for XGBoost, gain-based feature importance
(`XGBoostOutcomeModel.feature_importances()`) is reported rather than a
full SHAP analysis - per the project's own principle of not adding
explainability tooling without a clear reason, a deeper SHAP investigation
of dependence/interaction effects is deferred: it's most useful for
understanding a model that's actually earning its complexity, and right
now this one isn't. Worth revisiting if hyperparameter tuning changes that
picture. Logistic regression's standardized coefficients (reported above)
serve the equivalent role for that model - both explainability methods
here converge on the identical conclusion: `elo_diff` dominates every
other engineered feature by a wide margin.

## Calibration analysis (Phase 7)

**Why check this separately from log loss/Brier/RPS at all?** Those
metrics reward good calibration *on average*, but a model can post a
solid aggregate score while still being subtly miscalibrated in exactly
the probability range that matters most once it's driving 10,000+
simulated seasons (Phase 8). If a model says "70% chance of a home win"
across a group of matches but the true rate there is actually 55%, every
simulated season built from it quietly overstates how often that kind of
team wins - not by much in any one match, but compounded across a 38-game
season and thousands of simulated seasons, that turns into a materially
wrong title/relegation probability. `src/evaluation/calibration.py` checks
this directly: bin each model's predicted "home win" probability into
deciles, and compare the average prediction in each bin against the
actual observed home-win frequency there (a *reliability diagram*),
summarized by Expected Calibration Error (ECE) - full mathematical
explanation in the module docstring.

### Result: the calibration ranking is NOT the same as the log-loss ranking

| Model | ECE (home win) | (for reference) Log Loss rank |
|---|---|---|
| **Poisson (Dixon-Coles)** | **0.0156** | 3rd of 4 |
| XGBoost | 0.0185 | 4th of 4 |
| Logistic Regression | 0.0203 | 2nd of 4 |
| Elo | 0.0243 | **1st of 4** |

**Elo - the model with the best log loss - is the worst-calibrated of the
four**, specifically underconfident in its higher-confidence predictions:
in the 0.6-0.7 predicted-probability bin, Elo says ~64.6% but home wins
actually happen 69.4% of the time there (908 matches); in the 0.7-0.8 bin,
it says ~74.0% but the true rate is 79.4% (500 matches) - both sizeable
samples, not noise. Poisson (Dixon-Coles), despite a very slightly worse
log loss, tracks the diagonal much more closely across almost every bin.
All four models are reasonably well-calibrated in absolute terms (every
ECE is under 0.025 on a 0-1 scale) - this isn't "Elo is broken," it's "the
model that best minimizes average error isn't automatically the model
whose confidence levels can be trusted most literally," which is precisely
the distinction log loss alone can't surface and the reason this project
treats calibration as a separate, required check rather than assuming a
good log loss implies good calibration.

Full per-bin tables: `experiments/calibration_ece.csv`; reliability
diagram: `experiments/calibration_home_win.png`; reproduce with
`python -m experiments.run_phase7_calibration`.

**This is relevant to an earlier decision worth revisiting, not
overturning unilaterally:** the plan going into the Monte Carlo simulator
(Phase 8) was to default to Elo, since it has the best aggregate log
loss/Brier/RPS/accuracy, while keeping the simulator model-agnostic so any
model's probabilities can be swapped in. This calibration result doesn't
change the "keep it swappable" part, but it's a genuine trade-off worth
being aware of when picking the *default*: Elo's raw win/loss/draw
accuracy is best on average, but Poisson's probabilities are the most
literally trustworthy at any given confidence level - which is arguably
more important for a system whose whole output is "probability of X" at
scale. Worth a second look before finalizing which model the simulator
defaults to.

## Monte Carlo simulation engine (Phase 8)

`src/simulation/engine.py` turns a set of remaining fixtures' expected
goals into thousands of simulated final league tables in one pass: sample
every fixture's home/away goals for every simulation at once (two
vectorized `rng.poisson()` calls), convert to points, scatter-accumulate
each team's points/goals across every simulation with `np.add.at`, and
rank every simulation's final table with a single combined-integer
`argsort` (points → goal difference → goals scored, matching the real
Premier League order minus head-to-head - the same documented
simplification `src/features/league_table.py` uses for historical
features, and for the same reason: head-to-head isn't cleanly
vectorizable and almost never decides the outcomes this project reports).
`src/simulation/summary.py` turns that into the actual headline numbers:
title/Champions-League/relegation probability, expected position, and a
full points distribution per team, with European qualification cutoffs
configurable rather than hardcoded (the real UEFA rules are more complex
than a fixed cutoff and change year to year - this is a labeled
simplification, not a claim of exact accuracy).

**Design choice, and why:** as discussed above, this project's simulator
defaults to the Poisson/Dixon-Coles model specifically so it can sample
real scorelines (needed for goal-difference-aware standings and the
project's "Arsenal wins 2-1" style output) rather than just a W/D/L label.
Home and away goals are sampled **independently** from their marginal
Poisson distributions rather than jointly from the Dixon-Coles-adjusted
grid - a deliberate, quantified trade: joint sampling at 100,000
simulations × ~100 remaining fixtures × ~121 possible scorelines would
need well over a gigabyte of intermediate probability mass, for a
correlation effect Phase 4 already measured as barely moving aggregate
log loss (0.9891 → 0.9890). Full reasoning in the module docstring.

### Performance benchmark

| Simulations | Time | Throughput |
|---|---|---|
| 10,000 | 0.17s | ~57,000/sec |
| 50,000 | 0.86s | ~58,000/sec |
| 100,000 | 1.62s | ~62,000/sec |

(20-team league, ~100 remaining fixtures - a realistic mid-season
scenario.) Scales linearly, and 100,000 simulations in well under 2
seconds comfortably answers the project plan's question of whether that
scale is computationally reasonable.

### End-to-end validation against a real, fully-known season

Rather than only unit-testing the simulator's internal mechanics (which
`tests/test_simulation_engine.py` / `test_simulation_summary.py` do -
point assignment, valid-table invariants, reproducibility), the whole
pipeline was validated against a real season whose outcome is already
known: the **2020-21 season, frozen at a January 1, 2021 cutoff**. The
Poisson model was trained only on matches before that date (never seeing
the second half of the season), the current table was built from the 155
matches actually played by that point, and the remaining 225 fixtures
were simulated 20,000 times - exactly the shape Phase 9's live forecast
will eventually take, just checked against a season we already know the
answer to. Reproduce with `python -m experiments.run_phase8_simulation_validation`.

**Results**: Manchester City - the eventual champions - had by far the
highest simulated title probability (51.7%), roughly 1.5x Liverpool's
(35.5%) and 6.6x Manchester United's (7.8%). More strikingly, **all three
teams that were actually relegated that season** (Fulham, West Brom,
Sheffield United) were exactly the three teams with the highest simulated
relegation probabilities (77.7%, 83.7%, 91.6%) - not just "in the
relegation zone on average," but correctly ranked as the three most
likely relegation candidates specifically. The misses are informative
rather than concerning: West Ham's real second-half surge to a 6th-place
finish (expected position 11.3 from the model) is a well-documented
overperformance no pre-cutoff model could have seen coming, and
Liverpool's actual 69 points falling well short of their 77.3 expected
tracks their real, well-documented injury crisis that spring - exactly
the kind of genuine, irreducible football unpredictability this project's
probabilistic framing is built to represent honestly rather than paper
over with false certainty.

## Repository structure

```
data/
  raw/                 # untouched downloads (gitignored)
  processed/           # cleaned matches.parquet (gitignored, regeneratable)
  external/            # team_name_map.csv and similar small config-like data
notebooks/             # exploration only - no load-bearing logic lives here
src/
  data/                # ingestion, cleaning, validation
  features/            # Elo, rolling stats, league table state, dataset assembly
  models/               # baseline, Elo, Poisson/Dixon-Coles, logistic, XGBoost (all done, Phases 3-6)
  evaluation/           # metrics (log loss/Brier/RPS), expanding-window backtest, calibration
  simulation/            # Monte Carlo season simulator (engine + summary stats)
  visualization/          # chart helpers
experiments/            # experiment-runner scripts + versioned backtest results (results.csv)
tests/                  # pytest suite
app/
  backend/               # (Phase 9) FastAPI - no modeling logic, calls src/
  frontend/               # (Phase 10) React + TypeScript dashboard
configs/                 # data.yaml and future model/simulation configs
```

## Setup

```bash
pip install -r requirements.txt
python -m src.data.ingest                    # data/raw/kaggle_epl/results.csv
python -m src.data.validate                  # data/processed/matches.parquet
python -m src.features.build_dataset         # data/processed/features.parquet
python -m experiments.run_phase6_xgboost     # experiments/results.csv
python -m experiments.run_phase7_calibration # experiments/calibration_ece.csv, calibration_home_win.png
python -m experiments.run_phase8_simulation_validation  # simulator validated against a real known season
pytest tests/
```
