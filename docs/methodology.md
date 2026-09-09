# Methodology

Full phase-by-phase writeup of the data pipeline, modeling, evaluation, and
simulation methodology behind [premier-league-forecast](../README.md) —
moved here to keep the top-level README short. Deployment details (Render/
Vercel setup, two real production bugs found and fixed) are in
[deployment.md](deployment.md).

## Project status

All 12 originally-planned phases complete, plus two follow-on phases:

- [x] Phase 1 — Data ingestion & cleaning
- [x] Phase 2 — Feature engineering (Elo, rolling form, league context)
- [x] Phase 3 — Baselines & evaluation infrastructure (log loss/Brier/RPS, expanding-window backtest)
- [x] Phase 4 — Poisson / Dixon-Coles goal model
- [x] Phase 5 — Logistic regression (Model 2, curated feature set)
- [x] Phase 6 — XGBoost (Model 4) + feature-importance explainability
- [x] Phase 7 — Calibration analysis (reliability diagrams)
- [x] Phase 8 — Monte Carlo simulation engine
- [x] Phase 9 — Current-season forecast (live data refresh)
- [x] Phase 10 — Hyperparameter tuning
- [x] Phase 11 — API (FastAPI)
- [x] Phase 12 — Frontend dashboard (React)
- [x] Phase 13 — Simulation calibration backtest
- [x] Phase 14 — Live deployment (Docker + Render + Vercel, see [deployment.md](deployment.md))

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

**Operational source: three combined sources**, since no single one covers
1993-94 through the present:

1. [`irkaal/english-premier-league-results`](https://www.kaggle.com/datasets/irkaal/english-premier-league-results)
   (Kaggle) — mirrors football-data.co.uk's own E0 files. Used only for its
   1993-94..1999-00 seasons.
2. [`marcohuiii/english-premier-league-epl-match-data-2000-2025`](https://www.kaggle.com/datasets/marcohuiii/english-premier-league-epl-match-data-2000-2025)
   (Kaggle) — a newer, more complete mirror, 2000-01..2024-25 (slightly
   truncated near the very end).
3. [`openfootball/football.json`](https://github.com/openfootball/football.json)
   (GitHub, public domain, **no API key required**) — fills the 2025-26 gap
   neither Kaggle source covers, and is also the **live source for the
   current season** (Phase 9), since it publishes the full fixture list with
   results filled in as matches are played.

If football-data.co.uk becomes reachable again, only `src/data/ingest.py`
and `configs/data.yaml` need to change — everything from
`src/data/validate.py` onward consumes whatever files land in each source's
raw directory, regardless of where they came from.

**Two other Kaggle datasets were checked and rejected as the 2025-26
source** before openfootball was found: one claimed full-season coverage but
had real results for only 3 of 38 matchweeks despite listing a full-season
date range (a fixture-list-only upload, never updated after the season
started); another was periodic table snapshots, not match-level results.
Both were verified by actually downloading and inspecting them, not by
trusting their descriptions.

### Seasons and coverage

**33 seasons, 1993-94 through 2025-26, 12,704 matches — the full available
history through the most recently completed season, with zero data-completeness
warnings.** Match counts per season exactly match known Premier League
history: 462 matches/season for the 22-team era (1993-94, 1994-95), 380
every season since. `src/data/validate.py::merge_sources` combines the three
sources **per season**, keeping whichever source has the most complete data
for that specific season, not simply whichever source is newest overall —
see "Data quality issues found and fixed" below for why that distinction
mattered in practice. `src/data/validate.py::cross_validate_overlap` checks
every pair of sources against each other wherever their seasons overlap: **0
score disagreements across 8,549 overlapping matches** (8,199 between the two
Kaggle sources, 350 between the newer Kaggle source and openfootball's
2024-25 file) — strong independent confirmation of data quality.

### Variables available

Every match has: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`/`FTAG` (full-time
goals), `FTR` (H/D/A). From 1993-94: half-time goals/result (`HTHG`/`HTAG`/
`HTR`) for most matches (924 rows across the earliest seasons lack it).
**From 2000-01 onward only:** referee, shots/shots-on-target, corners, fouls,
and cards (`HS`,`AS`,`HST`,`AST`,`HC`,`AC`,`HF`,`AF`,`HY`,`AY`,`HR`,`AR`) —
entirely absent (null) before 2000-01, matching football-data.co.uk's own
known history. These extended columns are preserved in the processed dataset
but are not part of the core model input set per the project's Step 1 scope
decision — they're a possible future extension, not core scope.

### Licensing

football-data.co.uk's terms permit personal, non-commercial, educational use
with attribution to the site. This project is portfolio/educational use
only, not a commercial product, and stays inside those terms; the Kaggle
mirror inherits the same restriction.

### Download, storage, and reproducibility

- `data/raw/` holds the untouched downloaded files (gitignored — regenerated
  by the ingestion script, never hand-edited, never committed).
- `data/processed/matches.parquet` holds the cleaned, canonicalized,
  chronologically-sorted dataset every downstream module reads. Committed as
  a deliberate exception (see `.gitignore`) since the deployed backend reads
  it directly with no ingestion pipeline available in that environment.
- `data/external/team_name_map.csv` is a small, versioned seed list mapping
  known club-name aliases (e.g. "Man Utd" → "Manchester United") to a
  canonical name, since team naming isn't always consistent.

### Live updates (Phase 9)

The current in-progress season refreshes via the same
[openfootball/football.json](https://github.com/openfootball/football.json)
source used to fill the 2025-26 historical gap — no API key needed. It
publishes the full season's fixture list with results filled in as matches
are played, so `src/live_forecast.py` reads it directly for both "what's
happened so far" (current table state, via the same `LeagueTableTracker`
Phase 2's historical features use) and "what's left to play" (the
simulator's remaining-fixture input), kept entirely separate from the frozen
historical ingestion path (see `configs/data.yaml` → `live_source`). Unlike
every other raw source in this project, the live file is re-fetched on
every run rather than cached — it changes every time a match is played.

## Data quality issues found and fixed during Phase 1

Documented here deliberately, since catching and explaining exactly this
kind of issue is a core goal of this project:

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
   pandas auto-guess the date format resolves day/month ambiguity in ISO
   8601 strings by scanning the array for *some* row unambiguous enough to
   lock in a format, then applying that format to the whole batch. A batch
   containing only ambiguous rows (every date component ≤ 12) silently
   misparses e.g. `"1993-09-01T00:00:00Z"` as 9 Jan instead of 1 Sep — and
   because correctness depended on which other rows happened to share the
   batch, this is exactly the kind of bug that would resurface on a small
   incremental live-data update even though it "worked" on the full
   historical file. Fixed by detecting ISO input explicitly (the "T"
   separator) and parsing each format with an explicit rule rather than
   guessing. See `src/data/validate.py::parse_dates` and the
   `TestParseDates` tests.

3. **"Prefer the newer source" was the wrong merge rule.** When the
   historical dataset was later expanded to 3 sources, the first version of
   `merge_sources` picked, for any season two sources both covered,
   whichever source was listed later (generally newer/more complete).
   That's wrong: the newer, generally-more-complete 2000-2025 Kaggle source
   turned out to be missing 45 matches each for specifically 2003-04 and
   2004-05 (335/380), seasons the older 1993-2000 source has completely
   (380/380) — caught immediately by the same per-season match-count sanity
   check from bug #1, applied to the newly-merged data. Fixed by choosing
   per season based on actual match count, not source recency, with source
   order only breaking an exact tie. Regression tests lock in both
   directions (`TestMergeSources`).

4. **Adding a new source silently split a club's history in two.** The same
   expansion introduced "Leicester City FC" and "Southampton FC" as team
   names distinct from the already-canonical "Leicester City" and
   "Southampton" — the new source's naming convention wasn't covered by
   `team_name_map.csv`, which had only been built against the sources known
   at the time. First surfaced as an anomaly in an Elo sanity check (two
   unfamiliar-looking teams in the bottom-5 ratings), not by manual
   inspection of the 53-line team list — which is exactly why
   `sanity_check` now runs `find_near_duplicate_team_names` automatically
   on every build rather than relying on someone reading the printed list
   closely enough to notice two names that differ only by a trailing "FC".

All four were caught by the pipeline's own automated checks — per-season
match-count validation, cross-checking the derived `Season` column against
the source's own `Season` column (0 mismatches), and now automated
near-duplicate team-name detection — not by assuming the data or the code
was correct.

## Feature engineering (Phase 2)

`src/features/build_dataset.py` assembles `data/processed/features.parquet`
(12,704 rows × 77 columns) from `matches.parquet`, combining three
independent feature sources, each leakage-tested on its own:

- **Elo ratings** (`src/features/elo.py`) — one overall rating per team,
  updated sequentially match-by-match, with a home-advantage constant, an
  optional goal-margin multiplier, season-boundary regression toward the
  mean, and a mean-minus-penalty starting rating for teams with no prior
  history.
- **Rolling form** (`src/features/rolling.py`) — 3/5/10-match rolling
  goals-for/against, points-per-game, and win/draw rate per team, plus
  rest-days-since-last-match and relative attack-vs-defence "matchup"
  features, all computed with a rolling-then-shift pattern so a match's
  features reflect only strictly prior matches.
- **League table state** (`src/features/league_table.py`) — pre-match
  points, goal difference, games played, and table position, reset every
  season. Position is intentionally computed with a same-date-batched
  sequential pass rather than a vectorized cumsum: this dataset only
  records a match's date (not kickoff time), so two matches sharing a date
  must see an identical pre-match snapshot regardless of row order.

**Sanity check against real data:** the final rows of the feature matrix
(April 2022) show Manchester City on 73 points in 1st and Liverpool on 72
points in 2nd — recovering, purely from raw match results, the historically
famous single-point-margin 2021-22 title race.

**A second real bug was found and fixed the same way as Phase 1** — by a
unit test, not by inspection: in `EloRatingSystem`, the very first match of
the entire dataset gave the away team a "newly promoted" rating penalty
instead of the initial rating. The home team's lookup inserted into the
ratings dict first, so by the time the away team was checked a moment
later in the same match, the dict no longer looked empty. Fixed by
snapshotting "does the league have any ratings yet" once per match, before
either side is looked up — see `src/features/elo.py::EloRatingSystem.process_match`.

210+ tests pass across the full suite (`pytest tests/`), all against
synthetic data — no test depends on the downloaded dataset being present.

## Model evaluation (Phase 3)

**Why not select models by accuracy?** Accuracy only asks "was the top
pick right" — it's blind to *how confident* a model was, which is exactly
what the Monte Carlo simulator needs to sample from honestly. Concretely:
"always predict home win" scores **45.8% accuracy** on this dataset —
deceptively close to far more sophisticated models — purely because home
win is the single most common outcome in football (~46% of matches
historically). This project selects and compares models on **log loss,
Brier score, and RPS** (see `src/evaluation/metrics.py`), with accuracy
reported only for interpretability.

**Backtesting methodology:** `src/evaluation/backtest.py` implements
expanding-window, season-by-season backtesting — train on the first 10
seasons, predict the next, retrain including it, predict the next, and so
on through 19 held-out seasons. Every metric is a match-count-weighted
average across all 19, never a single lucky/unlucky split.

### Model comparison (Phases 3-5)

> The numbers below (and the calibration results further down) were
> computed on the 29-season dataset (1993-94..2021-22), before the
> historical data was later expanded to 33 seasons through 2025-26. The
> live forecast fits its own model fresh on the full current dataset
> regardless; the qualitative conclusions (Elo edges out the others,
> XGBoost underperforms, calibration ranking differs from log-loss ranking)
> are unlikely to flip from 4 additional seasons.

| Model | Log Loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| Baseline (historical frequency) | 1.0651 | 0.6434 | 0.2281 | 45.70% |
| **Elo** (`elo_diff` → multinomial logit) | **0.9825** | **0.5850** | **0.1999** | **53.07%** |
| Poisson (independent) | 0.9891 | 0.5894 | 0.2021 | 52.57% |
| Poisson (Dixon-Coles) | 0.9890 | 0.5893 | 0.2020 | 52.54% |
| Logistic Regression (11 features) | 0.9848 | 0.5855 | 0.2000 | 53.03% |
| XGBoost (~50 features) | 0.9907 | 0.5898 | 0.2017 | 52.22% |

Both Elo and Poisson clear the baseline by a wide, consistent margin.
Between Elo and Poisson the picture is closer and worth reporting honestly:

- **Elo wins on every aggregate metric, but only narrowly** — and only
  **12 of the 19** individual held-out seasons; Poisson (Dixon-Coles) wins
  outright in the other **7**, including several recent seasons.
- **The Dixon-Coles correlation adjustment barely moves the needle** here
  (log loss 0.98910 → 0.98904, ~0.006% relative change), even though the
  fitted `rho` (~ -0.04) is in a sensible, literature-consistent range.
- **Log loss alone doesn't settle which model belongs in this project.**
  Poisson gives something Elo structurally cannot: actual expected-goals
  numbers and a full scoreline probability grid — needed for the
  "expected goals: 1.72 / 1.31" / "most likely scorelines" match-prediction
  output, and for sampling realistic scorelines in the Monte Carlo
  simulator, not just a W/D/L outcome.

Reproduce with `python -m experiments.run_phase6_xgboost`.

**Model 1 (Elo) design note:** rather than hand-picking a fixed formula to
split Elo's two-outcome "expected score" into three W/D/L probabilities,
`src/models/elo_model.py` fits a multinomial logistic regression with
`elo_diff` as its only input — effectively just calibrating a rating gap
into a probability triple (~4 free parameters), refit fresh on each
backtest fold's training window.

**Model 3 (Poisson/Dixon-Coles) design notes:**
`src/models/poisson_model.py` fits per-team attack/defense strength
parameters (Maher, 1982) via maximum likelihood, with an optional
Dixon-Coles (1997) low-score correlation correction.
- **Two-stage fitting for speed.** Jointly optimizing all ~100 parameters
  via generic numerical-gradient optimization took **60-114 seconds per
  fit**, impractical for a 19-fold backtest. Deriving an **analytical
  gradient** for the (dominant) independent-Poisson likelihood, then
  fitting `rho` separately in a fast 1-D search, cut that to **~0.4
  seconds** (~285x) with fitted parameters essentially unchanged.
- **Identifiability.** Attack/defense parameters are only determined up to
  a joint additive shift, so the model fixes one reference team's attack
  rating to 0.

**Model 2 (logistic regression):** `src/models/logistic_model.py` fits a
multinomial logistic regression over 11 curated features. Despite eleven
inputs against Elo's one, it barely beats Elo alone (0.9848 vs 0.9825).
Standardized coefficients show why: `elo_diff`'s coefficient magnitude is
roughly 2-40x larger than every other feature's — Elo alone is already
capturing nearly all of the linearly-usable signal in this feature set.

**Phase 6 (XGBoost) result — a genuine, reported-honestly negative
finding.** `src/models/xgboost_model.py` gave gradient boosting the
richest feature set of any model here (~50 columns). If the closeness
between the linear-ish models really were "these have hit a ceiling a
non-linear model could break through," XGBoost should have won. **It's
actually the worst performer among every non-baseline model** — log loss
0.9907. Its own feature importances still rank `elo_diff` far above
everything else (gain 0.186 vs. 0.017-0.024 for the next dozen features),
consistent with overfitting a large feature set on a comparatively small
dataset with default, untuned hyperparameters. Not treated as a dead end —
hyperparameter tuning was untried at this point — but reported plainly
rather than dropped from the comparison.

## Calibration analysis (Phase 7)

**Why check this separately from log loss/Brier/RPS?** Those metrics
reward good calibration *on average*, but a model can post a solid
aggregate score while still being subtly miscalibrated in exactly the
probability range that matters most once it's driving thousands of
simulated seasons. `src/evaluation/calibration.py` bins each model's
predicted "home win" probability into deciles and compares the average
prediction in each bin against the actual observed frequency (a
*reliability diagram*), summarized by Expected Calibration Error (ECE).

### Result: the calibration ranking is NOT the same as the log-loss ranking

| Model | ECE (home win) | Log Loss rank |
|---|---|---|
| **Poisson (Dixon-Coles)** | **0.0156** | 3rd of 4 |
| XGBoost | 0.0185 | 4th of 4 |
| Logistic Regression | 0.0203 | 2nd of 4 |
| Elo | 0.0243 | **1st of 4** |

**Elo — the model with the best log loss — is the worst-calibrated of the
four**, specifically underconfident in its higher-confidence predictions:
in the 0.6-0.7 predicted-probability bin, Elo says ~64.6% but home wins
actually happen 69.4% of the time there (908 matches). Poisson
(Dixon-Coles), despite a very slightly worse log loss, tracks the diagonal
much more closely. All four models are reasonably well-calibrated in
absolute terms (every ECE under 0.025) — the takeaway is "the model that
best minimizes average error isn't automatically the model whose
confidence levels can be trusted most literally," which log loss alone
can't surface.

Full per-bin tables: `experiments/calibration_ece.csv`; reliability
diagram: `experiments/calibration_home_win.png`; reproduce with
`python -m experiments.run_phase7_calibration`.

## Monte Carlo simulation engine (Phase 8)

`src/simulation/engine.py` turns a set of remaining fixtures' expected
goals into thousands of simulated final league tables in one pass: sample
every fixture's home/away goals for every simulation at once (two
vectorized `rng.poisson()` calls), convert to points, scatter-accumulate
each team's points/goals across every simulation with `np.add.at`, and
rank every simulation's final table with a single combined-integer
`argsort` (points → goal difference → goals scored, minus head-to-head —
a documented simplification, not vectorizable and rarely decisive).
`src/simulation/summary.py` turns that into title/Champions-League/
relegation probability, expected position, and a full points distribution
per team.

**Design choice:** the simulator samples real scorelines (needed for
goal-difference-aware standings), sampling home/away goals
**independently** from their marginal Poisson distributions rather than
jointly from the Dixon-Coles-adjusted grid — joint sampling at 100,000
simulations × ~100 fixtures × ~121 possible scorelines would need well
over a gigabyte of intermediate probability mass, for a correlation effect
already measured as barely moving aggregate log loss.

### Performance benchmark

| Simulations | Time | Throughput |
|---|---|---|
| 10,000 | 0.17s | ~57,000/sec |
| 50,000 | 0.86s | ~58,000/sec |
| 100,000 | 1.62s | ~62,000/sec |

(20-team league, ~100 remaining fixtures.) Scales linearly.

### End-to-end validation against a real, fully-known season

The whole pipeline was validated against the **2020-21 season, frozen at a
January 1, 2021 cutoff**: trained only on matches before that date, current
table built from the 155 matches actually played, remaining 225 fixtures
simulated 20,000 times. Reproduce with
`python -m experiments.run_phase8_simulation_validation`.

**Results**: Manchester City — the eventual champions — had by far the
highest simulated title probability (51.7%), roughly 1.5x Liverpool's
(35.5%). **All three teams actually relegated that season** (Fulham, West
Brom, Sheffield United) were exactly the three teams with the highest
simulated relegation probabilities (77.7%, 83.7%, 91.6%). The misses are
informative rather than concerning: West Ham's real second-half surge to
6th (expected position 11.3 from the model) is a well-documented
overperformance no pre-cutoff model could see coming — exactly the kind of
genuine football unpredictability this project's probabilistic framing is
built to represent honestly.

## Current-season live forecast (Phase 9)

`src/live_forecast.py` fits the Poisson/Dixon-Coles model on the full
historical dataset plus the current season's matches so far, builds the
current table state, simulates the remaining season 50,000 times, and
reports the same headline statistics as the Phase 8 validation — for a
season still being played, not one whose outcome is already known.
Reproduce with `python -m experiments.run_phase9_live_forecast`.

This is exactly the probabilistic framing the whole project is built
around: a team having a high simulated title probability ten matches into
a season mostly reflects last season's form carried forward, which is
honest, not a flaw — a well-calibrated forecast this early should be
uncertain, and keeps updating automatically as more matches feed into the
same pipeline.

### Matchweek history

`GET /standings/history` lets the frontend show, for any matchweek that's
fully completed so far, both the table AND the model's own simulated
forecast exactly as they stood right after that matchweek finished - via
a dropdown on the standings page. The live data source
(`openfootball/football.json`) tags every match with a "round" field
("Matchday N"), and always returns the complete season-to-date match list
on every fetch — so instead of saving a snapshot each time a matchweek
finishes (which would need somewhere durable to write it, a real problem
on Render's free tier: the container's filesystem is wiped on every cold
restart, which happens routinely after 15 minutes of inactivity), the
whole history is recomputed from the live source on every cache refresh.

Two genuinely different things get recomputed, at two very different
costs:

- **The table** (`src/features/league_table.py::standings_by_matchweek`):
  replaying the same lightweight sequential tracker used everywhere else
  in this project up to each complete round. Costs nothing meaningful (a
  few hundred matches through a Python dict, not the Monte Carlo
  simulator) and can never lose history to a restart.
- **The forecast** (`src/live_forecast.py::_matchweek_forecasts`): a
  genuine refit of the Poisson/Dixon-Coles model plus a full Monte Carlo
  re-simulation, using only data that would have been available right
  after that matchweek - not the current forecast replayed backward.
  This is real, repeated work (confirmed by direct benchmarking: ~2s per
  matchweek at a reduced 10,000 simulations, given how many remaining
  fixtures an early matchweek still has to simulate), so unlike the
  table, it's cached ACROSS refreshes at the `app/backend/cache.py`
  level: a completed matchweek's forecast never changes once computed
  (it depends only on fixed past data), so each refresh only pays the
  simulation cost for a matchweek that just newly became complete, not
  for every matchweek all over again.

A matchweek only appears once every one of its fixtures has a result —
"the table after matchweek N" isn't well-defined while N is still being
played. Only available for the live current season: the Kaggle-sourced
historical seasons have no round field at all.

## Hyperparameter tuning (Phase 10)

`src/evaluation/tuning.py` implements nested validation: all 33 seasons
split chronologically into an initial training block (10 seasons), a
validation block used ONLY for tuning (5 seasons), and a final test block
(18 seasons) the tuning process never sees — so hyperparameters aren't fit
to the same seasons used to report how good they are.

### Result: every model improved, and the ranking held

| Model | Default log loss | Tuned log loss | Δ |
|---|---|---|---|
| **Elo** | 0.9814 | **0.9771** | −0.0043 |
| Logistic Regression | 0.9782 | 0.9779 | −0.0003 |
| XGBoost | 0.9844 | 0.9814 | −0.0030 |
| Poisson (Dixon-Coles) | 0.9923 | 0.9861 | −0.0062 |

**Elo still wins after every other model gets tuned its own fair shot** —
a more robust version of the "simple approach wins" finding, since it
rules out "Elo only won because nobody tuned the others."

### A real train/deploy mismatch, found and fixed

Applying Poisson's tuned hyperparameters to the live forecast produced a
visibly wrong result: **Hull City — newly promoted, one match played —
came out as the #2 title favorite at 30.7%.** Root cause: tuning only ever
validated *whole-season-ahead* predictions, never "predict the rest of a
season after 1-2 matchdays" — the live forecast's actual scenario. Fixed
by having the live forecast explicitly use the original, more conservative
values (`xi=0.3`, `promoted_penalty=0.4`) rather than the tuned class
default — see `src/live_forecast.py`. Reproduce tuning with
`python -m experiments.run_phase10_tuning` (a few minutes; XGBoost's
25-candidate random search is the slow part).

## API (Phase 11)

`app/backend/` is a thin FastAPI layer with no modeling or simulation
logic of its own — every endpoint calls into `src/live_forecast.py`. A
forecast build costs a few real seconds (live fetch + refit + 50,000
simulations), so `app/backend/cache.py` holds one in-memory copy behind a
lock, refreshed automatically every `FORECAST_REFRESH_INTERVAL_HOURS`
(default 3) via the FastAPI lifespan context manager, plus a non-blocking
warmup on startup. A failed background refresh is caught and logged rather
than crashing the loop — `GET /meta` reports whether the last attempt
succeeded.

| Endpoint | Returns |
|---|---|
| `GET /health` | Liveness check |
| `GET /meta` | Season, simulation count, matches played/remaining, generation time |
| `GET /standings` | Current table |
| `GET /forecast` | Title/Champions-League/relegation probability + expected position/points, every team |
| `GET /teams/{team}` | The `/forecast` row for one team |
| `GET /teams/{team}/position-distribution` | P(finish in position N) for every N |
| `GET /matches/upcoming` | Remaining fixtures with expected goals |
| `GET /matches/predict?home=X&away=Y` | Expected goals, W/D/L probabilities, top-5 scorelines |
| `POST /simulation/refresh` | Forces an on-demand re-fetch + refit + re-simulation |

Unknown team names return 404, `/matches/predict` with two identical teams
returns 400, missing required params return 422. `tests/test_api.py` covers
the HTTP layer against a real fitted model on synthetic data.

## Frontend dashboard (Phase 12)

`app/frontend/` is a React + TypeScript dashboard (Vite, React Router,
Recharts) — a standings/forecast page and a per-team detail page with a
finishing-position distribution chart, points-range visual, and upcoming
fixtures. No modeling logic here either. Chart design uses a validated
(colorblind-safety-checked), light/dark-aware categorical/status/sequential
palette. See `app/frontend/README.md` for details.

## Simulation calibration backtest (Phase 13)

Phases 3-7 validated the *match-level* model's calibration. Phase 8
validated the *simulator*, but only against one historical instance. This
phase asks the real question: across many different forecasts, does "this
team has a 65% title probability" correspond to winning the title ~65% of
the time?

`src/simulation/calibration_backtest.py` replays the live pipeline against
every season from 2003-04 onward at three cutoffs each (25%/50%/75%
through the season), comparing simulated probabilities against what
actually happened. Training data is strictly seasons before the target
season — guarded by a dedicated regression test that spies on which
seasons the model was actually fit on.

**Results** (`experiments/run_phase13_simulation_calibration.py`, 1,380
pooled (season, cutoff, team) instances):

| Outcome | ECE |
|---|---|
| Title | 0.009 |
| Top-4 (Champions League) | 0.016 |
| Relegation | 0.020 |

![Simulator calibration](../experiments/simulation_calibration.png)

All three curves track the diagonal closely — the strongest evidence in
the project that the full pipeline, not just the underlying match model in
isolation, produces honest probabilities.
