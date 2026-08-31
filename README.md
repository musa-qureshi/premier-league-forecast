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

🚧 **Phase 2 (feature engineering) complete.** See the roadmap below.

- [x] Phase 1 — Data ingestion & cleaning
- [x] Phase 2 — Feature engineering (Elo, rolling form, league context)
- [ ] Phase 3 — Baselines
- [ ] Phase 4 — Statistical models (Elo-derived, logistic regression, Poisson/Dixon-Coles)
- [ ] Phase 5 — ML model (XGBoost) + explainability
- [ ] Phase 6 — Evaluation (backtesting, calibration)
- [ ] Phase 7 — Monte Carlo simulation engine
- [ ] Phase 8 — Current-season forecast (live data refresh)
- [ ] Phase 9 — API (FastAPI)
- [ ] Phase 10 — Frontend dashboard (React)

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

68 tests pass across the full suite (`pytest tests/`), all against
synthetic data - no test depends on the downloaded dataset being present.

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
  models/               # (Phase 3+) baseline, Elo, logistic, Poisson, XGBoost
  evaluation/           # (Phase 6) metrics, calibration, backtesting
  simulation/            # (Phase 7) Monte Carlo season simulator
  visualization/          # chart helpers
experiments/            # versioned experiment/backtest results
tests/                  # pytest suite
app/
  backend/               # (Phase 9) FastAPI - no modeling logic, calls src/
  frontend/               # (Phase 10) React + TypeScript dashboard
configs/                 # data.yaml and future model/simulation configs
```

## Setup

```bash
pip install -r requirements.txt
python -m src.data.ingest             # data/raw/kaggle_epl/results.csv
python -m src.data.validate           # data/processed/matches.parquet
python -m src.features.build_dataset  # data/processed/features.parquet
pytest tests/
```
