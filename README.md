# Premier League Probabilistic Forecast & Monte Carlo Season Simulator

A probabilistic ML system for Premier League match and season forecasting:
Elo ratings, Poisson/Dixon-Coles goal modeling, logistic regression and
XGBoost, calibration-focused evaluation with time-based backtesting, and a
vectorized Monte Carlo season simulator producing title/top-4/relegation
probabilities. Portfolio project — every output is a probabilistic
forecast, not a prediction of what *will* happen.

**Live demo:** [premier-league-forecast.vercel.app](https://premier-league-forecast.vercel.app)
· [API docs](https://premier-league-forecast-api.onrender.com/docs)

## What it does

- Ingests and cleans 33 seasons (1993-94 → 2025-26, 12,704 matches) from three merged/cross-validated sources.
- Engineers Elo ratings, rolling form, and league-table-state features (all leakage-tested).
- Fits and backtests four model families season-by-season: Elo, Poisson/Dixon-Coles, logistic regression, XGBoost.
- Evaluates on log loss / Brier / RPS / calibration — not just accuracy — with nested-validation hyperparameter tuning.
- Simulates full seasons (50,000+ Monte Carlo trials, ~60k sims/sec) into title/European/relegation probabilities.
- Validates the simulator's own calibration across 69 historical (season, cutoff) instances, not just one example.
- Refreshes a live current-season forecast automatically and serves it via a FastAPI backend + React dashboard.
- Lets you browse both the table and the model's own forecast as they stood after any completed matchweek, each refit and re-simulated on only the data available at that point.
- Deployed: Docker backend on Render, static frontend on Vercel.

## Results

Expanding-window backtest, 19 held-out seasons, match-count-weighted:

| Model | Log Loss | Brier | RPS | Accuracy |
|---|---|---|---|---|
| Baseline (historical frequency) | 1.0651 | 0.6434 | 0.2281 | 45.7% |
| **Elo** | **0.9825** | **0.5850** | **0.1999** | **53.1%** |
| Poisson (Dixon-Coles) | 0.9890 | 0.5893 | 0.2020 | 52.5% |
| Logistic Regression | 0.9848 | 0.5855 | 0.2000 | 53.0% |
| XGBoost | 0.9907 | 0.5898 | 0.2017 | 52.2% |

Elo edges out the rest, including XGBoost with ~50 engineered features —
reported honestly rather than picking the more sophisticated-looking
model. Match-level calibration ranking (ECE) doesn't track log-loss
ranking (Poisson calibrates best, Elo worst, despite Elo's better log
loss) — the reason this project checks calibration as its own step.
Season-level simulator calibration (title/top-4/relegation probabilities
checked against 69 real historical instances): ECE of 0.009 / 0.016 /
0.020 — all three reliability curves track the diagonal closely.

Full methodology, per-phase writeups, and reproduction commands for every
number above: [`docs/methodology.md`](docs/methodology.md) (and each
module's own docstring) — kept out of this README to keep it a README.

## Notable things found and fixed along the way

- A COVID-delayed season boundary silently misfiled 66 matches into the wrong season — caught by an automated match-count sanity check.
- Tuned hyperparameters (validated only for whole-season predictions) made a newly-promoted team with one match played a 30.7% title favorite in the live forecast — traced to a train/deploy mismatch and fixed with a more conservative live-forecast config.
- A production OOM crash on Render's free tier (512MB) traced to `int64` simulation arrays; fixed by downcasting to `int16` (958MB → 450MB peak, zero behavior change).
- A timestamp serialized without a timezone marker made the frontend's "updated N ago" display permanently stuck, off by the visitor's UTC offset.

## Architecture

```
src/            data ingestion/cleaning, feature engineering, models,
                evaluation, Monte Carlo simulation, live-forecast pipeline
app/backend/    FastAPI - no modeling logic, calls into src/
app/frontend/   React + TypeScript dashboard, calls the API only
experiments/    reproducible experiment scripts + saved results
tests/          pytest suite (219+ tests, all on synthetic data)
docs/           full phase-by-phase methodology writeup
Dockerfile, render.yaml, requirements-api.txt   backend deployment
```

## Setup

```bash
pip install -r requirements.txt

# One-time: Kaggle API token at kaggle.com -> Settings -> API,
# placed at ~/.kaggle/kaggle.json (see src/data/ingest.py docstring).
python -m src.data.ingest      # downloads raw sources
python -m src.data.validate    # -> data/processed/matches.parquet
python -m src.features.build_dataset  # -> data/processed/features.parquet
pytest tests/
```

Run the API + dashboard locally:

```bash
uvicorn app.backend.main:app --reload   # http://127.0.0.1:8000/docs
cd app/frontend && npm install && npm run dev
```

Build and run the deployable backend image:

```bash
docker build -t plforecast-api .
docker run -p 8000:8000 plforecast-api
```

Deployment steps (Render + Vercel) are in [`docs/deployment.md`](docs/deployment.md).
