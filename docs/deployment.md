# Deployment

How [premier-league-forecast](../README.md) is deployed (backend on
Render, frontend on Vercel), plus two real production incidents found and
fixed after the first deploy. See [methodology.md](methodology.md) for the
modeling/evaluation writeup.

A project a reviewer can actually open in a browser is worth more than the
same project sitting in a cloned repo, so the backend and frontend are each
independently deployable to free hosting tiers — the backend as a Docker
container (works on Render, Railway, Fly.io, or any other host that builds
from a `Dockerfile`), the frontend as a static Vite build (works on Vercel,
Netlify, or any static host). Neither needs a paid tier for a portfolio's
level of traffic.

## Backend → Render (or any Docker host)

The root [`Dockerfile`](../Dockerfile) builds a standalone image for
`app/backend/` alone. Two things make it noticeably leaner than just
containerizing the whole project:

- **[`requirements-api.txt`](../requirements-api.txt)**, a verified minimal
  subset of the repo-root `requirements.txt` — only what
  `app/backend/main.py`'s import graph actually reaches at runtime
  (`fastapi`, `uvicorn`, `pandas`, `numpy`, `pyarrow`, `scipy`, `requests`,
  `pyyaml`). `scikit-learn`, `xgboost`, `statsmodels`, `matplotlib`/
  `seaborn`, `shap`, and `kaggle` are all in the full requirements.txt but
  never imported by anything the live API path calls. This cut the built
  image from 2.19GB to 855MB and the dependency-install step from ~142s to
  ~48s.
- **`data/processed/matches.parquet`** is committed as a deliberate,
  documented exception to `.gitignore` — it's small (~230KB) and the
  deployed container has no Kaggle credentials and can't reach
  football-data.co.uk anyway.

`app/backend/main.py`'s `lifespan` fires a non-blocking cache-warmup task
at startup (`_warm_cache_on_startup()`) — the app still starts accepting
connections (health checks included) immediately, but the first real
visitor no longer eats the full build cost (live fetch + refit + 50,000
simulations) themselves the way a purely-lazy design would on a
just-booted container.

To deploy (needs your own free Render account):

1. Push this repo to GitHub.
2. Render dashboard → **New +** → **Blueprint** → point it at this repo.
   [`render.yaml`](../render.yaml) declares the whole service (Docker
   build, free plan, `/health` as the health-check path). (No blueprint
   access? **New +** → **Web Service** → this repo → environment
   **Docker** — Render detects the root `Dockerfile` automatically.)
3. Once deployed, sanity-check: `curl https://<your-url>/health` should
   return `{"status":"ok"}`.

Render's free web services spin down after 15 minutes idle, so a visitor
after a quiet period waits ~30-50s for the container to boot before the
warmup task even starts — a real limitation of the free tier, not
something this project's code can hide. A paid "always-on" instance
removes it entirely.

## Frontend → Vercel (or any static host)

No Docker needed — `app/frontend/` is a standard Vite + React app, and
Vercel's zero-config Vite detection handles the build. This is a monorepo
(frontend isn't at the repo root), so the one thing to set explicitly is
the project's root directory:

1. Vercel dashboard → **Add New** → **Project** → import this repo.
2. **Root Directory**: `app/frontend`.
3. **Environment Variables**: `VITE_API_BASE` = your Render backend URL
   (e.g. `https://premier-league-forecast-api.onrender.com`, no trailing
   slash). Without this the deployed frontend falls back to `api.ts`'s dev
   default of `http://127.0.0.1:8000`, which won't resolve from a
   visitor's browser.
4. Deploy.

The backend's CORS policy is wide open (`allow_origins=["*"]`) precisely
so this works with zero backend-side configuration — reasonable here since
the API has no auth, no user-specific state, and serves only public
football data.

## Incident: free-tier OOM crash

The first real deployment crashed — `/health` returned a fast `502`
(nothing there to answer, not a slow timeout), pointing at the process
dying outright. Reproduced locally by running the built image under the
same `--memory=512m` cap Render's free tier imposes: confirmed, OOM-killed
a few seconds after boot, every time.

**Cause**: `src/simulation/engine.py` simulates every remaining fixture's
scoreline for every simulated season as one big `(n_simulations x
n_fixtures)` array, and `rng.poisson()` always returns `int64` (8 bytes).
At production scale early in a season (50,000 simulations × ~370 remaining
fixtures), the four such arrays this module builds measured a combined
**~958MB peak RSS** — confirmed by direct profiling
(`resource.getrusage().ru_maxrss`) inside the container, not guessed at.

**Fix**: cast those arrays down to `int16` immediately after generation. A
football score is never within three orders of magnitude of int16's
~32,000 headroom, so this changes zero simulated values — a storage-width
fix, not a behavior change. Cut the measured peak from 958MB to **~450MB**,
re-verified against the real containerized app surviving multiple real
requests under the identical 512MB cap.

## Incident: stuck "updated N hours ago"

The frontend's "updated N ago" display looked permanently frozen at "3
hours ago" regardless of how recently the forecast had actually refreshed.

**Cause**: the backend built `generated_at` with a plain
`pd.Timestamp.now()` — no timezone attached — and serialized it as-is
(e.g. `2026-09-01T08:04:33`, no "Z"/offset). A browser's `new Date(...)`
interprets an ISO timestamp with no timezone marker as *its own local
time*, not UTC — silently adding the visitor's UTC offset as phantom
staleness. Since `timeAgo()` only shows whole hours past 60 minutes, a
constant few-hour offset error looks "stuck" even as the real, much
smaller elapsed time keeps ticking underneath it.

**Fix**: `pd.Timestamp.now(tz="UTC")` at both construction sites in
`src/live_forecast.py`, so `generated_at` serializes with an explicit
`+00:00` — verified via the real container's `/meta` response. Matches a
pattern already used correctly elsewhere in the codebase
(`cache.py`'s `datetime.now(timezone.utc)`).
