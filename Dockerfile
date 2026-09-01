# Backend-only image: app/backend/ (FastAPI) + the src/ pipeline it calls
# into. The frontend is a separate static build (see app/frontend/README.md)
# meant for a static host (Vercel etc.), not this image.
#
# No modeling/training happens at build or boot time - matches.parquet is
# already a committed, processed artifact (see .gitignore's explicit
# exception for it) and every model here fits in-process in well under a
# second (Phase 10). Nothing here needs a GPU or a build farm.
FROM python:3.13-slim

WORKDIR /app

# requirements-api.txt, NOT the repo-root requirements.txt - a deliberately
# minimal, verified subset (see that file's own header comment for how it
# was derived and why). The full requirements.txt pulls in scikit-learn,
# xgboost, statsmodels, matplotlib/seaborn, shap, and kaggle - none of
# which the live API actually imports at runtime - and installing all of
# that here would balloon both build time and image size for no benefit.
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Only what app/backend/main.py's import graph actually reaches at
# runtime: src/ (the pipeline), app/backend/ (the API layer), configs/
# (data.yaml), and the two small data files build_current_forecast() reads
# directly. Deliberately NOT data/raw/, notebooks/, experiments/, tests/,
# or app/frontend/ - see .dockerignore.
COPY src/ src/
COPY app/backend/ app/backend/
COPY configs/ configs/
COPY data/external/ data/external/
COPY data/processed/matches.parquet data/processed/matches.parquet

EXPOSE 8000

# $PORT is set by most PaaS hosts (Render included) at runtime and can
# vary per deploy - 8000 is only the local fallback. Shell form (not exec
# JSON-array form) specifically so this env var actually gets expanded.
CMD uvicorn app.backend.main:app --host 0.0.0.0 --port ${PORT:-8000}
