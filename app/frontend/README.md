# Frontend (Phase 12)

React + TypeScript dashboard for the Premier League forecast API
(`app/backend/`). Built with Vite, React Router, and Recharts.

## Run locally

Needs the backend running first (see the project root README, "API (Phase
11)"):

```bash
# from the repo root, in one terminal
uvicorn app.backend.main:app --reload

# in another terminal
cd app/frontend
npm install
npm run dev
```

Then open the URL Vite prints (typically `http://localhost:5173`).

By default the frontend expects the API at `http://127.0.0.1:8000` (the
default `uvicorn --reload` port). To point it elsewhere, create
`app/frontend/.env.local` (gitignored) with:

```
VITE_API_BASE=http://127.0.0.1:8123
```

## Pages

- `/` - current standings combined with the simulator's forecast
  (title/top-4/relegation probability, expected position/points) for
  every team, as a sortable-by-eye table with probability meters.
- `/teams/:team` - one team's detail: stat tiles, a finishing-position
  distribution chart, a points-distribution range, and its next 5
  remaining fixtures with expected goals.

## Design notes

- Color palette and chart mark specs follow the project's dataviz
  skill reference palette (`src/theme.css`) - categorical/status/
  sequential roles as CSS custom properties, validated for colorblind
  safety via that skill's `validate_palette.js`, with separate light
  and dark values (`prefers-color-scheme` + a `data-theme` override).
- The finishing-position distribution is a bar chart (magnitude per
  discrete category - the right form per the dataviz skill), not a line;
  the points distribution is a labeled range, not a histogram, since the
  API only exposes summary statistics (mean/median/p05/p95), not the raw
  simulation draws - a histogram would fabricate detail the data doesn't
  support.
- No modeling/simulation logic lives here - every number comes from the
  API (`src/api.ts`), which itself has no modeling logic either (see the
  backend's own README section).

## Verification

No frontend unit test suite (Vitest/RTL) is included - the app was instead
verified against the real running backend with a headless-browser script
(navigate, screenshot, click into a team, screenshot, check for console
errors), in both light and dark color schemes. That's a reasonable scope
call for a presentation layer over an already-extensively-tested API and
model layer (199 backend tests), not an oversight; adding a proper
component test suite would be a reasonable next step if this were headed
toward production rather than a portfolio deliverable.
