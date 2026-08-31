"""Unit tests for app/backend/main.py.

The API's get_forecast() call is monkeypatched to return a small synthetic
LiveForecast fixture (built from a genuinely fitted DixonColesModel on toy
data, not a mock of the model itself) - consistent with this project's
rule that no test depends on the downloaded dataset or network access.
This tests the HTTP layer (routing, request validation, response shape,
404 handling) against real model/simulation output, not a stub.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import app.backend.main as main_module
from src.live_forecast import LiveForecast
from src.models.poisson_model import DixonColesModel
from src.simulation.engine import CurrentTableState, Fixture, simulate_final_tables
from src.simulation.summary import summarize_simulation


def _toy_forecast() -> LiveForecast:
    teams = ["Alpha", "Bravo", "Charlie", "Delta"]
    rng = np.random.default_rng(0)
    rows = []
    date = pd.Timestamp("2020-08-01")
    for _ in range(20):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                rows.append({
                    "Date": date, "HomeTeam": home, "AwayTeam": away,
                    "FTHG": rng.poisson(1.3), "FTAG": rng.poisson(1.1),
                })
        date += pd.Timedelta(days=7)
    historical = pd.DataFrame(rows)

    model = DixonColesModel(use_correlation=True, xi=0.0).fit(historical)

    current = CurrentTableState(
        teams=teams,
        points={"Alpha": 6, "Bravo": 3, "Charlie": 1, "Delta": 0},
        goals_for={"Alpha": 5, "Bravo": 3, "Charlie": 2, "Delta": 1},
        goals_against={"Alpha": 1, "Bravo": 3, "Charlie": 3, "Delta": 5},
    )
    fixtures = [Fixture("Alpha", "Delta", 1.8, 0.7), Fixture("Bravo", "Charlie", 1.2, 1.1)]
    positions, points = simulate_final_tables(current, fixtures, n_simulations=500, seed=0)
    summary = summarize_simulation(positions, points)

    standings = pd.DataFrame({
        "points": current.points, "played": {t: 3 for t in teams},
        "goals_for": current.goals_for, "goals_against": current.goals_against,
        "goal_difference": {t: current.goals_for[t] - current.goals_against[t] for t in teams},
    })
    standings.index.name = "team"
    standings = standings.sort_values("points", ascending=False)

    remaining_fixtures = pd.DataFrame([
        {"HomeTeam": "Alpha", "AwayTeam": "Delta", "expected_home_goals": 1.8, "expected_away_goals": 0.7},
        {"HomeTeam": "Bravo", "AwayTeam": "Charlie", "expected_home_goals": 1.2, "expected_away_goals": 1.1},
    ])

    return LiveForecast(
        season="2099-00", generated_at=pd.Timestamp("2099-01-01"),
        n_simulations=500, n_played=12, n_remaining=2,
        standings=standings, remaining_fixtures=remaining_fixtures,
        positions=positions, points=points, summary=summary, model=model,
    )


@pytest.fixture
def client(monkeypatch):
    forecast = _toy_forecast()
    monkeypatch.setattr(main_module, "get_forecast", lambda force_refresh=False: forecast)
    return TestClient(main_module.app)


class TestHealth:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestMeta:
    def test_meta_fields(self, client):
        response = client.get("/meta")
        assert response.status_code == 200
        body = response.json()
        assert body["season"] == "2099-00"
        assert body["n_simulations"] == 500


class TestStandings:
    def test_returns_all_teams(self, client):
        response = client.get("/standings")
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 4
        assert {row["team"] for row in body} == {"Alpha", "Bravo", "Charlie", "Delta"}

    def test_sorted_by_points_descending(self, client):
        body = client.get("/standings").json()
        points = [row["points"] for row in body]
        assert points == sorted(points, reverse=True)


class TestForecastAll:
    def test_returns_all_teams_with_probabilities(self, client):
        body = client.get("/forecast").json()
        assert len(body) == 4
        for row in body:
            assert 0.0 <= row["title_probability"] <= 1.0
            assert 0.0 <= row["relegation_probability"] <= 1.0


class TestTeamForecast:
    def test_known_team_returns_200(self, client):
        response = client.get("/teams/Alpha")
        assert response.status_code == 200
        assert response.json()["team"] == "Alpha"

    def test_unknown_team_returns_404(self, client):
        response = client.get("/teams/Nonexistent")
        assert response.status_code == 404


class TestPositionDistribution:
    def test_distribution_sums_to_one(self, client):
        response = client.get("/teams/Alpha/position-distribution")
        assert response.status_code == 200
        dist = response.json()["distribution"]
        assert sum(dist.values()) == pytest.approx(1.0, abs=1e-6)

    def test_unknown_team_returns_404(self, client):
        response = client.get("/teams/Nonexistent/position-distribution")
        assert response.status_code == 404


class TestUpcomingMatches:
    def test_returns_remaining_fixtures(self, client):
        body = client.get("/matches/upcoming").json()
        assert len(body) == 2
        assert body[0]["home_team"] == "Alpha"
        assert body[0]["away_team"] == "Delta"


class TestPredictMatch:
    def test_valid_matchup_returns_valid_probabilities(self, client):
        response = client.get("/matches/predict", params={"home": "Alpha", "away": "Delta"})
        assert response.status_code == 200
        body = response.json()
        total = body["home_win_probability"] + body["draw_probability"] + body["away_win_probability"]
        assert total == pytest.approx(1.0, abs=1e-6)
        assert len(body["most_likely_scorelines"]) == 5

    def test_unknown_team_returns_404(self, client):
        response = client.get("/matches/predict", params={"home": "Alpha", "away": "Nonexistent"})
        assert response.status_code == 404

    def test_same_team_twice_returns_400(self, client):
        response = client.get("/matches/predict", params={"home": "Alpha", "away": "Alpha"})
        assert response.status_code == 400

    def test_missing_query_params_returns_422(self, client):
        response = client.get("/matches/predict", params={"home": "Alpha"})
        assert response.status_code == 422


class TestRefreshSimulation:
    def test_refresh_calls_get_forecast_with_force_refresh(self, client, monkeypatch):
        calls = []

        def fake_get_forecast(force_refresh=False):
            calls.append(force_refresh)
            return _toy_forecast()

        monkeypatch.setattr(main_module, "get_forecast", fake_get_forecast)
        response = client.post("/simulation/refresh")
        assert response.status_code == 200
        assert calls == [True]
