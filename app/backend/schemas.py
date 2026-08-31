"""Pydantic response models for the API. Pure data shape definitions -
no modeling or computation happens here, only typed structure for
FastAPI's automatic request validation and OpenAPI docs generation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TeamStanding(BaseModel):
    team: str
    points: int
    played: int
    goals_for: int
    goals_against: int
    goal_difference: int


class TeamForecast(BaseModel):
    team: str
    title_probability: float = Field(..., description="Simulated probability of finishing 1st")
    champions_league_probability: float = Field(..., description="Simulated probability of a top-4 finish")
    european_qualification_probability: float = Field(
        ..., description="Simulated probability of any European qualification spot (config-dependent cutoff)"
    )
    relegation_probability: float
    expected_position: float = Field(..., description="Mean finishing position across all simulations")
    expected_points: float
    median_points: float
    points_p05: float = Field(..., description="5th percentile of simulated final points")
    points_p95: float = Field(..., description="95th percentile of simulated final points")


class PositionDistribution(BaseModel):
    team: str
    distribution: dict[str, float] = Field(
        ..., description="Finishing position (as a string key, 1-indexed) -> simulated probability"
    )


class RemainingFixture(BaseModel):
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float


class Scoreline(BaseModel):
    home_goals: int
    away_goals: int
    probability: float


class MatchPrediction(BaseModel):
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    home_win_probability: float
    draw_probability: float
    away_win_probability: float
    most_likely_scorelines: list[Scoreline]


class ForecastMeta(BaseModel):
    season: str
    generated_at: str
    n_simulations: int
    n_played: int
    n_remaining: int
