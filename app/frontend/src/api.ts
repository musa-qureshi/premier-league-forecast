// Typed client for the FastAPI backend (app/backend/main.py). Types here
// mirror app/backend/schemas.py field-for-field - if a backend response
// shape changes, update both together.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export interface TeamStanding {
  team: string;
  points: number;
  played: number;
  goals_for: number;
  goals_against: number;
  goal_difference: number;
}

export interface TeamForecast {
  team: string;
  title_probability: number;
  champions_league_probability: number;
  european_qualification_probability: number;
  relegation_probability: number;
  expected_position: number;
  expected_points: number;
  median_points: number;
  points_p05: number;
  points_p95: number;
}

export interface MatchweekStandings {
  matchweek: number;
  standings: TeamStanding[];
  forecast: TeamForecast[];
}

export interface PositionDistribution {
  team: string;
  distribution: Record<string, number>;
}

export interface RemainingFixture {
  home_team: string;
  away_team: string;
  expected_home_goals: number;
  expected_away_goals: number;
}

export interface Scoreline {
  home_goals: number;
  away_goals: number;
  probability: number;
}

export interface MatchPrediction {
  home_team: string;
  away_team: string;
  expected_home_goals: number;
  expected_away_goals: number;
  home_win_probability: number;
  draw_probability: number;
  away_win_probability: number;
  most_likely_scorelines: Scoreline[];
}

export interface ForecastMeta {
  season: string;
  generated_at: string;
  n_simulations: number;
  n_played: number;
  n_remaining: number;
  refresh_interval_hours: number;
  last_background_refresh_attempt_at: string | null;
  last_background_refresh_error: string | null;
}

async function getJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

async function postJSON<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { method: "POST" });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  meta: () => getJSON<ForecastMeta>("/meta"),
  standings: () => getJSON<TeamStanding[]>("/standings"),
  standingsHistory: () => getJSON<MatchweekStandings[]>("/standings/history"),
  forecastAll: () => getJSON<TeamForecast[]>("/forecast"),
  teamForecast: (team: string) => getJSON<TeamForecast>(`/teams/${encodeURIComponent(team)}`),
  positionDistribution: (team: string) =>
    getJSON<PositionDistribution>(`/teams/${encodeURIComponent(team)}/position-distribution`),
  upcomingMatches: () => getJSON<RemainingFixture[]>("/matches/upcoming"),
  predictMatch: (home: string, away: string) =>
    getJSON<MatchPrediction>(
      `/matches/predict?home=${encodeURIComponent(home)}&away=${encodeURIComponent(away)}`
    ),
  refreshSimulation: () => postJSON<ForecastMeta>("/simulation/refresh"),
};
