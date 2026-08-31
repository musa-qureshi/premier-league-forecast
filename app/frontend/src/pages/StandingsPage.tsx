import { Link } from "react-router-dom";
import { api, type TeamForecast, type TeamStanding } from "../api";
import { ProbabilityMeter } from "../components/ProbabilityMeter";
import { useFetch } from "../hooks/useFetch";

/** Home page: the current table plus, side by side, what the simulator
 * says about how each team's season ends up. Deliberately a TABLE (not a
 * bar chart of 20 series) - this is categorical, many-column, team-
 * indexed data, exactly the case the dataviz skill calls out as "the
 * answer is sometimes not a chart". Probability columns get a meter
 * *and* the numeric percentage as text side by side - never color alone. */
export function StandingsPage() {
  const standings = useFetch(() => api.standings(), []);
  const forecast = useFetch(() => api.forecastAll(), []);

  if (standings.loading || forecast.loading) {
    return <p className="status-text">Loading current forecast&hellip;</p>;
  }
  if (standings.error || forecast.error) {
    return (
      <p className="status-text status-text--error">
        Couldn't reach the API ({standings.error ?? forecast.error}). Is the backend running
        (<code>uvicorn app.backend.main:app --reload</code>)?
      </p>
    );
  }

  const forecastByTeam = new Map<string, TeamForecast>(
    (forecast.data ?? []).map((f) => [f.team, f])
  );

  const rows: (TeamStanding & { forecast?: TeamForecast })[] = (standings.data ?? []).map(
    (s) => ({ ...s, forecast: forecastByTeam.get(s.team) })
  );

  return (
    <div>
      <h1>Current standings &amp; forecast</h1>
      <div className="table-scroll">
        <table className="standings-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Team</th>
              <th>P</th>
              <th>GD</th>
              <th>Pts</th>
              <th>Title probability</th>
              <th>Top-4 probability</th>
              <th>Relegation probability</th>
              <th>Exp. position</th>
              <th>Exp. points</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={row.team}>
                <td>{i + 1}</td>
                <td>
                  <Link to={`/teams/${encodeURIComponent(row.team)}`}>{row.team}</Link>
                </td>
                <td>{row.played}</td>
                <td>{row.goal_difference > 0 ? `+${row.goal_difference}` : row.goal_difference}</td>
                <td>
                  <strong>{row.points}</strong>
                </td>
                <td>
                  {row.forecast && (
                    <ProbabilityMeter probability={row.forecast.title_probability} tone="good" />
                  )}
                </td>
                <td>
                  {row.forecast && (
                    <ProbabilityMeter
                      probability={row.forecast.champions_league_probability}
                      tone="good"
                    />
                  )}
                </td>
                <td>
                  {row.forecast && (
                    <ProbabilityMeter
                      probability={row.forecast.relegation_probability}
                      tone="critical"
                    />
                  )}
                </td>
                <td>{row.forecast?.expected_position.toFixed(1)}</td>
                <td>{row.forecast?.expected_points.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
