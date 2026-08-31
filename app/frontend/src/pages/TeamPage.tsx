import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { PointsRange } from "../components/PointsRange";
import { PositionDistributionChart } from "../components/PositionDistributionChart";
import { StatTile } from "../components/StatTile";
import { useFetch } from "../hooks/useFetch";

export function TeamPage() {
  const { team = "" } = useParams<{ team: string }>();

  const forecast = useFetch(() => api.teamForecast(team), [team]);
  const standings = useFetch(() => api.standings(), []);
  const distribution = useFetch(() => api.positionDistribution(team), [team]);
  const upcoming = useFetch(() => api.upcomingMatches(), []);

  if (forecast.loading || standings.loading) {
    return <p className="status-text">Loading&hellip;</p>;
  }
  if (forecast.error) {
    return (
      <div>
        <p className="status-text status-text--error">{forecast.error}</p>
        <Link to="/">&larr; Back to standings</Link>
      </div>
    );
  }

  const f = forecast.data!;
  const teamStanding = (standings.data ?? []).find((s) => s.team === team);
  const allTeamFixtures = (upcoming.data ?? []).filter(
    (m) => m.home_team === team || m.away_team === team
  );
  const NEXT_N_FIXTURES = 5;
  const teamFixtures = allTeamFixtures.slice(0, NEXT_N_FIXTURES);

  return (
    <div>
      <Link to="/">&larr; Back to standings</Link>
      <h1>{team}</h1>

      <div className="stat-tile-grid">
        {teamStanding && (
          <>
            <StatTile label="Current points" value={String(teamStanding.points)} />
            <StatTile
              label="Goal difference"
              value={
                teamStanding.goal_difference > 0
                  ? `+${teamStanding.goal_difference}`
                  : String(teamStanding.goal_difference)
              }
            />
          </>
        )}
        <StatTile
          label="Title probability"
          value={`${(f.title_probability * 100).toFixed(1)}%`}
          tone="good"
        />
        <StatTile
          label="Top-4 probability"
          value={`${(f.champions_league_probability * 100).toFixed(1)}%`}
          tone="good"
        />
        <StatTile
          label="Relegation probability"
          value={`${(f.relegation_probability * 100).toFixed(1)}%`}
          tone={f.relegation_probability > 0.05 ? "critical" : "neutral"}
        />
        <StatTile label="Expected final position" value={f.expected_position.toFixed(1)} />
        <StatTile label="Expected final points" value={f.expected_points.toFixed(1)} />
      </div>

      <section>
        <h2>Finishing-position distribution</h2>
        <p className="section-caption">
          Across every simulated season, how often {team} finished in each position.
          Shaded bands mark the title position, the European qualification zone, and
          the relegation zone.
        </p>
        {distribution.loading && <p className="status-text">Loading&hellip;</p>}
        {distribution.data && (
          <PositionDistributionChart distribution={distribution.data.distribution} />
        )}
      </section>

      <section>
        <h2>Points distribution</h2>
        <PointsRange
          p05={f.points_p05}
          median={f.median_points}
          p95={f.points_p95}
          mean={f.expected_points}
        />
      </section>

      <section>
        <h2>Upcoming fixtures</h2>
        <p className="section-caption">
          Next {teamFixtures.length} of {allTeamFixtures.length} remaining fixtures this season,
          with each side's expected goals from the fitted model.
        </p>
        {upcoming.loading && <p className="status-text">Loading&hellip;</p>}
        {teamFixtures.length === 0 && !upcoming.loading && (
          <p className="status-text">No remaining fixtures.</p>
        )}
        <ul className="fixture-list">
          {teamFixtures.map((m, i) => {
            const isHome = m.home_team === team;
            const opponent = isHome ? m.away_team : m.home_team;
            return (
              <li key={i}>
                <span className="fixture-venue">{isHome ? "vs" : "at"}</span>{" "}
                <Link to={`/teams/${encodeURIComponent(opponent)}`}>{opponent}</Link>
                <span className="fixture-xg">
                  xG {m.expected_home_goals.toFixed(2)}&ndash;{m.expected_away_goals.toFixed(2)}
                </span>
              </li>
            );
          })}
        </ul>
      </section>
    </div>
  );
}
