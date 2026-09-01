import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";
import { PointsRange } from "../components/PointsRange";
import { PositionDistributionChart } from "../components/PositionDistributionChart";
import { StatTile } from "../components/StatTile";
import { TeamBadge } from "../components/TeamBadge";
import { useFetch } from "../hooks/useFetch";

const FIXTURES_PER_PAGE = 5;

export function TeamPage() {
  const { team = "" } = useParams<{ team: string }>();
  const [fixturePage, setFixturePage] = useState(0);

  const forecast = useFetch(() => api.teamForecast(team), [team]);
  const standings = useFetch(() => api.standings(), []);
  const distribution = useFetch(() => api.positionDistribution(team), [team]);
  const upcoming = useFetch(() => api.upcomingMatches(), []);

  if (forecast.loading || standings.loading) {
    return <div className="page-shell"><p className="status-text">Loading&hellip;</p></div>;
  }
  if (forecast.error) {
    return (
      <div className="page-shell">
        <p className="status-text status-text--error">{forecast.error}</p>
        <Link to="/" className="back-link">&larr; Back to standings</Link>
      </div>
    );
  }

  const f = forecast.data!;
  const teamStanding = (standings.data ?? []).find((s) => s.team === team);
  const allTeamFixtures = (upcoming.data ?? []).filter(
    (m) => m.home_team === team || m.away_team === team
  );

  const pageStart = fixturePage * FIXTURES_PER_PAGE;
  const pageFixtures = allTeamFixtures.slice(pageStart, pageStart + FIXTURES_PER_PAGE);
  const hasPrev = fixturePage > 0;
  const hasNext = pageStart + FIXTURES_PER_PAGE < allTeamFixtures.length;

  return (
    <div className="page-shell">
      <Link to="/" className="back-link">&larr; Back to standings</Link>

      <div className="team-heading">
        <TeamBadge team={team} size={48} />
        <h1>{team}</h1>
      </div>

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

      <section className="section-card">
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

      <section className="section-card">
        <h2>Points distribution</h2>
        <PointsRange
          p05={f.points_p05}
          median={f.median_points}
          p95={f.points_p95}
          mean={f.expected_points}
        />
      </section>

      <section className="section-card">
        <div className="section-header-row">
          <div>
            <h2>Upcoming fixtures</h2>
            <p className="section-caption">
              {allTeamFixtures.length === 0
                ? "No remaining fixtures."
                : `Fixtures ${pageStart + 1}–${Math.min(pageStart + FIXTURES_PER_PAGE, allTeamFixtures.length)} of ${allTeamFixtures.length} remaining, with each side's expected goals from the fitted model.`}
            </p>
          </div>
          {allTeamFixtures.length > FIXTURES_PER_PAGE && (
            <div className="fixture-pager">
              <button type="button" disabled={!hasPrev} onClick={() => setFixturePage((p) => p - 1)}>
                &larr; Previous
              </button>
              <button type="button" disabled={!hasNext} onClick={() => setFixturePage((p) => p + 1)}>
                Next &rarr;
              </button>
            </div>
          )}
        </div>
        {upcoming.loading && <p className="status-text">Loading&hellip;</p>}
        <ul className="fixture-list">
          {pageFixtures.map((m, i) => {
            const isHome = m.home_team === team;
            const opponent = isHome ? m.away_team : m.home_team;
            return (
              <li key={i}>
                <TeamBadge team={opponent} size={24} />
                <span className="fixture-venue">{isHome ? "(H)" : "(A)"}</span>
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
