import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, type TeamForecast, type TeamStanding } from "../api";
import { ProbabilityMeter } from "../components/ProbabilityMeter";
import { RefreshStatus } from "../components/RefreshStatus";
import { TeamBadge } from "../components/TeamBadge";
import { useFetch } from "../hooks/useFetch";

type Row = TeamStanding & { forecast?: TeamForecast };

type SortKey = "table" | "title_probability" | "relegation_probability" | "expected_position" | "expected_points";

interface SortConfig {
  key: SortKey;
  direction: "asc" | "desc";
}

// Each sortable column's natural first-click direction - e.g. "expected
// position" wants ascending (1st is best) by default, while points and
// probabilities want descending (highest first) - including the table's
// own points column, which is why its default is "desc", not "asc": a
// league table leads with the top team, not the bottom one.
const DEFAULT_DIRECTION: Record<SortKey, "asc" | "desc"> = {
  table: "desc",
  title_probability: "desc",
  relegation_probability: "desc",
  expected_position: "asc",
  expected_points: "desc",
};

const COLUMN_LABELS: Record<Exclude<SortKey, "table">, string> = {
  title_probability: "Title probability",
  relegation_probability: "Relegation probability",
  expected_position: "Exp. position",
  expected_points: "Exp. points",
};

function sortRows(rows: Row[], sort: SortConfig): Row[] {
  const sorted = [...rows];
  // "asc" always means "ascending by value" (lowest first) for every
  // column, table included: asc on points genuinely means worst-team-
  // first, matching what the arrow indicator shows. The DEFAULT for the
  // table column is "desc" precisely so the page loads leader-first.
  const factor = sort.direction === "asc" ? 1 : -1;

  sorted.sort((a, b) => {
    if (sort.key === "table") {
      return (a.points - b.points || a.goal_difference - b.goal_difference) * factor;
    }
    const av = a.forecast?.[sort.key];
    const bv = b.forecast?.[sort.key];
    if (av == null || bv == null) return 0;
    return (av - bv) * factor;
  });
  return sorted;
}

function SortableHeader({
  columnKey,
  label,
  active,
  direction,
  onClick,
  align = "left",
}: {
  columnKey: SortKey;
  label: string;
  active: boolean;
  direction: "asc" | "desc";
  onClick: (key: SortKey) => void;
  align?: "left" | "right";
}) {
  return (
    <th
      className={`sortable-header ${active ? "sortable-header--active" : ""} ${align === "right" ? "num" : ""}`}
    >
      <button type="button" onClick={() => onClick(columnKey)}>
        {label}
        <span className="sort-arrow">{active ? (direction === "asc" ? "↑" : "↓") : ""}</span>
      </button>
    </th>
  );
}

const POLL_INTERVAL_MS = 5 * 60 * 1000; // re-check every 5 minutes

export function StandingsPage() {
  // No manual refresh control - the backend refreshes itself automatically
  // (see app/backend/cache.py). This tick just re-fetches periodically so
  // "Updated N ago" and the table stay honest for a tab left open across
  // one of those automatic refreshes, rather than freezing at page-load.
  const [pollTick, setPollTick] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setPollTick((t) => t + 1), POLL_INTERVAL_MS);
    return () => clearInterval(id);
  }, []);

  const standings = useFetch(() => api.standings(), [pollTick]);
  const forecast = useFetch(() => api.forecastAll(), [pollTick]);
  const meta = useFetch(() => api.meta(), [pollTick]);
  const [sort, setSort] = useState<SortConfig>({ key: "table", direction: DEFAULT_DIRECTION.table });

  const rows: Row[] = useMemo(() => {
    const forecastByTeam = new Map<string, TeamForecast>((forecast.data ?? []).map((f) => [f.team, f]));
    const base = (standings.data ?? []).map((s) => ({ ...s, forecast: forecastByTeam.get(s.team) }));
    return sortRows(base, sort);
  }, [standings.data, forecast.data, sort]);

  function handleSort(key: SortKey) {
    setSort((prev) =>
      prev.key === key
        ? { key, direction: prev.direction === "asc" ? "desc" : "asc" }
        : { key, direction: DEFAULT_DIRECTION[key] }
    );
  }

  if (standings.loading || forecast.loading) {
    return <div className="page-shell"><p className="status-text">Loading current forecast&hellip;</p></div>;
  }
  if (standings.error || forecast.error) {
    return (
      <div className="page-shell">
        <p className="status-text status-text--error">
          Couldn't reach the API ({standings.error ?? forecast.error}). Is the backend running
          (<code>uvicorn app.backend.main:app --reload</code>)?
        </p>
      </div>
    );
  }

  return (
    <div className="page-shell">
      <div className="page-heading">
        <div className="page-heading-row">
          <div>
            <h1>Current standings &amp; forecast</h1>
            <p className="page-subtitle">
              Simulated across 50,000 seasons from the current table and remaining fixtures.
            </p>
          </div>
          <RefreshStatus meta={meta.data} />
        </div>
      </div>
      <div className="table-card">
        <div className="table-scroll">
          <table className="standings-table">
            <thead>
              <tr>
                <th className="col-rank">#</th>
                <th className="col-team">Team</th>
                <th className="num">P</th>
                <th className="num">GD</th>
                <th
                  className={`sortable-header num ${sort.key === "table" ? "sortable-header--active" : ""}`}
                >
                  <button type="button" onClick={() => handleSort("table")}>
                    Pts
                    <span className="sort-arrow">
                      {sort.key === "table" ? (sort.direction === "asc" ? "↑" : "↓") : ""}
                    </span>
                  </button>
                </th>
                <SortableHeader
                  columnKey="title_probability"
                  label="Title"
                  active={sort.key === "title_probability"}
                  direction={sort.direction}
                  onClick={handleSort}
                />
                <th>Top-4</th>
                <SortableHeader
                  columnKey="relegation_probability"
                  label="Relegation"
                  active={sort.key === "relegation_probability"}
                  direction={sort.direction}
                  onClick={handleSort}
                />
                <SortableHeader
                  columnKey="expected_position"
                  label="Exp. pos."
                  active={sort.key === "expected_position"}
                  direction={sort.direction}
                  onClick={handleSort}
                  align="right"
                />
                <SortableHeader
                  columnKey="expected_points"
                  label="Exp. points"
                  active={sort.key === "expected_points"}
                  direction={sort.direction}
                  onClick={handleSort}
                  align="right"
                />
              </tr>
            </thead>
            <tbody>
              {rows.map((row, i) => (
                <tr key={row.team}>
                  <td className="col-rank">{i + 1}</td>
                  <td className="col-team">
                    <Link to={`/teams/${encodeURIComponent(row.team)}`} className="team-link">
                      <TeamBadge team={row.team} />
                      {row.team}
                    </Link>
                  </td>
                  <td className="num">{row.played}</td>
                  <td className="num">{row.goal_difference > 0 ? `+${row.goal_difference}` : row.goal_difference}</td>
                  <td className="num">
                    <strong>{row.points}</strong>
                  </td>
                  <td>
                    {row.forecast && <ProbabilityMeter probability={row.forecast.title_probability} tone="good" />}
                  </td>
                  <td>
                    {row.forecast && (
                      <ProbabilityMeter probability={row.forecast.champions_league_probability} tone="good" />
                    )}
                  </td>
                  <td>
                    {row.forecast && (
                      <ProbabilityMeter probability={row.forecast.relegation_probability} tone="critical" />
                    )}
                  </td>
                  <td className="num">{row.forecast?.expected_position.toFixed(1)}</td>
                  <td className="num">
                    {row.forecast?.expected_points.toFixed(1)}
                    {row.forecast && (
                      <span className="points-range-inline">
                        {row.forecast.points_p05.toFixed(0)}&ndash;{row.forecast.points_p95.toFixed(0)}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="table-caption">
        Click a column header ({Object.values(COLUMN_LABELS).join(", ")}) to sort by it.
      </p>
    </div>
  );
}
