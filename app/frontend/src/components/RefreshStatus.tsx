import type { ForecastMeta } from "../api";
import { timeAgo } from "../utils/time";

interface RefreshStatusProps {
  meta: ForecastMeta | null;
}

/** Answers "does this update automatically" directly in the UI: the
 * server refreshes itself in the background on a fixed interval (see
 * app/backend/cache.py) - purely informational, no manual trigger here,
 * since the automatic schedule is the whole point. */
export function RefreshStatus({ meta }: RefreshStatusProps) {
  if (!meta) return null;
  return (
    <div className="refresh-status">
      <span className="refresh-status-text">
        Updated {timeAgo(meta.generated_at)}
        <span className="refresh-status-dot">&middot;</span>
        auto-refreshes every {meta.refresh_interval_hours}h
        {meta.last_background_refresh_error && (
          <span className="refresh-status-error"> (last auto-refresh failed)</span>
        )}
      </span>
    </div>
  );
}
