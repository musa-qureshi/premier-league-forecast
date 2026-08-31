interface PointsRangeProps {
  p05: number;
  median: number;
  p95: number;
  mean: number;
}

/** Not a chart, deliberately: the API only exposes four summary numbers
 * (mean/median/p05/p95), not the raw simulation draws, so a histogram
 * would be fabricating detail that isn't there. A labeled range track -
 * the 5th-95th percentile band with the median marked - communicates
 * "how much does the simulated outcome vary" honestly, at the precision
 * the data actually supports. */
export function PointsRange({ p05, median, p95, mean }: PointsRangeProps) {
  const span = p95 - p05 || 1;
  const medianPct = ((median - p05) / span) * 100;

  return (
    <div className="points-range">
      <div className="points-range-labels">
        <span>{p05.toFixed(0)} pts (5th pct.)</span>
        <span>{p95.toFixed(0)} pts (95th pct.)</span>
      </div>
      <div className="points-range-track">
        <div className="points-range-band" />
        <div className="points-range-median" style={{ left: `${medianPct}%` }} title={`Median: ${median.toFixed(0)} pts`} />
      </div>
      <div className="points-range-caption">
        Median <strong>{median.toFixed(0)}</strong> pts &middot; mean <strong>{mean.toFixed(1)}</strong> pts
      </div>
    </div>
  );
}
