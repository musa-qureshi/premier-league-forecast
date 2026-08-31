interface StatTileProps {
  label: string;
  value: string;
  tone?: "good" | "critical" | "neutral";
}

/** Stat tile: label (sentence case, no trailing colon), value (semibold,
 * proportional figures - never tabular-nums at display size). */
export function StatTile({ label, value, tone = "neutral" }: StatTileProps) {
  return (
    <div className="stat-tile">
      <div className="stat-tile-label">{label}</div>
      <div className={`stat-tile-value stat-tile-value--${tone}`}>{value}</div>
    </div>
  );
}
