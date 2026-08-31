interface ProbabilityMeterProps {
  probability: number; // 0-1
  tone: "good" | "critical" | "neutral";
  label?: string;
}

const TONE_FILL: Record<ProbabilityMeterProps["tone"], string> = {
  good: "var(--status-good)",
  critical: "var(--status-critical)",
  neutral: "var(--series-1)",
};

/** A meter: fill carries the value, unfilled track is a lighter step of
 * the surface so state reads across the whole bar. The percentage is
 * ALWAYS rendered as text alongside it - color never carries meaning
 * alone (dataviz skill: status colors ship with an icon/label, never
 * color-only; the aqua/light-mode contrast WARN on this palette also
 * obligates visible labels). */
export function ProbabilityMeter({ probability, tone, label }: ProbabilityMeterProps) {
  const pct = Math.round(probability * 1000) / 10; // one decimal
  return (
    <div className="probability-meter" title={label}>
      <div className="probability-meter-track">
        <div
          className="probability-meter-fill"
          style={{ width: `${Math.max(pct, pct > 0 ? 2 : 0)}%`, background: TONE_FILL[tone] }}
        />
      </div>
      <span className="probability-meter-value">{pct.toFixed(1)}%</span>
    </div>
  );
}
