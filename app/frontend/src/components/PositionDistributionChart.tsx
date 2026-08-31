import { Bar, BarChart, CartesianGrid, ReferenceArea, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

interface PositionDistributionChartProps {
  distribution: Record<string, number>; // position (string) -> probability
}

interface TooltipPayloadItem {
  payload: { position: number; probability: number };
}

function ChartTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadItem[] }) {
  if (!active || !payload?.length) return null;
  const { position, probability } = payload[0].payload;
  return (
    <div className="chart-tooltip">
      <strong>Position {position}</strong>
      <div>{(probability * 100).toFixed(1)}% of simulations</div>
    </div>
  );
}

/** Single-series magnitude-per-category chart (dataviz skill: a bar chart
 * is the right form here, not a line - position is a discrete, unordered-
 * in-value category, not a continuous quantity). One sequential hue (no
 * legend needed - a single series's identity is the chart title, not a
 * swatch), ≤24px bars with a 4px rounded top / square baseline, hairline
 * recessive gridlines, hover tooltip. The three shaded reference bands
 * (title / European qualification / relegation zone) are real, meaningful
 * boundaries a reader of this chart would want to see, not decoration. */
export function PositionDistributionChart({ distribution }: PositionDistributionChartProps) {
  const data = Object.entries(distribution)
    .map(([position, probability]) => ({ position: Number(position), probability }))
    .sort((a, b) => a.position - b.position);

  const maxPosition = data.length > 0 ? Math.max(...data.map((d) => d.position)) : 20;

  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 12, left: 0, bottom: 8 }} barCategoryGap={4}>
        <ReferenceArea x1={0.5} x2={1.5} fill="var(--status-good)" fillOpacity={0.08} />
        <ReferenceArea x1={1.5} x2={4.5} fill="var(--series-1)" fillOpacity={0.06} />
        <ReferenceArea
          x1={maxPosition - 2.5}
          x2={maxPosition + 0.5}
          fill="var(--status-critical)"
          fillOpacity={0.08}
        />
        <CartesianGrid vertical={false} stroke="var(--gridline)" strokeDasharray="0" />
        <XAxis
          dataKey="position"
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={{ stroke: "var(--baseline)" }}
          tickLine={false}
        />
        <YAxis
          tickFormatter={(v: number) => `${Math.round(v * 100)}%`}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          axisLine={false}
          tickLine={false}
          width={40}
        />
        <Tooltip content={<ChartTooltip />} cursor={{ fill: "var(--gridline)", opacity: 0.4 }} />
        <Bar dataKey="probability" fill="var(--seq-450)" radius={[4, 4, 0, 0]} maxBarSize={24} />
      </BarChart>
    </ResponsiveContainer>
  );
}
