interface MatchweekPickerProps {
  matchweeks: number[];
  selected: number | "current";
  onChange: (value: number | "current") => void;
}

/** Switches the standings table between the live forecast ("Current") and
 * a snapshot of the table exactly as it stood after one of this season's
 * completed matchweeks. A <select>, not a row of pill buttons, since the
 * matchweek count grows to 38 over a season - a button row that wide
 * would either wrap awkwardly or need its own scroll container, while a
 * dropdown stays exactly the same size regardless of how far the season
 * has progressed. */
export function MatchweekPicker({ matchweeks, selected, onChange }: MatchweekPickerProps) {
  return (
    <label className="matchweek-picker">
      <span className="matchweek-picker-label">View table as of:</span>
      <select
        value={selected}
        onChange={(e) => onChange(e.target.value === "current" ? "current" : Number(e.target.value))}
      >
        <option value="current">Current</option>
        {matchweeks.map((mw) => (
          <option key={mw} value={mw}>
            After Matchweek {mw}
          </option>
        ))}
      </select>
    </label>
  );
}
