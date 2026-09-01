// Purely decorative team identity (no crest images available without an
// external asset source) - a deterministic initials badge, NOT a data
// encoding, so it draws from its own small rotation rather than the
// dataviz skill's reserved categorical/status palette (those colors carry
// data meaning elsewhere in this app; reusing them here would make a
// badge color look like it means something statistical, which it doesn't).
const BADGE_COLORS = [
  "#2a5d8c", "#8c4a2a", "#2a8c5d", "#6b2a8c", "#8c2a4a",
  "#2a768c", "#8c6b2a", "#4a2a8c", "#2a8c76", "#8c2a2a",
];

function hashString(value: string): number {
  let hash = 0;
  for (let i = 0; i < value.length; i++) {
    hash = (hash << 5) - hash + value.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash);
}

function initials(team: string): string {
  const words = team.replace(/^AFC\s+/, "").split(/\s+/).filter(Boolean);
  if (words.length === 1) return words[0].slice(0, 2).toUpperCase();
  return (words[0][0] + words[words.length - 1][0]).toUpperCase();
}

interface TeamBadgeProps {
  team: string;
  size?: number;
}

export function TeamBadge({ team, size = 28 }: TeamBadgeProps) {
  const color = BADGE_COLORS[hashString(team) % BADGE_COLORS.length];
  return (
    <span
      className="team-badge"
      style={{ width: size, height: size, background: color, fontSize: size * 0.38 }}
      aria-hidden="true"
    >
      {initials(team)}
    </span>
  );
}
