"""Cleans, canonicalizes, and validates the raw match CSVs into a single
chronologically-sorted dataset.

This module is the boundary between "whatever format the data source
happens to ship" and "the clean schema every downstream module can rely on".
Nothing outside src/data/ should ever read data/raw/ directly - features,
models, and the simulator all read data/processed/matches.parquet.

Design note: the transformation functions below (assign_season,
canonicalize_teams, derive_result, check_result_consistency) are kept as
small pure functions - taking and returning DataFrames/Series rather than
reading files - specifically so they can be unit tested with tiny synthetic
DataFrames (see tests/test_data_validate.py) without needing real downloaded
data or network access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"

# Required for every match; a source row missing any of these is dropped.
CORE_COLUMNS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Kept when present, but not required - older seasons (pre-2000) lack most
# of these in football-data.co.uk-derived data.
OPTIONAL_COLUMNS = [
    "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF", "HY", "AY", "HR", "AR",
]


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def discover_csvs(raw_dir: Path) -> list[Path]:
    files = sorted(raw_dir.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {raw_dir}. Run "
            f"`python -m src.data.ingest` first."
        )
    return files


def read_raw_csvs(files: list[Path]) -> pd.DataFrame:
    """Reads and concatenates the raw source files, keeping only rows that
    have the columns we require and only the columns we recognize."""
    frames = []
    for f in files:
        try:
            df = pd.read_csv(f, encoding="latin-1")
        except Exception as e:
            print(f"[validate] skipping unreadable file {f.name}: {e}")
            continue
        # The Kaggle mirror names the match-date column "DateTime" (ISO
        # 8601 timestamps) rather than football-data.co.uk's original
        # "Date" (DD/MM/YY). Normalize to "Date" so the rest of the
        # pipeline doesn't care which source shipped the file.
        if "Date" not in df.columns and "DateTime" in df.columns:
            df = df.rename(columns={"DateTime": "Date"})
        missing = [c for c in CORE_COLUMNS if c not in df.columns]
        if missing:
            print(f"[validate] skipping {f.name}: missing required columns {missing}")
            continue
        keep = CORE_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]
        frames.append(df[keep].copy())

    if not frames:
        raise ValueError(
            "None of the source CSVs had the required columns "
            f"{CORE_COLUMNS}. Check the downloaded data's format."
        )
    return pd.concat(frames, ignore_index=True)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Parses the Date column, which may hold either ISO 8601 timestamps
    (the Kaggle mirror's "DateTime" column, e.g. "1993-09-01T00:00:00Z") or
    raw football-data.co.uk-style DD/MM/YYYY strings.

    Deliberately does NOT hand the whole column to pandas' automatic format
    guessing (with or without format="mixed"): that guesser resolves
    ambiguous numeric date components (day vs. month, both <= 12) by
    scanning the array for *some* row unambiguous enough to lock in a
    format, then applying that format to every row. That makes correctness
    depend on which other rows happen to be present in the batch being
    parsed - verified to silently produce wrong dates (e.g. "1993-09-01"
    parsed as 9 Jan instead of 1 Sep) on a small or homogeneous batch, which
    is exactly what a unit test batch or a small incremental live-data
    update (Phase 7) looks like. Instead we detect which format a row is in
    from an unambiguous marker (the ISO "T" separator) and parse each group
    with an explicit format, so correctness never depends on batch
    composition.
    """
    df = df.copy()
    date_str = df["Date"].astype(str)
    is_iso = date_str.str.contains("T", na=False)

    parsed = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
    if is_iso.any():
        parsed.loc[is_iso] = pd.to_datetime(
            date_str[is_iso], format="ISO8601", errors="coerce", utc=True
        )
    if (~is_iso).any():
        parsed.loc[~is_iso] = pd.to_datetime(
            date_str[~is_iso], dayfirst=True, errors="coerce", utc=True
        )

    # Drop to day-level, timezone-naive dates: kickoff time doesn't matter
    # for season-level modeling, and mixing tz-aware/naive timestamps across
    # data sources (this Kaggle mirror vs. a future live-data source) is a
    # reliable source of silent comparison bugs if left tz-aware.
    df["Date"] = parsed.dt.tz_localize(None).dt.normalize()
    n_bad = df["Date"].isna().sum()
    if n_bad:
        print(f"[validate] dropping {n_bad} row(s) with an unparseable date")
        df = df.dropna(subset=["Date"])
    return df


def assign_season(date: pd.Series) -> pd.Series:
    """Premier League seasons normally run August through May, so a match
    played in January 2024 belongs to the 2023-24 season and one played in
    September 2024 belongs to 2024-25.

    The cutover is August (month >= 8), not July. This matters more than it
    looks: the COVID-disrupted 2019-20 season finished in late July 2020
    (matches played behind closed doors to complete the fixture list), and
    an earlier July-based cutover misclassified those ~66 trailing matches
    as belonging to 2020-21 - inflating that season to 446 matches / 23
    teams (20 real 2020-21 teams plus the 3 clubs relegated after 2019-20
    whose final matches leaked in) instead of the correct 380/20. The
    Premier League has never started a season before August, so month >= 8
    has no equivalent failure mode in the other direction.
    """
    start_year = date.dt.year.where(date.dt.month >= 8, date.dt.year - 1)
    return start_year.astype(str) + "-" + (start_year + 1).astype(str).str[-2:]


def canonicalize_teams(df: pd.DataFrame, team_map: dict[str, str]) -> pd.DataFrame:
    """Applies the alias -> canonical-name mapping to both HomeTeam and
    AwayTeam. Any name not present as a key in `team_map` is left as-is -
    the map only needs to cover known inconsistencies, not every team."""
    df = df.copy()
    df["HomeTeam"] = df["HomeTeam"].str.strip().replace(team_map)
    df["AwayTeam"] = df["AwayTeam"].str.strip().replace(team_map)
    return df


def derive_result(fthg: pd.Series, ftag: pd.Series) -> pd.Series:
    """Recomputes H/D/A from the goal columns rather than trusting the
    source's FTR column directly - this is a cheap way to catch transcription
    errors in the source data (see check_result_consistency)."""
    result = pd.Series("D", index=fthg.index)
    result[fthg > ftag] = "H"
    result[fthg < ftag] = "A"
    return result


def check_result_consistency(ftr: pd.Series, derived_ftr: pd.Series) -> int:
    """Returns the number of rows where the source's FTR column disagrees
    with the goal-derived result. Non-zero counts are logged, and the
    derived result is what actually gets used - the goals are the more
    fundamental, harder-to-transcribe-wrong signal."""
    return int((ftr != derived_ftr).sum())


def load_team_name_map(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    return dict(zip(df["alias"], df["canonical"]))


def load_and_clean(raw_dir: Path, team_map_path: Path) -> pd.DataFrame:
    files = discover_csvs(raw_dir)
    matches = read_raw_csvs(files)
    matches = parse_dates(matches)

    matches = matches.dropna(subset=["FTHG", "FTAG", "HomeTeam", "AwayTeam"])
    matches["FTHG"] = matches["FTHG"].astype(int)
    matches["FTAG"] = matches["FTAG"].astype(int)

    team_map = load_team_name_map(team_map_path)
    matches = canonicalize_teams(matches, team_map)

    derived = derive_result(matches["FTHG"], matches["FTAG"])
    n_disagree = check_result_consistency(matches["FTR"], derived)
    if n_disagree:
        print(f"[validate] {n_disagree} row(s) had FTR inconsistent with "
              f"goals; using the goal-derived result")
    matches["FTR"] = derived

    matches["Season"] = assign_season(matches["Date"])
    matches = matches.drop_duplicates(subset=["Date", "HomeTeam", "AwayTeam"])
    matches = matches.sort_values("Date").reset_index(drop=True)
    matches["match_id"] = matches.index

    return matches


def sanity_check(matches: pd.DataFrame) -> None:
    """Cheap structural checks, run at the end of the pipeline, that catch a
    broken/incomplete source file before it silently corrupts every
    downstream feature and model. Deliberately not exhaustive - this is a
    smoke test, not a full data-quality audit."""
    assert matches["Date"].is_monotonic_increasing, "matches must be chronologically sorted"
    assert matches["FTR"].isin(["H", "D", "A"]).all(), "FTR must be H/D/A"
    assert (matches["FTHG"] >= 0).all() and (matches["FTAG"] >= 0).all(), "goals cannot be negative"
    assert matches["match_id"].is_unique, "match_id must be unique"

    for season, g in matches.groupby("Season"):
        n_teams = pd.concat([g["HomeTeam"], g["AwayTeam"]]).nunique()
        expected = n_teams * (n_teams - 1)  # every team plays every other team home + away
        if len(g) != expected:
            print(f"[validate] WARNING season {season}: {len(g)} matches "
                  f"but {n_teams} teams implies {expected} - possible "
                  f"missing/duplicate rows or a mid-season data gap")


def build(raw_dir: Path | None = None, out_path: Path | None = None,
          team_map_path: Path | None = None) -> pd.DataFrame:
    config = load_config()
    raw_dir = raw_dir or (PROJECT_ROOT / config["raw_dir"])
    out_path = out_path or (PROJECT_ROOT / config["processed_path"])
    team_map_path = team_map_path or (PROJECT_ROOT / config["team_name_map_path"])

    matches = load_and_clean(raw_dir, team_map_path)
    sanity_check(matches)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    matches.to_parquet(out_path, index=False)

    print(f"[validate] wrote {len(matches)} matches spanning "
          f"{matches['Season'].nunique()} seasons "
          f"({matches['Season'].min()} to {matches['Season'].max()}) -> {out_path}")

    unique_teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    print(f"[validate] {len(unique_teams)} unique team names after "
          f"canonicalization - review this list for lingering duplicates "
          f"(e.g. the same club under two spellings):")
    for t in unique_teams:
        print(f"  - {t}")

    return matches


if __name__ == "__main__":
    build()
