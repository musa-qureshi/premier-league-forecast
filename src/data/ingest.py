"""Downloads the raw historical Premier League match data.

Data provenance
----------------
The canonical historical source for this kind of project is
football-data.co.uk, which publishes one CSV per season per English
football division (Premier League = "E0") going back to 1993-94, including
match odds and (from 2000-01 onward) shots/corners/cards.

That source is unreachable from this project's development environment:
connection attempts fail at the TLS handshake stage across four
independent client stacks (Windows schannel/curl, .NET/PowerShell, Python +
OpenSSL with relaxed cipher settings, and a server-side fetch service),
confirmed on more than one occasion, while other HTTPS hosts (google.com,
kaggle.com, football-data.org) work fine. That pattern points to the site
itself being unreachable or blocking automated clients, not a local
network/proxy problem.

Rather than build the pipeline against an untestable source, this project
downloads from three sources that together cover the full history through
the present season (see configs/data.yaml for the full reasoning behind
each and src/data/validate.py for how they're normalized and merged):

  - "irkaal/english-premier-league-results" (Kaggle): 1993-94 through a
    truncated 2021-22. Only its pre-2000 seasons are actually used.
  - "marcohuiii/english-premier-league-epl-match-data-2000-2025" (Kaggle):
    2000-01 through a slightly truncated 2024-25. Preferred over the first
    source for every season it covers.
  - openfootball/football.json (GitHub, no API key needed): fills the
    2025-26 gap neither Kaggle source covers, and is also the live source
    for the current season (Phase 9) - it exposes the full remaining
    fixture list directly, not just completed results.

If football-data.co.uk becomes reachable later, only this file and
configs/data.yaml need to change - everything from src/data/validate.py
onward consumes whatever files land in each source's raw_dir, regardless
of where they came from.

Licensing: football-data.co.uk's terms permit personal, non-commercial,
educational use with attribution to the site. This project is educational /
portfolio use only, not a commercial product, and stays inside those terms.

Setup required to run this script
----------------------------------
1. Create a Kaggle account (free) if you don't have one.
2. Go to kaggle.com -> your profile -> Settings -> API -> "Create New Token".
   This downloads a `kaggle.json` file containing {"username": ..., "key": ...}.
   (If Kaggle's UI instead shows you a bare token string starting "KGAT_"
   rather than downloading a file, build kaggle.json yourself: {"username":
   "<your kaggle username>", "key": "<that token>"}.)
3. Place it at:
     Windows:      C:\\Users\\<you>\\.kaggle\\kaggle.json
     Linux/macOS:  ~/.kaggle/kaggle.json
   or set the KAGGLE_USERNAME and KAGGLE_KEY environment variables instead.
4. `pip install kaggle` (already in requirements.txt).
5. Run: python -m src.data.ingest
"""

from __future__ import annotations

import sys
from pathlib import Path

import requests
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "data.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def download(dataset_slug: str, dest: Path, force: bool = False) -> Path:
    """Download and unzip a Kaggle dataset into `dest`.

    Idempotent by default: if `dest` already contains files, skips the
    network call instead of re-downloading every run. Pass force=True to
    re-fetch (e.g. after the source dataset has been updated upstream).
    """
    dest.mkdir(parents=True, exist_ok=True)

    if any(dest.iterdir()) and not force:
        print(f"[ingest] {dest} already has files, skipping download "
              f"(pass --force to re-download).")
        return dest

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        raise RuntimeError(
            "Could not authenticate with the Kaggle API. Create an API "
            "token at kaggle.com -> Settings -> API -> 'Create New Token', "
            "and place kaggle.json at ~/.kaggle/kaggle.json (or set "
            "KAGGLE_USERNAME / KAGGLE_KEY environment variables). "
            f"Underlying error: {e}"
        ) from e

    api = KaggleApi()
    api.authenticate()
    print(f"[ingest] downloading '{dataset_slug}' -> {dest}")
    api.dataset_download_files(dataset_slug, path=str(dest), unzip=True, quiet=False)

    downloaded = sorted(dest.rglob("*.csv"))
    print(f"[ingest] done - {len(downloaded)} CSV file(s) in {dest}")
    for f in downloaded:
        print(f"  - {f.relative_to(dest)}")
    return dest


def download_json(url: str, dest: Path, filename: str, force: bool = False) -> Path:
    """Downloads a single JSON file via plain HTTP GET - used for the
    openfootball/football.json source, which needs no API key/auth at
    all, unlike the Kaggle sources above."""
    dest.mkdir(parents=True, exist_ok=True)
    out_file = dest / filename

    if out_file.exists() and not force:
        print(f"[ingest] {out_file} already exists, skipping download "
              f"(pass --force to re-download).")
        return out_file

    print(f"[ingest] downloading '{url}' -> {out_file}")
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    out_file.write_bytes(response.content)
    print(f"[ingest] done - {len(response.content):,} bytes")
    return out_file


def main() -> None:
    config = load_config()
    force = "--force" in sys.argv

    for source in config["kaggle_sources"]:
        dest = PROJECT_ROOT / source["raw_dir"]
        download(source["slug"], dest, force=force)

    for source in config.get("openfootball_sources", []):
        dest = PROJECT_ROOT / source["raw_dir"]
        download_json(source["url"], dest, filename="matches.json", force=force)


if __name__ == "__main__":
    main()
