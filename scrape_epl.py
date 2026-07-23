"""
Scrape Understat EPL player-aggregated stats via understatapi.
Writes: understat_epl_YYYY_latest.csv
"""

import sys
from datetime import datetime

import pandas as pd
from understatapi import UnderstatClient

# CONFIG — flip to 2026 when 2026/27 season begins
SEASON_YEAR = 2025
OUTPUT_FILE = f"understat_epl_{SEASON_YEAR}_latest.csv"


def main():
    print(f"Fetching EPL {SEASON_YEAR}/{SEASON_YEAR+1} via understatapi")

    with UnderstatClient() as client:
        players = client.league(league="EPL").get_player_data(season=str(SEASON_YEAR))

    if not players:
        raise RuntimeError("No player data returned")

    print(f"Retrieved {len(players)} players")

    df = pd.DataFrame(players)
    df["scrape_timestamp_utc"] = datetime.utcnow().isoformat()
    df["season"] = f"{SEASON_YEAR}/{str(SEASON_YEAR + 1)[-2:]}"

    # Cast numeric columns
    numeric_cols = ["games", "time", "goals", "assists", "shots", "key_passes",
                    "yellow_cards", "red_cards", "npg", "xG", "xA", "npxG",
                    "xGChain", "xGBuildup"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by npxG desc for stable git diffs
    if "npxG" in df.columns:
        df = df.sort_values("npxG", ascending=False)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE} ({len(df)} rows × {len(df.columns)} cols)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
