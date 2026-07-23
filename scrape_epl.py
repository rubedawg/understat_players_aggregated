"""
Scrape Understat EPL player-aggregated stats for the current season.
Writes: understat_epl_YYYY_latest.csv

Season year = calendar start year (e.g. 2026 = 2026/27 season).
Update SEASON_YEAR each August or wire to date logic.
"""

import json
import re
import sys
from datetime import datetime

import pandas as pd
import requests
from bs4 import BeautifulSoup

# CONFIG — update each August
SEASON_YEAR = 2026  # 2026 = 2026/27 season

URL = f"https://understat.com/league/EPL/{SEASON_YEAR}"
OUTPUT_FILE = f"understat_epl_{SEASON_YEAR}_latest.csv"


def fetch_page(url):
    """Fetch the Understat league page HTML."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def extract_players_data(html):
    """
    Understat embeds data as JSON strings inside <script> tags via
    JSON.parse('escaped_string'). We find the playersData assignment,
    unescape it, and load as JSON.
    """
    soup = BeautifulSoup(html, "html.parser")

    for script in soup.find_all("script"):
        if script.string and "playersData" in script.string:
            # Match JSON.parse('...')
            match = re.search(
                r"playersData\s*=\s*JSON\.parse\('(.+?)'\)", script.string
            )
            if not match:
                continue

            # Understat escapes with \x hex codes — decode them
            escaped = match.group(1)
            decoded = escaped.encode("utf-8").decode("unicode_escape")
            return json.loads(decoded)

    raise RuntimeError("Could not find playersData in page HTML")


def normalize(players):
    """Cast numeric columns to correct types."""
    df = pd.DataFrame(players)

    int_cols = ["games", "time", "goals", "assists", "shots", "key_passes",
                "yellow_cards", "red_cards", "npg"]
    float_cols = ["xG", "xA", "npxG", "xGChain", "xGBuildup"]

    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["scrape_timestamp_utc"] = datetime.utcnow().isoformat()
    df["season"] = f"{SEASON_YEAR}/{str(SEASON_YEAR + 1)[-2:]}"
    return df


def main():
    print(f"Fetching {URL}")
    html = fetch_page(URL)

    players = extract_players_data(html)
    print(f"Extracted {len(players)} player records")

    df = normalize(players)

    # Sort by npxG descending for stable output (helps git diffs)
    if "npxG" in df.columns:
        df = df.sort_values("npxG", ascending=False)

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Wrote {OUTPUT_FILE} ({len(df)} rows, {len(df.columns)} cols)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
