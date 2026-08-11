"""Loading the bundled transfer and FIFA rating archives."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from fta.matching import lastname, normalize_text

REPO_ROOT = Path(__file__).resolve().parents[2]
TRANSFERS_ZIP = REPO_ROOT / "data" / "transfers.zip"
RATINGS_ZIP = REPO_ROOT / "data" / "ratings.zip"

# Columns actually used downstream. Reading only these keeps the five 8 MB
# player files from ballooning memory and makes the dependency explicit.
FIFA_COLS = [
    "short_name",
    "long_name",
    "club",
    "age",
    "overall",
    "potential",
    "player_positions",
    "contract_valid_until",
    "attacking_finishing",
    "attacking_crossing",
    "mentality_positioning",
    "mentality_vision",
]


def load_transfers(path: Path = TRANSFERS_ZIP) -> pd.DataFrame:
    """Load the transfer records and derive ``Season_transferred``."""
    with zipfile.ZipFile(path) as zf:
        with zf.open("top250-00-19.csv") as fh:
            df = pd.read_csv(fh)
    df["Season_transferred"] = df["Season"].str.split("-").str[0].astype("int64")
    return df


def load_fifa(years=(2015, 2016, 2017, 2018, 2019), path: Path = RATINGS_ZIP) -> pd.DataFrame:
    """Load FIFA editions, tagged with the edition year.

    ``Season`` on the returned frame is the *transfer season the edition may
    legitimately describe*: FIFA 2016 ships in September 2015 and so describes
    players as they were going into the 2015 summer window. See
    ``fta.config.FIFA_EDITION_OFFSET`` for why this matters.
    """
    frames = []
    with zipfile.ZipFile(path) as zf:
        names = set(zf.namelist())
        for year in years:
            fname = f"players_{year - 2000}.csv"
            if fname not in names:
                continue
            with zf.open(fname) as fh:
                head = pd.read_csv(fh, nrows=0)
            usecols = [c for c in FIFA_COLS if c in head.columns]
            nat_col = next(
                (c for c in ("nationality_name", "nationality") if c in head.columns), None
            )
            if nat_col:
                usecols.append(nat_col)
            with zf.open(fname) as fh:
                df = pd.read_csv(fh, usecols=usecols)
            if nat_col:
                df = df.rename(columns={nat_col: "Nationality"})
            df["fifa_edition"] = year
            # FIFA edition Y is released in autumn Y-1, so it is the last
            # snapshot taken strictly before the summer window of season Y-1.
            df["Season"] = year - 1
            frames.append(df)

    fifa = pd.concat(frames, ignore_index=True)
    fifa["full_name"] = fifa["long_name"].fillna(fifa["short_name"])
    fifa["Lastname"] = fifa["short_name"].map(lastname)
    # Nationality is left verbatim: REGION_MAP keys are the dataset's own
    # spellings ("Republic of Ireland", "China PR", "DR Congo") and casefolding
    # or title-casing them silently drops those rows into the unmapped bucket.
    if "club" in fifa.columns:
        fifa["club"] = fifa["club"].map(normalize_text)
    return fifa
