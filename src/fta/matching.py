"""Joining transfer records to FIFA player-season rows.

The join is where this audit's headline finding was most fragile, so it gets
its own tested module.

The original notebook (cell 15) built its FIFA lookup with::

    fifa_nat.sort_values('overall', ascending=False)
            .drop_duplicates(subset=['Lastname', 'Season_transferred'])

That is a surname-only key resolved in favour of the *highest rated* namesake.
Surname collisions are not uniformly distributed: Silva, Santos, Fernandes,
Rodriguez, Garcia and Sanchez are borne by dozens of players per FIFA edition,
while Oxlade-Chamberlain is borne by one. So the rule systematically imports
the best same-surname player's ability into Brazilian, Portuguese and Spanish
records - inflating the ability inputs for precisely the regional group the
audit then reports as *undervalued*. Ability inflated up, predicted fee pushed
up, residual pushed up, undervaluation understated (or, if the fee does not
follow, the group's residual is shifted by an artefact of the join).

This module therefore:

* keeps the ambiguity instead of silently resolving it (``annotate_ambiguity``),
* offers a strict join requiring full-name agreement plus club agreement, and
* lets the caller measure the ambiguous-match rate by region so the bias can be
  quantified rather than assumed away.
"""

from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd


def normalize_text(s):
    """Accent-strip and casefold a name or club string for joining."""
    if not isinstance(s, str):
        return s
    stripped = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return " ".join(stripped.lower().split())


def lastname(name):
    """Last whitespace-separated token of a name, normalized."""
    if not isinstance(name, str) or not name.strip():
        return name
    return normalize_text(name.strip().split(" ")[-1])


def annotate_ambiguity(fifa: pd.DataFrame, key=("Lastname", "Season")) -> pd.DataFrame:
    """Add ``n_namesakes``/``is_ambiguous``/``overall_spread`` to a FIFA table.

    ``overall_spread`` is the ability range across the namesakes sharing a key.
    It is the size of the error the highest-rated-namesake rule can introduce
    for a single row, which is what makes the bias quantifiable.
    """
    key = list(key)
    out = fifa.copy()
    grouped = out.groupby(key)["overall"]
    out["n_namesakes"] = grouped.transform("size")
    out["overall_spread"] = grouped.transform("max") - grouped.transform("min")
    out["is_ambiguous"] = out["n_namesakes"] > 1
    return out


def highest_rated_namesake(fifa: pd.DataFrame, key=("Lastname", "Season")) -> pd.DataFrame:
    """Reproduce the original (biased) resolution rule.

    Kept deliberately, and named for what it does, so the audit can quantify the
    difference between it and the strict join rather than just asserting one is
    better.
    """
    return (
        fifa.sort_values("overall", ascending=False)
        .drop_duplicates(subset=list(key))
        .reset_index(drop=True)
    )


def strict_join(
    transfers: pd.DataFrame,
    fifa: pd.DataFrame,
    season_col: str = "Season_transferred",
) -> pd.DataFrame:
    """Join transfers to FIFA rows requiring identity evidence beyond a surname.

    Accepted evidence, in descending order of strength - the first tier that
    matches wins, and a row that matches no tier is left unmatched rather than
    guessed:

    1. full normalized name + season,
    2. surname + season + club agreement with either the selling or buying club,
    3. surname + season *only when that surname is unique in that FIFA edition*.

    Tier 3 is the crucial change. The old code applied a surname key to
    everybody and resolved collisions by rating; here a colliding surname with
    no club evidence simply does not match.
    """
    fifa = annotate_ambiguity(fifa)
    t = transfers.copy()

    t["_full"] = t["Name"].map(normalize_text)
    t["_last"] = t["Name"].map(lastname)
    for col in ("Team_from", "Team_to"):
        if col in t.columns:
            t["_" + col.lower()] = t[col].map(normalize_text)
        else:
            t["_" + col.lower()] = np.nan

    f = fifa.copy()
    f["_full"] = f["full_name"].map(normalize_text) if "full_name" in f.columns else f["short_name"].map(normalize_text)
    f["_last"] = f["Lastname"]
    f["_club"] = f["club"].map(normalize_text) if "club" in f.columns else np.nan

    value_cols = [c for c in ("overall", "potential", "Nationality", "age", "n_namesakes", "overall_spread", "is_ambiguous") if c in f.columns]

    # Tier 1: full name + season.
    t1 = f.drop_duplicates(subset=["_full", "Season"])[["_full", "Season", "_club"] + value_cols]
    m1 = t.merge(t1, left_on=["_full", season_col], right_on=["_full", "Season"], how="left")
    m1["match_tier"] = np.where(m1["overall"].notna(), "full_name", "unmatched")

    unmatched = m1["overall"].isna()

    # Tier 2: surname + season + club agreement (either direction).
    t2 = f[["_last", "Season", "_club"] + value_cols].drop_duplicates(subset=["_last", "Season", "_club"])
    for side in ("_team_from", "_team_to"):
        if not unmatched.any():
            break
        sub = m1.loc[unmatched, ["_last", season_col, side]].copy()
        joined = sub.merge(
            t2,
            left_on=["_last", season_col, side],
            right_on=["_last", "Season", "_club"],
            how="left",
        ).set_index(sub.index)
        fill = joined["overall"].notna()
        for col in value_cols:
            m1.loc[sub.index[fill], col] = joined.loc[fill, col].values
        m1.loc[sub.index[fill], "match_tier"] = "surname+club"
        unmatched = m1["overall"].isna()

    # Tier 3: surname + season, only for surnames unique in that edition.
    if unmatched.any():
        unique_only = f[~f["is_ambiguous"]][["_last", "Season"] + value_cols]
        unique_only = unique_only.drop_duplicates(subset=["_last", "Season"])
        sub = m1.loc[unmatched, ["_last", season_col]].copy()
        joined = sub.merge(
            unique_only, left_on=["_last", season_col], right_on=["_last", "Season"], how="left"
        ).set_index(sub.index)
        fill = joined["overall"].notna()
        for col in value_cols:
            m1.loc[sub.index[fill], col] = joined.loc[fill, col].values
        m1.loc[sub.index[fill], "match_tier"] = "unique_surname"

    m1.loc[m1["overall"].isna(), "match_tier"] = "unmatched"
    return m1.drop(columns=[c for c in ("_full", "_last", "_team_from", "_team_to", "_club", "Season") if c in m1.columns])


def ambiguity_rate_by(
    transfers: pd.DataFrame,
    fifa: pd.DataFrame,
    by: str,
    season_col: str = "Season_transferred",
) -> pd.DataFrame:
    """Share of transfer rows whose surname key is ambiguous, broken down by ``by``.

    This is the number the audit needed and never produced: if the ambiguous
    rate is materially higher for South American / Iberian names, then any
    surname-keyed join distorts that group's ability inputs specifically.
    """
    fifa = annotate_ambiguity(fifa)
    ambiguous_keys = set(
        map(tuple, fifa.loc[fifa["is_ambiguous"], ["Lastname", "Season"]].drop_duplicates().values)
    )
    spread = (
        fifa.loc[fifa["is_ambiguous"]]
        .groupby(["Lastname", "Season"])["overall_spread"]
        .max()
    )

    t = transfers.copy()
    t["_last"] = t["Name"].map(lastname)
    keys = list(zip(t["_last"], t[season_col]))
    t["_ambiguous"] = [k in ambiguous_keys for k in keys]
    t["_spread"] = [spread.get(k, np.nan) for k in keys]

    out = (
        t.groupby(by)
        .agg(
            n_transfers=("_ambiguous", "size"),
            n_ambiguous=("_ambiguous", "sum"),
            mean_overall_spread=("_spread", "mean"),
        )
        .reset_index()
    )
    out["ambiguous_rate"] = out["n_ambiguous"] / out["n_transfers"]
    return out.sort_values("ambiguous_rate", ascending=False).reset_index(drop=True)


def attrition_table(
    matched: pd.DataFrame,
    by: str,
    matched_flag_col: str = "match_tier",
) -> pd.DataFrame:
    """Match rate broken down by ``by``.

    The joins discard roughly five in six transfer rows, and the survivors are
    not a random sample - obscure leagues and non-Latin names drop out first.
    Reporting the rate by league / region / season is the minimum honest
    accounting.
    """
    df = matched.copy()
    df["_matched"] = df[matched_flag_col] != "unmatched"
    out = (
        df.groupby(by)
        .agg(n_rows=("_matched", "size"), n_matched=("_matched", "sum"))
        .reset_index()
    )
    out["match_rate"] = out["n_matched"] / out["n_rows"]
    return out.sort_values("match_rate").reset_index(drop=True)
