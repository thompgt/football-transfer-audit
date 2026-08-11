"""Known-answer tests for the transfer <-> FIFA join.

These exist because the join was where the audit's headline finding was most
fragile and there was no test of any kind in the repository.
"""

from __future__ import annotations

import pandas as pd
import pytest

from fta.matching import (
    ambiguity_rate_by,
    annotate_ambiguity,
    attrition_table,
    highest_rated_namesake,
    lastname,
    normalize_text,
    strict_join,
)


def test_normalize_text_strips_accents_and_case():
    assert normalize_text("Luís Figo") == "luis figo"
    assert normalize_text("Hernán Crespo") == "hernan crespo"
    assert normalize_text("  Real   Madrid ") == "real madrid"


def test_lastname_takes_final_token():
    assert lastname("Alex Oxlade-Chamberlain") == "oxlade-chamberlain"
    assert lastname("Thiago Emiliano da Silva") == "silva"
    assert lastname("Ronaldinho") == "ronaldinho"


@pytest.fixture
def fifa():
    """Three Silvas and one unique surname, all in the same edition."""
    return pd.DataFrame(
        {
            "short_name": ["Thiago Silva", "Bernardo Silva", "Joe Silva", "A. Oxlade-Chamberlain"],
            "full_name": [
                "thiago emiliano da silva",
                "bernardo mota veiga de carvalho e silva",
                "joe silva",
                "alex oxlade-chamberlain",
            ],
            "Lastname": ["silva", "silva", "silva", "oxlade-chamberlain"],
            "club": ["paris saint-germain", "manchester city", "hartlepool", "arsenal"],
            "Nationality": ["Brazil", "Portugal", "England", "England"],
            "overall": [88, 86, 54, 82],
            "potential": [88, 90, 58, 87],
            "Season": [2017, 2017, 2017, 2017],
        }
    )


def test_annotate_ambiguity_counts_namesakes_and_spread(fifa):
    out = annotate_ambiguity(fifa)
    silvas = out[out["Lastname"] == "silva"]
    assert set(silvas["n_namesakes"]) == {3}
    assert silvas["is_ambiguous"].all()
    # 88 (Thiago) - 54 (Joe): the size of the ability error the old rule could
    # inject into a single row.
    assert set(silvas["overall_spread"]) == {34}

    ox = out[out["Lastname"] == "oxlade-chamberlain"]
    assert not ox["is_ambiguous"].any()
    assert ox["overall_spread"].iloc[0] == 0


def test_highest_rated_namesake_reproduces_the_biased_rule(fifa):
    """The old rule collapses three players into the best-rated one."""
    collapsed = highest_rated_namesake(fifa)
    silva = collapsed[collapsed["Lastname"] == "silva"]
    assert len(silva) == 1
    assert silva["overall"].iloc[0] == 88
    # Joe Silva, a 54-rated third-tier player, would have been handed Thiago
    # Silva's 88 rating by this join.
    assert silva["short_name"].iloc[0] == "Thiago Silva"


def test_strict_join_matches_on_full_name(fifa):
    transfers = pd.DataFrame(
        {
            "Name": ["Bernardo Mota Veiga de Carvalho e Silva"],
            "Team_from": ["AS Monaco"],
            "Team_to": ["Manchester City"],
            "Season_transferred": [2017],
        }
    )
    out = strict_join(transfers, fifa)
    assert out["match_tier"].iloc[0] == "full_name"
    assert out["overall"].iloc[0] == 86
    assert out["Nationality"].iloc[0] == "Portugal"


def test_strict_join_uses_club_evidence_for_a_colliding_surname(fifa):
    """A bare surname is not identity - but surname plus club is."""
    transfers = pd.DataFrame(
        {
            "Name": ["Thiago Silva"],
            "Team_from": ["Paris Saint-Germain"],
            "Team_to": ["Chelsea"],
            "Season_transferred": [2017],
        }
    )
    out = strict_join(transfers, fifa)
    assert out["match_tier"].iloc[0] == "surname+club"
    assert out["overall"].iloc[0] == 88


def test_strict_join_refuses_an_ambiguous_surname_with_no_club_evidence(fifa):
    """The regression this whole module exists to prevent.

    A transfer whose only identifying token is a surname shared by three
    players must NOT be silently assigned the best-rated one.
    """
    transfers = pd.DataFrame(
        {
            "Name": ["R. Silva"],
            "Team_from": ["Vitoria Guimaraes"],
            "Team_to": ["Sporting CP"],
            "Season_transferred": [2017],
        }
    )
    out = strict_join(transfers, fifa)
    assert out["match_tier"].iloc[0] == "unmatched"
    assert pd.isna(out["overall"].iloc[0])

    # The old rule, by contrast, would have handed this row an 88.
    old = highest_rated_namesake(fifa)
    assert old.loc[old["Lastname"] == "silva", "overall"].iloc[0] == 88


def test_strict_join_allows_a_unique_surname_without_club_evidence(fifa):
    transfers = pd.DataFrame(
        {
            "Name": ["Oxlade-Chamberlain"],
            "Team_from": ["Somewhere Unlisted"],
            "Team_to": ["Elsewhere Unlisted"],
            "Season_transferred": [2017],
        }
    )
    out = strict_join(transfers, fifa)
    assert out["match_tier"].iloc[0] == "unique_surname"
    assert out["overall"].iloc[0] == 82


def test_ambiguity_rate_by_region_separates_colliding_surnames(fifa):
    transfers = pd.DataFrame(
        {
            "Name": ["Joao Silva", "Pedro Silva", "Alex Oxlade-Chamberlain"],
            "Region": ["South America", "South America", "W. Europe"],
            "Season_transferred": [2017, 2017, 2017],
        }
    )
    out = ambiguity_rate_by(transfers, fifa, by="Region")
    rates = dict(zip(out["Region"], out["ambiguous_rate"]))
    assert rates["South America"] == 1.0
    assert rates["W. Europe"] == 0.0
    # And the magnitude of the potential distortion is carried alongside.
    spread = dict(zip(out["Region"], out["mean_overall_spread"]))
    assert spread["South America"] == 34


def test_attrition_table_reports_match_rate_by_group():
    matched = pd.DataFrame(
        {
            "League_from": ["Serie A", "Serie A", "Liga MX", "Liga MX", "Liga MX"],
            "match_tier": ["full_name", "surname+club", "unmatched", "unmatched", "full_name"],
        }
    )
    out = attrition_table(matched, by="League_from")
    rates = dict(zip(out["League_from"], out["match_rate"]))
    assert rates["Serie A"] == 1.0
    assert rates["Liga MX"] == pytest.approx(1 / 3)
    # Worst-matched group sorts first, so attrition is the first thing you see.
    assert out["League_from"].iloc[0] == "Liga MX"
