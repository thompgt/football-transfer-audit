"""Nationality to origin-region mapping used for the fairness disaggregation."""

from __future__ import annotations

REGION_MAP = {
    # South America
    "Brazil": "South America", "Argentina": "South America", "Colombia": "South America",
    "Uruguay": "South America", "Chile": "South America", "Paraguay": "South America",
    "Ecuador": "South America", "Peru": "South America", "Venezuela": "South America",
    "Bolivia": "South America",
    # Western Europe
    "France": "W. Europe", "Spain": "W. Europe", "Germany": "W. Europe",
    "Italy": "W. Europe", "Portugal": "W. Europe", "Netherlands": "W. Europe",
    "Belgium": "W. Europe", "England": "W. Europe", "Switzerland": "W. Europe",
    "Austria": "W. Europe", "Scotland": "W. Europe", "Wales": "W. Europe",
    "Republic of Ireland": "W. Europe", "Northern Ireland": "W. Europe",
    "Denmark": "W. Europe", "Sweden": "W. Europe", "Norway": "W. Europe",
    "Finland": "W. Europe", "Iceland": "W. Europe", "Luxembourg": "W. Europe",
    # Eastern / South-Eastern Europe
    "Poland": "E. Europe", "Czech Republic": "E. Europe", "Croatia": "E. Europe",
    "Serbia": "E. Europe", "Bosnia Herzegovina": "E. Europe", "Romania": "E. Europe",
    "Hungary": "E. Europe", "Slovakia": "E. Europe", "Slovenia": "E. Europe",
    "Bulgaria": "E. Europe", "Ukraine": "E. Europe", "Russia": "E. Europe",
    "Greece": "E. Europe", "Turkey": "E. Europe", "Albania": "E. Europe",
    "Montenegro": "E. Europe", "Georgia": "E. Europe", "Belarus": "E. Europe",
    "FYR Macedonia": "E. Europe", "North Macedonia": "E. Europe", "Kosovo": "E. Europe",
    "Armenia": "E. Europe", "Azerbaijan": "E. Europe",
    # Africa
    "Senegal": "Africa", "Ivory Coast": "Africa", "Côte d'Ivoire": "Africa",
    "Ghana": "Africa", "Nigeria": "Africa", "Cameroon": "Africa",
    "DR Congo": "Africa", "Mali": "Africa", "Guinea": "Africa", "Gabon": "Africa",
    "Morocco": "Africa", "Algeria": "Africa", "Tunisia": "Africa", "Egypt": "Africa",
    "Congo": "Africa", "Burkina Faso": "Africa", "Togo": "Africa", "Benin": "Africa",
    "Angola": "Africa", "Zambia": "Africa", "Kenya": "Africa", "South Africa": "Africa",
    "Cape Verde": "Africa", "Guinea Bissau": "Africa", "Equatorial Guinea": "Africa",
    "Central African Rep.": "Africa", "Mozambique": "Africa", "Zimbabwe": "Africa",
    "Uganda": "Africa", "Sierra Leone": "Africa", "Liberia": "Africa",
    "Comoros": "Africa", "Madagascar": "Africa", "Niger": "Africa", "Chad": "Africa",
    # Rest of the Americas
    "Mexico": "Americas (other)", "United States": "Americas (other)",
    "Canada": "Americas (other)", "Jamaica": "Americas (other)",
    "Costa Rica": "Americas (other)", "Honduras": "Americas (other)",
    "Panama": "Americas (other)", "Trinidad & Tobago": "Americas (other)",
    "Curacao": "Americas (other)", "Haiti": "Americas (other)",
    "Dominican Republic": "Americas (other)", "El Salvador": "Americas (other)",
    "Guatemala": "Americas (other)",
    # Asia & Oceania
    "Japan": "Asia & Oceania", "South Korea": "Asia & Oceania",
    "Korea Republic": "Asia & Oceania", "Australia": "Asia & Oceania",
    "Iran": "Asia & Oceania", "China PR": "Asia & Oceania", "China": "Asia & Oceania",
    "New Zealand": "Asia & Oceania", "Saudi Arabia": "Asia & Oceania",
    "Israel": "Asia & Oceania", "Uzbekistan": "Asia & Oceania",
    "Syria": "Asia & Oceania", "Iraq": "Asia & Oceania", "Qatar": "Asia & Oceania",
    "United Arab Emirates": "Asia & Oceania", "India": "Asia & Oceania",
    "Thailand": "Asia & Oceania", "Philippines": "Asia & Oceania",
}

UNKNOWN_REGION = "Unmapped"


def to_region(nationality) -> str:
    """Map a nationality string to an origin region.

    Unmapped and missing nationalities land in ``Unmapped`` rather than a
    catch-all called "Other". The previous code folded *both* genuinely small
    regions and rows where the nationality join simply failed into one "Other"
    bucket, and then reported that bucket's mean residual as a finding. A
    bucket whose membership is "the join did not work" cannot support a claim
    about a group of players.
    """
    import pandas as pd

    if nationality is None or (isinstance(nationality, float) and pd.isna(nationality)):
        return UNKNOWN_REGION
    return REGION_MAP.get(str(nationality), UNKNOWN_REGION)
