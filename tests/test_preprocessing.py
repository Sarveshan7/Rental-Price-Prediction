import pandas as pd

from src.preprocessing import (
    extract_rent,
    extract_size,
    flag_nearby_trains,
    remove_bathroom_outliers,
)


def test_extract_rent_parses_currency_string():
    assert extract_rent("RM 1 200 per month") == 1200


def test_extract_rent_returns_none_when_no_match():
    assert extract_rent("not a rent string") is None


def test_extract_size_parses_sqft_string():
    assert extract_size("850 sq. ft.") == 850


def test_flag_nearby_trains_detects_keyword():
    assert flag_nearby_trains("Near KTM/LRT, Playground") == "yes"
    assert flag_nearby_trains("Playground, Pool") == "no"


def test_flag_nearby_trains_handles_non_string():
    assert flag_nearby_trains(None) is None


def test_remove_bathroom_outliers_drops_implausible_rows():
    df = pd.DataFrame(
        {
            "bathroom": [1, 2, 10],
            "rooms": [1, 2, 1],
        }
    )
    cleaned = remove_bathroom_outliers(df)
    assert len(cleaned) == 2
    assert 10 not in cleaned["bathroom"].values
