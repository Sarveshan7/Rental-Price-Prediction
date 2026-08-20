"""
Data cleaning steps for the KL/Selangor apartment rental dataset.

Mirrors the cleaning logic from notebooks/Rental_Price_Prediction.ipynb,
turned into reusable, testable functions.
"""
import re

import numpy as np
import pandas as pd

DROP_COLUMNS = [
    "ads_id",
    "prop_name",
    "completion_year",
    "facilities",
    "additional_facilities",
    "region",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw scraped listings CSV."""
    return pd.read_csv(path)


def drop_duplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()


def flag_nearby_trains(text) -> str:
    """Return 'yes' if a listing mentions being near a KTM/LRT station."""
    pattern = re.compile(r"\bNear KTM/LRT\b")
    try:
        return "yes" if pattern.search(text) else "no"
    except TypeError:
        return text


def add_nearby_trains_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["nearby_trains"] = df["additional_facilities"].apply(flag_nearby_trains)
    return df


def drop_unused_columns(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    columns = columns or DROP_COLUMNS
    return df.drop(columns=columns, errors="ignore")


def extract_rent(value) -> int | None:
    """Pull the numeric monthly rent out of strings like 'RM 1 200 per month'."""
    match = re.search(r"RM (.*?) per", str(value))
    if match:
        return int(match.group(1).replace(" ", ""))
    return None


def extract_size(value) -> int | None:
    """Pull the numeric size (sqft) out of strings like '850 sq. ft.'."""
    match = re.search(r"(.*?) sq", str(value))
    if match:
        return int(match.group(1).replace(" ", ""))
    return None


def clean_rent_and_size(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["monthly_rent"] = df["monthly_rent"].astype(str).apply(extract_rent)
    df["size"] = df["size"].astype(str).apply(extract_size)
    df["location"] = df["location"].apply(lambda x: x.split("-")[-1].strip())
    return df


def drop_missing_rent(df: pd.DataFrame) -> pd.DataFrame:
    return df.dropna(subset=["monthly_rent"])


def filter_rent_range(df: pd.DataFrame, low: int = 100, high: int = 6000) -> pd.DataFrame:
    return df.query("monthly_rent > @low & monthly_rent < @high")


def filter_size_range(df: pd.DataFrame, low: int = 50, high: int = 2000) -> pd.DataFrame:
    return df.query("size > @low & size < @high")


def remove_bathroom_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop listings where bathroom count is implausibly higher than room count."""
    df = df.copy()
    df["bathroom"] = pd.to_numeric(df["bathroom"], errors="coerce")
    df["rooms"] = pd.to_numeric(df["rooms"], errors="coerce")
    remove = df[df["bathroom"] > df["rooms"] + 2]
    mask = df.index.isin(remove.index)
    return df[~mask]


def add_rent_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rent_per_sqft"] = df["monthly_rent"] / df["size"]
    return df


def remove_pps_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rent-per-sqft outliers within each location group (mean +/- 1 std)."""
    df_out = pd.DataFrame()
    for _, subdf in df.groupby("location"):
        m = np.mean(subdf.rent_per_sqft)
        st = np.std(subdf.rent_per_sqft)
        reduced_df = subdf[(subdf.rent_per_sqft > (m - st)) & (subdf.rent_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out


def remove_rare_categories(df: pd.DataFrame, column: str, min_count: int = 10) -> pd.DataFrame:
    counts = df[column].value_counts()
    rare = counts[counts < min_count].index
    return df[~df[column].isin(rare)]


def run_pipeline(raw_csv_path: str) -> pd.DataFrame:
    """Run the full cleaning pipeline end-to-end and return a tidy DataFrame."""
    df = load_raw_data(raw_csv_path)
    df = drop_duplicate_listings(df)
    df = add_nearby_trains_column(df)
    df = drop_unused_columns(df)
    df = clean_rent_and_size(df)
    df = drop_missing_rent(df)
    df = filter_rent_range(df)
    df = filter_size_range(df)
    df = remove_bathroom_outliers(df)
    df = add_rent_per_sqft(df)
    df = remove_pps_outliers(df)
    df = df.drop(columns=["rent_per_sqft"])
    df = df.dropna()
    df = remove_rare_categories(df, "location", min_count=10)
    df = remove_rare_categories(df, "property_type", min_count=10)
    return df
