"""
One-hot encoding and final feature-matrix construction.
"""
import pandas as pd

# Baseline category dropped for each one-hot encoded column (avoids the dummy
# variable trap). Matches the columns dropped in the original notebook.
BASELINE_CATEGORIES = [
    "location_City",
    "property_type_Others",
    "furnished_Not Furnished",
    "nearby_trains_no",
]


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns and append them to the DataFrame."""
    dummies = pd.get_dummies(df.select_dtypes(include="object"), dtype="int")
    dummies = dummies.drop(columns=BASELINE_CATEGORIES, errors="ignore")
    return pd.concat([df, dummies], axis="columns")


def build_feature_matrix(df: pd.DataFrame):
    """
    Encode categoricals, drop the now-redundant raw text columns, and split
    into feature matrix X and target y.
    """
    df = encode_categoricals(df)

    raw_categorical_cols = [
        col for col in ["location", "property_type", "furnished", "nearby_trains"]
        if col in df.columns
    ]
    df = df.drop(columns=raw_categorical_cols)

    X = df.drop(columns=["monthly_rent"])
    y = df["monthly_rent"]
    return X, y
