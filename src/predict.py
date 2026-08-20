"""
Load the trained model and predict monthly rent for a new listing.

Usage:
    python -m src.predict --location Cheras --property-type Apartment \\
        --furnished "Partially Furnished" --nearby-trains yes \\
        --rooms 3 --parking 1 --bathroom 2 --size 1000
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

MODELS_DIR = Path("models")


def load_artifacts():
    model = joblib.load(MODELS_DIR / "rent_model.joblib")
    with open(MODELS_DIR / "feature_columns.json") as f:
        columns = json.load(f)
    return model, columns


def predict_rent(
    model,
    columns,
    location: str,
    property_type: str,
    furnished: str,
    nearby_trains: str,
    rooms: float,
    parking: float,
    bathroom: float,
    size: float,
) -> float:
    """Build the one-hot feature vector for a single listing and predict rent."""
    feature_vector = np.zeros(len(columns))
    feature_vector[columns.index("rooms")] = rooms
    feature_vector[columns.index("parking")] = parking
    feature_vector[columns.index("bathroom")] = bathroom
    feature_vector[columns.index("size")] = size

    # Categories that were used as the encoding baseline (dropped during
    # training) simply leave every one-hot column at 0, matching sklearn's
    # dummy-variable-trap handling.
    for prefix, value in [
        ("location", location),
        ("property_type", property_type),
        ("furnished", furnished),
        ("nearby_trains", nearby_trains),
    ]:
        col_name = f"{prefix}_{value}"
        if col_name in columns:
            feature_vector[columns.index(col_name)] = 1

    return float(model.predict([feature_vector])[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict monthly rent for a listing.")
    parser.add_argument("--location", required=True)
    parser.add_argument("--property-type", required=True)
    parser.add_argument("--furnished", required=True)
    parser.add_argument("--nearby-trains", required=True, choices=["yes", "no"])
    parser.add_argument("--rooms", type=float, required=True)
    parser.add_argument("--parking", type=float, required=True)
    parser.add_argument("--bathroom", type=float, required=True)
    parser.add_argument("--size", type=float, required=True)
    args = parser.parse_args()

    model, columns = load_artifacts()
    rent = predict_rent(
        model,
        columns,
        args.location,
        args.property_type,
        args.furnished,
        args.nearby_trains,
        args.rooms,
        args.parking,
        args.bathroom,
        args.size,
    )
    print(f"Predicted monthly rent: RM {rent:,.2f}")
