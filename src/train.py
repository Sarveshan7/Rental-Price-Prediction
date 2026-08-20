"""
Train the rental price regression model and persist it to disk.

Usage:
    python -m src.train --data data/raw/mudah-apartment-kl-selangor.csv
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeRegressor

from src.features import build_feature_matrix
from src.preprocessing import run_pipeline

MODELS_DIR = Path("models")


def find_best_model(X, y) -> pd.DataFrame:
    """Compare a few regressors with grid search and return their best scores."""
    algos = {
        "linear_regression": {
            "model": LinearRegression(),
            "params": {
                "copy_X": [True, False],
                "fit_intercept": [True, False],
                "positive": [True, False],
            },
        },
        "lasso": {
            "model": Lasso(),
            "params": {
                "alpha": [1, 2],
                "selection": ["random", "cyclic"],
            },
        },
        "decision_tree": {
            "model": DecisionTreeRegressor(),
            "params": {
                "criterion": ["squared_error", "friedman_mse"],
                "splitter": ["best", "random"],
            },
        },
    }

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    scores = []
    for name, cfg in algos.items():
        gs = GridSearchCV(cfg["model"], cfg["params"], cv=cv, return_train_score=False)
        gs.fit(X, y)
        scores.append({"model": name, "best_score": gs.best_score_, "best_params": gs.best_params_})

    return pd.DataFrame(scores, columns=["model", "best_score", "best_params"])


def train(data_path: str, compare_models: bool = False) -> None:
    print(f"Loading and cleaning data from {data_path} ...")
    df = run_pipeline(data_path)
    X, y = build_feature_matrix(df)
    print(f"Feature matrix: {X.shape[0]} rows, {X.shape[1]} columns")

    if compare_models:
        print("Comparing candidate models with GridSearchCV (this can take a while) ...")
        print(find_best_model(X, y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    model = LinearRegression()
    model.fit(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    cv_scores = cross_val_score(LinearRegression(), X, y, cv=cv)

    print(f"Test R^2: {test_r2:.4f}")
    print(f"Cross-val R^2: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODELS_DIR / "rent_model.joblib")
    with open(MODELS_DIR / "feature_columns.json", "w") as f:
        json.dump(list(X.columns), f, indent=2)
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump({"test_r2": test_r2, "cv_r2_mean": cv_scores.mean()}, f, indent=2)

    print(f"Saved model to {MODELS_DIR / 'rent_model.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the rental price model.")
    parser.add_argument(
        "--data",
        default="data/raw/mudah-apartment-kl-selangor.csv",
        help="Path to the raw listings CSV.",
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Also run GridSearchCV over several model types before training the final model.",
    )
    args = parser.parse_args()
    train(args.data, compare_models=args.compare_models)
