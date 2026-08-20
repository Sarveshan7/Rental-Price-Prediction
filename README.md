# Rental Price Prediction

A machine learning project that predicts monthly rental prices for
apartments and condos in Kuala Lumpur and Selangor, Malaysia. Built on
10,000 real listings scraped from mudah.my, the project covers the full
pipeline: cleaning messy real-world data, engineering features from
categorical and text fields, and training a linear regression model to
estimate rent from attributes like location, size, room count, furnishing
status, and proximity to public transit.

The analysis started as exploratory work in a Jupyter notebook
(`notebooks/`) and was refactored into reusable, tested Python modules
(`src/`) for training and running predictions from the command line.

## Project structure

```
rental-price-prediction/
├── data/
│   ├── raw/            # place the source CSV here (not committed)
│   └── processed/      # cleaned data written by the pipeline (not committed)
├── models/              # trained model + metadata (not committed)
├── notebooks/
│   └── Rental_Price_Prediction.ipynb   # original exploratory analysis
├── src/
│   ├── preprocessing.py  # cleaning: dedup, parse rent/size, outlier removal
│   ├── features.py       # one-hot encoding, feature matrix construction
│   ├── train.py          # trains and saves the model
│   └── predict.py        # loads the saved model and predicts rent
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
└── .github/workflows/tests.yml
```

## Dataset

The listings come from a scrape of [mudah.my](https://www.mudah.my) apartment
rentals for KL/Selangor. Download the dataset from Kaggle:

[Rental Pricing Dataset, Malaysia](https://www.kaggle.com/datasets/ariewijaya/rent-pricing-kuala-lumpur-malaysi)

```
data/raw/mudah-apartment-kl-selangor.csv
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Train the model:**

```bash
python -m src.train --data data/raw/mudah-apartment-kl-selangor.csv
```

Add `--compare-models` to also run a `GridSearchCV` comparison across linear
regression, lasso, and a decision tree before fitting the final model.

This writes `models/rent_model.joblib`, `models/feature_columns.json`, and
`models/metrics.json`.

**Predict a rent:**

```bash
python -m src.predict \
  --location Cheras --property-type Apartment \
  --furnished "Partially Furnished" --nearby-trains yes \
  --rooms 3 --parking 1 --bathroom 2 --size 1000
```

**Run tests:**

```bash
pytest
```

**Explore the original analysis:**

```bash
jupyter notebook notebooks/Rental_Price_Prediction.ipynb
```

## Pipeline overview

1. **Preprocessing** (`src/preprocessing.py`): drop duplicates, flag listings
   near a KTM/LRT station, parse `monthly_rent` and `size` out of free-text
   fields, drop rows with missing rent, filter unrealistic rent/size ranges,
   remove bathroom/room-count outliers, and remove rent-per-sqft outliers
   within each location group.
2. **Feature engineering** (`src/features.py`): one-hot encode `location`,
   `property_type`, `furnished`, and `nearby_trains`, dropping one baseline
   category per feature to avoid the dummy-variable trap.
3. **Modeling** (`src/train.py`): fits a `LinearRegression` model, evaluated
   with a held-out test split and 5-fold shuffle cross-validation.

## License

See [LICENSE](LICENSE).
