from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "dataset" / "facility_hygiene_data.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "dataset" / "processed_hygiene_data.csv"
FEATURE_COLUMNS = [
    "cleanliness_score",
    "odor_score",
    "waste_level",
    "complaints",
    "footfall",
    "hours_since_cleaning",
]
SCORE_COLUMNS = ["cleanliness_score", "odor_score", "waste_level"]
TARGET_COLUMN = "hygiene_risk"


def main():
    data = pd.read_csv(RAW_DATA_PATH)

    print("Raw dataset shape:", data.shape)
    print("\nDataset information:")
    data.info()
    print("\nMissing values before cleaning:")
    print(data.isna().sum())
    print("\nData types before cleaning:")
    print(data.dtypes)

    data = data.drop_duplicates().copy()
    print("\nDuplicate records removed.")

    for column in FEATURE_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    for column in SCORE_COLUMNS:
        data.loc[~data[column].between(0, 10), column] = np.nan

    for column in ["complaints", "footfall", "hours_since_cleaning"]:
        data.loc[data[column] < 0, column] = np.nan

    valid_targets = ["Low", "Medium", "High"]
    data.loc[~data[TARGET_COLUMN].isin(valid_targets), TARGET_COLUMN] = np.nan
    data = data.dropna(subset=[TARGET_COLUMN]).copy()

    print("\nMissing values after invalid-value checks:")
    print(data.isna().sum())

    for column in FEATURE_COLUMNS:
        data[column] = data[column].fillna(data[column].median())

    print("\nMissing values after imputation:")
    print(data.isna().sum())
    print("\nData types after cleaning:")
    print(data.dtypes)
    print("\nCleaned dataset shape:", data.shape)
    print("\nFeature summary:")
    print(data[FEATURE_COLUMNS].describe().round(2))
    print("\nAverage feature values by hygiene risk:")
    print(data.groupby(TARGET_COLUMN)[FEATURE_COLUMNS].mean().round(2))
    print("\nHygiene risk counts:")
    print(data[TARGET_COLUMN].value_counts())

    data["cleaning_overdue"] = (data["hours_since_cleaning"] > 8).astype(int)
    features = data[FEATURE_COLUMNS + ["cleaning_overdue"]]
    target = data[TARGET_COLUMN].map({"Low": 0, "Medium": 1, "High": 2})
    processed_data = features.copy()
    processed_data[TARGET_COLUMN] = target
    processed_data.to_csv(PROCESSED_DATA_PATH, index=False)

    print("\nEncoded target values: Low=0, Medium=1, High=2")
    print("Processed dataset saved to:", PROCESSED_DATA_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
