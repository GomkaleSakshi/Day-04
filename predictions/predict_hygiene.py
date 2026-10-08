from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "logistic_regression.joblib"
COMPARISON_PATH = PROJECT_ROOT / "evaluation" / "model_comparison.txt"
OUTPUT_PATH = PROJECT_ROOT / "predictions" / "hygiene_predictions.csv"
FEATURE_COLUMNS = [
    "cleanliness_score",
    "odor_score",
    "waste_level",
    "complaints",
    "footfall",
    "hours_since_cleaning",
]
RISK_LABELS = {0: "Low", 1: "Medium", 2: "High"}


def main():
    comparison = COMPARISON_PATH.read_text(encoding="utf-8")
    best_model_line = next(
        line for line in comparison.splitlines() if line.startswith("Selected best model by weighted F1 score:")
    )
    best_model_name = best_model_line.split(":", maxsplit=1)[1].strip()
    model_filename = best_model_name.lower().replace(" ", "_") + ".joblib"
    model_path = PROJECT_ROOT / "models" / model_filename
    model = joblib.load(model_path)

    samples = pd.DataFrame(
        [
            [9.0, 1.0, 1.0, 0, 85, 2],
            [6.0, 5.0, 5.0, 3, 170, 7],
            [2.5, 9.0, 9.0, 8, 260, 18],
        ],
        columns=FEATURE_COLUMNS,
    )
    samples["cleaning_overdue"] = (samples["hours_since_cleaning"] > 8).astype(int)
    predictions = model.predict(samples)
    output = samples[FEATURE_COLUMNS].copy()
    output["predicted_hygiene_risk"] = [RISK_LABELS[int(value)] for value in predictions]
    output.to_csv(OUTPUT_PATH, index=False)

    print("Selected model:", best_model_name)
    print("Facility hygiene risk predictions:")
    print(output.to_string(index=False))
    print("\nPredictions saved to:", OUTPUT_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
