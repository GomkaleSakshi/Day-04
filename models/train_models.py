from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed_hygiene_data.csv"
MODEL_DIR = PROJECT_ROOT / "models"
COMPARISON_PATH = PROJECT_ROOT / "evaluation" / "model_comparison.txt"
TARGET_COLUMN = "hygiene_risk"
RISK_LABELS = ["Low", "Medium", "High"]


def main():
    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
    }
    results = {}

    for model_name, model in models.items():
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        results[model_name] = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "f1_score": f1_score(y_test, predictions, average="weighted", zero_division=0),
        }
        model_path = MODEL_DIR / (model_name.lower().replace(" ", "_") + ".joblib")
        joblib.dump(model, model_path)
        print(model_name, "test accuracy:", f"{results[model_name]['accuracy']:.3f}")
        print("Saved model to:", model_path.relative_to(PROJECT_ROOT))

    best_f1_score = max(metrics["f1_score"] for metrics in results.values())
    best_models = [
        name for name, metrics in results.items() if metrics["f1_score"] == best_f1_score
    ]
    best_model_name = (
        "Logistic Regression" if "Logistic Regression" in best_models else best_models[0]
    )
    selection_note = (
        "Both models tie on weighted F1; Logistic Regression is selected because it is simpler to explain."
        if len(best_models) > 1
        else f"{best_model_name} has the highest weighted F1 score."
    )
    lines = [
        "Model comparison (20% test set, random_state=42)",
        "Metrics use weighted averaging for the three risk classes.",
        "",
    ]
    for model_name, metrics in results.items():
        lines.append(model_name)
        for metric_name, value in metrics.items():
            lines.append(f"  {metric_name.replace('_', ' ').title()}: {value:.4f}")
        lines.append("")
    lines.append(f"Selected best model by weighted F1 score: {best_model_name}")
    lines.append(f"Selection note: {selection_note}")
    COMPARISON_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("Comparison saved to:", COMPARISON_PATH.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
