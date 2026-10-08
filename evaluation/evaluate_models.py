from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "dataset" / "processed_hygiene_data.csv"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
TARGET_COLUMN = "hygiene_risk"
RISK_LABELS = ["Low", "Medium", "High"]
MODELS = {
    "Logistic Regression": PROJECT_ROOT / "models" / "logistic_regression.joblib",
    "Decision Tree": PROJECT_ROOT / "models" / "decision_tree.joblib",
}


def save_eda_charts(data):
    raw_data = pd.read_csv(PROJECT_ROOT / "dataset" / "facility_hygiene_data.csv")
    cleaned = raw_data.drop_duplicates().copy()
    risk_order = ["Low", "Medium", "High"]
    plot_features = [
        ("cleanliness_score", "Cleanliness score by hygiene risk", "Cleanliness score"),
        ("odor_score", "Odor score by hygiene risk", "Odor score"),
        ("waste_level", "Waste level by hygiene risk", "Waste level"),
        ("hours_since_cleaning", "Time since cleaning by hygiene risk", "Hours since cleaning"),
    ]

    plt.figure(figsize=(7, 5))
    cleaned[TARGET_COLUMN].value_counts().reindex(risk_order).plot(
        kind="bar", color=["#5b9a75", "#e2ad4b", "#c85a54"]
    )
    plt.title("Hygiene risk distribution")
    plt.xlabel("Hygiene risk")
    plt.ylabel("Facility records")
    plt.tight_layout()
    plt.savefig(EVALUATION_DIR / "hygiene_risk_distribution.png", dpi=150)
    plt.close()

    for column, title, y_label in plot_features:
        grouped_values = [
            cleaned.loc[cleaned[TARGET_COLUMN] == risk, column].dropna()
            for risk in risk_order
        ]
        plt.figure(figsize=(7, 5))
        plt.boxplot(grouped_values, labels=risk_order)
        plt.title(title)
        plt.xlabel("Hygiene risk")
        plt.ylabel(y_label)
        plt.tight_layout()
        plt.savefig(EVALUATION_DIR / f"{column}_by_hygiene_risk.png", dpi=150)
        plt.close()


def main():
    data = pd.read_csv(DATA_PATH)
    features = data.drop(columns=[TARGET_COLUMN])
    target = data[TARGET_COLUMN]
    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    result_lines = ["Hygiene risk model evaluation", "Test set: 20%, random_state=42", ""]
    model_scores = {}

    for model_name, model_path in MODELS.items():
        model = joblib.load(model_path)
        predictions = model.predict(x_test)
        metrics = {
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision (weighted)": precision_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "Recall (weighted)": recall_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
            "F1 Score (weighted)": f1_score(
                y_test, predictions, average="weighted", zero_division=0
            ),
        }
        model_scores[model_name] = metrics["F1 Score (weighted)"]
        result_lines.append(model_name)
        for metric_name, value in metrics.items():
            result_lines.append(f"  {metric_name}: {value:.4f}")
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1, 2])
        result_lines.append("  Confusion matrix (rows=actual, columns=predicted):")
        result_lines.extend("    " + str(row.tolist()) for row in matrix)
        result_lines.append("")

        figure, axis = plt.subplots(figsize=(6, 5))
        image = axis.imshow(matrix, cmap="YlGn", interpolation="nearest")
        axis.set(
            xticks=[0, 1, 2],
            yticks=[0, 1, 2],
            xticklabels=RISK_LABELS,
            yticklabels=RISK_LABELS,
            xlabel="Predicted risk",
            ylabel="Actual risk",
            title=f"{model_name} confusion matrix",
        )
        figure.colorbar(image, ax=axis, label="Facility records")
        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                    color="white" if matrix[row_index, column_index] > threshold else "black",
                )
        figure.tight_layout()
        figure.savefig(
            EVALUATION_DIR / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png",
            dpi=150,
        )
        plt.close(figure)

    best_f1_score = max(model_scores.values())
    best_models = [name for name, score in model_scores.items() if score == best_f1_score]
    best_model_name = (
        "Logistic Regression" if "Logistic Regression" in best_models else best_models[0]
    )
    result_lines.append(f"Selected best model by weighted F1 score: {best_model_name}")
    if len(best_models) > 1:
        result_lines.append(
            "Selection note: Both models tie on weighted F1; Logistic Regression is selected because it is simpler to explain."
        )
    save_eda_charts(data)
    output_path = EVALUATION_DIR / "evaluation_results.txt"
    output_path.write_text("\n".join(result_lines) + "\n", encoding="utf-8")
    print("\n".join(result_lines))
    print("\nEvaluation results saved to:", output_path.relative_to(PROJECT_ROOT))
    print("EDA and confusion matrix charts saved in evaluation/")


if __name__ == "__main__":
    main()
