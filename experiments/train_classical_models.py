import os
import time
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from feature_pipeline import build_feature_pipeline, ARTIFACT_DIR, RESULT_DIR, ensure_dirs


def calc_metrics(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def save_confusion_matrix(y_true, y_pred, labels, file_name):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", annot=False, fmt="d")
    plt.title(file_name.replace(".png", ""))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, file_name), dpi=200)
    plt.close()


def main():
    ensure_dirs()
    bundle = build_feature_pipeline()

    X_train = bundle["X_train_sel"]
    X_test = bundle["X_test_sel"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]

    labels = sorted(y_train.unique().tolist())

    models = {
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=42),
            "param_grid": {
                "max_depth": [5, 10, 20, 30, 40, 50]
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "n_estimators": [50, 100, 150, 200],
                "max_depth": [None, 10, 20, 30],
            },
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": list(range(1, 21))
            },
        },
        "svm": {
            "estimator": SVC(random_state=42),
            "param_grid": {
                "C": [0.1, 1, 10, 100],
                "kernel": ["linear", "rbf", "sigmoid"],
            },
        },
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=1000, random_state=42),
            "param_grid": {
                "C": [0.1, 0.5, 1, 5, 10]
            },
        },
    }

    best_params_rows = []
    classification_rows = []
    training_time_rows = []
    prediction_time_rows = []

    best_model_name = None
    best_test_acc = -1
    best_model = None

    for model_name, config in models.items():
        print(f"\n=== Training {model_name} ===")

        start_train = time.time()
        grid = GridSearchCV(
            config["estimator"],
            config["param_grid"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        train_time = time.time() - start_train

        model = grid.best_estimator_

        # training prediction
        start_pred_train = time.time()
        y_train_pred = model.predict(X_train)
        train_pred_time = time.time() - start_pred_train

        # test prediction
        start_pred_test = time.time()
        y_test_pred = model.predict(X_test)
        test_pred_time = time.time() - start_pred_test

        train_metrics = calc_metrics(y_train, y_train_pred)
        test_metrics = calc_metrics(y_test, y_test_pred)

        save_confusion_matrix(
            y_test,
            y_test_pred,
            labels=labels,
            file_name=f"confusion_matrix_{model_name}.png"
        )

        joblib.dump(model, os.path.join(ARTIFACT_DIR, f"classifier_{model_name}.joblib"))

        best_params_rows.append({
            "algorithm": model_name,
            "best_params": str(grid.best_params_),
            "best_cv_accuracy": grid.best_score_,
        })

        classification_rows.append({
            "algorithm": model_name,
            "train_accuracy": train_metrics["accuracy"],
            "test_accuracy": test_metrics["accuracy"],
            "train_precision_macro": train_metrics["precision_macro"],
            "test_precision_macro": test_metrics["precision_macro"],
            "train_recall_macro": train_metrics["recall_macro"],
            "test_recall_macro": test_metrics["recall_macro"],
            "train_f1_macro": train_metrics["f1_macro"],
            "test_f1_macro": test_metrics["f1_macro"],
            "train_precision_weighted": train_metrics["precision_weighted"],
            "test_precision_weighted": test_metrics["precision_weighted"],
            "train_recall_weighted": train_metrics["recall_weighted"],
            "test_recall_weighted": test_metrics["recall_weighted"],
            "train_f1_weighted": train_metrics["f1_weighted"],
            "test_f1_weighted": test_metrics["f1_weighted"],
        })

        training_time_rows.append({
            "algorithm": model_name,
            "training_time_seconds": train_time
        })

        prediction_time_rows.append({
            "algorithm": model_name,
            "train_prediction_time_seconds": train_pred_time,
            "test_prediction_time_seconds": test_pred_time,
        })

        if test_metrics["accuracy"] > best_test_acc:
            best_test_acc = test_metrics["accuracy"]
            best_model_name = model_name
            best_model = model

    pd.DataFrame(best_params_rows).to_csv(
        os.path.join(RESULT_DIR, "best_params_classical.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(classification_rows).to_csv(
        os.path.join(RESULT_DIR, "classification_metrics_classical.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(training_time_rows).to_csv(
        os.path.join(RESULT_DIR, "training_time_classical.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    pd.DataFrame(prediction_time_rows).to_csv(
        os.path.join(RESULT_DIR, "prediction_time_classical.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if best_model is not None:
        joblib.dump(best_model, os.path.join(ARTIFACT_DIR, "best_classical_model.joblib"))
        with open(os.path.join(RESULT_DIR, "best_classical_model.txt"), "w", encoding="utf-8") as f:
            f.write(f"best_model_name={best_model_name}\n")
            f.write(f"best_test_accuracy={best_test_acc}\n")

    print("\nFinished classical models.")
    print(f"Best classical model: {best_model_name} ({best_test_acc:.4f})")


if __name__ == "__main__":
    main()