import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from feature_pipeline import build_feature_pipeline, ARTIFACT_DIR, RESULT_DIR, ensure_dirs

# optional ANN
HAS_TF = True
try:
    from tensorflow.keras.models import load_model
except Exception:
    HAS_TF = False


RANDOM_STATE = 42


def calc_metrics(y_true, y_pred):
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def save_confusion_matrix(y_true, y_pred, labels, file_name, title):
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", annot=False, fmt="d")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, file_name), dpi=200)
    plt.close()


def evaluate_sklearn_model(model_name, model, X_train, X_test, y_train, y_test, labels):
    start_train_pred = time.time()
    y_train_pred = model.predict(X_train)
    train_pred_time = time.time() - start_train_pred

    start_test_pred = time.time()
    y_test_pred = model.predict(X_test)
    test_pred_time = time.time() - start_test_pred

    train_metrics = calc_metrics(y_train, y_train_pred)
    test_metrics = calc_metrics(y_test, y_test_pred)

    save_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        labels=labels,
        file_name=f"confusion_matrix_eval_{model_name}.png",
        title=f"Confusion Matrix - {model_name}"
    )

    return {
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
        "train_prediction_time_seconds": float(train_pred_time),
        "test_prediction_time_seconds": float(test_pred_time),
    }


def evaluate_ann_model(bundle, labels):
    if not HAS_TF:
        print("skip ANN: tensorflow not available")
        return None, None

    ann_model_path = os.path.join(ARTIFACT_DIR, "best_model.h5")
    ann_label_encoder_path = os.path.join(ARTIFACT_DIR, "ann_label_encoder.pkl")

    if not os.path.exists(ann_model_path) or not os.path.exists(ann_label_encoder_path):
        print("skip ANN: best_model.h5 or ann_label_encoder.pkl not found")
        return None, None

    model = load_model(ann_model_path)
    label_encoder = joblib.load(ann_label_encoder_path)

    X_train = bundle["X_train_sel"].toarray()
    X_test = bundle["X_test_sel"].toarray()

    y_train = bundle["y_train"].to_numpy()
    y_test = bundle["y_test"].to_numpy()

    y_train_int = label_encoder.transform(y_train)
    y_test_int = label_encoder.transform(y_test)

    start_train_pred = time.time()
    y_train_pred_int = np.argmax(model.predict(X_train, verbose=0), axis=1)
    train_pred_time = time.time() - start_train_pred

    start_test_pred = time.time()
    y_test_pred_int = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_pred_time = time.time() - start_test_pred

    train_metrics = calc_metrics(y_train_int, y_train_pred_int)
    test_metrics = calc_metrics(y_test_int, y_test_pred_int)

    save_confusion_matrix(
        y_true=y_test_int,
        y_pred=y_test_pred_int,
        labels=list(range(len(label_encoder.classes_))),
        file_name="confusion_matrix_eval_ann.png",
        title="Confusion Matrix - ann"
    )

    result_row = {
        "algorithm": "ann",
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
        "train_prediction_time_seconds": float(train_pred_time),
        "test_prediction_time_seconds": float(test_pred_time),
    }

    prediction_df = pd.DataFrame({
        "algorithm": ["ann"] * len(y_test),
        "actual": y_test_int,
        "predicted": y_test_pred_int,
    })

    return result_row, prediction_df


def main():
    ensure_dirs()
    bundle = build_feature_pipeline(test_size=0.2, random_state=RANDOM_STATE)

    X_train = bundle["X_train_sel"]
    X_test = bundle["X_test_sel"]
    y_train = bundle["y_train"]
    y_test = bundle["y_test"]
    labels = sorted(y_train.unique().tolist())

    sklearn_model_files = {
        "decision_tree": os.path.join(ARTIFACT_DIR, "classifier_decision_tree.joblib"),
        "random_forest": os.path.join(ARTIFACT_DIR, "classifier_random_forest.joblib"),
        "knn": os.path.join(ARTIFACT_DIR, "classifier_knn.joblib"),
        "svm": os.path.join(ARTIFACT_DIR, "classifier_svm.joblib"),
        "logistic_regression": os.path.join(ARTIFACT_DIR, "classifier_logistic_regression.joblib"),
    }

    summary_rows = []
    prediction_rows = []

    for model_name, model_path in sklearn_model_files.items():
        if not os.path.exists(model_path):
            print(f"skip {model_name}: not found -> {model_path}")
            continue

        print(f"evaluate {model_name}")
        model = joblib.load(model_path)
        result_row = evaluate_sklearn_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            labels=labels
        )
        summary_rows.append(result_row)

        y_test_pred = model.predict(X_test)
        for actual, pred in zip(y_test, y_test_pred):
            prediction_rows.append({
                "algorithm": model_name,
                "actual": int(actual),
                "predicted": int(pred),
            })

    # ANN
    ann_row, ann_pred_df = evaluate_ann_model(bundle, labels)
    if ann_row is not None:
        summary_rows.append(ann_row)
        prediction_rows.extend(ann_pred_df.to_dict(orient="records"))

    summary_df = pd.DataFrame(summary_rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(
            by=["test_f1_weighted", "test_accuracy"],
            ascending=False
        )

    prediction_df = pd.DataFrame(prediction_rows)

    summary_df.to_csv(
        os.path.join(RESULT_DIR, "classification_evaluation_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    prediction_df.to_csv(
        os.path.join(RESULT_DIR, "classification_prediction_details.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if not summary_df.empty:
        best_model_name = summary_df.iloc[0]["algorithm"]
        best_score = summary_df.iloc[0]["test_f1_weighted"]
        with open(os.path.join(RESULT_DIR, "best_classifier_from_evaluation.txt"), "w", encoding="utf-8") as f:
            f.write(f"best_model_name={best_model_name}\n")
            f.write(f"best_test_f1_weighted={best_score}\n")

    print("saved -> results/classification_evaluation_summary.csv")
    print("saved -> results/classification_prediction_details.csv")


if __name__ == "__main__":
    main()