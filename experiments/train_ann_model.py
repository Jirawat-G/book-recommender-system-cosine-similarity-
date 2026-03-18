import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

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


def save_confusion_matrix(y_true, y_pred, file_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", annot=False, fmt="d")
    plt.title(file_name.replace(".png", ""))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, file_name), dpi=200)
    plt.close()


def build_ann(input_dim, n_classes, units, dropout_rate, learning_rate):
    model = Sequential([
        Dense(units, activation="relu", input_shape=(input_dim,)),
        Dropout(dropout_rate),
        Dense(units // 2, activation="relu"),
        Dropout(dropout_rate),
        Dense(n_classes, activation="softmax")
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    ensure_dirs()
    bundle = build_feature_pipeline()

    X_train = bundle["X_train_sel"].toarray()
    X_test = bundle["X_test_sel"].toarray()
    y_train = bundle["y_train"].to_numpy()
    y_test = bundle["y_test"].to_numpy()

    le = LabelEncoder()
    y_train_int = le.fit_transform(y_train)
    y_test_int = le.transform(y_test)

    y_train_enc = to_categorical(y_train_int)
    y_test_enc = to_categorical(y_test_int)

    n_classes = len(le.classes_)
    input_dim = X_train.shape[1]

    ann_rows = []
    best_ann_acc = -1
    best_params = None
    best_model = None
    best_y_test_pred = None
    best_train_time = None
    best_test_pred_time = None

    for units in [128, 256]:
        for dropout in [0.2, 0.3]:
            for learning_rate in [0.001, 0.01]:
                for epochs in [20, 50]:
                    for batch_size in [32, 64]:
                        print(
                            f"ANN -> units={units}, dropout={dropout}, "
                            f"lr={learning_rate}, epochs={epochs}, batch={batch_size}"
                        )

                        model = build_ann(
                            input_dim=input_dim,
                            n_classes=n_classes,
                            units=units,
                            dropout_rate=dropout,
                            learning_rate=learning_rate
                        )

                        start_train = time.time()
                        model.fit(
                            X_train,
                            y_train_enc,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            verbose=0
                        )
                        train_time = time.time() - start_train

                        # train pred
                        start_train_pred = time.time()
                        y_train_pred = np.argmax(model.predict(X_train, verbose=0), axis=1)
                        train_pred_time = time.time() - start_train_pred

                        # test pred
                        start_test_pred = time.time()
                        y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
                        test_pred_time = time.time() - start_test_pred

                        train_metrics = calc_metrics(y_train_int, y_train_pred)
                        test_metrics = calc_metrics(y_test_int, y_test_pred)

                        ann_rows.append({
                            "algorithm": "ann",
                            "units": units,
                            "dropout_rate": dropout,
                            "learning_rate": learning_rate,
                            "epochs": epochs,
                            "batch_size": batch_size,
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
                            "training_time_seconds": train_time,
                            "train_prediction_time_seconds": train_pred_time,
                            "test_prediction_time_seconds": test_pred_time,
                        })

                        if test_metrics["accuracy"] > best_ann_acc:
                            best_ann_acc = test_metrics["accuracy"]
                            best_params = {
                                "units": units,
                                "dropout_rate": dropout,
                                "learning_rate": learning_rate,
                                "epochs": epochs,
                                "batch_size": batch_size
                            }
                            best_model = model
                            best_y_test_pred = y_test_pred
                            best_train_time = train_time
                            best_test_pred_time = test_pred_time

    pd.DataFrame(ann_rows).to_csv(
        os.path.join(RESULT_DIR, "ann_all_results.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    if best_model is not None:
        best_model.save(os.path.join(ARTIFACT_DIR, "best_model.h5"))
        joblib.dump(le, os.path.join(ARTIFACT_DIR, "ann_label_encoder.pkl"))

        save_confusion_matrix(
            y_test_int,
            best_y_test_pred,
            "confusion_matrix_ann.png"
        )

        with open(os.path.join(RESULT_DIR, "best_ann_model.txt"), "w", encoding="utf-8") as f:
            f.write(f"best_test_accuracy={best_ann_acc}\n")
            f.write(f"best_params={best_params}\n")
            f.write(f"training_time_seconds={best_train_time}\n")
            f.write(f"test_prediction_time_seconds={best_test_pred_time}\n")

    print("\nFinished ANN.")
    print(f"Best ANN accuracy: {best_ann_acc:.4f}")
    print(f"Best ANN params: {best_params}")


if __name__ == "__main__":
    main()