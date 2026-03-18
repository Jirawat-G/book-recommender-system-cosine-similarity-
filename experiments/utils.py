import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


@dataclass
class DataBundle:
    df: pd.DataFrame
    vectorizer: TfidfVectorizer
    X_all: csr_matrix
    y_all: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    X_train: csr_matrix
    X_test: csr_matrix
    y_train: np.ndarray
    y_test: np.ndarray


def get_engine() -> object:
    db_url = os.getenv(
        "DATABASE_URL",
        "mysql+pymysql://root:password@127.0.0.1:3306/book_rec_db?charset=utf8mb4"
    )
    return create_engine(db_url)


def preprocess_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\sก-๙]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_books_from_db() -> pd.DataFrame:
    engine = get_engine()
    sql = """
    SELECT
        b.id,
        b.title,
        b.description,
        b.toc,
        b.course_id
    FROM books b
    WHERE b.is_active = 1
    ORDER BY b.id ASC
    """
    df = pd.read_sql(sql, engine)

    if df.empty:
        raise ValueError("ไม่พบข้อมูลหนังสือจากฐานข้อมูล")

    required_cols = ["id", "title", "description", "toc", "course_id"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"ไม่พบคอลัมน์ {col} ในข้อมูล")

    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int)

    if df["course_id"].nunique() < 2:
        raise ValueError("จำนวนคลาสไม่พอสำหรับ train")

    df["combined_text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["toc"].fillna("")
    ).apply(preprocess_text)

    return df.reset_index(drop=True)


def build_data_bundle(
    test_size: float = 0.2,
    random_state: int = 42
) -> DataBundle:
    df = load_books_from_db()

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95
    )
    X_all = vectorizer.fit_transform(df["combined_text"])
    y_all = df["course_id"].to_numpy()

    indices = np.arange(len(df))
    train_idx, test_idx, y_train, y_test = train_test_split(
        indices,
        y_all,
        test_size=test_size,
        random_state=random_state,
        stratify=y_all
    )

    X_train = X_all[train_idx]
    X_test = X_all[test_idx]

    return DataBundle(
        df=df,
        vectorizer=vectorizer,
        X_all=X_all,
        y_all=y_all,
        train_idx=train_idx,
        test_idx=test_idx,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def get_model_configs(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
    return {
        "decision_tree": {
            "estimator": DecisionTreeClassifier(random_state=random_state),
            "param_grid": {
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "max_features": ["sqrt", "log2"],
            },
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "param_grid": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "cosine"],
            },
        },
        "svm": {
            "estimator": SVC(probability=True, random_state=random_state),
            "param_grid": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "class_weight": [None, "balanced"],
            },
        },
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=3000, random_state=random_state),
            "param_grid": {
                "C": [0.1, 1, 10],
                "solver": ["liblinear", "lbfgs"],
                "class_weight": [None, "balanced"],
            },
        },
        "ann": {
            "estimator": MLPClassifier(max_iter=500, random_state=random_state),
            "param_grid": {
                "hidden_layer_sizes": [(100,), (128,), (100, 50)],
                "activation": ["relu", "tanh"],
                "alpha": [0.0001, 0.001, 0.01],
            },
        },
    }


def train_classifier_with_search(
    model_name: str,
    X_train: csr_matrix,
    y_train: np.ndarray,
    cv_splits: int = 5,
    scoring: str = "f1_weighted",
    n_jobs: int = -1
):
    configs = get_model_configs()
    if model_name not in configs:
        raise ValueError(f"ไม่รองรับ model_name={model_name}")

    config = configs[model_name]
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=config["estimator"],
        param_grid=config["param_grid"],
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=1
    )
    search.fit(X_train, y_train)
    return search


def evaluate_classifier(model: object, X_test: csr_matrix, y_test: np.ndarray) -> Dict[str, float]:
    y_pred = model.predict(X_test)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def get_top_k_predicted_classes(model: object, X_query: csr_matrix, k: int = 2) -> List[Tuple[int, float]]:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_query)[0]
        class_ids = model.classes_
        ranked = sorted(zip(class_ids, probs), key=lambda x: x[1], reverse=True)
        return [(int(cls), float(score)) for cls, score in ranked[:k]]

    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_query)
        class_ids = model.classes_
        if scores.ndim == 1:
            ranked_idx = np.argsort(scores)[::-1][:k]
            return [(int(class_ids[i]), float(scores[i])) for i in ranked_idx]
        scores = scores[0]
        ranked_idx = np.argsort(scores)[::-1][:k]
        return [(int(class_ids[i]), float(scores[i])) for i in ranked_idx]

    pred = model.predict(X_query)[0]
    return [(int(pred), 1.0)]


def recommend_cosine_only(data: DataBundle, query_index: int, top_n: int = 5) -> List[Dict]:
    query_vec = data.X_all[query_index]
    scores = cosine_similarity(query_vec, data.X_all)[0]

    candidates = []
    for idx, score in enumerate(scores):
        if idx == query_index:
            continue
        candidates.append({
            "row_index": idx,
            "book_id": int(data.df.iloc[idx]["id"]),
            "course_id": int(data.df.iloc[idx]["course_id"]),
            "score": float(score)
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]


def recommend_classify_only(model: object, data: DataBundle, query_index: int, top_n: int = 5):
    query_vec = data.X_all[query_index]
    top_classes = get_top_k_predicted_classes(model, query_vec, k=1)
    predicted_class = top_classes[0][0]

    df_candidates = data.df.copy()
    df_candidates["row_index"] = np.arange(len(df_candidates))
    df_candidates = df_candidates[
        (df_candidates["course_id"] == predicted_class) &
        (df_candidates["row_index"] != query_index)
    ]

    results = []
    for _, row in df_candidates.head(top_n).iterrows():
        results.append({
            "row_index": int(row["row_index"]),
            "book_id": int(row["id"]),
            "course_id": int(row["course_id"]),
            "score": 1.0
        })

    return top_classes, results


def recommend_hybrid(model: object, data: DataBundle, query_index: int, top_n: int = 5, top_k_classes: int = 2):
    query_vec = data.X_all[query_index]
    top_classes = get_top_k_predicted_classes(model, query_vec, k=top_k_classes)
    class_ids = [cls for cls, _ in top_classes]

    df_candidates = data.df.copy()
    df_candidates["row_index"] = np.arange(len(df_candidates))
    df_candidates = df_candidates[
        df_candidates["course_id"].isin(class_ids) &
        (df_candidates["row_index"] != query_index)
    ]

    if df_candidates.empty:
        return top_classes, []

    row_indices = df_candidates["row_index"].to_numpy()
    candidate_matrix = data.X_all[row_indices]
    scores = cosine_similarity(query_vec, candidate_matrix)[0]

    results = []
    for i, (_, row) in enumerate(df_candidates.iterrows()):
        results.append({
            "row_index": int(row["row_index"]),
            "book_id": int(row["id"]),
            "course_id": int(row["course_id"]),
            "score": float(scores[i])
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return top_classes, results[:top_n]


def evaluate_recommendation_result(true_course_id: int, results: List[Dict], k_values=None) -> Dict[str, float]:
    if k_values is None:
        k_values = [1, 3, 5]

    metrics = {}
    relevant_positions = [
        i for i, item in enumerate(results, start=1)
        if item["course_id"] == true_course_id
    ]

    for k in k_values:
        top_k = results[:k]
        hit = 1.0 if any(item["course_id"] == true_course_id for item in top_k) else 0.0
        precision = (
            sum(1 for item in top_k if item["course_id"] == true_course_id) / k
            if top_k else 0.0
        )
        metrics[f"hit@{k}"] = hit
        metrics[f"precision@{k}"] = precision

    metrics["mrr"] = 1.0 / min(relevant_positions) if relevant_positions else 0.0
    return metrics


def ensure_dirs():
    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("results", exist_ok=True)


def save_model_artifacts(model_name: str, data: DataBundle, fitted_search):
    ensure_dirs()
    best_model = fitted_search.best_estimator_

    joblib.dump(data.vectorizer, os.path.join("artifacts", "vectorizer.joblib"))
    joblib.dump(best_model, os.path.join("artifacts", f"classifier_{model_name}.joblib"))
    joblib.dump(data.X_all, os.path.join("artifacts", "book_matrix.joblib"))
    joblib.dump(data.df["id"].tolist(), os.path.join("artifacts", "book_ids.joblib"))

    metadata = {
        "model_name": model_name,
        "total_books": int(len(data.df)),
        "total_classes": int(data.df["course_id"].nunique()),
        "best_params": fitted_search.best_params_,
        "best_cv_score": float(fitted_search.best_score_),
    }
    with open(os.path.join("artifacts", f"metadata_{model_name}.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)