import os
import re
import json
import joblib
import pymysql
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity

try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "127.0.0.1"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "password"),
    "database": os.getenv("DB_NAME", "book_rec_db"),
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

SUPPORTED_STRATEGIES = {"cosine_only", "hybrid_top1", "hybrid_top2"}


def _safe_load_json(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_read_best_classifier() -> str:
    path = os.path.join(RESULTS_DIR, "best_classifier_from_evaluation.txt")
    if not os.path.exists(path):
        return "ann"

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("best_model_name="):
                return line.strip().split("=", 1)[1]
    return "ann"


def _safe_read_deployment_strategy() -> str:
    json_path = os.path.join(RESULTS_DIR, "final_experiment_summary.json")
    data = _safe_load_json(json_path)
    strategy = (
        data.get("deployment_recommendation", {})
        .get("strategy")
    )
    if strategy in SUPPORTED_STRATEGIES:
        return strategy
    return "hybrid_top2"


def preprocess_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\sก-๙]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl"))
selector = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_selector.pkl"))
book_matrix = selector.transform(vectorizer.transform([]))
book_ids = []
book_id_to_index = {}

book_matrix_path = os.path.join(ARTIFACTS_DIR, "book_matrix.joblib")
book_ids_path = os.path.join(ARTIFACTS_DIR, "book_ids.joblib")
if os.path.exists(book_matrix_path) and os.path.exists(book_ids_path):
    book_matrix = joblib.load(book_matrix_path)
    book_ids = joblib.load(book_ids_path)
    book_id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

DEFAULT_ALGORITHM = _safe_read_best_classifier()
DEFAULT_STRATEGY = _safe_read_deployment_strategy()

ann_model = None
ann_label_encoder = None
if DEFAULT_ALGORITHM == "ann":
    ann_model_path = os.path.join(ARTIFACTS_DIR, "best_model.h5")
    ann_label_encoder_path = os.path.join(ARTIFACTS_DIR, "ann_label_encoder.pkl")
    if load_model is not None and os.path.exists(ann_model_path) and os.path.exists(ann_label_encoder_path):
        ann_model = load_model(ann_model_path)
        ann_label_encoder = joblib.load(ann_label_encoder_path)

sklearn_classifier = None
if DEFAULT_ALGORITHM != "ann":
    classifier_path = os.path.join(ARTIFACTS_DIR, f"classifier_{DEFAULT_ALGORITHM}.joblib")
    if os.path.exists(classifier_path):
        sklearn_classifier = joblib.load(classifier_path)

if ann_model is None and sklearn_classifier is None:
    fallback_names = [
        "ann",
        "svm",
        "logistic_regression",
        "knn",
        "random_forest",
        "decision_tree",
    ]
    for model_name in fallback_names:
        if model_name == "ann":
            ann_model_path = os.path.join(ARTIFACTS_DIR, "best_model.h5")
            ann_label_encoder_path = os.path.join(ARTIFACTS_DIR, "ann_label_encoder.pkl")
            if load_model is not None and os.path.exists(ann_model_path) and os.path.exists(ann_label_encoder_path):
                ann_model = load_model(ann_model_path)
                ann_label_encoder = joblib.load(ann_label_encoder_path)
                DEFAULT_ALGORITHM = "ann"
                break
            continue

        classifier_path = os.path.join(ARTIFACTS_DIR, f"classifier_{model_name}.joblib")
        if os.path.exists(classifier_path):
            sklearn_classifier = joblib.load(classifier_path)
            DEFAULT_ALGORITHM = model_name
            break


app = FastAPI(title="Book Recommendation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RecommendRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500)
    top_n: int = Field(default=5, ge=1, le=20)
    strategy: str | None = Field(default=None)


class RecommendationConfigRequest(BaseModel):
    strategy: str = Field(pattern="^(cosine_only|hybrid_top1|hybrid_top2)$")


class AlgorithmInfo(BaseModel):
    algorithm: str
    strategy: str


def get_conn():
    return pymysql.connect(**DB_CONFIG)


def vectorize_query(normalized_query: str):
    query_vec = vectorizer.transform([normalized_query])
    return selector.transform(query_vec)


def get_top_k_predicted_classes(query_vec, top_k: int):
    if DEFAULT_ALGORITHM == "ann":
        if ann_model is None or ann_label_encoder is None:
            raise HTTPException(status_code=500, detail="ANN model artifacts not loaded")

        probs = ann_model.predict(query_vec.toarray(), verbose=0)[0]
        ranked_idx = np.argsort(probs)[::-1][:top_k]
        classes = ann_label_encoder.inverse_transform(ranked_idx)
        return [
            {
                "course_id": int(course_id),
                "course_name": "",
                "score": float(probs[idx]),
            }
            for course_id, idx in zip(classes, ranked_idx)
        ]

    if sklearn_classifier is None:
        raise HTTPException(status_code=500, detail="Classifier artifacts not loaded")

    if hasattr(sklearn_classifier, "predict_proba"):
        probs = sklearn_classifier.predict_proba(query_vec)[0]
        class_ids = sklearn_classifier.classes_
        ranked = sorted(zip(class_ids, probs), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "course_id": int(course_id),
                "course_name": "",
                "score": float(score),
            }
            for course_id, score in ranked
        ]

    if hasattr(sklearn_classifier, "decision_function"):
        scores = sklearn_classifier.decision_function(query_vec)
        class_ids = sklearn_classifier.classes_
        if scores.ndim == 1:
            ranked_idx = np.argsort(scores)[::-1][:top_k]
            return [
                {
                    "course_id": int(class_ids[i]),
                    "course_name": "",
                    "score": float(scores[i]),
                }
                for i in ranked_idx
            ]

        scores = scores[0]
        ranked_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {
                "course_id": int(class_ids[i]),
                "course_name": "",
                "score": float(scores[i]),
            }
            for i in ranked_idx
        ]

    pred = sklearn_classifier.predict(query_vec)[0]
    return [{"course_id": int(pred), "course_name": "", "score": 1.0}]


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "default_algorithm": DEFAULT_ALGORITHM,
        "default_strategy": DEFAULT_STRATEGY,
    }


@app.get("/api/model-info")
def model_info():
    return {
        "default_algorithm": DEFAULT_ALGORITHM,
        "default_strategy": DEFAULT_STRATEGY,
        "supported_strategies": sorted(SUPPORTED_STRATEGIES),
    }


@app.post("/api/recommendations")
def recommend(payload: RecommendRequest):
    strategy = payload.strategy or DEFAULT_STRATEGY
    if strategy not in SUPPORTED_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unsupported strategy: {strategy}")

    normalized_query = preprocess_text(payload.query)
    if not normalized_query:
        raise HTTPException(status_code=400, detail="Query is empty after preprocessing")

    query_vec = vectorize_query(normalized_query)

    predicted_classes = []
    course_ids = []
    top_k_classes = 0

    if strategy == "hybrid_top1":
        top_k_classes = 1
    elif strategy == "hybrid_top2":
        top_k_classes = 2

    if top_k_classes > 0:
        predicted_classes = get_top_k_predicted_classes(query_vec, top_k_classes)
        course_ids = [item["course_id"] for item in predicted_classes]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            books = []

            if strategy == "cosine_only" or not course_ids:
                cur.execute(
                    """
                    SELECT b.id, b.title, b.author, b.description, b.course_id, c.name_th AS course_name
                    FROM books b
                    JOIN courses c ON c.id = b.course_id
                    WHERE b.is_active = 1
                    """
                )
                books = cur.fetchall()
            else:
                placeholders = ",".join(["%s"] * len(course_ids))
                cur.execute(
                    f"""
                    SELECT b.id, b.title, b.author, b.description, b.course_id, c.name_th AS course_name
                    FROM books b
                    JOIN courses c ON c.id = b.course_id
                    WHERE b.is_active = 1
                      AND b.course_id IN ({placeholders})
                    """,
                    course_ids,
                )
                books = cur.fetchall()

                if len(books) < 5:
                    cur.execute(
                        """
                        SELECT b.id, b.title, b.author, b.description, b.course_id, c.name_th AS course_name
                        FROM books b
                        JOIN courses c ON c.id = b.course_id
                        WHERE b.is_active = 1
                        """
                    )
                    books = cur.fetchall()

            if course_ids:
                placeholders = ",".join(["%s"] * len(course_ids))
                cur.execute(
                    f"SELECT id, name_th FROM courses WHERE id IN ({placeholders})",
                    course_ids,
                )
                course_rows = cur.fetchall()
                course_map = {row["id"]: row["name_th"] for row in course_rows}
                for item in predicted_classes:
                    item["course_name"] = course_map.get(item["course_id"], "")

        candidate_idx = []
        candidate_books = []
        for book in books:
            idx = book_id_to_index.get(book["id"])
            if idx is None:
                continue
            candidate_idx.append(idx)
            candidate_books.append(book)

        if not candidate_books:
            return {
                "query": payload.query,
                "normalized_query": normalized_query,
                "strategy": strategy,
                "algorithm": DEFAULT_ALGORITHM,
                "predicted_classes": predicted_classes,
                "results": [],
            }

        candidate_matrix = book_matrix[candidate_idx]
        scores = cosine_similarity(query_vec, candidate_matrix)[0]

        results = []
        for book, score in zip(candidate_books, scores):
            results.append(
                {
                    "book_id": book["id"],
                    "title": book["title"],
                    "author": book["author"],
                    "description": book["description"],
                    "course_id": book["course_id"],
                    "course_name": book["course_name"],
                    "score": float(score),
                }
            )

        results = sorted(results, key=lambda x: x["score"], reverse=True)[: payload.top_n]

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO search_logs (query_text, normalized_query, predicted_course_ids, result_count)
                VALUES (%s, %s, %s, %s)
                """,
                (
                    payload.query,
                    normalized_query,
                    ",".join(str(x["course_id"]) for x in predicted_classes),
                    len(results),
                ),
            )
            conn.commit()

        return {
            "query": payload.query,
            "normalized_query": normalized_query,
            "strategy": strategy,
            "algorithm": DEFAULT_ALGORITHM,
            "predicted_classes": predicted_classes,
            "results": results,
        }
    finally:
        conn.close()