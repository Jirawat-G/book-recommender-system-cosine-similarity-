import re
import joblib
import pymysql
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity

DB_CONFIG = {
    "host": "127.0.0.1",
    "user": "root",
    "password": "password",
    "database": "book_rec_db",
    "charset": "utf8mb4",
    "cursorclass": pymysql.cursors.DictCursor
}

vectorizer = joblib.load("artifacts/vectorizer.joblib")
classifier = joblib.load("artifacts/classifier.joblib")
book_matrix = joblib.load("artifacts/book_matrix.joblib")
book_ids = joblib.load("artifacts/book_ids.joblib")
book_id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

app = FastAPI()

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
    top_k_classes: int = Field(default=2, ge=1, le=5)


def preprocess_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\sก-๙]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_conn():
    return pymysql.connect(**DB_CONFIG)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/recommendations")
def recommend(payload: RecommendRequest):
    normalized_query = preprocess_text(payload.query)
    query_vec = vectorizer.transform([normalized_query])

    probs = classifier.predict_proba(query_vec)[0]
    class_ids = classifier.classes_

    ranked_classes = sorted(
        zip(class_ids, probs),
        key=lambda x: x[1],
        reverse=True
    )[:payload.top_k_classes]

    course_ids = [int(course_id) for course_id, _ in ranked_classes]

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT b.id, b.title, b.author, b.description, b.course_id, c.name_th AS course_name
                FROM books b
                JOIN courses c ON c.id = b.course_id
                WHERE b.is_active = 1
                  AND b.course_id IN ({",".join(["%s"] * len(course_ids))})
                """,
                course_ids
            )
            books = cur.fetchall()

            if not books:
                cur.execute("""
                    SELECT b.id, b.title, b.author, b.description, b.course_id, c.name_th AS course_name
                    FROM books b
                    JOIN courses c ON c.id = b.course_id
                    WHERE b.is_active = 1
                """)
                books = cur.fetchall()

            cur.execute(
                f"""
                SELECT id, name_th
                FROM courses
                WHERE id IN ({",".join(["%s"] * len(course_ids))})
                """,
                course_ids
            )
            course_rows = cur.fetchall()
            course_map = {row["id"]: row["name_th"] for row in course_rows}

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
                "predicted_classes": [],
                "results": []
            }

        candidate_matrix = book_matrix[candidate_idx]
        scores = cosine_similarity(query_vec, candidate_matrix)[0]

        results = []
        for book, score in zip(candidate_books, scores):
            results.append({
                "book_id": book["id"],
                "title": book["title"],
                "author": book["author"],
                "description": book["description"],
                "course_id": book["course_id"],
                "course_name": book["course_name"],
                "score": float(score)
            })

        results = sorted(results, key=lambda x: x["score"], reverse=True)[:payload.top_n]

        predicted_classes = [
            {
                "course_id": int(course_id),
                "course_name": course_map.get(int(course_id), ""),
                "score": float(score)
            }
            for course_id, score in ranked_classes
        ]

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
                    len(results)
                )
            )
            conn.commit()

        return {
            "query": payload.query,
            "normalized_query": normalized_query,
            "predicted_classes": predicted_classes,
            "results": results
        }

    finally:
        conn.close()