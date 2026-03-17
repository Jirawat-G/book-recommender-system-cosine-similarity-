import os
import re
import joblib
import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DB_URL = "mysql+pymysql://root:password@127.0.0.1:3306/book_rec_db?charset=utf8mb4"
ARTIFACT_DIR = "artifacts"

def preprocess_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^\w\sก-๙]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    engine = create_engine(DB_URL)

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

    print(df.head())
    print(df["course_id"].unique())
    print(df["course_id"].nunique())

    if df.empty:
        raise ValueError("ไม่พบข้อมูลหนังสือในฐานข้อมูล")

    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int)

    if df["course_id"].nunique() < 2:
        raise ValueError(f"class ไม่พอ: {df['course_id'].unique().tolist()}")

    df["text"] = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["toc"].fillna("")
    ).apply(preprocess_text)

    X_text = df["text"]
    y = df["course_id"]

    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X_text)

    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_vec, y)

    joblib.dump(vectorizer, os.path.join(ARTIFACT_DIR, "vectorizer.joblib"))
    joblib.dump(classifier, os.path.join(ARTIFACT_DIR, "classifier.joblib"))
    joblib.dump(X_vec, os.path.join(ARTIFACT_DIR, "book_matrix.joblib"))
    joblib.dump(df["id"].tolist(), os.path.join(ARTIFACT_DIR, "book_ids.joblib"))

    print("train complete")

if __name__ == "__main__":
    main()