import os
import pickle
from typing import Dict

import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.model_selection import train_test_split

from preprocessing import combine_text, clean_text


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:password@127.0.0.1:3306/book_rec_db?charset=utf8mb4"
)

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def ensure_dirs():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)


def load_data_from_db() -> pd.DataFrame:
    engine = create_engine(DATABASE_URL)

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
        raise ValueError("ไม่พบข้อมูลหนังสือในฐานข้อมูล")

    df["course_id"] = pd.to_numeric(df["course_id"], errors="coerce")
    df = df.dropna(subset=["course_id"]).copy()
    df["course_id"] = df["course_id"].astype(int)

    if df["course_id"].nunique() < 2:
        raise ValueError("จำนวนคลาสไม่พอสำหรับการฝึกโมเดล")

    df["combined_text"] = df.apply(lambda row: combine_text(row.to_dict()), axis=1)
    df["clean_text"] = df["combined_text"].apply(clean_text)

    return df.reset_index(drop=True)


def build_feature_pipeline(
    test_size: float = 0.2,
    random_state: int = 42
) -> Dict:
    ensure_dirs()

    df = load_data_from_db()

    X = df["clean_text"]
    y = df["course_id"]

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X,
        y,
        df.index.to_numpy(),
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    selector = SelectPercentile(chi2, percentile=100)
    X_train_sel = selector.fit_transform(X_train_tfidf, y_train)
    X_test_sel = selector.transform(X_test_tfidf)

    selected_feature_names = tfidf.get_feature_names_out()[selector.get_support()]

    # save artifacts ตาม flow เดิม
    pickle.dump(tfidf, open(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"), "wb"))
    pickle.dump(selector, open(os.path.join(ARTIFACT_DIR, "feature_selector.pkl"), "wb"))
    pickle.dump(selected_feature_names, open(os.path.join(ARTIFACT_DIR, "selected_feature_names.pkl"), "wb"))

    return {
        "df": df,
        "X": X,
        "y": y,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "idx_train": idx_train,
        "idx_test": idx_test,
        "tfidf": tfidf,
        "selector": selector,
        "selected_feature_names": selected_feature_names,
        "X_train_sel": X_train_sel,
        "X_test_sel": X_test_sel,
    }