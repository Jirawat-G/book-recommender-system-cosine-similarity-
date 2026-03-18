import os
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from feature_pipeline import build_feature_pipeline, RESULT_DIR, ARTIFACT_DIR, ensure_dirs


SAMPLE_QUERIES = [
    "Python Programming",
    "Non Linear Equation",
    "JavaScript Frameworks",
    "Software Engineering",
]


def recommend_books(query: str, df, tfidf, selector, top_n: int = 5):
    query_vec = tfidf.transform([query])
    query_sel = selector.transform(query_vec)

    all_doc_vec = selector.transform(tfidf.transform(df["clean_text"]))
    scores = cosine_similarity(query_sel, all_doc_vec)[0]

    temp = df.copy()
    temp["cosine_similarity"] = scores
    temp = temp.sort_values(by="cosine_similarity", ascending=False)

    return temp.head(top_n)[["id", "title", "course_id", "cosine_similarity"]]


def main():
    ensure_dirs()
    bundle = build_feature_pipeline()

    df = bundle["df"]
    tfidf = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.pkl"))
    selector = joblib.load(os.path.join(ARTIFACT_DIR, "feature_selector.pkl"))

    rows = []

    for query in SAMPLE_QUERIES:
        top_books = recommend_books(query, df, tfidf, selector, top_n=5)

        for rank, (_, row) in enumerate(top_books.iterrows(), start=1):
            rows.append({
                "query": query,
                "rank": rank,
                "book_id": int(row["id"]),
                "title": row["title"],
                "course_id": int(row["course_id"]),
                "cosine_similarity": float(row["cosine_similarity"]),
            })

    result_df = pd.DataFrame(rows)
    result_df.to_csv(
        os.path.join(RESULT_DIR, "recommendation_query_examples.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    print("saved -> results/recommendation_query_examples.csv")


if __name__ == "__main__":
    main()