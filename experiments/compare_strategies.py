import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from feature_pipeline import build_feature_pipeline, ARTIFACT_DIR, RESULT_DIR, ensure_dirs
from tensorflow.keras.models import load_model

RANDOM_STATE = 42
TOP_N = 5
MIN_CANDIDATES = 5


def load_best_classifier_name():
    path = os.path.join(RESULT_DIR, "best_classifier_from_evaluation.txt")
    if not os.path.exists(path):
        raise FileNotFoundError("ไม่พบ results/best_classifier_from_evaluation.txt กรุณารัน evaluate_classification.py ก่อน")

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("best_model_name="):
            return line.strip().split("=", 1)[1]

    raise ValueError("ไม่พบ best_model_name ใน best_classifier_from_evaluation.txt")


def get_relevant_row_indices(df, query_index):
    query_course_id = int(df.iloc[query_index]["course_id"])
    df_temp = df.copy()
    df_temp["row_index"] = np.arange(len(df_temp))

    relevant_rows = df_temp[
        (df_temp["course_id"] == query_course_id) &
        (df_temp["row_index"] != query_index)
    ]["row_index"].tolist()

    return relevant_rows


def precision_recall_f1_at_k(result_indices, relevant_indices, k):
    top_k = result_indices[:k]
    relevant_set = set(relevant_indices)

    retrieved_relevant = sum(1 for idx in top_k if idx in relevant_set)
    precision = retrieved_relevant / k if k > 0 else 0.0

    total_relevant = len(relevant_indices)
    recall = retrieved_relevant / total_relevant if total_relevant > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def hit_at_k(result_indices, relevant_indices, k):
    relevant_set = set(relevant_indices)
    return 1.0 if any(idx in relevant_set for idx in result_indices[:k]) else 0.0


def mrr_score(result_indices, relevant_indices):
    relevant_set = set(relevant_indices)
    for rank, idx in enumerate(result_indices, start=1):
        if idx in relevant_set:
            return 1.0 / rank
    return 0.0


def calc_recommendation_metrics(result_indices, relevant_indices):
    metrics = {}
    for k in [1, 3, 5]:
        p, r, f1 = precision_recall_f1_at_k(result_indices, relevant_indices, k)
        metrics[f"precision@{k}"] = p
        metrics[f"recall@{k}"] = r
        metrics[f"f1@{k}"] = f1
        metrics[f"hit@{k}"] = hit_at_k(result_indices, relevant_indices, k)

    metrics["mrr"] = mrr_score(result_indices, relevant_indices)
    return metrics


def build_bundle():
    bundle = build_feature_pipeline(test_size=0.2, random_state=RANDOM_STATE)
    bundle["X_all_sel"] = bundle["selector"].transform(
        bundle["tfidf"].transform(bundle["df"]["clean_text"])
    )
    return bundle


def load_predictor(best_model_name):
    if best_model_name == "ann":
        ann_model_path = os.path.join(ARTIFACT_DIR, "best_model.h5")
        ann_label_encoder_path = os.path.join(ARTIFACT_DIR, "ann_label_encoder.pkl")

        if not os.path.exists(ann_model_path) or not os.path.exists(ann_label_encoder_path):
            raise FileNotFoundError("ไม่พบ ANN artifacts")

        return {
            "type": "ann",
            "algorithm": "ann",
            "model": load_model(ann_model_path),
            "label_encoder": joblib.load(ann_label_encoder_path),
        }

    model_path = os.path.join(ARTIFACT_DIR, f"classifier_{best_model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ไม่พบ model file: {model_path}")

    return {
        "type": "sklearn",
        "algorithm": best_model_name,
        "model": joblib.load(model_path),
    }


def get_top_k_predicted_classes(predictor, query_vec, k):
    if predictor["type"] == "sklearn":
        model = predictor["model"]

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(query_vec)[0]
            class_ids = model.classes_
            ranked = sorted(zip(class_ids, probs), key=lambda x: x[1], reverse=True)
            return [(int(cls), float(score)) for cls, score in ranked[:k]]

        if hasattr(model, "decision_function"):
            scores = model.decision_function(query_vec)
            class_ids = model.classes_

            if scores.ndim == 1:
                ranked_idx = np.argsort(scores)[::-1][:k]
                return [(int(class_ids[i]), float(scores[i])) for i in ranked_idx]

            scores = scores[0]
            ranked_idx = np.argsort(scores)[::-1][:k]
            return [(int(class_ids[i]), float(scores[i])) for i in ranked_idx]

        pred = model.predict(query_vec)[0]
        return [(int(pred), 1.0)]

    # ANN
    probs = predictor["model"].predict(query_vec.toarray(), verbose=0)[0]
    ranked_idx = np.argsort(probs)[::-1][:k]
    classes = predictor["label_encoder"].inverse_transform(ranked_idx)
    return [(int(cls), float(probs[idx])) for cls, idx in zip(classes, ranked_idx)]


def cosine_only_recommendation(bundle, query_index, top_n=TOP_N):
    query_vec = bundle["X_all_sel"][query_index]
    scores = cosine_similarity(query_vec, bundle["X_all_sel"])[0]

    pairs = []
    for idx, score in enumerate(scores):
        if idx == query_index:
            continue
        pairs.append((idx, float(score)))

    pairs.sort(key=lambda x: x[1], reverse=True)
    return [], pairs[:top_n]


def hybrid_recommendation(bundle, predictor, query_index, top_k_classes, top_n=TOP_N):
    query_vec = bundle["X_all_sel"][query_index]
    predicted_classes = get_top_k_predicted_classes(predictor, query_vec, top_k_classes)

    class_ids = [cls for cls, _ in predicted_classes]
    df = bundle["df"].copy()
    df["row_index"] = np.arange(len(df))

    candidate_df = df[
        (df["course_id"].isin(class_ids)) &
        (df["row_index"] != query_index)
    ]

    if len(candidate_df) < MIN_CANDIDATES:
        candidate_df = df[df["row_index"] != query_index]

    candidate_indices = candidate_df["row_index"].tolist()
    candidate_matrix = bundle["X_all_sel"][candidate_indices]
    scores = cosine_similarity(query_vec, candidate_matrix)[0]

    pairs = list(zip(candidate_indices, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    return predicted_classes, pairs[:top_n]


def evaluate_strategy(bundle, predictor, strategy_name, top_k_classes=None):
    df = bundle["df"]
    test_indices = bundle["idx_test"]

    metric_sum = {
        "precision@1": 0.0, "recall@1": 0.0, "f1@1": 0.0, "hit@1": 0.0,
        "precision@3": 0.0, "recall@3": 0.0, "f1@3": 0.0, "hit@3": 0.0,
        "precision@5": 0.0, "recall@5": 0.0, "f1@5": 0.0, "hit@5": 0.0,
        "mrr": 0.0,
    }

    detail_rows = []

    for query_index in test_indices:
        relevant_indices = get_relevant_row_indices(df, query_index)

        if strategy_name == "cosine_only":
            predicted_classes, pairs = cosine_only_recommendation(bundle, query_index, top_n=TOP_N)
        elif strategy_name == "hybrid_top1":
            predicted_classes, pairs = hybrid_recommendation(bundle, predictor, query_index, top_k_classes=1, top_n=TOP_N)
        elif strategy_name == "hybrid_top2":
            predicted_classes, pairs = hybrid_recommendation(bundle, predictor, query_index, top_k_classes=2, top_n=TOP_N)
        else:
            raise ValueError(f"unsupported strategy: {strategy_name}")

        result_indices = [idx for idx, _ in pairs]
        metrics = calc_recommendation_metrics(result_indices, relevant_indices)

        for key in metric_sum:
            metric_sum[key] += metrics[key]

        detail_rows.append({
            "strategy": strategy_name,
            "algorithm": predictor["algorithm"] if predictor else "",
            "query_row_index": int(query_index),
            "query_book_id": int(df.iloc[query_index]["id"]),
            "query_title": str(df.iloc[query_index]["title"]),
            "query_course_id": int(df.iloc[query_index]["course_id"]),
            "predicted_classes": "|".join(f"{cls}:{score:.6f}" for cls, score in predicted_classes),
            "top_result_row_index": int(result_indices[0]) if result_indices else None,
            "top_result_book_id": int(df.iloc[result_indices[0]]["id"]) if result_indices else None,
            "top_result_course_id": int(df.iloc[result_indices[0]]["course_id"]) if result_indices else None,
            **metrics
        })

    total_queries = len(test_indices)
    summary_row = {
        "strategy": strategy_name,
        "algorithm": predictor["algorithm"] if predictor else "",
        "top_k_classes": 0 if strategy_name == "cosine_only" else (1 if strategy_name == "hybrid_top1" else 2),
        **{k: v / total_queries for k, v in metric_sum.items()}
    }

    return summary_row, detail_rows


def main():
    ensure_dirs()
    best_model_name = load_best_classifier_name()
    bundle = build_bundle()
    predictor = load_predictor(best_model_name)

    summary_rows = []
    detail_rows = []

    for strategy_name in ["cosine_only", "hybrid_top1", "hybrid_top2"]:
        print(f"evaluate strategy: {strategy_name}")
        summary_row, rows = evaluate_strategy(bundle, predictor, strategy_name)
        summary_rows.append(summary_row)
        detail_rows.extend(rows)

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_df = summary_df.sort_values(
        by=["f1@5", "hit@5", "mrr"],
        ascending=False
    ).reset_index(drop=True)

    summary_df.to_csv(
        os.path.join(RESULT_DIR, "strategy_comparison_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    detail_df.to_csv(
        os.path.join(RESULT_DIR, "strategy_comparison_details.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    best_row = summary_df.iloc[0]
    with open(os.path.join(RESULT_DIR, "best_hybrid_strategy.txt"), "w", encoding="utf-8") as f:
        f.write(f"strategy={best_row['strategy']}\n")
        f.write(f"algorithm={best_row['algorithm']}\n")
        f.write(f"f1@5={best_row['f1@5']}\n")
        f.write(f"hit@5={best_row['hit@5']}\n")
        f.write(f"mrr={best_row['mrr']}\n")

    print("saved -> results/strategy_comparison_summary.csv")
    print("saved -> results/strategy_comparison_details.csv")
    print("saved -> results/best_hybrid_strategy.txt")


if __name__ == "__main__":
    main()