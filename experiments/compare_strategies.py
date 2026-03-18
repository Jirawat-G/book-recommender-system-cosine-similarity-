import os
import math
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity

from feature_pipeline import build_feature_pipeline, ARTIFACT_DIR, RESULT_DIR, ensure_dirs

HAS_TF = True
try:
    from tensorflow.keras.models import load_model
except Exception:
    HAS_TF = False


RANDOM_STATE = 42
TOP_N = 5
MIN_CANDIDATES = 5
TOP_K_LIST = [1, 2, 3]
INCLUDE_CLASSIFY_ONLY = True


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
    if k == 0:
        return 0.0, 0.0, 0.0

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


def cosine_only_recommendation(bundle, query_index, top_n=TOP_N):
    query_vec = bundle["X_test_sel"][0:0]  # placeholder, not used
    query_vec = bundle["selector"].transform(
        bundle["tfidf"].transform([bundle["df"].iloc[query_index]["clean_text"]])
    )
    # ใช้ representation เดียวกับ training
    all_doc_vec = bundle["selector"].transform(
        bundle["tfidf"].transform(bundle["df"]["clean_text"])
    )

    scores = cosine_similarity(query_vec, all_doc_vec)[0]

    candidates = []
    for idx, score in enumerate(scores):
        if idx == query_index:
            continue
        candidates.append((idx, float(score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


def cosine_only_recommendation_fast(bundle, query_index, top_n=TOP_N):
    query_vec = bundle["X_all_sel"][query_index]
    scores = cosine_similarity(query_vec, bundle["X_all_sel"])[0]

    candidates = []
    for idx, score in enumerate(scores):
        if idx == query_index:
            continue
        candidates.append((idx, float(score)))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_n]


def get_top_k_predicted_classes_sklearn(model, query_vec, k):
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


def get_top_k_predicted_classes_ann(model, label_encoder, query_vec_dense, k):
    probs = model.predict(query_vec_dense, verbose=0)[0]
    ranked_idx = np.argsort(probs)[::-1][:k]
    classes = label_encoder.inverse_transform(ranked_idx)
    return [(int(cls), float(probs[idx])) for cls, idx in zip(classes, ranked_idx)]


def hybrid_recommendation(bundle, query_index, predictor_type, predictor, top_k_classes, top_n=TOP_N):
    query_vec = bundle["X_all_sel"][query_index]

    if predictor_type == "sklearn":
        predicted_classes = get_top_k_predicted_classes_sklearn(predictor, query_vec, top_k_classes)
    elif predictor_type == "ann":
        predicted_classes = get_top_k_predicted_classes_ann(
            predictor["model"],
            predictor["label_encoder"],
            query_vec.toarray(),
            top_k_classes
        )
    else:
        raise ValueError("unsupported predictor_type")

    class_ids = [cls for cls, _ in predicted_classes]

    df = bundle["df"].copy()
    df["row_index"] = np.arange(len(df))

    candidate_df = df[
        (df["course_id"].isin(class_ids)) &
        (df["row_index"] != query_index)
    ]

    # fallback ถ้า candidate น้อยเกินไป
    if len(candidate_df) < MIN_CANDIDATES:
        candidate_df = df[df["row_index"] != query_index]

    candidate_indices = candidate_df["row_index"].tolist()
    candidate_matrix = bundle["X_all_sel"][candidate_indices]
    scores = cosine_similarity(query_vec, candidate_matrix)[0]

    pairs = list(zip(candidate_indices, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)

    return predicted_classes, pairs[:top_n]


def classify_only_recommendation(bundle, query_index, predictor_type, predictor, top_n=TOP_N):
    query_vec = bundle["X_all_sel"][query_index]

    if predictor_type == "sklearn":
        predicted_classes = get_top_k_predicted_classes_sklearn(predictor, query_vec, 1)
    elif predictor_type == "ann":
        predicted_classes = get_top_k_predicted_classes_ann(
            predictor["model"],
            predictor["label_encoder"],
            query_vec.toarray(),
            1
        )
    else:
        raise ValueError("unsupported predictor_type")

    predicted_class = predicted_classes[0][0]

    df = bundle["df"].copy()
    df["row_index"] = np.arange(len(df))

    candidate_df = df[
        (df["course_id"] == predicted_class) &
        (df["row_index"] != query_index)
    ]

    result_indices = candidate_df["row_index"].tolist()[:top_n]
    pairs = [(idx, 1.0) for idx in result_indices]

    return predicted_classes, pairs


def load_available_predictors():
    predictors = []

    sklearn_model_files = {
        "decision_tree": os.path.join(ARTIFACT_DIR, "classifier_decision_tree.joblib"),
        "random_forest": os.path.join(ARTIFACT_DIR, "classifier_random_forest.joblib"),
        "knn": os.path.join(ARTIFACT_DIR, "classifier_knn.joblib"),
        "svm": os.path.join(ARTIFACT_DIR, "classifier_svm.joblib"),
        "logistic_regression": os.path.join(ARTIFACT_DIR, "classifier_logistic_regression.joblib"),
    }

    for model_name, model_path in sklearn_model_files.items():
        if os.path.exists(model_path):
            predictors.append({
                "algorithm": model_name,
                "type": "sklearn",
                "model": joblib.load(model_path),
            })

    if HAS_TF:
        ann_model_path = os.path.join(ARTIFACT_DIR, "best_model.h5")
        ann_label_encoder_path = os.path.join(ARTIFACT_DIR, "ann_label_encoder.pkl")
        if os.path.exists(ann_model_path) and os.path.exists(ann_label_encoder_path):
            predictors.append({
                "algorithm": "ann",
                "type": "ann",
                "model": {
                    "model": load_model(ann_model_path),
                    "label_encoder": joblib.load(ann_label_encoder_path),
                },
            })

    return predictors


def build_bundle_for_strategy():
    bundle = build_feature_pipeline(test_size=0.2, random_state=RANDOM_STATE)
    bundle["X_all_sel"] = bundle["selector"].transform(
        bundle["tfidf"].transform(bundle["df"]["clean_text"])
    )
    return bundle


def main():
    ensure_dirs()
    bundle = build_bundle_for_strategy()
    predictors = load_available_predictors()

    if not predictors:
        raise ValueError("ไม่พบ model artifacts กรุณารัน train_classical_models.py และ/หรือ train_ann_model.py ก่อน")

    df = bundle["df"]
    test_indices = bundle["idx_test"]

    summary_rows = []
    detail_rows = []

    # baseline: cosine only
    cosine_metric_sum = {
        "precision@1": 0.0, "recall@1": 0.0, "f1@1": 0.0, "hit@1": 0.0,
        "precision@3": 0.0, "recall@3": 0.0, "f1@3": 0.0, "hit@3": 0.0,
        "precision@5": 0.0, "recall@5": 0.0, "f1@5": 0.0, "hit@5": 0.0,
        "mrr": 0.0,
    }

    for query_index in test_indices:
        relevant_indices = get_relevant_row_indices(df, query_index)
        pairs = cosine_only_recommendation_fast(bundle, query_index, top_n=TOP_N)
        result_indices = [idx for idx, _ in pairs]
        metrics = calc_recommendation_metrics(result_indices, relevant_indices)

        for k in cosine_metric_sum:
            cosine_metric_sum[k] += metrics[k]

        detail_rows.append({
            "strategy": "cosine_only",
            "algorithm": "",
            "query_row_index": int(query_index),
            "query_book_id": int(df.iloc[query_index]["id"]),
            "query_title": str(df.iloc[query_index]["title"]),
            "query_course_id": int(df.iloc[query_index]["course_id"]),
            "predicted_classes": "",
            "top_result_row_index": int(result_indices[0]) if result_indices else None,
            "top_result_book_id": int(df.iloc[result_indices[0]]["id"]) if result_indices else None,
            "top_result_course_id": int(df.iloc[result_indices[0]]["course_id"]) if result_indices else None,
            **metrics
        })

    total_queries = len(test_indices)
    summary_rows.append({
        "strategy": "cosine_only",
        "algorithm": "",
        "top_k_classes": 0,
        **{k: v / total_queries for k, v in cosine_metric_sum.items()}
    })

    # classifier-based strategies
    for predictor in predictors:
        algorithm = predictor["algorithm"]
        predictor_type = predictor["type"]
        predictor_model = predictor["model"]

        # hybrid top-k
        for top_k in TOP_K_LIST:
            metric_sum = {
                "precision@1": 0.0, "recall@1": 0.0, "f1@1": 0.0, "hit@1": 0.0,
                "precision@3": 0.0, "recall@3": 0.0, "f1@3": 0.0, "hit@3": 0.0,
                "precision@5": 0.0, "recall@5": 0.0, "f1@5": 0.0, "hit@5": 0.0,
                "mrr": 0.0,
            }

            for query_index in test_indices:
                relevant_indices = get_relevant_row_indices(df, query_index)

                predicted_classes, pairs = hybrid_recommendation(
                    bundle=bundle,
                    query_index=query_index,
                    predictor_type=predictor_type,
                    predictor=predictor_model,
                    top_k_classes=top_k,
                    top_n=TOP_N
                )

                result_indices = [idx for idx, _ in pairs]
                metrics = calc_recommendation_metrics(result_indices, relevant_indices)

                for k in metric_sum:
                    metric_sum[k] += metrics[k]

                detail_rows.append({
                    "strategy": f"hybrid_top{top_k}",
                    "algorithm": algorithm,
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

            summary_rows.append({
                "strategy": f"hybrid_top{top_k}",
                "algorithm": algorithm,
                "top_k_classes": top_k,
                **{k: v / total_queries for k, v in metric_sum.items()}
            })

        # classify only top1
        if INCLUDE_CLASSIFY_ONLY:
            metric_sum = {
                "precision@1": 0.0, "recall@1": 0.0, "f1@1": 0.0, "hit@1": 0.0,
                "precision@3": 0.0, "recall@3": 0.0, "f1@3": 0.0, "hit@3": 0.0,
                "precision@5": 0.0, "recall@5": 0.0, "f1@5": 0.0, "hit@5": 0.0,
                "mrr": 0.0,
            }

            for query_index in test_indices:
                relevant_indices = get_relevant_row_indices(df, query_index)

                predicted_classes, pairs = classify_only_recommendation(
                    bundle=bundle,
                    query_index=query_index,
                    predictor_type=predictor_type,
                    predictor=predictor_model,
                    top_n=TOP_N
                )

                result_indices = [idx for idx, _ in pairs]
                metrics = calc_recommendation_metrics(result_indices, relevant_indices)

                for k in metric_sum:
                    metric_sum[k] += metrics[k]

                detail_rows.append({
                    "strategy": "classify_only_top1",
                    "algorithm": algorithm,
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

            summary_rows.append({
                "strategy": "classify_only_top1",
                "algorithm": algorithm,
                "top_k_classes": 1,
                **{k: v / total_queries for k, v in metric_sum.items()}
            })

    summary_df = pd.DataFrame(summary_rows)
    detail_df = pd.DataFrame(detail_rows)

    summary_df = summary_df.sort_values(
        by=["strategy", "f1@5", "hit@5", "mrr"],
        ascending=[True, False, False, False]
    )

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

    # best hybrid summary
    hybrid_df = summary_df[summary_df["strategy"].str.startswith("hybrid_top", na=False)].copy()
    if not hybrid_df.empty:
        hybrid_best = hybrid_df.sort_values(by=["f1@5", "hit@5", "mrr"], ascending=False).iloc[0]
        with open(os.path.join(RESULT_DIR, "best_hybrid_strategy.txt"), "w", encoding="utf-8") as f:
            f.write(f"strategy={hybrid_best['strategy']}\n")
            f.write(f"algorithm={hybrid_best['algorithm']}\n")
            f.write(f"f1@5={hybrid_best['f1@5']}\n")
            f.write(f"hit@5={hybrid_best['hit@5']}\n")
            f.write(f"mrr={hybrid_best['mrr']}\n")

    print("saved -> results/strategy_comparison_summary.csv")
    print("saved -> results/strategy_comparison_details.csv")


if __name__ == "__main__":
    main()