# experiments/finalize_experiment_results.py

import os
import json
import pandas as pd

from protocol_config import (
    CLASSIFICATION_PRIMARY_METRIC,
    CLASSIFICATION_SECONDARY_METRIC,
    STRATEGY_PRIMARY_METRIC,
    STRATEGY_SECONDARY_METRIC,
    STRATEGY_TERTIARY_METRIC,
    PRECISION_ORIENTED_METRIC,
    HIT_ORIENTED_METRIC,
    DEPLOYMENT_MAX_F1_DROP_FOR_TOP2,
    PREFER_TOP2_IF_HIT_IMPROVES,
    COSINE_ONLY_NAME,
    HYBRID_PREFIX,
    CLASSIFY_ONLY_NAME,
)

BASE_DIR = os.path.dirname(__file__)
RESULT_DIR = os.path.join(BASE_DIR, "..", "results")


def load_csv(file_name: str) -> pd.DataFrame:
    path = os.path.join(RESULT_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}")
    return pd.read_csv(path)


def rank_classifiers(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [CLASSIFICATION_PRIMARY_METRIC, CLASSIFICATION_SECONDARY_METRIC]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"classification summary ไม่มีคอลัมน์ {col}")

    ranked = df.sort_values(
        by=[CLASSIFICATION_PRIMARY_METRIC, CLASSIFICATION_SECONDARY_METRIC],
        ascending=[False, False]
    ).reset_index(drop=True)

    ranked["rank"] = range(1, len(ranked) + 1)
    cols = ["rank"] + [c for c in ranked.columns if c != "rank"]
    return ranked[cols]


def rank_strategies(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [STRATEGY_PRIMARY_METRIC, STRATEGY_SECONDARY_METRIC, STRATEGY_TERTIARY_METRIC]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"strategy summary ไม่มีคอลัมน์ {col}")

    ranked = df.sort_values(
        by=[STRATEGY_PRIMARY_METRIC, STRATEGY_SECONDARY_METRIC, STRATEGY_TERTIARY_METRIC],
        ascending=[False, False, False]
    ).reset_index(drop=True)

    ranked["rank"] = range(1, len(ranked) + 1)
    cols = ["rank"] + [c for c in ranked.columns if c != "rank"]
    return ranked[cols]


def build_classifier_summary_tables(classification_df: pd.DataFrame):
    ranked = rank_classifiers(classification_df)

    best_row = ranked.iloc[0].to_dict()

    compact_cols = [
        "rank",
        "algorithm",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
        "test_precision_weighted",
        "test_recall_weighted",
        "test_f1_weighted",
        "test_prediction_time_seconds",
    ]
    compact_cols = [c for c in compact_cols if c in ranked.columns]
    compact = ranked[compact_cols].copy()

    return ranked, compact, best_row


def build_strategy_summary_tables(strategy_df: pd.DataFrame):
    ranked = rank_strategies(strategy_df)

    # precision-oriented best
    precision_best = strategy_df.sort_values(
        by=[PRECISION_ORIENTED_METRIC, STRATEGY_SECONDARY_METRIC, STRATEGY_TERTIARY_METRIC],
        ascending=[False, False, False]
    ).iloc[0].to_dict()

    # hit-oriented best
    hit_best = strategy_df.sort_values(
        by=[HIT_ORIENTED_METRIC, STRATEGY_PRIMARY_METRIC, STRATEGY_TERTIARY_METRIC],
        ascending=[False, False, False]
    ).iloc[0].to_dict()

    compact_cols = [
        "rank",
        "strategy",
        "algorithm",
        "top_k_classes",
        "precision@1", "recall@1", "f1@1", "hit@1",
        "precision@3", "recall@3", "f1@3", "hit@3",
        "precision@5", "recall@5", "f1@5", "hit@5",
        "mrr",
    ]
    compact_cols = [c for c in compact_cols if c in ranked.columns]
    compact = ranked[compact_cols].copy()

    return ranked, compact, precision_best, hit_best


def extract_hybrid_rows(strategy_df: pd.DataFrame) -> pd.DataFrame:
    return strategy_df[
        strategy_df["strategy"].astype(str).str.startswith(HYBRID_PREFIX, na=False)
    ].copy()


def summarize_topk_tradeoff(strategy_df: pd.DataFrame) -> pd.DataFrame:
    hybrid_df = extract_hybrid_rows(strategy_df)
    if hybrid_df.empty:
        return pd.DataFrame()

    cols = [
        "strategy",
        "algorithm",
        "top_k_classes",
        "precision@5",
        "recall@5",
        "f1@5",
        "hit@5",
        "mrr",
    ]
    cols = [c for c in cols if c in hybrid_df.columns]
    hybrid_df = hybrid_df[cols].copy()

    hybrid_df = hybrid_df.sort_values(
        by=["algorithm", "top_k_classes", "f1@5"],
        ascending=[True, True, False]
    ).reset_index(drop=True)

    return hybrid_df


def choose_deployment_strategy(strategy_df: pd.DataFrame) -> dict:
    hybrid_df = extract_hybrid_rows(strategy_df)
    if hybrid_df.empty:
        # fallback ไปใช้ best overall
        ranked = rank_strategies(strategy_df)
        best = ranked.iloc[0].to_dict()
        return {
            "deployment_strategy": best["strategy"],
            "deployment_algorithm": best.get("algorithm", ""),
            "reason": "ไม่พบ hybrid strategy จึงใช้ strategy ที่ดีที่สุดจากผลรวม",
            "reference_metrics": {
                "f1@5": best.get("f1@5"),
                "hit@5": best.get("hit@5"),
                "mrr": best.get("mrr"),
            }
        }

    # best top1
    top1_df = hybrid_df[hybrid_df["top_k_classes"] == 1].copy()
    top2_df = hybrid_df[hybrid_df["top_k_classes"] == 2].copy()

    top1_best = None
    top2_best = None

    if not top1_df.empty:
        top1_best = top1_df.sort_values(
            by=["f1@5", "hit@5", "mrr"],
            ascending=[False, False, False]
        ).iloc[0].to_dict()

    if not top2_df.empty:
        top2_best = top2_df.sort_values(
            by=["f1@5", "hit@5", "mrr"],
            ascending=[False, False, False]
        ).iloc[0].to_dict()

    # ถ้าไม่มี top1 หรือ top2 ให้ fallback best hybrid overall
    if top1_best is None or top2_best is None:
        best_hybrid = hybrid_df.sort_values(
            by=["f1@5", "hit@5", "mrr"],
            ascending=[False, False, False]
        ).iloc[0].to_dict()
        return {
            "deployment_strategy": best_hybrid["strategy"],
            "deployment_algorithm": best_hybrid.get("algorithm", ""),
            "reason": "ไม่มีข้อมูล top1/top2 ครบ จึงเลือก hybrid ที่ดีที่สุดโดยรวม",
            "reference_metrics": {
                "f1@5": best_hybrid.get("f1@5"),
                "hit@5": best_hybrid.get("hit@5"),
                "mrr": best_hybrid.get("mrr"),
            }
        }

    f1_drop = float(top1_best["f1@5"]) - float(top2_best["f1@5"])
    hit_gain = float(top2_best["hit@5"]) - float(top1_best["hit@5"])

    if PREFER_TOP2_IF_HIT_IMPROVES and f1_drop <= DEPLOYMENT_MAX_F1_DROP_FOR_TOP2 and hit_gain > 0:
        chosen = top2_best
        reason = (
            f"เลือก top2 เพราะ f1@5 ลดลงเพียง {f1_drop:.6f} "
            f"แต่ hit@5 เพิ่มขึ้น {hit_gain:.6f} จึงเหมาะกับการใช้งานจริงมากกว่า"
        )
    else:
        chosen = top1_best
        reason = (
            f"เลือก top1 เพราะ f1@5 ดีกว่าอย่างชัดเจน "
            f"(ส่วนต่างจาก top2 = {f1_drop:.6f})"
        )

    return {
        "deployment_strategy": chosen["strategy"],
        "deployment_algorithm": chosen.get("algorithm", ""),
        "reason": reason,
        "reference_metrics": {
            "f1@5": chosen.get("f1@5"),
            "hit@5": chosen.get("hit@5"),
            "mrr": chosen.get("mrr"),
        }
    }


def compare_baselines(strategy_df: pd.DataFrame) -> pd.DataFrame:
    compare_rows = []

    cosine_df = strategy_df[strategy_df["strategy"] == COSINE_ONLY_NAME].copy()
    classify_only_df = strategy_df[strategy_df["strategy"] == CLASSIFY_ONLY_NAME].copy()
    hybrid_df = extract_hybrid_rows(strategy_df).copy()

    if not cosine_df.empty:
        best_cosine = cosine_df.sort_values(
            by=["f1@5", "hit@5", "mrr"],
            ascending=[False, False, False]
        ).iloc[0]
        compare_rows.append({
            "group": "baseline",
            "label": "best_cosine_only",
            "strategy": best_cosine["strategy"],
            "algorithm": best_cosine.get("algorithm", ""),
            "f1@5": best_cosine.get("f1@5"),
            "hit@5": best_cosine.get("hit@5"),
            "mrr": best_cosine.get("mrr"),
        })

    if not classify_only_df.empty:
        best_classify_only = classify_only_df.sort_values(
            by=["f1@5", "hit@5", "mrr"],
            ascending=[False, False, False]
        ).iloc[0]
        compare_rows.append({
            "group": "baseline",
            "label": "best_classify_only",
            "strategy": best_classify_only["strategy"],
            "algorithm": best_classify_only.get("algorithm", ""),
            "f1@5": best_classify_only.get("f1@5"),
            "hit@5": best_classify_only.get("hit@5"),
            "mrr": best_classify_only.get("mrr"),
        })

    if not hybrid_df.empty:
        best_hybrid = hybrid_df.sort_values(
            by=["f1@5", "hit@5", "mrr"],
            ascending=[False, False, False]
        ).iloc[0]
        compare_rows.append({
            "group": "baseline",
            "label": "best_hybrid",
            "strategy": best_hybrid["strategy"],
            "algorithm": best_hybrid.get("algorithm", ""),
            "f1@5": best_hybrid.get("f1@5"),
            "hit@5": best_hybrid.get("hit@5"),
            "mrr": best_hybrid.get("mrr"),
        })

    return pd.DataFrame(compare_rows)


def save_text_summary(best_classifier: dict, precision_best: dict, hit_best: dict, deployment_choice: dict):
    path = os.path.join(RESULT_DIR, "final_experiment_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=== FINAL EXPERIMENT SUMMARY ===\n\n")

        f.write("[Best Classifier]\n")
        f.write(f"algorithm={best_classifier.get('algorithm')}\n")
        f.write(f"{CLASSIFICATION_PRIMARY_METRIC}={best_classifier.get(CLASSIFICATION_PRIMARY_METRIC)}\n")
        f.write(f"{CLASSIFICATION_SECONDARY_METRIC}={best_classifier.get(CLASSIFICATION_SECONDARY_METRIC)}\n\n")

        f.write("[Best Precision-Oriented Strategy]\n")
        f.write(f"strategy={precision_best.get('strategy')}\n")
        f.write(f"algorithm={precision_best.get('algorithm')}\n")
        f.write(f"f1@5={precision_best.get('f1@5')}\n")
        f.write(f"hit@5={precision_best.get('hit@5')}\n")
        f.write(f"mrr={precision_best.get('mrr')}\n\n")

        f.write("[Best Hit-Oriented Strategy]\n")
        f.write(f"strategy={hit_best.get('strategy')}\n")
        f.write(f"algorithm={hit_best.get('algorithm')}\n")
        f.write(f"f1@5={hit_best.get('f1@5')}\n")
        f.write(f"hit@5={hit_best.get('hit@5')}\n")
        f.write(f"mrr={hit_best.get('mrr')}\n\n")

        f.write("[Deployment Recommendation]\n")
        f.write(f"deployment_strategy={deployment_choice.get('deployment_strategy')}\n")
        f.write(f"deployment_algorithm={deployment_choice.get('deployment_algorithm')}\n")
        f.write(f"reason={deployment_choice.get('reason')}\n")
        ref = deployment_choice.get("reference_metrics", {})
        f.write(f"f1@5={ref.get('f1@5')}\n")
        f.write(f"hit@5={ref.get('hit@5')}\n")
        f.write(f"mrr={ref.get('mrr')}\n")


def main():
    classification_df = load_csv("classification_evaluation_summary.csv")
    strategy_df = load_csv("strategy_comparison_summary.csv")

    # classifier summaries
    classifier_ranked, classifier_compact, best_classifier = build_classifier_summary_tables(classification_df)

    # strategy summaries
    strategy_ranked, strategy_compact, precision_best, hit_best = build_strategy_summary_tables(strategy_df)
    topk_tradeoff_df = summarize_topk_tradeoff(strategy_df)
    baseline_compare_df = compare_baselines(strategy_df)
    deployment_choice = choose_deployment_strategy(strategy_df)

    # save csv
    classifier_ranked.to_csv(
        os.path.join(RESULT_DIR, "final_classifier_ranking.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    classifier_compact.to_csv(
        os.path.join(RESULT_DIR, "final_classifier_compact_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    strategy_ranked.to_csv(
        os.path.join(RESULT_DIR, "final_strategy_ranking.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    strategy_compact.to_csv(
        os.path.join(RESULT_DIR, "final_strategy_compact_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    topk_tradeoff_df.to_csv(
        os.path.join(RESULT_DIR, "hybrid_topk_tradeoff_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    baseline_compare_df.to_csv(
        os.path.join(RESULT_DIR, "baseline_vs_hybrid_summary.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # save json
    summary_json = {
        "best_classifier": best_classifier,
        "best_precision_oriented_strategy": precision_best,
        "best_hit_oriented_strategy": hit_best,
        "deployment_recommendation": deployment_choice,
    }

    with open(os.path.join(RESULT_DIR, "final_experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    save_text_summary(best_classifier, precision_best, hit_best, deployment_choice)

    print("saved -> results/final_classifier_ranking.csv")
    print("saved -> results/final_classifier_compact_table.csv")
    print("saved -> results/final_strategy_ranking.csv")
    print("saved -> results/final_strategy_compact_table.csv")
    print("saved -> results/hybrid_topk_tradeoff_summary.csv")
    print("saved -> results/baseline_vs_hybrid_summary.csv")
    print("saved -> results/final_experiment_summary.json")
    print("saved -> results/final_experiment_summary.txt")


if __name__ == "__main__":
    main()