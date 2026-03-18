import os
import json
import pandas as pd

RESULT_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def load_csv(file_name: str) -> pd.DataFrame:
    path = os.path.join(RESULT_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"ไม่พบไฟล์ {path}")
    return pd.read_csv(path)


def main():
    classification_df = load_csv("classification_evaluation_summary.csv")
    strategy_df = load_csv("strategy_comparison_summary.csv")

    # best classifier
    classifier_ranked = classification_df.sort_values(
        by=["test_f1_weighted", "test_accuracy"],
        ascending=False
    ).reset_index(drop=True)
    classifier_ranked["rank"] = range(1, len(classifier_ranked) + 1)

    best_classifier = classifier_ranked.iloc[0].to_dict()

    # best strategy by F1@5
    strategy_ranked_by_f1 = strategy_df.sort_values(
        by=["f1@5", "hit@5", "mrr"],
        ascending=False
    ).reset_index(drop=True)
    strategy_ranked_by_f1["rank"] = range(1, len(strategy_ranked_by_f1) + 1)
    best_strategy_by_f1 = strategy_ranked_by_f1.iloc[0].to_dict()

    # best strategy by Hit@5
    strategy_ranked_by_hit = strategy_df.sort_values(
        by=["hit@5", "f1@5", "mrr"],
        ascending=False
    ).reset_index(drop=True)
    best_strategy_by_hit = strategy_ranked_by_hit.iloc[0].to_dict()

    # deployment recommendation แบบง่าย
    top1_row = strategy_df[strategy_df["strategy"] == "hybrid_top1"]
    top2_row = strategy_df[strategy_df["strategy"] == "hybrid_top2"]

    if not top1_row.empty and not top2_row.empty:
        top1 = top1_row.iloc[0]
        top2 = top2_row.iloc[0]

        if float(top2["hit@5"]) > float(top1["hit@5"]) and float(top2["f1@5"]) >= float(top1["f1@5"]) - 0.03:
            deployment_strategy = "hybrid_top2"
            deployment_reason = "เลือก top2 เพราะช่วยเพิ่มโอกาสพบหนังสือที่เกี่ยวข้อง และค่า F1@5 ลดลงไม่มาก"
        else:
            deployment_strategy = "hybrid_top1"
            deployment_reason = "เลือก top1 เพราะให้ค่า F1@5 ดีกว่าอย่างชัดเจน"
    else:
        deployment_strategy = best_strategy_by_f1["strategy"]
        deployment_reason = "ใช้ strategy ที่ดีที่สุดตาม F1@5"

    summary_json = {
        "best_classifier": {
            "algorithm": best_classifier["algorithm"],
            "test_accuracy": best_classifier["test_accuracy"],
            "test_f1_weighted": best_classifier["test_f1_weighted"],
        },
        "best_strategy_by_f1": {
            "strategy": best_strategy_by_f1["strategy"],
            "algorithm": best_strategy_by_f1["algorithm"],
            "f1@5": best_strategy_by_f1["f1@5"],
            "hit@5": best_strategy_by_f1["hit@5"],
            "mrr": best_strategy_by_f1["mrr"],
        },
        "best_strategy_by_hit": {
            "strategy": best_strategy_by_hit["strategy"],
            "algorithm": best_strategy_by_hit["algorithm"],
            "f1@5": best_strategy_by_hit["f1@5"],
            "hit@5": best_strategy_by_hit["hit@5"],
            "mrr": best_strategy_by_hit["mrr"],
        },
        "deployment_recommendation": {
            "strategy": deployment_strategy,
            "algorithm": best_classifier["algorithm"],
            "reason": deployment_reason,
        }
    }

    # save ranking tables
    classifier_ranked.to_csv(
        os.path.join(RESULT_DIR, "final_classifier_ranking.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    strategy_ranked_by_f1.to_csv(
        os.path.join(RESULT_DIR, "final_strategy_ranking.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # compact tables
    classifier_compact = classifier_ranked[
        [
            "rank", "algorithm",
            "test_accuracy",
            "test_precision_weighted",
            "test_recall_weighted",
            "test_f1_weighted",
        ]
    ].copy()

    strategy_compact = strategy_ranked_by_f1[
        [
            "rank", "strategy", "algorithm", "top_k_classes",
            "precision@5", "recall@5", "f1@5", "hit@5", "mrr"
        ]
    ].copy()

    classifier_compact.to_csv(
        os.path.join(RESULT_DIR, "final_classifier_compact_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )
    strategy_compact.to_csv(
        os.path.join(RESULT_DIR, "final_strategy_compact_table.csv"),
        index=False,
        encoding="utf-8-sig"
    )

    # text summary
    with open(os.path.join(RESULT_DIR, "final_experiment_summary.txt"), "w", encoding="utf-8") as f:
        f.write("=== FINAL EXPERIMENT SUMMARY ===\n\n")

        f.write("[Best Classifier]\n")
        f.write(f"algorithm={best_classifier['algorithm']}\n")
        f.write(f"test_accuracy={best_classifier['test_accuracy']}\n")
        f.write(f"test_f1_weighted={best_classifier['test_f1_weighted']}\n\n")

        f.write("[Best Strategy by F1@5]\n")
        f.write(f"strategy={best_strategy_by_f1['strategy']}\n")
        f.write(f"algorithm={best_strategy_by_f1['algorithm']}\n")
        f.write(f"f1@5={best_strategy_by_f1['f1@5']}\n")
        f.write(f"hit@5={best_strategy_by_f1['hit@5']}\n")
        f.write(f"mrr={best_strategy_by_f1['mrr']}\n\n")

        f.write("[Best Strategy by Hit@5]\n")
        f.write(f"strategy={best_strategy_by_hit['strategy']}\n")
        f.write(f"algorithm={best_strategy_by_hit['algorithm']}\n")
        f.write(f"f1@5={best_strategy_by_hit['f1@5']}\n")
        f.write(f"hit@5={best_strategy_by_hit['hit@5']}\n")
        f.write(f"mrr={best_strategy_by_hit['mrr']}\n\n")

        f.write("[Deployment Recommendation]\n")
        f.write(f"strategy={deployment_strategy}\n")
        f.write(f"algorithm={best_classifier['algorithm']}\n")
        f.write(f"reason={deployment_reason}\n")

    with open(os.path.join(RESULT_DIR, "final_experiment_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2)

    print("saved -> results/final_classifier_ranking.csv")
    print("saved -> results/final_classifier_compact_table.csv")
    print("saved -> results/final_strategy_ranking.csv")
    print("saved -> results/final_strategy_compact_table.csv")
    print("saved -> results/final_experiment_summary.txt")
    print("saved -> results/final_experiment_summary.json")


if __name__ == "__main__":
    main()