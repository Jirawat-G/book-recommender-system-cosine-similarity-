import os
import sys
import subprocess
from pathlib import Path


def run_step(step_name: str, command: list[str], cwd: Path) -> None:
    print(f"\n==> {step_name}")
    print(" ".join(command))
    result = subprocess.run(command, cwd=str(cwd))
    if result.returncode != 0:
        raise SystemExit(f"ERROR: step failed -> {step_name}")


def remove_file(path: Path) -> None:
    if path.exists():
        path.unlink()
        print(f"removed: {path.name}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    results_dir = project_root / "results"

    print("==> Start experiments pipeline")
    print(f"script_dir   : {script_dir}")
    print(f"project_root : {project_root}")
    print(f"results_dir  : {results_dir}")

    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)

    print("\n==> Python")
    print(sys.executable)
    print(sys.version)

    print("\n==> Clear old result summaries")
    files_to_remove = [
        "classification_evaluation_summary.csv",
        "classification_prediction_details.csv",
        "strategy_comparison_summary.csv",
        "strategy_comparison_details.csv",
        "final_classifier_ranking.csv",
        "final_classifier_compact_table.csv",
        "final_strategy_ranking.csv",
        "final_strategy_compact_table.csv",
        "final_experiment_summary.txt",
        "final_experiment_summary.json",
        "best_classifier_from_evaluation.txt",
        "best_hybrid_strategy.txt",
    ]

    for file_name in files_to_remove:
        remove_file(results_dir / file_name)

    # ถ้าคุณอยากล้างไฟล์อื่นเพิ่ม ค่อยเติมชื่อใน list นี้
    print("\n==> Run experiments")
    run_step("Train classical models", [sys.executable, "train_classical_models.py"], script_dir)
    run_step("Train ANN model", [sys.executable, "train_ann_model.py"], script_dir)
    run_step("Evaluate classification", [sys.executable, "evaluate_classification.py"], script_dir)
    run_step("Compare strategies", [sys.executable, "compare_strategies.py"], script_dir)
    run_step("Finalize experiment results", [sys.executable, "finalize_experiment_results.py"], script_dir)

    print("\n==> Validate outputs")
    expected_files = [
        "classification_evaluation_summary.csv",
        "strategy_comparison_summary.csv",
        "best_classifier_from_evaluation.txt",
        "final_experiment_summary.txt",
    ]

    missing_files = []
    for file_name in expected_files:
        path = results_dir / file_name
        if path.exists():
            print(f"OK   : {file_name}")
        else:
            print(f"MISS : {file_name}")
            missing_files.append(file_name)

    if missing_files:
        raise SystemExit(f"ERROR: missing expected output files -> {missing_files}")

    print("\n==> best_classifier_from_evaluation.txt")
    print((results_dir / "best_classifier_from_evaluation.txt").read_text(encoding="utf-8"))

    print("\n==> final_experiment_summary.txt")
    print((results_dir / "final_experiment_summary.txt").read_text(encoding="utf-8"))

    print("\n==> Done")


if __name__ == "__main__":
    main()