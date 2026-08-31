"""Runs the full model comparison (Phases 3-6: baselines, Elo, Poisson,
logistic regression, XGBoost) through the expanding-window backtest and
writes experiments/results.csv / results_summary.csv - supersedes
run_phase5_logistic.py's output with the same files, now including
Model 4 (XGBoost).
"""

from functools import partial
from pathlib import Path

import pandas as pd

from src.evaluation.backtest import expanding_window_backtest, summarize
from src.models.baseline import FrequencyBaseline
from src.models.elo_model import EloOutcomeModel
from src.models.logistic_model import LogisticOutcomeModel
from src.models.poisson_model import DixonColesModel
from src.models.xgboost_model import XGBoostOutcomeModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_TRAIN_SEASONS = 10


def main() -> None:
    features = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")

    models = {
        "Baseline (frequency)": FrequencyBaseline,
        "Elo": EloOutcomeModel,
        "Poisson (independent)": partial(DixonColesModel, use_correlation=False, xi=0.3),
        "Poisson (Dixon-Coles)": partial(DixonColesModel, use_correlation=True, xi=0.3),
        "Logistic Regression": LogisticOutcomeModel,
        "XGBoost": XGBoostOutcomeModel,
    }

    all_rows = []
    summaries = {}
    for name, factory in models.items():
        print(f"\n=== {name} ===")
        result = expanding_window_backtest(features, factory, min_train_seasons=MIN_TRAIN_SEASONS)
        result.insert(0, "model", name)
        all_rows.append(result)

        summary = summarize(result)
        summaries[name] = summary
        print(f"weighted overall: log_loss={summary['log_loss']:.4f}  "
              f"brier={summary['brier_score']:.4f}  rps={summary['rps']:.4f}  "
              f"accuracy={summary['accuracy']:.4f}  ({summary['n_matches']} matches, "
              f"{summary['n_seasons']} seasons)")

    detailed = pd.concat(all_rows, ignore_index=True)
    out_path = PROJECT_ROOT / "experiments" / "results.csv"
    detailed.to_csv(out_path, index=False)
    print(f"\n[experiments] wrote per-season results -> {out_path}")

    summary_df = pd.DataFrame(summaries).T
    summary_df.index.name = "model"
    summary_out = PROJECT_ROOT / "experiments" / "results_summary.csv"
    summary_df.to_csv(summary_out)
    print(f"[experiments] wrote summary -> {summary_out}\n")
    print(summary_df.to_string())

    print("\n=== Logistic Regression standardized coefficients (fit on full dataset) ===")
    full_logreg = LogisticOutcomeModel().fit(features)
    print(full_logreg.coefficients().T.to_string())

    print("\n=== XGBoost feature importances, top 15 (fit on full dataset) ===")
    full_xgb = XGBoostOutcomeModel().fit(features)
    print(full_xgb.feature_importances().head(15).to_string())


if __name__ == "__main__":
    main()
