"""Phase 7: calibration analysis. For each model, checks whether its
predicted "home win" probabilities are honest - not just whether they're
accurate on average (already checked via log loss/Brier/RPS in Phases 3-6)
- by comparing predicted probability against actual observed frequency in
bins. Produces a reliability diagram (experiments/calibration_home_win.png)
and an Expected Calibration Error per model.

Home win is used as the reference outcome: it's the largest class (avoids
the extra noise of computing a curve from the smaller draw/away classes'
sparser bins) and the one the project's own match-prediction examples
center on ("Arsenal win: 47%").
"""

from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.backtest import expanding_window_predictions
from src.evaluation.calibration import calibration_curve, expected_calibration_error
from src.models.elo_model import EloOutcomeModel
from src.models.logistic_model import LogisticOutcomeModel
from src.models.poisson_model import DixonColesModel
from src.models.xgboost_model import XGBoostOutcomeModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_TRAIN_SEASONS = 10

MODELS = {
    "Elo": EloOutcomeModel,
    "Poisson (Dixon-Coles)": partial(DixonColesModel, use_correlation=True, xi=0.3),
    "Logistic Regression": LogisticOutcomeModel,
    "XGBoost": XGBoostOutcomeModel,
}


def main() -> None:
    features = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "features.parquet")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    ece_rows = []
    for name, factory in MODELS.items():
        print(f"=== {name} ===")
        preds = expanding_window_predictions(features, factory, min_train_seasons=MIN_TRAIN_SEASONS)
        y_home = (preds["FTR"] == "H").astype(int)
        p_home = preds["H"]

        curve = calibration_curve(y_home, p_home, n_bins=10)
        ece = expected_calibration_error(y_home, p_home, n_bins=10)
        ece_rows.append({"model": name, "ece_home_win": ece})
        print(curve.to_string(index=False))
        print(f"ECE (home win): {ece:.4f}\n")

        ax.plot(curve["mean_predicted"], curve["observed_frequency"], marker="o", label=name)

    ax.set_xlabel("Predicted probability of home win")
    ax.set_ylabel("Observed frequency of home win")
    ax.set_title("Reliability diagram: home win (19 held-out backtest seasons)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()

    out_path = PROJECT_ROOT / "experiments" / "calibration_home_win.png"
    fig.savefig(out_path, dpi=150)
    print(f"[calibration] wrote reliability diagram -> {out_path}")

    ece_df = pd.DataFrame(ece_rows).set_index("model").sort_values("ece_home_win")
    ece_out = PROJECT_ROOT / "experiments" / "calibration_ece.csv"
    ece_df.to_csv(ece_out)
    print(f"[calibration] wrote ECE summary -> {ece_out}\n")
    print(ece_df.to_string())


if __name__ == "__main__":
    main()
