"""Phase 10: hyperparameter tuning, with a dedicated validation block kept
separate from the final reported test seasons (see src/evaluation/tuning.py
for the full methodology and why it matters).

Season split (33 total seasons, 1993-94..2025-26):
    initial training : first 10 seasons  (1993-94..2002-03)
    validation block : next 5 seasons    (2003-04..2007-08)  <- tuning only
    final test block : remaining seasons (2008-09..2025-26)  <- reported results

Elo is tuned first, and its winning hyperparameters are used to regenerate
elo_diff across the FULL dataset before tuning logistic regression/XGBoost,
since both consume that feature - see module docstring in
src/evaluation/tuning.py for why Elo's tuning has this different shape.
Poisson tunes purely at the model level and needs no feature regeneration.

For every model, the SAME final test block is used to compare "tuned" vs
"default (as shipped in Phases 3-6)" hyperparameters, so the improvement
number is a fair like-for-like comparison, not a comparison against the
README's original numbers (which used a different season split entirely -
see "Model comparison" in README.md for that caveat).
"""

import itertools
import random
from functools import partial
from pathlib import Path

import pandas as pd
import yaml

from src.evaluation.tuning import final_test_backtest, tune_elo_features, tune_model
from src.features.build_dataset import build_features
from src.models.elo_model import EloOutcomeModel
from src.models.logistic_model import LogisticOutcomeModel
from src.models.poisson_model import DixonColesModel
from src.models.xgboost_model import XGBoostOutcomeModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INITIAL_TRAIN_SEASONS = 10
N_VALIDATION_SEASONS = 5

DEFAULT_ELO_CONFIG = {
    "initial_rating": 1500.0, "k_factor": 20.0, "home_advantage": 100.0,
    "use_goal_difference_multiplier": True, "promoted_team_penalty": 150.0,
    "season_shrinkage": 0.33,
}


def tune_elo(matches: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    candidates = [
        {**DEFAULT_ELO_CONFIG, "k_factor": k, "home_advantage": h,
         "season_shrinkage": s, "promoted_team_penalty": p}
        for k in (10.0, 20.0, 30.0)
        for h in (50.0, 100.0, 150.0)
        for s in (0.0, 0.33, 0.5)
        for p in (100.0, 150.0, 200.0)
    ]
    return tune_elo_features(
        matches, candidates, model_factory_builder=lambda params: EloOutcomeModel,
        initial_train_seasons=INITIAL_TRAIN_SEASONS, n_validation_seasons=N_VALIDATION_SEASONS,
    )


def tune_poisson(matches: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    candidates = [
        {"xi": xi, "promoted_penalty": pp, "use_correlation": uc}
        for xi in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0)
        for pp in (0.2, 0.4, 0.6)
        for uc in (True, False)
    ]
    return tune_model(
        matches, candidates, model_factory_builder=lambda params: partial(DixonColesModel, **params),
        initial_train_seasons=INITIAL_TRAIN_SEASONS, n_validation_seasons=N_VALIDATION_SEASONS,
    )


def tune_logistic(features: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    candidates = [{"C": c} for c in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)]
    return tune_model(
        features, candidates, model_factory_builder=lambda params: partial(LogisticOutcomeModel, **params),
        initial_train_seasons=INITIAL_TRAIN_SEASONS, n_validation_seasons=N_VALIDATION_SEASONS,
    )


def tune_xgboost(features: pd.DataFrame, n_random_candidates: int = 25) -> tuple[dict, pd.DataFrame]:
    rng = random.Random(0)
    grid = {
        "n_estimators": [100, 200, 300, 400],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    }
    all_combos = list(itertools.product(*grid.values()))
    sampled = rng.sample(all_combos, k=min(n_random_candidates, len(all_combos)))
    candidates = [dict(zip(grid.keys(), combo)) for combo in sampled]
    return tune_model(
        features, candidates, model_factory_builder=lambda params: partial(XGBoostOutcomeModel, **params),
        initial_train_seasons=INITIAL_TRAIN_SEASONS, n_validation_seasons=N_VALIDATION_SEASONS,
    )


def compare_on_final_test(name, tuned_data, tuned_factory, default_data, default_factory) -> None:
    """`tuned_data`/`default_data` are allowed to differ (needed for Elo,
    where what's tuned is the elo_diff FEATURE, not the model - the
    "tuned" and "default" runs must each see the feature set built with
    their own Elo config, not one shared feature set)."""
    from src.evaluation.backtest import summarize
    tuned = summarize(final_test_backtest(tuned_data, tuned_factory, INITIAL_TRAIN_SEASONS, N_VALIDATION_SEASONS))
    default = summarize(final_test_backtest(default_data, default_factory, INITIAL_TRAIN_SEASONS, N_VALIDATION_SEASONS))
    print(f"\n{name}: final test block ({tuned['n_seasons']:.0f} seasons, {tuned['n_matches']} matches)")
    print(f"  default : log_loss={default['log_loss']:.4f}  brier={default['brier_score']:.4f}  "
          f"rps={default['rps']:.4f}  accuracy={default['accuracy']:.4f}")
    print(f"  tuned   : log_loss={tuned['log_loss']:.4f}  brier={tuned['brier_score']:.4f}  "
          f"rps={tuned['rps']:.4f}  accuracy={tuned['accuracy']:.4f}")


def main() -> None:
    matches = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "matches.parquet")

    print("=== Tuning Elo (feature-generation hyperparameters) ===")
    best_elo, elo_results = tune_elo(matches)
    print(f"best: {best_elo}")
    print(elo_results.sort_values("log_loss").head(5).to_string(index=False))

    print("\n=== Tuning Poisson/Dixon-Coles ===")
    best_poisson, poisson_results = tune_poisson(matches)
    print(f"best: {best_poisson}")
    print(poisson_results.sort_values("log_loss").head(5).to_string(index=False))

    # Regenerate features with the TUNED Elo config, so logistic
    # regression and XGBoost tune against the improved elo_diff feature.
    print("\n=== Regenerating features.parquet with tuned Elo config ===")
    features = build_features(matches, elo_config=best_elo)
    default_elo_features = build_features(matches, elo_config=DEFAULT_ELO_CONFIG)

    print("\n=== Tuning Logistic Regression ===")
    best_logistic, logistic_results = tune_logistic(features)
    print(f"best: {best_logistic}")
    print(logistic_results.sort_values("log_loss").to_string(index=False))

    print("\n=== Tuning XGBoost (random search, 25 candidates) ===")
    best_xgb, xgb_results = tune_xgboost(features)
    print(f"best: {best_xgb}")
    print(xgb_results.sort_values("log_loss").head(5).to_string(index=False))

    print("\n\n########## Final test block: tuned vs default ##########")
    # Elo the MODEL has no hyperparameters of its own - what's being
    # compared here is the tuned-Elo-config feature set vs the
    # default-Elo-config feature set, both run through the same model.
    compare_on_final_test("Elo", features, EloOutcomeModel, default_elo_features, EloOutcomeModel)
    compare_on_final_test(
        "Poisson", matches, partial(DixonColesModel, **best_poisson),
        matches, partial(DixonColesModel, use_correlation=True, xi=0.3),
    )
    compare_on_final_test(
        "Logistic Regression", features, partial(LogisticOutcomeModel, **best_logistic),
        features, LogisticOutcomeModel,
    )
    compare_on_final_test(
        "XGBoost", features, partial(XGBoostOutcomeModel, **best_xgb),
        features, XGBoostOutcomeModel,
    )

    # Persist the tuned Elo config as the project's new default - every
    # downstream module (Phase 2 feature pipeline, Phase 9 live forecast)
    # reads configs/elo.yaml, so this is what makes the tuning "stick".
    elo_yaml_path = PROJECT_ROOT / "configs" / "elo.yaml"
    print(f"\n[tuning] NOTE: best Elo config found: {best_elo}")
    print(f"[tuning] NOTE: best Poisson config found: {best_poisson}")
    print(f"[tuning] NOTE: best Logistic C found: {best_logistic}")
    print(f"[tuning] NOTE: best XGBoost config found: {best_xgb}")
    print(f"[tuning] (configs/elo.yaml not overwritten automatically - "
          f"review results above before updating it by hand)")


if __name__ == "__main__":
    main()
