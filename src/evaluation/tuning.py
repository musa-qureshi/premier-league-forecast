"""Hyperparameter tuning, with a dedicated validation window separate from
the final backtest seasons.

Why not just pick whichever hyperparameters score best on the same 19
held-out seasons Phases 3-6 already reported results on? Because that
would be leakage one level up: instead of fitting a model's *parameters*
to data it shouldn't see, you'd be fitting its *hyperparameters* to the
exact seasons used to report "how good is this model" - the reported
number would flatter the model by construction, not measure it honestly.
The standard fix (nested validation / walk-forward with a held-out tuning
block) is what this module implements: all 33 seasons are split into three
blocks in chronological order -

    [ initial training ] [ validation block ] [ final test block ]

Hyperparameter search only ever looks at the validation block (never the
final test block). Once the best hyperparameters are chosen there, they're
frozen, and the FINAL reported numbers come from an expanding-window
backtest over the final test block only - seasons the tuning process never
saw, exactly preserving the "no lucky/leaked split" guarantee
src/evaluation/backtest.py was built for in the first place.

A second wrinkle specific to this project: Elo's hyperparameters
(k_factor, home_advantage, season_shrinkage, promoted_team_penalty) aren't
parameters of a *model* the way XGBoost's are - they're parameters used to
compute the `elo_diff` FEATURE itself (src/features/elo.py), consumed
identically by the Elo model, logistic regression, and XGBoost. Tuning
them means regenerating that feature per candidate value, not just
refitting a model against a fixed feature matrix - see tune_elo_features()
below, which is why it has a different shape than the other tuning
functions in this module (tune_poisson, tune_logistic, tune_xgboost),
which all tune purely at the model level against an already-fixed feature
matrix.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from src.evaluation.backtest import expanding_window_backtest, summarize
from src.features.elo import compute_elo_features


def season_blocks(
    features: pd.DataFrame, initial_train_seasons: int, n_validation_seasons: int
) -> tuple[list[str], list[str], list[str]]:
    """Splits the sorted season list into (initial_training, validation,
    final_test) blocks. Only `initial_train_seasons` + `n_validation_seasons`
    together define the validation cutoff - everything after that is the
    final test block, whatever remains."""
    seasons = sorted(features["Season"].unique())
    validation_end = initial_train_seasons + n_validation_seasons
    return (
        seasons[:initial_train_seasons],
        seasons[initial_train_seasons:validation_end],
        seasons[validation_end:],
    )


def _score_candidates(
    tuning_data: pd.DataFrame,
    candidates: list[dict],
    model_factory_builder: Callable[[dict], Callable],
    initial_train_seasons: int,
    metric: str,
) -> pd.DataFrame:
    """Shared scoring loop: for each candidate hyperparameter dict, runs
    an expanding-window backtest restricted to `tuning_data` (already
    sliced to the initial-training + validation blocks only) and records
    its validation-block metric."""
    rows = []
    for params in candidates:
        factory = model_factory_builder(params)
        result = expanding_window_backtest(tuning_data, factory, min_train_seasons=initial_train_seasons)
        summary = summarize(result)
        rows.append({**params, **summary})
    return pd.DataFrame(rows)


def _best_params(results: pd.DataFrame, param_names: list[str], metric: str) -> dict:
    # log_loss/brier_score/rps are all "lower is better"; accuracy is the
    # only "higher is better" metric this module might be asked to use.
    ascending = metric != "accuracy"
    best_idx = results[metric].sort_values(ascending=ascending).index[0]
    # Deliberately index each column separately (results[k].loc[best_idx])
    # rather than extracting the whole row via .iloc[0]/.loc[best_idx]:
    # pulling a single row out of a DataFrame with mixed column dtypes
    # (e.g. int n_estimators alongside float learning_rate) silently
    # upcasts every value in that row to a common dtype - usually
    # float64. That's harmless for Elo/Poisson/logistic-regression's
    # float hyperparameters, but XGBoost's n_estimators/max_depth strictly
    # require a Python int, and silently handing it numpy.float64(400.0)
    # crashes deep inside the XGBoost training loop with a genuinely
    # confusing error far from this function - caught by actually running
    # this against XGBoost's real hyperparameter grid, not a synthetic
    # all-float test case that would never have hit the bug.
    return {k: results[k].loc[best_idx].item() if hasattr(results[k].loc[best_idx], "item")
            else results[k].loc[best_idx] for k in param_names}


def tune_elo_features(
    matches: pd.DataFrame,
    candidates: list[dict],
    model_factory_builder: Callable[[dict], Callable],
    initial_train_seasons: int,
    n_validation_seasons: int,
    metric: str = "log_loss",
) -> tuple[dict, pd.DataFrame]:
    """Tunes Elo's feature-generation hyperparameters (k_factor,
    home_advantage, season_shrinkage, promoted_team_penalty). Each
    candidate regenerates elo_diff from scratch via compute_elo_features -
    unlike the other tune_* functions, which reuse one fixed feature
    matrix, this one can't, because the thing being tuned IS the feature.
    `model_factory_builder(params)` here typically ignores `params` (Elo's
    hyperparameters live in the feature, not the model) and just returns
    EloOutcomeModel - kept as a parameter for interface consistency with
    the other tune_* functions and in case a future model wants to
    consume both the Elo params and its own.
    """
    _, validation_seasons, _ = season_blocks(matches, initial_train_seasons, n_validation_seasons)
    tuning_cutoff_seasons = matches["Season"].isin(
        sorted(matches["Season"].unique())[: initial_train_seasons + n_validation_seasons]
    )
    tuning_matches = matches[tuning_cutoff_seasons]

    rows = []
    for params in candidates:
        elo_features = compute_elo_features(tuning_matches, config=params)
        factory = model_factory_builder(params)
        result = expanding_window_backtest(elo_features, factory, min_train_seasons=initial_train_seasons)
        summary = summarize(result)
        rows.append({**params, **summary})

    results = pd.DataFrame(rows)
    best = _best_params(results, list(candidates[0].keys()), metric)
    return best, results


def tune_model(
    features: pd.DataFrame,
    candidates: list[dict],
    model_factory_builder: Callable[[dict], Callable],
    initial_train_seasons: int,
    n_validation_seasons: int,
    metric: str = "log_loss",
) -> tuple[dict, pd.DataFrame]:
    """Tunes a model whose hyperparameters live entirely in the model
    (Poisson, logistic regression, XGBoost) against an already-fixed
    feature matrix - no feature regeneration needed, unlike tune_elo_features."""
    _, _, _ = season_blocks(features, initial_train_seasons, n_validation_seasons)
    initial_and_validation_seasons = sorted(features["Season"].unique())[
        : initial_train_seasons + n_validation_seasons
    ]
    tuning_data = features[features["Season"].isin(initial_and_validation_seasons)]

    results = _score_candidates(tuning_data, candidates, model_factory_builder, initial_train_seasons, metric)
    best = _best_params(results, list(candidates[0].keys()), metric)
    return best, results


def final_test_backtest(
    features: pd.DataFrame,
    model_factory,
    initial_train_seasons: int,
    n_validation_seasons: int,
) -> pd.DataFrame:
    """Runs the honest, tuning-untouched backtest: expanding window
    starting right where the validation block ends, so every season it
    scores is one the tuning process in tune_model/tune_elo_features never
    looked at."""
    validation_end = initial_train_seasons + n_validation_seasons
    return expanding_window_backtest(features, model_factory, min_train_seasons=validation_end)
