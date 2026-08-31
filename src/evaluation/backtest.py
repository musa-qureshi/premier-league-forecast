"""Expanding-window, season-by-season backtesting harness.

Why not a single train/test split? A single split (e.g. train on
1993-2017, test on 2018) tells you how a model did in exactly one season,
which could be a lucky or unlucky year regardless of the model's real
skill. This project instead retrains before every held-out season,
expanding the training window by one season each time, and reports metrics
season by season plus an aggregate - the expanding-window backtesting
scheme standard in the football-forecasting research literature (e.g.
Constantinou & Fenton), and a direct answer to this project's requirement
to avoid ever reporting a single lucky split.

This harness is also the top-level guarantee that Phase 2's leakage-safety
work is actually being used correctly: every fold's model only ever sees
strictly earlier seasons at fit() time, and predict_proba() is only ever
called on the held-out season's own pre-match features (already themselves
leakage-safe by construction - see src/features/).
"""

from __future__ import annotations

from typing import Protocol

import pandas as pd

from src.evaluation.metrics import evaluate_all


class OutcomeModel(Protocol):
    def fit(self, train: pd.DataFrame) -> "OutcomeModel": ...
    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame: ...


def _expanding_folds(features: pd.DataFrame, model_factory, min_train_seasons: int):
    """Shared fold-generation logic behind both expanding_window_backtest
    (aggregated per-season metrics) and expanding_window_predictions (raw
    per-match predictions, needed for calibration analysis) - factored out
    once so the two can never silently diverge in how a fold is defined."""
    seasons = sorted(features["Season"].unique())
    if len(seasons) <= min_train_seasons:
        raise ValueError(
            f"Not enough seasons ({len(seasons)}) for "
            f"min_train_seasons={min_train_seasons}."
        )

    for i in range(min_train_seasons, len(seasons)):
        test_season = seasons[i]
        train_seasons = seasons[:i]

        train = features[features["Season"].isin(train_seasons)]
        test = features[features["Season"] == test_season]

        model = model_factory()
        model.fit(train)
        proba = model.predict_proba(test)

        yield test_season, train_seasons, test, proba


def expanding_window_backtest(
    features: pd.DataFrame,
    model_factory,
    min_train_seasons: int = 10,
) -> pd.DataFrame:
    """For each season after the first `min_train_seasons`, fits a fresh
    model instance (via model_factory(), a zero-argument callable
    returning an unfit model - a fresh instance each fold so no state
    leaks between folds) on every strictly-earlier season, predicts the
    held-out season, and scores it against the actual results. Returns one
    row per held-out season with every metric plus season/training-size
    bookkeeping columns.
    """
    rows = []
    for test_season, train_seasons, test, proba in _expanding_folds(features, model_factory, min_train_seasons):
        metrics = evaluate_all(test["FTR"], proba)
        metrics["season"] = test_season
        metrics["n_train_seasons"] = len(train_seasons)
        rows.append(metrics)

    cols = ["season", "n_train_seasons", "n_matches", "log_loss", "brier_score", "rps", "accuracy"]
    return pd.DataFrame(rows)[cols]


def expanding_window_predictions(
    features: pd.DataFrame,
    model_factory,
    min_train_seasons: int = 10,
) -> pd.DataFrame:
    """Same fold structure as expanding_window_backtest, but returns the
    raw match-level predicted probabilities (concatenated across every
    held-out season) instead of aggregated per-season metrics - needed for
    calibration analysis (src/evaluation/calibration.py) and any other
    per-match inspection, where a season-level summary isn't enough."""
    chunks = []
    for test_season, _, test, proba in _expanding_folds(features, model_factory, min_train_seasons):
        chunk = proba.copy()
        chunk["FTR"] = test["FTR"].to_numpy()
        chunk["Season"] = test_season
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


def summarize(backtest_results: pd.DataFrame) -> dict:
    """Aggregates metrics across all held-out seasons, weighted by the
    number of matches in each season (seasons in this dataset don't all
    have the same match count - see Phase 1: 462 for the 22-team era, 380
    since, 309 for the truncated final season - so an unweighted mean-of-
    means would silently over-count small seasons)."""
    weights = backtest_results["n_matches"]
    summary = {}
    for metric in ("log_loss", "brier_score", "rps", "accuracy"):
        summary[metric] = float((backtest_results[metric] * weights).sum() / weights.sum())
    summary["n_seasons"] = len(backtest_results)
    summary["n_matches"] = int(weights.sum())
    return summary
