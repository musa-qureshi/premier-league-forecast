"""Model 1: Elo-derived match outcome probabilities.

Elo's own expected_score() (src/features/elo.py) gives a single number: the
probability of a "win" in the chess sense, where a draw counts as half a
win. Football has three genuine outcomes, not two, so that number alone
isn't yet a P(Home)/P(Draw)/P(Away) triple - it has to be split into three.

The approach used here: fit a multinomial logistic regression with
`elo_diff` as its ONLY input feature. This is deliberately the simplest
possible way to turn a rating gap into three calibrated probabilities:

    P(class) = softmax(beta_class * elo_diff + intercept_class)

With just one feature, this model has essentially 4 free parameters (an
intercept and a slope for two of the three classes, the third being fixed
by the softmax constraint) - it isn't "doing machine learning" in any deep
sense, it's calibrating a single number into a probability triple, which is
exactly what Model 1 is supposed to be: a step up from the frequency
baseline that uses team strength, but nothing else. The intercepts also
absorb the average home-advantage effect automatically (since HomeTeam is
always the "home" side in this data), without needing to hand-tune a
separate home-advantage constant here on top of the one already baked into
how the Elo ratings themselves were updated (src/features/elo.py).

This is kept deliberately distinct from Model 2 (logistic regression, Phase
4), which will use many more features - Model 1 exists to answer "how much
of the story does Elo alone tell", not to be a smaller version of Model 2.

An alternative, non-fitted approach would hand-pick a fixed formula
mapping Elo's win-probability into a H/D/A split (e.g. via an assumed
draw margin) - rejected here because "hand-pick a constant" is not
obviously simpler than "fit 4 parameters by maximum likelihood", and the
fitted version lets the backtest harness re-calibrate it per training
window rather than trusting a constant chosen once and never revisited.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.evaluation.metrics import CLASS_ORDER


class EloOutcomeModel:
    def __init__(self) -> None:
        self._clf: LogisticRegression | None = None

    def fit(self, train: pd.DataFrame) -> "EloOutcomeModel":
        X = train[["elo_diff"]].to_numpy()
        y = train["FTR"].to_numpy()
        # sklearn's default "lbfgs" solver fits a true multinomial (softmax)
        # model natively for a multi-class target - the explicit
        # multi_class="multinomial" flag used to be required to get that
        # behavior instead of one-vs-rest, but became the automatic default
        # and was removed as a constructor argument in newer sklearn.
        self._clf = LogisticRegression(max_iter=1000)
        self._clf.fit(X, y)
        return self

    def predict_proba(self, test: pd.DataFrame) -> pd.DataFrame:
        if self._clf is None:
            raise RuntimeError("Call fit() before predict_proba().")
        X = test[["elo_diff"]].to_numpy()
        proba = self._clf.predict_proba(X)
        # sklearn orders columns by self._clf.classes_ (alphabetical for
        # string labels: A, D, H - which happens to already match
        # CLASS_ORDER, but assert it rather than assume a library detail
        # silently stays true).
        assert list(self._clf.classes_) == CLASS_ORDER, (
            f"Expected sklearn class order {CLASS_ORDER}, got {list(self._clf.classes_)}"
        )
        return pd.DataFrame(proba, columns=CLASS_ORDER, index=test.index)
