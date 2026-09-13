"""Turning a ranking into a probability.

S3 ended with proof that the models are miscalibrated rather than a suspicion:
on the test block the constant rule scores a better Brier (0.0935) than the t0
logistic (0.0961), because both models learn a 14.70% base rate and score a
10.19% one. Ranking well and being right about the probability are different
things, and every euro figure depends on the second.

Two rules shape everything here:

- The calibrator is fitted on validation, never on training. A model calibrated
  on the data it was fitted to looks perfectly calibrated by construction - it
  has already seen those labels.
- The choice between methods is made on a slice of validation the calibrator
  did not see: April fits, May judges. Picking the method by looking at the
  whole block would be choosing it on the same data that grades it.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src import config, evaluate, features, train

# Inner cut of the validation block: April fits the calibrators, May compares
# them. Both months sit in the same logistics regime (D-08), so the comparison
# is not confounded by one of them being a crisis.
METHOD_CHOICE_CUT = pd.Timestamp("2018-05-01")

# Probabilities are pushed off 0 and 1 before taking log-odds, which are
# infinite there. 1e-6 is far below any threshold this project will use.
_EPS = 1e-6


def _log_odds(p: np.ndarray) -> np.ndarray:
    """log(p / (1-p)), the scale a sigmoid calibrator is linear on."""
    clipped = np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)
    return np.log(clipped / (1 - clipped))


@dataclass(frozen=True)
class Calibrator:
    """A fitted mapping from raw model scores to calibrated probabilities."""

    method: str
    transform: Callable[[np.ndarray], np.ndarray]

    def __call__(self, scores) -> np.ndarray:
        return np.asarray(self.transform(np.asarray(scores, dtype=float)), dtype=float)


def fit_sigmoid(scores, y) -> Calibrator:
    """Platt scaling: one logistic regression on the model's own log-odds.

    Two parameters, a slope and an intercept. The intercept is what fixes a
    base-rate shift, which is the shape of miscalibration D-13 measured, and two
    parameters cannot memorise 1,600 positives. Being strictly increasing, it
    leaves the ranking - and therefore PR-AUC and recall@k - untouched.
    """
    z = _log_odds(scores).reshape(-1, 1)
    # C=np.inf is "no regularisation" in the current sklearn API; penalty=None
    # says the same thing and is deprecated since 1.8. Two parameters fitted on
    # 13,612 rows do not need shrinking towards anything.
    model = LogisticRegression(C=np.inf).fit(z, np.asarray(y).astype(int))
    return Calibrator("sigmoid", lambda s: model.predict_proba(_log_odds(s).reshape(-1, 1))[:, 1])


def fit_isotonic(scores, y) -> Calibrator:
    """Isotonic regression: any non-decreasing step function.

    Free to bend however the data asks, which is its strength and its risk: with
    1,614 positives it can fit the noise of the block it learns from. It is also
    only weakly monotone - it maps whole ranges of scores onto one value - so it
    creates ties, and ties can move PR-AUC slightly even though no order is
    reversed.
    """
    model = IsotonicRegression(out_of_bounds="clip").fit(
        np.asarray(scores, dtype=float), np.asarray(y).astype(float)
    )
    return Calibrator("isotonic", model.predict)


METHODS = {"sigmoid": fit_sigmoid, "isotonic": fit_isotonic}


def reliability(y_true, p, bins: int = 10) -> pd.DataFrame:
    """The calibration curve as a table: predicted against observed, by bin.

    Equal-count bins rather than equal-width. The predictions crowd into the low
    end - most orders are low risk - and equal-width bins would put almost
    everything in the first one and leave the rest reading off a handful of
    orders.
    """
    y = np.asarray(y_true).astype(float)
    scores = np.asarray(p, dtype=float)
    edges = pd.qcut(scores, bins, labels=False, duplicates="drop")

    table = pd.DataFrame({"bin": edges, "predicted": scores, "observed": y})
    summary = table.groupby("bin", observed=True).agg(
        orders=("observed", "size"),
        predicted=("predicted", "mean"),
        observed=("observed", "mean"),
    )
    summary["gap"] = summary["predicted"] - summary["observed"]
    return summary


def calibration_error(y_true, p, bins: int = 10) -> float:
    """Mean absolute gap between predicted and observed, weighted by bin size.

    Brier says how good the probabilities are overall, mixing calibration with
    how well the model separates the classes. This isolates the calibration half:
    of the orders this model calls 30% risk, how many really end badly?
    """
    summary = reliability(y_true, p, bins)
    weights = summary["orders"] / summary["orders"].sum()
    return float((summary["gap"].abs() * weights).sum())


def scores_by_block(moment: str, matrix: pd.DataFrame) -> dict[str, np.ndarray]:
    """Raw logistic scores for every block, from a single fit on training."""
    columns = features.feature_columns(moment)
    rows = matrix[matrix["split"] == "train"]
    model = train.fit_logistic(rows[columns], rows["y"])
    return {
        block: model.predict_proba(matrix.loc[matrix["split"] == block, columns])[:, 1]
        for block in config.SPLIT_ORDER
    }


def choose_method(moment: str, matrix: pd.DataFrame) -> pd.DataFrame:
    """Fit every method on April, grade them on May, report what each scored.

    Returns the comparison rather than just the winner: a method chosen without
    showing the numbers behind the choice is a method nobody can argue with.
    """
    scores = scores_by_block(moment, matrix)["val"]
    validation = matrix[matrix["split"] == "val"]
    fits = (validation["order_purchase_timestamp"] < METHOD_CHOICE_CUT).to_numpy()
    y = validation["y"].to_numpy()

    rows = [
        {
            "method": "none",
            "brier": evaluate.brier(y[~fits], scores[~fits]),
            "calibration_error": calibration_error(y[~fits], scores[~fits]),
            "pr_auc": evaluate.pr_auc(y[~fits], scores[~fits]),
        }
    ]
    for name, fit in METHODS.items():
        calibrator = fit(scores[fits], y[fits])
        judged = calibrator(scores[~fits])
        rows.append(
            {
                "method": name,
                "brier": evaluate.brier(y[~fits], judged),
                "calibration_error": calibration_error(y[~fits], judged),
                "pr_auc": evaluate.pr_auc(y[~fits], judged),
            }
        )
    return pd.DataFrame(rows).set_index("method")


# Chosen on the May half of validation, which the calibrators never saw: the
# sigmoid matches isotonic on Brier and beats it on calibration error, and it
# leaves PR-AUC untouched to the fifth decimal where isotonic costs 0.012 at t0
# and 0.015 at t1 by turning ranges of scores into ties. See D-14.
CHOSEN_METHOD = "sigmoid"


def fit_calibrator(moment: str, matrix: pd.DataFrame, method: str = CHOSEN_METHOD) -> Calibrator:
    """Fit the chosen calibrator on the whole validation block.

    The method was picked on half of validation; refitting on all of it doubles
    the data behind the two parameters that ship. Training is never used here -
    the model has already seen those labels.
    """
    scores = scores_by_block(moment, matrix)["val"]
    y = matrix.loc[matrix["split"] == "val", "y"].to_numpy()
    return METHODS[method](scores, y)


def calibrated_scores(
    moment: str, matrix: pd.DataFrame, method: str = CHOSEN_METHOD
) -> dict[str, np.ndarray]:
    """Calibrated probabilities per block, plus the calibrator that made them."""
    raw = scores_by_block(moment, matrix)
    calibrator = fit_calibrator(moment, matrix, method)
    return {"calibrator": calibrator, **{block: calibrator(s) for block, s in raw.items()}}


def calibration_summary(matrix: pd.DataFrame | None = None, block: str = "test") -> pd.DataFrame:
    """What calibration bought, and what it cost, on a block it never saw."""
    matrix = features.load_features() if matrix is None else matrix
    y = matrix.loc[matrix["split"] == block, "y"].to_numpy()

    rows = []
    for moment in ("t0", "t1"):
        raw = scores_by_block(moment, matrix)[block]
        calibrated = calibrated_scores(moment, matrix)[block]
        for label, scores in (("raw", raw), ("calibrated", calibrated)):
            rows.append(
                {
                    "moment": moment,
                    "scores": label,
                    "mean_predicted": float(np.mean(scores)),
                    "observed": float(y.mean()),
                    "brier": evaluate.brier(y, scores),
                    "calibration_error": calibration_error(y, scores),
                    "pr_auc": evaluate.pr_auc(y, scores),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Run with `python -m src.calibrate`."""
    matrix = features.load_features()
    for moment in ("t0", "t1"):
        print(f"\n{moment} · choosing the method on May, fitted on April")
        print(choose_method(moment, matrix).round(5).to_string())

    print("\n\ncalibrated on validation, measured on test")
    print(calibration_summary(matrix).round(5).to_string(index=False))


if __name__ == "__main__":
    main()
