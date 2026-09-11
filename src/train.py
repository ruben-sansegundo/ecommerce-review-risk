"""Baselines first, then anything that claims to beat them.

Rule 5 of CLAUDE.md asks a complex model to beat a trivial one by an explicit
number. That only works if both are fitted on the same rows, scored on the same
rows and measured by the same code, so every model in the project is built and
run from here.

Nothing in this module looks at the test block. Model choice happens on
validation; test is read once, at the end, with the choices already frozen.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler

from src import config, evaluate, features

# Deciles, to match how the signal was read in S2: the relationship between
# these features and the target is not linear, it lives in the tails.
N_BINS = 10

# Categories below this many orders in training are pooled. product_category has
# ~70 levels and a long tail; one column each would be mostly empty.
MIN_CATEGORY_FREQUENCY = 50

# The strongest single feature at each moment, by univariate AUC on training
# (docs/features.md). Standing in for the rule an analyst would write with no
# model at all.
SINGLE_FEATURE = {"t0": "freight_total", "t1": "handover_days"}

BASELINES = ["constant", "single feature", "logistic", "logistic binned"]


def split_frames(moment: str, matrix: pd.DataFrame | None = None) -> dict:
    """Features and target per block, restricted to what `moment` may see."""
    matrix = features.load_features() if matrix is None else matrix
    columns = features.feature_columns(moment)
    return {
        block: (rows[columns], rows["y"].to_numpy())
        for block, rows in matrix.groupby("split", observed=True)
        if block in config.SPLIT_ORDER
    }


def column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Which columns go through the numeric branch and which through one-hot."""
    categorical = [c for c in X.columns if c in features.CATEGORICAL]
    return [c for c in X.columns if c not in categorical], categorical


def logistic_pipeline(X: pd.DataFrame, binned: bool = False) -> Pipeline:
    """Logistic regression with the preprocessing it cannot do without.

    Three branches, because the model needs three different things from the
    data and one of them is usually forgotten:

    - the numeric columns, median-imputed and then either scaled or cut into
      deciles. Binning is what lets a linear model express "flat for nine
      deciles and then a jump", which is the shape S2 measured.
    - a missing indicator, kept separate from the imputation. That a seller has
      no tenure means it is new, and replacing it with the median throws that
      away - imputing the value is not the same as pretending it was never
      missing.
    - the categoricals, one-hot with a frequency floor. No target encoding:
      that is this project's own leak, done without a backward window.

    No class_weight, by rule 7: the imbalance is handled by the threshold, and
    reweighting would distort the probabilities the euro figures depend on.
    """
    numeric, categorical = column_groups(X)

    if binned:
        shape = KBinsDiscretizer(
            n_bins=N_BINS, encode="onehot-dense", strategy="quantile", subsample=None
        )
    else:
        shape = StandardScaler()

    numeric_branch = Pipeline([("impute", SimpleImputer(strategy="median")), ("shape", shape)])
    preprocessing = ColumnTransformer(
        [
            ("numeric", numeric_branch, numeric),
            ("missing", MissingIndicator(features="missing-only"), numeric),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_CATEGORY_FREQUENCY,
                    sparse_output=False,
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("prepare", preprocessing),
            ("model", LogisticRegression(max_iter=2000, random_state=config.SEED)),
        ]
    )


def baseline_scores(name: str, moment: str, frames: dict, block: str = "val") -> np.ndarray:
    """Fit `name` on training and score `block` with it."""
    X_train, y_train = frames["train"]
    X_eval, _ = frames[block]

    if name == "constant":
        # The floor: one number for everybody, the training base rate. PR-AUC
        # lands on the prevalence of the block being scored, by construction.
        return np.full(len(X_eval), y_train.mean())

    if name == "single feature":
        column = SINGLE_FEATURE[moment]
        return X_eval[column].fillna(X_train[column].median()).to_numpy()

    if name in ("logistic", "logistic binned"):
        pipeline = logistic_pipeline(X_train, binned=name.endswith("binned"))
        pipeline.fit(X_train, y_train)
        return pipeline.predict_proba(X_eval)[:, 1]

    raise ValueError(f"unknown model {name!r}")


def baseline_table(block: str = "val", matrix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every baseline at both decision moments, in one table."""
    matrix = features.load_features() if matrix is None else matrix
    rows = []
    for moment in ("t0", "t1"):
        frames = split_frames(moment, matrix)
        _, y_eval = frames[block]
        for name in BASELINES:
            scores = baseline_scores(name, moment, frames, block)
            rows.append(evaluate.score_model(name, moment, y_eval, scores))
    return evaluate.comparison_table(rows)


def main() -> None:
    """Run with `make train` or `python -m src.train`."""
    table = baseline_table("val")
    shown = ["model", "moment", "prevalence", "pr_auc", "recall@10%", "lift@10%", "brier"]

    print("baselines, scored on validation (2018-04 .. 2018-05)\n")
    print(table[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
