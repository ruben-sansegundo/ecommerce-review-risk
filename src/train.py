"""Baselines first, then anything that claims to beat them.

Rule 5 of CLAUDE.md asks a complex model to beat a trivial one by an explicit
number. That only works if both are fitted on the same rows, scored on the same
rows and measured by the same code, so every model in the project is built and
run from here.

Nothing in this module looks at the test block. Model choice happens on
validation; test is read once, at the end, with the choices already frozen.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
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
MODELS = [*BASELINES, "lightgbm"]

# The last two months of training stand in as an inner validation set, used for
# nothing but deciding how many trees. See D-12.
INNER_VALIDATION_START = pd.Timestamp("2018-02-01")
EARLY_STOPPING_ROUNDS = 100

# Deliberately conservative. 64,360 training rows, 32 features and no single
# variable above 0.59 univariate AUC: the risk here is memorising noise, not
# underfitting. No class_weight and no is_unbalance, by rule 7.
LGBM_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "subsample_freq": 1,
    "n_estimators": 3000,
    "random_state": config.SEED,
    # Never -1. This CPU is heterogeneous - performance, efficiency and
    # low-power cores in one package - and OpenMP hands every thread an equal
    # share, so at each barrier the fast cores sit waiting for the slow ones.
    # With no core left over for the rest of the system it degrades into spin
    # waiting: measured on this machine, 50 trees take 0.15s on 8 threads and
    # 28s on 16. Half the cores leaves headroom and stays portable. See D-12.
    "n_jobs": max(1, (os.cpu_count() or 2) // 2),
    "verbosity": -1,
}


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


def lightgbm_model(n_estimators: int | None = None) -> LGBMClassifier:
    """The tree model, with the rounds left open until the inner split decides."""
    params = dict(LGBM_PARAMS)
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    return LGBMClassifier(**params)


def lightgbm_rounds(moment: str, matrix: pd.DataFrame) -> int:
    """How many trees to grow, decided inside the training block.

    Stopping on the validation block would let the model choose when to stop by
    watching the very block that judges it afterwards. So the last two months of
    training play that role instead, and validation stays untouched until the
    comparison. See D-12.
    """
    columns = features.feature_columns(moment)
    rows = matrix[matrix["split"] == "train"]
    inner = rows["order_purchase_timestamp"] >= INNER_VALIDATION_START

    model = lightgbm_model()
    model.fit(
        rows.loc[~inner, columns],
        rows.loc[~inner, "y"],
        eval_X=rows.loc[inner, columns],
        eval_y=rows.loc[inner, "y"],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    return int(model.best_iteration_)


def lightgbm_scores(
    moment: str, matrix: pd.DataFrame, block: str = "val", rounds: int | None = None
) -> np.ndarray:
    """Fit on the whole training block and score another one.

    Refitting on everything after the inner split has chosen the rounds is the
    point: the two logistics crises of 2017-11 and 2018-02/03 live in training
    because D-08 put them there, and a model that has never seen a saturated
    network will fail exactly when intervening matters most.
    """
    columns = features.feature_columns(moment)
    rounds = lightgbm_rounds(moment, matrix) if rounds is None else rounds
    rows = matrix[matrix["split"] == "train"]

    model = lightgbm_model(rounds).fit(rows[columns], rows["y"])
    return model.predict_proba(matrix.loc[matrix["split"] == block, columns])[:, 1]


def model_scores(name: str, moment: str, matrix: pd.DataFrame, block: str = "val") -> np.ndarray:
    """Scores from any model in MODELS, baselines and tree alike."""
    if name == "lightgbm":
        return lightgbm_scores(moment, matrix, block)
    return baseline_scores(name, moment, split_frames(moment, matrix), block)


def model_table(block: str = "val", matrix: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every model at both decision moments, in the one table S3 exists to produce."""
    matrix = features.load_features() if matrix is None else matrix
    rows = []
    for moment in ("t0", "t1"):
        y_block = matrix.loc[matrix["split"] == block, "y"].to_numpy()
        for name in MODELS:
            scores = model_scores(name, moment, matrix, block)
            rows.append(evaluate.score_model(name, moment, y_block, scores))
    return evaluate.comparison_table(rows)


def main() -> None:
    """Run with `make train` or `python -m src.train`."""
    table = model_table("val")
    shown = ["model", "moment", "prevalence", "pr_auc", "recall@10%", "lift@10%", "brier"]

    print("every model, scored on validation (2018-04 .. 2018-05)\n")
    print(table[shown].to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
