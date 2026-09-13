"""Shared fixtures.

The synthetic matrix lives here because two modules need it: the model tests
and the calibration tests. It is shaped like the real feature matrix - same
columns, same dtypes, same block layout in time - and nothing in it comes from
data/raw, so the suite stays fast and runs on a clean clone.
"""

import numpy as np
import pandas as pd

from src.features import CATEGORICAL, T0_FEATURES, T1_FEATURES


def synthetic_matrix(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """A feature matrix shaped like the real one, with a target worth learning."""
    rng = np.random.default_rng(seed)
    columns = {}
    for column in T0_FEATURES + T1_FEATURES:
        if column in CATEGORICAL:
            columns[column] = pd.Categorical(rng.choice(["a", "b", "c"], rows))
        else:
            columns[column] = rng.normal(size=rows)

    matrix = pd.DataFrame(columns)
    matrix["split"] = np.repeat(["train", "val", "test"], [rows // 2, rows // 4, rows // 4])
    # Purchase dates laid out like the real blocks, so the inner split that
    # chooses the number of trees has training rows on both sides of its cut.
    matrix["order_purchase_timestamp"] = pd.concat(
        [
            pd.Series(pd.date_range("2017-01-01", "2018-03-31", periods=rows // 2)),
            pd.Series(pd.date_range("2018-04-01", "2018-05-31", periods=rows // 4)),
            pd.Series(pd.date_range("2018-06-01", "2018-08-31", periods=rows // 4)),
        ],
        ignore_index=True,
    )
    # The target leans on one t0 feature, so a fitted model has something real
    # to find and a leak has something to give itself away with.
    risk = 1 / (1 + np.exp(-matrix["freight_total"]))
    matrix["y"] = rng.uniform(size=rows) < risk
    return matrix
