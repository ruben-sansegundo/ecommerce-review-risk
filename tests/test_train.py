"""Tests for the model layer.

The matrix here is synthetic: what these check is the wiring - which columns a
moment is allowed to see, what the floor is worth - and none of that needs the
real dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src import evaluate, train
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
    # The target leans on one t0 feature, so a fitted model has something real
    # to find and a leak has something to give itself away with.
    risk = 1 / (1 + np.exp(-matrix["freight_total"]))
    matrix["y"] = rng.uniform(size=rows) < risk
    return matrix


def test_each_moment_only_sees_the_columns_it_is_allowed_to():
    matrix = synthetic_matrix()
    assert list(train.split_frames("t0", matrix)["train"][0].columns) == T0_FEATURES
    assert list(train.split_frames("t1", matrix)["train"][0].columns) == T0_FEATURES + T1_FEATURES


def test_the_blocks_come_back_whole_and_separate():
    matrix = synthetic_matrix()
    frames = train.split_frames("t0", matrix)

    assert set(frames) == {"train", "val", "test"}
    assert sum(len(X) for X, _ in frames.values()) == len(matrix)


def test_the_constant_rule_scores_the_training_base_rate():
    matrix = synthetic_matrix()
    frames = train.split_frames("t0", matrix)
    scores = train.baseline_scores("constant", "t0", frames, "val")

    assert np.unique(scores).size == 1
    assert scores[0] == pytest.approx(frames["train"][1].mean())


def test_the_floor_is_worth_exactly_the_prevalence():
    """The property that makes the constant rule a floor rather than a model."""
    matrix = synthetic_matrix()
    frames = train.split_frames("t0", matrix)
    _, y_val = frames["val"]
    scores = train.baseline_scores("constant", "t0", frames, "val")

    assert evaluate.pr_auc(y_val, scores) == pytest.approx(y_val.mean())


def test_a_t0_model_cannot_be_moved_by_a_t1_column():
    """The leakage proof of S2, aimed at the model instead of the features.

    Every t1 column is replaced with nonsense. A model fitted at t0 must score
    the block identically - if any of those columns reaches the pipeline, the
    two runs stop matching.
    """
    matrix = synthetic_matrix()
    honest = train.baseline_scores("logistic", "t0", train.split_frames("t0", matrix), "val")

    poisoned_matrix = matrix.copy()
    for column in T1_FEATURES:
        poisoned_matrix[column] = np.where(poisoned_matrix["y"], 999.0, -999.0)
    poisoned = train.baseline_scores(
        "logistic", "t0", train.split_frames("t0", poisoned_matrix), "val"
    )

    np.testing.assert_array_equal(honest, poisoned)


def test_an_unknown_model_is_refused():
    matrix = synthetic_matrix()
    with pytest.raises(ValueError, match="unknown model"):
        train.baseline_scores(
            "gradient boosted wishful thinking", "t0", train.split_frames("t0", matrix), "val"
        )
