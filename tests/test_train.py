"""Tests for the model layer.

The matrix here is synthetic: what these check is the wiring - which columns a
moment is allowed to see, what the floor is worth - and none of that needs the
real dataset.
"""

import numpy as np
import pytest
from conftest import synthetic_matrix

from src import evaluate, train
from src.features import T0_FEATURES, T1_FEATURES


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


def test_the_number_of_trees_is_decided_without_looking_at_validation():
    """The inner split lives inside training; validation must not move it.

    Poisoning validation is the check: if early stopping ever watched that
    block, the round count would change with it.
    """
    matrix = synthetic_matrix()
    honest = train.lightgbm_rounds("t0", matrix)

    poisoned = matrix.copy()
    validation = poisoned["split"] == "val"
    poisoned.loc[validation, "freight_total"] = np.where(
        poisoned.loc[validation, "y"], 999.0, -999.0
    )

    assert train.lightgbm_rounds("t0", poisoned) == honest


def test_the_tree_is_refitted_on_the_whole_training_block():
    """Rounds come from the inner split; the model that ships sees every row.

    The two logistics crises sit in training because D-08 put them there. A
    model fitted only on the inner split would never have seen them.
    """
    matrix = synthetic_matrix()
    rounds = train.lightgbm_rounds("t0", matrix)
    # The same frame with the tail of training removed, validation left alone.
    early = matrix["order_purchase_timestamp"] < train.INNER_VALIDATION_START
    inner_only = matrix[(matrix["split"] != "train") | early]

    full = train.lightgbm_scores("t0", matrix, "val", rounds=rounds)
    partial = train.lightgbm_scores("t0", inner_only, "val", rounds=rounds)

    assert not np.allclose(full, partial)


def test_a_t0_tree_cannot_be_moved_by_a_t1_column():
    """The same leakage proof the logistic gets, for the model that may ship."""
    matrix = synthetic_matrix()
    honest = train.lightgbm_scores("t0", matrix, "val", rounds=20)

    poisoned_matrix = matrix.copy()
    for column in T1_FEATURES:
        poisoned_matrix[column] = np.where(poisoned_matrix["y"], 999.0, -999.0)
    poisoned = train.lightgbm_scores("t0", poisoned_matrix, "val", rounds=20)

    np.testing.assert_array_equal(honest, poisoned)
