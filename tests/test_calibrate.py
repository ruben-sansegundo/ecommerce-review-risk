"""Tests for calibration.

Most of these need only scores and labels, not a model: calibration is a map
from one to the other, and testing it on invented numbers is both faster and
clearer about what is being claimed.
"""

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_matrix

from src import calibrate, config, evaluate


def scores_and_labels(rows: int = 4000, shift: float = 0.0, seed: int = config.SEED):
    """Scores that rank well but sit at the wrong level, like the real ones.

    `shift` pushes the scores up in log-odds, which is exactly the failure D-13
    measured: a model trained at a 14.70% base rate scoring a 10.19% block.
    """
    rng = np.random.default_rng(seed)
    truth = rng.beta(2, 12, size=rows)
    y = rng.uniform(size=rows) < truth
    odds = np.log(truth / (1 - truth)) + shift
    return 1 / (1 + np.exp(-odds)), y


def test_a_sigmoid_leaves_the_ranking_exactly_where_it_was():
    """The property that decided the method: no order changes, so no metric
    that depends only on order changes either."""
    scores, y = scores_and_labels(shift=0.8)
    calibrated = calibrate.fit_sigmoid(scores, y)(scores)

    assert np.array_equal(np.argsort(scores), np.argsort(calibrated))
    assert evaluate.pr_auc(y, calibrated) == pytest.approx(evaluate.pr_auc(y, scores))


def test_isotonic_keeps_the_order_but_creates_ties():
    """Why it was not chosen. Nothing is reversed, but whole ranges of scores
    collapse onto one value, and ties are not free when you act on the top."""
    scores, y = scores_and_labels(shift=0.8)
    calibrated = calibrate.fit_isotonic(scores, y)(scores)

    assert np.all(np.diff(calibrated[np.argsort(scores)]) >= 0)
    assert len(np.unique(calibrated)) < len(np.unique(scores)) / 10


def test_calibration_moves_the_level_without_being_asked_to():
    """The whole point: predicted risk should average out to observed risk."""
    scores, y = scores_and_labels(shift=0.8)
    calibrated = calibrate.fit_sigmoid(scores, y)(scores)

    assert scores.mean() > y.mean() + 0.03  # the miscalibration is really there
    assert calibrated.mean() == pytest.approx(y.mean(), abs=0.005)


def test_calibration_error_is_zero_when_the_probabilities_are_honest():
    rng = np.random.default_rng(config.SEED)
    truth = rng.uniform(0.01, 0.6, size=40_000)
    y = rng.uniform(size=40_000) < truth

    assert calibrate.calibration_error(y, truth) < 0.01


def test_calibration_error_grows_with_the_shift():
    honest, y = scores_and_labels()
    inflated, _ = scores_and_labels(shift=1.0)

    assert calibrate.calibration_error(y, inflated) > calibrate.calibration_error(y, honest)


def test_the_reliability_table_accounts_for_every_order():
    scores, y = scores_and_labels()
    table = calibrate.reliability(y, scores, bins=10)

    assert table["orders"].sum() == len(y)
    assert table["gap"].to_numpy() == pytest.approx(
        (table["predicted"] - table["observed"]).to_numpy()
    )


def test_the_test_block_takes_no_part_in_the_calibration():
    """Structural, rather than a matter of reading the code carefully.

    Flip every test label and the calibrator must not move a digit; flip the
    validation labels and it must. Training labels are deliberately left alone:
    those do reach the calibrator, through the model whose scores it corrects.
    """
    matrix = synthetic_matrix()
    honest = calibrate.fit_calibrator("t0", matrix)

    def flipped(block):
        poisoned = matrix.copy()
        rows = poisoned["split"] == block
        poisoned.loc[rows, "y"] = ~poisoned.loc[rows, "y"]
        return calibrate.fit_calibrator("t0", poisoned)

    probe = np.linspace(0.01, 0.9, 25)
    np.testing.assert_allclose(honest(probe), flipped("test")(probe))
    assert not np.allclose(honest(probe), flipped("val")(probe))


def test_the_method_is_chosen_on_a_half_of_validation_it_never_saw():
    """April fits, May grades, and the table reports the uncalibrated row too so
    the choice can be argued with rather than taken on trust."""
    comparison = calibrate.choose_method("t0", synthetic_matrix(rows=2000))

    assert list(comparison.index) == ["none", "sigmoid", "isotonic"]
    assert set(comparison.columns) == {"brier", "calibration_error", "pr_auc"}


def test_april_and_may_land_on_different_sides_of_the_cut():
    matrix = synthetic_matrix()
    validation = matrix[matrix["split"] == "val"]
    fits = validation["order_purchase_timestamp"] < calibrate.METHOD_CHOICE_CUT

    assert fits.sum() > 0
    assert (~fits).sum() > 0
    assert pd.Timestamp("2018-04-01") <= validation["order_purchase_timestamp"].min()
