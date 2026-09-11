"""Tests for the metrics module.

Every case here is hand-sized and its expected value is worked out by hand in
the test itself. A metric that only agrees with its own implementation is not
checked at all.
"""

from dataclasses import replace

import numpy as np
import pytest

from src import config
from src.evaluate import (
    best_threshold,
    brier,
    comparison_table,
    cost_per_order,
    do_nothing_cost,
    expected_cost,
    lift_at_k,
    net_savings,
    pr_auc,
    precision_at_k,
    prevalence,
    recall_at_k,
    score_model,
    sensitivity_grid,
)

# Four orders covering the four cells of the cost matrix under a 0.5 cutoff:
# acted-positive, acted-negative, ignored-positive, ignored-negative.
Y = np.array([1, 0, 1, 0])
P = np.array([0.9, 0.8, 0.1, 0.2])


def perfect(y):
    """Scores that rank every positive above every negative."""
    return np.asarray(y, dtype=float)


def test_prevalence_is_the_share_of_positives():
    assert prevalence([1, 0, 0, 0]) == 0.25


def test_a_perfect_ranking_scores_one():
    y = [1, 1, 0, 0, 0, 0]
    assert pr_auc(y, perfect(y)) == pytest.approx(1.0)


def test_a_constant_score_lands_on_the_prevalence():
    # The floor of the metric: knowing nothing is worth exactly the base rate.
    y = [1] * 20 + [0] * 80
    assert pr_auc(y, np.full(100, 0.5)) == pytest.approx(0.20, abs=0.01)


def test_an_inverted_ranking_scores_below_the_prevalence():
    # Graded scores, not a flipped 0/1: with only two tied levels every ordering
    # inside a level cancels out and average precision lands back on the base
    # rate. Being wrong on purpose only shows up when the ranking is graded.
    y = [1] * 20 + [0] * 80
    assert pr_auc(y, np.arange(100, dtype=float)) < 0.20


def test_recall_at_k_counts_positives_inside_the_top_slice():
    # Top 20% of 10 orders is 2 rows; the ranking puts one positive in them.
    y = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    scores = [0.9, 0.8, 0.1, 0, 0, 0, 0, 0, 0, 0]
    assert recall_at_k(y, scores, 0.20) == pytest.approx(0.5)


def test_recall_at_one_hundred_percent_captures_everything():
    y = [1, 0, 1, 0]
    assert recall_at_k(y, [0.4, 0.3, 0.2, 0.1], 1.0) == pytest.approx(1.0)


def test_the_top_slice_is_never_empty():
    # 1% of 4 orders rounds up to one row, not to zero.
    assert precision_at_k([1, 0, 0, 0], [0.9, 0.1, 0.1, 0.1], 0.01) == pytest.approx(1.0)


def test_lift_is_precision_over_the_base_rate():
    y = [1] * 10 + [0] * 90
    scores = perfect(y)
    k = 0.10
    assert lift_at_k(y, scores, k) == pytest.approx(precision_at_k(y, scores, k) / 0.10)
    # A perfect ranking fills the top 10% with the 10% of orders that are
    # positive, so it is exactly 1 / prevalence times richer than the base.
    assert lift_at_k(y, scores, k) == pytest.approx(10.0)


def test_recall_is_undefined_without_positives():
    assert np.isnan(recall_at_k([0, 0, 0], [0.9, 0.5, 0.1], 0.5))


def test_ties_are_broken_at_random_but_reproducibly():
    # A constant score in chronological order: all positives sit at the front.
    y = np.array([1] * 10 + [0] * 90)
    flat = np.full(100, 0.3)

    # Dataset order would hand the top 10% every positive - recall 1.0 for a
    # rule that knows nothing. Random tie-breaking puts it near the base rate.
    assert recall_at_k(y, flat, 0.10) < 0.5
    assert recall_at_k(y, flat, 0.10) == recall_at_k(y, flat, 0.10)
    assert recall_at_k(y, flat, 0.10, seed=1) != recall_at_k(y, flat, 0.10, seed=7)


def test_brier_is_the_mean_squared_error_of_the_probabilities():
    assert brier([1, 0], [0.75, 0.25]) == pytest.approx(0.0625)


def test_brier_refuses_scores_that_are_not_probabilities():
    assert np.isnan(brier([1, 0], [12.0, -3.0]))


def test_the_cost_matrix_matches_d04_cell_by_cell():
    costs = config.DEFAULT_COSTS
    # do nothing / review OK, do nothing / negative, intervene / OK, intervene / negative
    cells = cost_per_order([0, 1, 0, 1], [0, 0, 1, 1], costs)
    assert cells == pytest.approx([0.0, 40.0, 3.0, 31.0])


def test_expected_cost_adds_up_the_four_cells():
    # Cutoff 0.5 acts on the first two orders: 31 + 3 + 40 + 0.
    assert expected_cost(Y, P, threshold=0.5) == pytest.approx(74.0)


def test_doing_nothing_costs_the_negatives_reviews_it_lets_through():
    assert do_nothing_cost(Y) == pytest.approx(80.0)
    assert net_savings(Y, P, threshold=0.5) == pytest.approx(6.0)


def test_the_default_cutoff_is_the_break_even_threshold_of_d04():
    assert config.DEFAULT_COSTS.threshold == pytest.approx(0.25)
    assert expected_cost(Y, P) == pytest.approx(expected_cost(Y, P, threshold=0.25))


def test_costs_given_per_order_reduce_to_the_scalar_case():
    per_order = replace(config.DEFAULT_COSTS, c_neg=np.full(4, 40.0))
    assert expected_cost(Y, P, 0.5, per_order) == pytest.approx(expected_cost(Y, P, 0.5))


def test_costs_given_per_order_can_differ_row_by_row():
    # The first order is worth ten times the others; it is acted on and positive,
    # so its cell is 3 + 0.7 * 400 = 283 instead of 31.
    per_order = replace(config.DEFAULT_COSTS, c_neg=np.array([400.0, 40.0, 40.0, 40.0]))
    assert expected_cost(Y, P, 0.5, per_order) == pytest.approx(74.0 + 252.0)


def test_the_best_threshold_acts_on_the_positives_and_nobody_else():
    y = np.array([1] * 10 + [0] * 90)
    cutoff, savings = best_threshold(y, perfect(y))

    # Each caught positive saves 40 - 31 = 9 EUR; each false alarm costs 3, so
    # the optimum on a perfect ranking is the 10 positives and nothing more.
    assert savings == pytest.approx(90.0)
    assert cutoff == pytest.approx(1.0)


def test_the_best_threshold_intervenes_on_nobody_when_nothing_is_worth_it():
    # No positives at all: every intervention is a pure 3 EUR loss.
    cutoff, savings = best_threshold(np.zeros(5), np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
    assert savings == pytest.approx(0.0)
    assert cutoff > 0.9


def test_the_best_threshold_is_never_worse_than_the_fixed_one():
    rng = np.random.default_rng(config.SEED)
    p = rng.uniform(size=500)
    y = rng.uniform(size=500) < p

    _, best = best_threshold(y, p)
    assert best >= net_savings(y, p) - 1e-9


def test_a_scored_row_carries_its_own_base_rate():
    y = [1] * 10 + [0] * 90
    row = score_model("logistic", "t0", y, perfect(y))

    assert row["model"] == "logistic"
    assert row["moment"] == "t0"
    assert row["orders"] == 100
    assert row["prevalence"] == pytest.approx(0.10)
    assert row["pr_auc"] == pytest.approx(1.0)
    assert row["recall@10%"] == pytest.approx(1.0)
    assert row["lift@10%"] == pytest.approx(10.0)


def test_the_comparison_table_ranks_each_moment_by_pr_auc():
    y = [1] * 10 + [0] * 90
    rows = [
        score_model("weak", "t0", y, np.full(100, 0.5)),
        score_model("strong", "t0", y, perfect(y)),
    ]
    table = comparison_table(rows)
    assert list(table["model"]) == ["strong", "weak"]


def test_the_sensitivity_grid_covers_every_combination():
    grid = sensitivity_grid(Y, P, [1, 3], [20, 40], [0.15, 0.30])
    assert len(grid) == 8
    row = grid.iloc[0]
    assert row["threshold"] == pytest.approx(row["c_int"] / (row["effectiveness"] * row["c_neg"]))


@pytest.mark.parametrize(
    "y, scores",
    [
        ([1, 0, 1], [0.5, 0.5]),
        ([1, 0], [0.5, np.nan]),
        ([], []),
    ],
)
def test_malformed_input_is_rejected_rather_than_scored(y, scores):
    with pytest.raises(ValueError):
        pr_auc(y, scores)
