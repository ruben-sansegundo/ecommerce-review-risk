"""Tests for the euro layer.

The arithmetic of D-04 is simple enough that every expected value below is
worked out by hand in the test. That matters more here than anywhere else in
the project: these are the numbers that would be quoted in a meeting.
"""

from dataclasses import replace

import numpy as np
import pytest
from conftest import synthetic_matrix

from src import business, config, evaluate

# Ten orders, three of which end in a negative review, scored perfectly.
Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
PERFECT = Y.astype(float)


def test_a_perfect_model_saves_nine_euros_per_negative_review():
    """Acting on a review that was coming costs 3 and avoids 40 three times in
    ten: 3 + 0.7 * 40 = 31 against 40, so 9 saved, and no false alarms to pay."""
    report = business.policy(Y, PERFECT)

    assert report["flagged"] == 3
    assert report["false_alarms"] == 0
    assert report["savings_eur"] == pytest.approx(27.0)


def test_the_counts_account_for_every_order():
    scores = np.linspace(0, 1, 10)
    report = business.policy(Y, scores)

    assert report["caught"] + report["missed"] == Y.sum()
    assert report["caught"] + report["false_alarms"] == report["flagged"]
    assert report["flagged_share"] == pytest.approx(report["flagged"] / len(Y))


def test_a_threshold_nobody_reaches_changes_nothing():
    report = business.policy(Y, PERFECT, threshold=1.1)

    assert report["flagged"] == 0
    assert report["savings_eur"] == pytest.approx(0.0)


def test_savings_per_thousand_orders_scale_with_the_block():
    report = business.policy(Y, PERFECT)
    expected = report["savings_eur"] / len(Y) * 1000

    assert report["savings_per_1000_orders"] == pytest.approx(expected)


def test_intervening_on_everyone_loses_money_at_this_base_rate():
    """The claim behind needing a model at all.

    Acting on all n orders costs 3n and saves 0.3 * 40 = 12 on each of the P
    that would have gone bad. At a 10% base rate that is 1.2 against 3 per
    order, so blanket action destroys value and only targeting can recover it.
    """
    rate = 0.10
    rng = np.random.default_rng(config.SEED)
    y = rng.uniform(size=10_000) < rate

    table = business.blanket_policies(y).set_index("policy")
    expected = 12 * y.sum() - 3 * len(y)

    assert table.loc["do nothing", "savings_eur"] == 0.0
    assert table.loc["intervene on everyone", "savings_eur"] == pytest.approx(expected)
    assert table.loc["intervene on everyone", "savings_eur"] < 0


def test_a_better_intervention_lowers_the_bar_for_using_it():
    """The threshold is c_int / (e * C_neg), so it has to move with e."""
    grid = business.effectiveness_scenarios(
        synthetic_matrix(), block="test", t0_effectiveness=(0.30, 0.60)
    )

    assert grid["t0_threshold"].to_list() == pytest.approx([0.25, 0.125])
    assert grid["t0_threshold"].is_monotonic_decreasing


def test_the_operating_point_is_reported_for_both_moments():
    table = business.policy_table(synthetic_matrix(), block="test")

    assert table["moment"].to_list() == ["t0", "t1"]
    assert (table["threshold"] == config.DEFAULT_COSTS.threshold).all()


def test_the_break_even_is_where_the_savings_really_cross_zero():
    """The closed form against brute force.

    e = c_int / (C_neg * precision) is derived by hand in the docstring; this
    checks the derivation against the arithmetic it claims to summarise.
    """
    rng = np.random.default_rng(config.SEED)
    truth = rng.beta(2, 10, size=5000)
    y = rng.uniform(size=5000) < truth

    crossing = business.break_even(y, truth)["effectiveness"]
    costs = replace(config.DEFAULT_COSTS, effectiveness=crossing)

    # At the crossing the policy is worth nothing - but the acting set stays the
    # one the standing threshold chose, which is the whole point of the question.
    savings = evaluate.net_savings(y, truth, config.DEFAULT_COSTS.threshold, costs)
    assert savings == pytest.approx(0.0, abs=1e-6)


def test_a_sharper_model_survives_worse_assumptions():
    """Break-even depends on precision and on nothing else about the block."""
    rng = np.random.default_rng(config.SEED)
    y = rng.uniform(size=4000) < 0.12
    sharp = np.where(y, 0.9, np.where(rng.uniform(size=4000) < 0.1, 0.9, 0.01))
    blunt = np.where(rng.uniform(size=4000) < 0.5, 0.9, 0.01)

    assert (
        business.break_even(y, sharp)["effectiveness"]
        < (business.break_even(y, blunt)["effectiveness"])
    )


def test_break_even_is_undefined_when_nothing_is_flagged():
    report = business.break_even(Y, np.zeros(len(Y)))

    assert np.isnan(report["effectiveness"])
    assert np.isnan(report["precision"])
