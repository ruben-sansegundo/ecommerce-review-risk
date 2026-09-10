"""Tests for the spine: population, target and the two decision moments.

The frames here are tiny and invented. build_spine is a pure function, so none
of these tests touch data/raw - they check the rules, not the dataset.
"""

import pandas as pd
import pytest

from src.features import build_spine, first_review_per_order

ORDER_DATES = ["order_purchase_timestamp", "order_approved_at", "order_delivered_carrier_date"]


def order(
    order_id="o1",
    customer_id=None,
    purchase="2017-06-01 10:00",
    approved="2017-06-01 12:00",
    carrier="2017-06-03 09:00",
):
    """One plausible order, inside the analysis window and correctly sequenced."""
    return {
        "order_id": order_id,
        "customer_id": customer_id or f"c-{order_id}",
        "order_purchase_timestamp": purchase,
        "order_approved_at": approved,
        "order_delivered_carrier_date": carrier,
    }


def review(review_id="r1", order_id="o1", created="2017-06-12", score=5):
    return {
        "review_id": review_id,
        "order_id": order_id,
        "review_score": score,
        "review_creation_date": created,
    }


def orders_frame(records):
    df = pd.DataFrame(records)
    for column in ORDER_DATES:
        df[column] = pd.to_datetime(df[column])
    return df


def reviews_frame(records):
    df = pd.DataFrame(records)
    df["review_creation_date"] = pd.to_datetime(df["review_creation_date"])
    return df


def customers_frame(orders):
    """One customer row per order, as the real table has."""
    return pd.DataFrame(
        {
            "customer_id": orders["customer_id"],
            "customer_unique_id": ["person-" + c for c in orders["customer_id"]],
        }
    )


def spine_of(order_records, review_records):
    orders = orders_frame(order_records)
    return build_spine(orders, reviews_frame(review_records), customers_frame(orders))


def test_keeps_the_earliest_review_not_the_worst():
    """The 1-star review arrived later, so it is information from after t1."""
    reviews = reviews_frame(
        [
            review(review_id="r2", created="2017-06-20", score=1),
            review(review_id="r1", created="2017-06-12", score=4),
        ]
    )
    kept = first_review_per_order(reviews)
    assert list(kept["review_id"]) == ["r1"]
    assert kept["review_score"].item() == 4


def test_ties_are_broken_by_review_id_not_by_row_order():
    records = [
        review(review_id="r-b", created="2017-06-12", score=2),
        review(review_id="r-a", created="2017-06-12", score=5),
    ]
    forwards = first_review_per_order(reviews_frame(records))
    backwards = first_review_per_order(reviews_frame(records[::-1]))
    assert forwards["review_id"].item() == backwards["review_id"].item() == "r-a"


def test_t1_is_the_carrier_handover_when_the_sequence_is_normal():
    spine = spine_of([order()], [review()])
    assert spine["t1"].item() == pd.Timestamp("2017-06-03 09:00")
    assert not spine["t1_was_inverted"].item()


def test_t1_is_clamped_and_flagged_when_the_handover_precedes_approval():
    """A bookkeeping artefact, kept rather than dropped - but never before t0."""
    spine = spine_of(
        [order(approved="2017-06-02 12:00", carrier="2017-06-01 09:00")],
        [review()],
    )
    assert spine["t1"].item() == spine["t0"].item() == pd.Timestamp("2017-06-02 12:00")
    assert spine["t1_was_inverted"].item()


@pytest.mark.parametrize("missing", ["order_approved_at", "order_delivered_carrier_date"])
def test_orders_that_never_reached_a_decision_moment_are_excluded(missing):
    record = order()
    record[missing] = None
    spine = spine_of(
        [record, order(order_id="o2")], [review(), review(review_id="r2", order_id="o2")]
    )
    assert list(spine["order_id"]) == ["o2"]


def test_orders_without_a_review_are_excluded():
    spine = spine_of([order(), order(order_id="o2")], [review(order_id="o2")])
    assert list(spine["order_id"]) == ["o2"]


@pytest.mark.parametrize("purchase", ["2016-12-31 23:59", "2018-09-01 00:00"])
def test_orders_outside_the_analysis_window_are_excluded(purchase):
    """Both ends of the window: 2016 is a pilot, September 2018 a truncated export."""
    spine = spine_of(
        [order(purchase=purchase, approved=purchase, carrier=purchase), order(order_id="o2")],
        [review(), review(review_id="r2", order_id="o2")],
    )
    assert list(spine["order_id"]) == ["o2"]


@pytest.mark.parametrize(
    ("score", "expected"),
    [(1, True), (2, True), (3, False), (4, False), (5, False)],
)
def test_target_is_true_only_at_one_and_two_stars(score, expected):
    spine = spine_of([order()], [review(score=score)])
    assert spine["y"].item() is expected


def test_the_spine_carries_no_post_decision_column():
    """Leakage guard.

    order_delivered_customer_date predicts the target almost perfectly and
    exists in the same source table, one column away from the ones we do keep.
    Pinning the column list makes adding it a test failure rather than a
    surprisingly good model.
    """
    spine = spine_of([order()], [review()])
    assert set(spine.columns) == {
        "order_id",
        "customer_unique_id",
        "order_purchase_timestamp",
        "t0",
        "t1",
        "t1_was_inverted",
        "review_creation_date",
        "review_score",
        "y",
    }


def test_a_fanned_out_merge_is_caught():
    """A duplicated customer row silently doubles the order; the guard must fire."""
    orders = orders_frame([order()])
    customers = pd.concat([customers_frame(orders)] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="more than one row"):
        build_spine(orders, reviews_frame([review()]), customers)


def test_the_funnel_reports_every_exclusion_step():
    spine = spine_of(
        [
            order(),
            order(order_id="o2", carrier=None),
            order(
                order_id="o3",
                purchase="2016-05-01 10:00",
                approved="2016-05-01 12:00",
                carrier="2016-05-03 09:00",
            ),
        ],
        [review(), review(review_id="r3", order_id="o3")],
    )
    assert spine.attrs["funnel"] == {
        "all orders": 3,
        "reached t0 and t1": 2,  # o2 never shipped
        "with a review": 2,
        "inside the analysis window": 1,  # o3 predates the window
    }
