"""Tests for the spine: population, target and the two decision moments.

The frames here are tiny and invented. build_spine is a pure function, so none
of these tests touch data/raw - they check the rules, not the dataset.
"""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import OlistTables
from src.features import (
    T0_FEATURES,
    T1_FEATURES,
    build_features,
    build_spine,
    feature_columns,
    first_review_per_order,
    haversine_km,
    order_item_attributes,
    seller_history,
    shrinkage_from_moments,
    temporal_split,
)

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
        "split",
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


@pytest.mark.parametrize(
    ("purchase", "expected"),
    [
        ("2017-01-01 00:00", "train"),
        ("2018-03-31 23:59", "train"),  # last instant before the validation cut
        ("2018-04-01 00:00", "val"),  # the cut itself belongs to the later block
        ("2018-05-31 23:59", "val"),
        ("2018-06-01 00:00", "test"),
        ("2018-08-31 23:59", "test"),
    ],
)
def test_split_boundaries_are_left_closed(purchase, expected):
    labels = temporal_split(pd.Series([pd.Timestamp(purchase)]))
    assert labels.item() == expected


def test_split_is_an_ordered_categorical():
    """Otherwise groupby and plots alphabetise the blocks into test, train, val."""
    labels = temporal_split(pd.Series(pd.to_datetime(["2018-07-01", "2017-05-01"])))
    assert labels.cat.ordered
    assert list(labels.cat.categories) == config.SPLIT_ORDER


def test_the_blocks_never_overlap_in_time():
    """The guard against a random split sneaking back in.

    Rule 1 of CLAUDE.md: train on the older months, evaluate on the recent ones.
    Any shuffling would put a late purchase in train or an early one in test,
    and these inequalities would stop holding.
    """
    dates = pd.Series(pd.date_range("2017-01-15", "2018-08-15", freq="17D"))
    frame = pd.DataFrame({"purchase": dates.sample(frac=1, random_state=config.SEED)})
    frame["split"] = temporal_split(frame["purchase"])

    latest = frame.groupby("split", observed=True)["purchase"].max()
    earliest = frame.groupby("split", observed=True)["purchase"].min()
    assert latest["train"] < earliest["val"]
    assert latest["val"] < earliest["test"]


def test_the_spine_carries_its_split():
    """Persisted with the data, so no two scripts can derive different blocks."""
    spine = spine_of(
        [
            order(),
            order(
                order_id="o2",
                purchase="2018-07-01 10:00",
                approved="2018-07-01 12:00",
                carrier="2018-07-03 09:00",
            ),
        ],
        [review(), review(review_id="r2", order_id="o2", created="2018-07-12")],
    )
    assert dict(zip(spine["order_id"], spine["split"], strict=True)) == {
        "o1": "train",
        "o2": "test",
    }


# --- feature building -------------------------------------------------------


def toy_tables(orders_frame_, reviews_frame_):
    """The nine Olist tables, shrunk to what two orders need.

    Written out rather than sampled from data/raw so the tests keep running
    without the dataset, and so each value is chosen to make one case obvious.
    """
    orders = orders_frame_.copy()
    orders["order_estimated_delivery_date"] = pd.to_datetime("2017-06-20")
    # Deliberately absurd: nothing downstream may depend on this column.
    orders["order_delivered_customer_date"] = pd.to_datetime("2017-06-15")

    items = pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "order_item_id": [1, 2],
            "product_id": ["p-cheap", "p-dear"],
            "seller_id": ["s1", "s2"],
            "shipping_limit_date": pd.to_datetime(["2017-06-05", "2017-06-05"]),
            "price": [10.0, 90.0],
            "freight_value": [5.0, 15.0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p-cheap", "p-dear"],
            "product_category_name": ["livros", "moveis"],
            "product_name_lenght": [40, 50],
            "product_description_lenght": [200, 400],
            "product_photos_qty": [1, 3],
            "product_weight_g": [100.0, 900.0],
            "product_length_cm": [10.0, 20.0],
            "product_height_cm": [10.0, 20.0],
            "product_width_cm": [10.0, 20.0],
        }
    )
    payments = pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "payment_sequential": [1, 2],
            "payment_type": ["credit_card", "voucher"],
            "payment_installments": [3, 1],
            "payment_value": [100.0, 20.0],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": orders["customer_id"],
            "customer_unique_id": ["person-" + c for c in orders["customer_id"]],
            "customer_zip_code_prefix": [1000] * len(orders),
            "customer_city": ["sao paulo"] * len(orders),
            "customer_state": ["SP"] * len(orders),
        }
    )
    sellers = pd.DataFrame(
        {
            "seller_id": ["s1", "s2"],
            "seller_zip_code_prefix": [2000, 3000],
            "seller_city": ["sao paulo", "rio"],
            "seller_state": ["SP", "RJ"],
        }
    )
    geolocation = pd.DataFrame(
        {
            "geolocation_zip_code_prefix": [1000, 1000, 1000, 2000, 3000],
            # Prefix 1000 carries three points, one of them a bad geocode in the
            # northern hemisphere. The median must ignore it; a mean would not.
            "geolocation_lat": [-23.55, -23.55, 40.0, -23.50, -22.91],
            "geolocation_lng": [-46.63, -46.63, 10.0, -46.60, -43.20],
            "geolocation_city": ["sao paulo", "sao paulo", "nowhere", "sao paulo", "rio"],
            "geolocation_state": ["SP", "SP", "XX", "SP", "RJ"],
        }
    )
    return OlistTables(
        orders=orders,
        order_items=items,
        order_payments=payments,
        order_reviews=reviews_frame_,
        products=products,
        sellers=sellers,
        customers=customers,
        geolocation=geolocation,
        product_category_name_translation=pd.DataFrame(
            columns=["product_category_name", "product_category_name_english"]
        ),
    )


def toy_spine_and_tables():
    orders = orders_frame([order()])
    reviews = reviews_frame([review()])
    tables = toy_tables(orders, reviews)
    spine = build_spine(tables.orders, tables.order_reviews, tables.customers)
    return spine, tables


def test_build_features_ignores_the_actual_delivery_date():
    """The leakage proof, not the leakage promise.

    order_delivered_customer_date exists at neither t0 nor t1 and predicts the
    target almost perfectly. Poisoning it must leave every feature untouched -
    if any code path ever reads it, this comparison stops matching.
    """
    spine, tables = toy_spine_and_tables()
    honest = build_features(spine, tables)

    poisoned = tables.orders.copy()
    poisoned["order_delivered_customer_date"] = pd.to_datetime("1999-01-01")
    features = build_features(spine, OlistTables(**{**tables.__dict__, "orders": poisoned}))

    pd.testing.assert_frame_equal(honest, features)


def test_every_built_column_is_declared_at_one_moment():
    """A feature that belongs to neither list is a feature nobody has vetted."""
    spine, tables = toy_spine_and_tables()
    built = build_features(spine, tables)
    assert set(built.columns) == set(spine.columns) | set(T0_FEATURES) | set(T1_FEATURES)


def test_the_two_moments_claim_disjoint_features():
    assert not set(T0_FEATURES) & set(T1_FEATURES)
    assert len(set(T0_FEATURES)) == len(T0_FEATURES)
    assert feature_columns("t1") == T0_FEATURES + T1_FEATURES
    assert feature_columns("t0") == T0_FEATURES


def test_an_unknown_moment_is_refused():
    with pytest.raises(ValueError, match="unknown moment"):
        feature_columns("t2")


def test_t1_features_are_absent_from_the_t0_column_list():
    """The t0 model must not see the carrier handover, however convenient."""
    for column in ["handover_days", "handover_vs_limit_days", "remaining_days_at_t1"]:
        assert column not in feature_columns("t0")


def test_haversine_matches_a_known_distance():
    """Sao Paulo to Rio de Janeiro is about 357 km in a straight line."""
    km = haversine_km(-23.55, -46.63, -22.91, -43.20)
    assert 350 < km < 365


def test_zip_coordinates_ignore_a_bad_geocode():
    """Prefix 1000 has one stray point in the northern hemisphere."""
    spine, tables = toy_spine_and_tables()
    built = build_features(spine, tables)
    # Sao Paulo to Rio, not Sao Paulo to a stray point in the Mediterranean.
    assert 350 < built["distance_km"].item() < 365
    assert not built["same_state"].item()


def test_an_all_missing_weight_stays_missing():
    """groupby.sum() reports 0 for an all-NaN group: a weightless parcel."""
    items = pd.DataFrame(
        {
            "order_id": ["o1"],
            "order_item_id": [1],
            "product_id": ["p1"],
            "seller_id": ["s1"],
            "shipping_limit_date": pd.to_datetime(["2017-06-05"]),
            "price": [10.0],
            "freight_value": [5.0],
        }
    )
    products = pd.DataFrame(
        {
            "product_id": ["p1"],
            "product_category_name": ["livros"],
            "product_description_lenght": [100],
            "product_photos_qty": [1],
            "product_weight_g": [float("nan")],
            "product_length_cm": [10.0],
            "product_height_cm": [10.0],
            "product_width_cm": [10.0],
        }
    )
    assert pd.isna(order_item_attributes(items, products)["weight_g"].item())


def test_the_dominant_seller_is_the_one_behind_the_dearest_item():
    """Multi-seller orders are common; D-06 picks the seller by item value."""
    spine, tables = toy_spine_and_tables()
    built = build_features(spine, tables)
    assert built["seller_state"].item() == "RJ"  # s2, in Rio, sold the 90.0 item
    assert built["product_category"].item() == "moveis"
    assert built["n_sellers"].item() == 2


# --- seller history ---------------------------------------------------------
#
# One market (seller "sm", two orders reviewed in January) and one seller under
# test ("s"), whose orders are timed so that every clock matters:
#
#   o-a  ships 03-03, LATE, its 1-star review is written 03-15
#   o-d  ships 03-06, on time, its 5-star review is written 03-20
#   o-b  scored at t0 = 03-10  <-- here, o-a and o-d have shipped but neither
#                                  review exists yet
#   o-c  scored at t0 = 04-01  <-- here, all three reviews exist
#
# Every expected value below is worked out by hand from that timeline.


def history_rows(order_id, seller, approved, carrier, limit, review_at, score):
    """One order, as the three raw tables see it."""
    return (
        {
            "order_id": order_id,
            "customer_id": f"c-{order_id}",
            "order_purchase_timestamp": approved,
            "order_approved_at": approved,
            "order_delivered_carrier_date": carrier,
        },
        {
            "order_id": order_id,
            "order_item_id": 1,
            "product_id": "p",
            "seller_id": seller,
            "shipping_limit_date": limit,
            "price": 100.0,
            "freight_value": 10.0,
        },
        {
            "review_id": f"r-{order_id}",
            "order_id": order_id,
            "review_score": score,
            "review_creation_date": review_at,
        },
    )


TIMELINE = [
    history_rows("o-m1", "sm", "2017-01-01", "2017-01-02", "2017-01-10", "2017-01-05", 1),
    history_rows("o-m2", "sm", "2017-01-01", "2017-01-02", "2017-01-10", "2017-01-05", 5),
    history_rows("o-a", "s", "2017-03-01", "2017-03-03", "2017-03-02", "2017-03-15", 1),
    history_rows("o-d", "s", "2017-03-05", "2017-03-06", "2017-03-10", "2017-03-20", 5),
    history_rows("o-b", "s", "2017-03-10", "2017-03-11", "2017-03-20", "2017-03-25", 5),
    history_rows("o-c", "s", "2017-04-01", "2017-04-02", "2017-04-10", "2017-04-20", 5),
]


def history_tables(timeline=TIMELINE):
    orders = orders_frame([row[0] for row in timeline])
    items = pd.DataFrame([row[1] for row in timeline])
    items["shipping_limit_date"] = pd.to_datetime(items["shipping_limit_date"])
    reviews = reviews_frame([row[2] for row in timeline])
    empty = pd.DataFrame()
    return OlistTables(
        orders=orders,
        order_items=items,
        order_reviews=reviews,
        customers=customers_frame(orders),
        order_payments=empty,
        products=empty,
        sellers=empty,
        geolocation=empty,
        product_category_name_translation=empty,
    )


def history_of(tables=None):
    tables = tables or history_tables()
    spine = build_spine(tables.orders, tables.order_reviews, tables.customers)
    return seller_history(spine, tables).set_index("order_id")


def test_an_order_never_enters_its_own_seller_history():
    """The first order of a seller has to see an empty history, not itself."""
    row = history_of().loc["o-a"]
    assert row["seller_prior_orders"] == 0
    assert row["seller_prior_reviews"] == 0
    assert pd.isna(row["seller_tenure_days"])


def test_a_prior_order_counts_as_shipped_but_not_yet_as_reviewed():
    """The double clock, which rule 3 of CLAUDE.md does not cover on its own.

    At t0 of o-b the seller has shipped twice, and that is knowable. Neither
    review has been written yet, and using them would be reading the future.
    """
    row = history_of().loc["o-b"]
    assert row["seller_prior_orders"] == 2
    assert row["seller_prior_reviews"] == 0
    assert row["seller_tenure_days"] == pytest.approx(7.0)


def test_a_label_joins_the_history_once_its_review_exists():
    """By o-c the same three orders have all been reviewed, one of them badly."""
    row = history_of().loc["o-c"]
    assert row["seller_prior_orders"] == 3
    assert row["seller_prior_reviews"] == 3
    # Market rate at t0 of o-c: 2 negatives out of 5 known reviews.
    assert row["seller_neg_rate"] == pytest.approx((1 + 25 * 0.4) / (3 + 25))


def test_a_seller_without_known_labels_falls_back_to_the_market():
    """Not a 0% negative rate: an unknown rate, answered with the market's."""
    rows = history_of()
    market_rate = 1 / 2  # o-m1 and o-m2, one of them negative, both known by 01-06
    assert rows.loc["o-a", "seller_neg_rate"] == pytest.approx(market_rate)
    assert rows.loc["o-b", "seller_neg_rate"] == pytest.approx(market_rate)


def test_the_late_handover_rate_runs_on_the_operational_clock_alone():
    """It needs no review at all: a late shipment is known the day it ships."""
    row = history_of().loc["o-b"]
    # Seller: 1 late of 2 shipped. Market at 03-10: 1 late of 4 with a deadline.
    assert row["seller_late_handover_rate"] == pytest.approx((1 + 3 * 0.25) / (2 + 3))


def test_poisoning_reviews_that_are_still_in_the_future_changes_nothing():
    """The leakage proof of S2, aimed at the aggregates instead of the matrix.

    Every review written after t0 of o-b is flipped to one star. Those reviews
    do not exist at that moment, so not one of o-b's aggregates may move.
    """
    honest = history_of()

    tables = history_tables()
    reviews = tables.order_reviews.copy()
    future = reviews["review_creation_date"] > pd.Timestamp("2017-03-10")
    reviews.loc[future, "review_score"] = 1
    poisoned = history_of(OlistTables(**{**tables.__dict__, "order_reviews": reviews}))

    assert future.sum() == 4  # o-a, o-d, o-b and o-c all sit in the future here
    pd.testing.assert_series_equal(honest.loc["o-b"], poisoned.loc["o-b"])
    # And the order that can see those labels does move, or the test proves
    # nothing: a comparison that passes when the code is broken is decoration.
    assert honest.loc["o-c", "seller_neg_rate"] != poisoned.loc["o-c", "seller_neg_rate"]


def test_the_market_prior_looks_backwards_too():
    """A late burst of negatives must not touch what came before it."""
    honest = history_of()

    late_burst = [
        history_rows(f"o-z{i}", "sz", "2018-01-01", "2018-01-02", "2018-01-10", "2018-01-05", 1)
        for i in range(20)
    ]
    flooded = history_of(history_tables(TIMELINE + late_burst))

    pd.testing.assert_frame_equal(honest, flooded.loc[honest.index])


def test_shrinkage_from_moments_recovers_a_known_prior():
    """Beta-binomial sellers with a prior strength of 30, recovered from counts."""
    rng = np.random.default_rng(config.SEED)
    strength, mean = 30.0, 0.15
    rates = rng.beta(mean * strength, (1 - mean) * strength, size=4000)
    trials = pd.Series(rng.integers(20, 200, size=4000))
    successes = pd.Series(rng.binomial(trials, rates))

    assert shrinkage_from_moments(successes, trials) == pytest.approx(strength, rel=0.15)


def test_shrinkage_is_enormous_when_every_seller_is_really_the_same():
    """Sellers drawn from one single rate: nothing to tell apart, so shrink hard."""
    rng = np.random.default_rng(config.SEED)
    trials = pd.Series(np.full(2000, 50))
    successes = pd.Series(rng.binomial(50, 0.15, size=2000))

    assert shrinkage_from_moments(successes, trials) > 1_000


def test_shrinkage_is_infinite_when_the_spread_is_all_noise():
    """Rates too alike even for coin flips: the estimate has nothing left to divide by."""
    trials = pd.Series(np.full(500, 50))
    successes = pd.Series(np.full(500, 7))

    assert shrinkage_from_moments(successes, trials) == float("inf")
