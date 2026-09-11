"""Feature engineering for the review-risk model.

Every value produced here has to survive one question: if I were scoring this
order at its decision moment, would this value already exist? docs/features.md
answers it feature by feature, and tests/test_features.py enforces it.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src import config, data

SPINE_PATH = config.DATA_INTERIM / "spine.parquet"


def first_review_per_order(reviews: pd.DataFrame) -> pd.DataFrame:
    """Keep one review per order: the earliest one by creation date.

    Keeping the worst score, or the latest, would use information from after the
    moment the model is applied. See docs/decisiones.md, D-06.

    review_id breaks ties so the result never depends on the order pandas
    happened to read the rows in.
    """
    return reviews.sort_values(["review_creation_date", "review_id"]).drop_duplicates(
        "order_id", keep="first"
    )


def handover_moment(approved: pd.Series, carrier: pd.Series) -> pd.Series:
    """t1, the handover to the carrier, never allowed to precede t0.

    1,350 orders record the handover before the approval, which reads as a
    bookkeeping artefact rather than a real sequence. D-07 clamps instead of
    dropping them. Written once because the seller history has to date its
    events exactly as the spine does.
    """
    return carrier.where(carrier >= approved, approved)


def build_spine(
    orders: pd.DataFrame,
    reviews: pd.DataFrame,
    customers: pd.DataFrame,
) -> pd.DataFrame:
    """One row per order: the population, the target and the two decision moments.

    Pure on purpose - it reads no files, so a test can call it with a handful of
    made-up rows. The exclusion counts travel back in .attrs["funnel"] rather
    than being printed, leaving the caller to decide whether to show them.
    """
    funnel = {"all orders": len(orders)}

    # An order joins the sample only if both decision moments exist. t1 is the
    # handover to the carrier, so orders cancelled or gone unavailable before
    # shipping never reach it. The 14 orders with a handover but no approval
    # timestamp go too: inventing t0 would mean inventing the moment a model
    # is supposed to run.
    df = orders[
        orders["order_approved_at"].notna() & orders["order_delivered_carrier_date"].notna()
    ]
    funnel["reached t0 and t1"] = len(df)

    review = first_review_per_order(reviews)
    df = df.merge(
        review[["order_id", "review_score", "review_creation_date"]],
        on="order_id",
        how="inner",
    )
    funnel["with a review"] = len(df)

    df = df[
        df["order_purchase_timestamp"].between(
            config.ANALYSIS_START, config.ANALYSIS_END, inclusive="left"
        )
    ]
    funnel["inside the analysis window"] = len(df)

    # customer_id is one per order; customer_unique_id is the actual person.
    df = df.merge(customers[["customer_id", "customer_unique_id"]], on="customer_id", how="left")

    approved = df["order_approved_at"]
    carrier = df["order_delivered_carrier_date"]

    spine = pd.DataFrame(
        {
            "order_id": df["order_id"],
            "customer_unique_id": df["customer_unique_id"],
            "order_purchase_timestamp": df["order_purchase_timestamp"],
            "t0": approved,
            "t1": handover_moment(approved, carrier),
            "t1_was_inverted": carrier < approved,
            "review_creation_date": df["review_creation_date"],
            "review_score": df["review_score"].astype("int8"),
            "y": df["review_score"] <= 2,
        }
    ).reset_index(drop=True)

    spine["split"] = temporal_split(spine["order_purchase_timestamp"])

    # A duplicated key here would mean a merge fanned out - silent, and fatal to
    # every count downstream.
    if spine["order_id"].duplicated().any():
        raise ValueError("build_spine produced more than one row for some order")

    spine.attrs["funnel"] = funnel
    return spine


def temporal_split(purchase: pd.Series) -> pd.Series:
    """Label each order train / val / test by its purchase date.

    Cut by date, never at random: rule 1 of CLAUDE.md. The split runs on the
    purchase timestamp rather than on t0 or t1 so that both decision moments
    land in exactly the same blocks - otherwise the two models would be judged
    on drifting populations and the comparison would stop measuring the value
    of the extra signal. See docs/decisiones.md, D-08.

    The result is an *ordered* categorical, so groupby and plots keep the blocks
    in chronological order instead of alphabetising them into test, train, val.
    """
    labels = pd.Series("train", index=purchase.index, dtype=object)
    labels[purchase >= config.VAL_START] = "val"
    labels[purchase >= config.TEST_START] = "test"
    return pd.Series(
        pd.Categorical(labels, categories=config.SPLIT_ORDER, ordered=True),
        index=purchase.index,
        name="split",
    )


def monthly_target_summary(spine: pd.DataFrame) -> pd.DataFrame:
    """Orders, positives and prevalence per calendar month.

    Deliberately free of anything post-decision: this module never touches
    order_delivered_customer_date, not even to explain the past. The delivery
    diagnosis lives in notebooks/01_eda.ipynb, where the caveat sits next to it.
    """
    month = spine["order_purchase_timestamp"].dt.to_period("M")
    summary = spine.groupby(month, observed=True).agg(
        orders=("y", "size"), positives=("y", "sum"), prevalence=("y", "mean")
    )
    summary.index.name = "month"
    return summary


def split_summary(spine: pd.DataFrame) -> pd.DataFrame:
    """Size, positives and prevalence per block - reported next to every metric.

    Prevalence drifts across the blocks because it tracks the delivery-delay
    rate, so the test figure is the one any test metric must be read against.
    """
    summary = spine.groupby("split", observed=True).agg(
        months=("order_purchase_timestamp", lambda c: c.dt.to_period("M").nunique()),
        orders=("y", "size"),
        positives=("y", "sum"),
        prevalence=("y", "mean"),
        first=("order_purchase_timestamp", "min"),
        last=("order_purchase_timestamp", "max"),
    )
    return summary


# Features by decision moment. Membership is data, not a convention kept in
# someone's head: the t0 model trains on T0_FEATURES, the t1 model on
# T0_FEATURES + T1_FEATURES, and a test refuses any column claimed by neither.
# docs/features.md carries the same table in prose.
T0_FEATURES = [
    "promised_days",
    "approval_latency_h",
    "seller_margin_days",
    "distance_km",
    "same_state",
    "n_items",
    "n_sellers",
    "n_products",
    "price_total",
    "freight_total",
    "freight_ratio",
    "max_item_price",
    "weight_g",
    "volume_cm3",
    "photos",
    "description_len",
    "installments",
    "payment_value",
    "payment_type",
    "product_category",
    "customer_state",
    "seller_state",
    # Seller history, as of t0. Backward-looking by construction; see D-10.
    "seller_prior_orders",
    "seller_tenure_days",
    "seller_prior_reviews",
    "seller_neg_rate",
    "seller_late_handover_rate",
]

T1_FEATURES = [
    "handover_days",
    "handover_vs_limit_days",
    "handover_late",
    "remaining_days_at_t1",
    "elapsed_share",
]

CATEGORICAL = ["payment_type", "product_category", "customer_state", "seller_state"]

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1, lng1, lat2, lng2):
    """Great-circle distance in km between two arrays of coordinates.

    Straight-line distance, not road distance - Brazil is wide enough that the
    two differ, but the correlation is high and no routing data ships with the
    dataset.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(np.asarray(lng2) - np.asarray(lng1))
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def zip_coordinates(geolocation: pd.DataFrame) -> pd.DataFrame:
    """One coordinate per zip-code prefix: the median of its points.

    Median rather than mean because geolocation carries badly geocoded strays
    that drag an average across the country.
    """
    return geolocation.groupby("geolocation_zip_code_prefix")[
        ["geolocation_lat", "geolocation_lng"]
    ].median()


def order_item_attributes(items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Per-order rollup of the item lines, plus the dominant seller.

    An order can span several sellers and products; the dominant seller is the
    one behind the most expensive item, as fixed in docs/decisiones.md D-06.
    """
    items = items.merge(products, on="product_id", how="left")
    items["item_volume_cm3"] = (
        items["product_length_cm"] * items["product_height_cm"] * items["product_width_cm"]
    )

    rollup = items.groupby("order_id").agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        n_products=("product_id", "nunique"),
        price_total=("price", "sum"),
        freight_total=("freight_value", "sum"),
        max_item_price=("price", "max"),
        # min_count=1 keeps an all-missing group as NaN. Plain sum() would report
        # a weightless parcel, which no model can tell from a genuine light one.
        weight_g=("product_weight_g", lambda c: c.sum(min_count=1)),
        volume_cm3=("item_volume_cm3", lambda c: c.sum(min_count=1)),
        photos=("product_photos_qty", "mean"),
        description_len=("product_description_lenght", "mean"),
        # The earliest deadline binds: the order ships when its slowest seller does.
        shipping_limit=("shipping_limit_date", "min"),
    )

    dominant = dominant_item(items)[["seller_id", "product_category_name"]].rename(
        columns={"product_category_name": "product_category"}
    )
    return rollup.join(dominant)


def dominant_item(items: pd.DataFrame) -> pd.DataFrame:
    """The item line an order is named after: the most expensive one.

    The convention of D-06, written once. Both the per-order rollup and the
    seller history call it, so an order is attributed to the same seller in
    the features and in the aggregates that feed them.
    """
    return (
        items.sort_values(["price", "order_item_id"], ascending=[False, True])
        .drop_duplicates("order_id")
        .set_index("order_id")
    )


def payment_attributes(payments: pd.DataFrame) -> pd.DataFrame:
    """Per-order payment rollup.

    An order can be split across several payment rows; payment_type is taken
    from the first sequential entry and the instalment count from the largest.
    """
    ordered = payments.sort_values(["order_id", "payment_sequential"])
    return ordered.groupby("order_id").agg(
        payment_type=("payment_type", "first"),
        installments=("payment_installments", "max"),
        payment_value=("payment_value", "sum"),
    )


FEATURES_PATH = config.DATA_PROCESSED / "features.parquet"

_DAY = pd.Timedelta(days=1)


# --- seller history: what was already known about the seller at t0 -----------

SELLER_HISTORY = [
    "seller_prior_orders",
    "seller_tenure_days",
    "seller_prior_reviews",
    "seller_neg_rate",
    "seller_late_handover_rate",
]

# Fictitious observations of the marketplace average that a seller has to
# outweigh before its own rate is believed: with m of them, a seller needs m
# observations of its own to sit halfway between the market and what it
# actually scored. Both values are estimated from the training block by
# shrinkage_from_moments, not chosen by hand. See D-10.
#
# They differ by an order of magnitude, and that is the finding rather than an
# inconsistency: lateness varies far more between sellers than the negative
# review rate does, so a handful of shipments already pins a seller down while
# a handful of reviews says almost nothing.
SHRINKAGE_NEG = 25.0
SHRINKAGE_LATE = 3.0

# review_creation_date has day granularity: a review "created" at midnight would
# otherwise count for an order scored at noon that same day. When the only
# granularity available is the day, round in the direction that takes
# information away rather than the one that hands it over.
LABEL_DELAY = pd.Timedelta(days=1)


def shrinkage_from_moments(successes: pd.Series, trials: pd.Series) -> float:
    """How much to shrink a rate, estimated rather than picked by hand.

    The spread of the rates one actually observes is two things added together:
    the real spread between sellers, and the coin-flip noise of estimating a
    rate from few trials. Subtract the noise, and what is left is how much
    sellers genuinely differ - which is exactly what decides how much evidence
    of its own a seller needs before its rate is believed.

    Sellers that really differ a lot need little shrinkage; sellers that are all
    much alike need a lot, because then any deviation is probably noise. The
    method-of-moments estimate of the Beta-binomial prior strength says it in
    one line: m = p(1-p) / var(true) - 1.

    Returns infinity when the observed spread is entirely explained by noise:
    no seller has earned its own rate, so every one of them should fall back to
    the market. Callers cap it; the value is a diagnostic, not a constant to
    wire in unexamined.
    """
    p = successes.sum() / trials.sum()
    noise = (p * (1 - p) / trials).mean()
    between = (successes / trials).var() - noise
    if between <= 0:
        return float("inf")
    return float(p * (1 - p) / between - 1)


def seller_history_base(tables: data.OlistTables) -> pd.DataFrame:
    """One row per shipped order: what it contributes to its seller's history.

    Built from every order that reached the handover, not from the spine. The
    analysis population leaves out 1,022 of them - the 2016 pilot, orders whose
    review never arrived, the ones outside the window - but the seller really
    did ship those and their reviews really did exist. Dropping them would
    confuse the population criterion with what was knowable at the time, and
    would leave every seller active in January 2017 with no history at all.
    """
    orders = tables.orders
    shipped = orders[
        orders["order_approved_at"].notna() & orders["order_delivered_carrier_date"].notna()
    ]
    items = tables.order_items
    seller = dominant_item(items)["seller_id"]
    # The earliest deadline binds, exactly as in the per-order rollup.
    limit = items.groupby("order_id")["shipping_limit_date"].min()
    review = first_review_per_order(tables.order_reviews)

    df = shipped[["order_id", "order_approved_at", "order_delivered_carrier_date"]].copy()
    df["seller_id"] = df["order_id"].map(seller)
    df["shipping_limit"] = df["order_id"].map(limit)
    df = df.merge(
        review[["order_id", "review_creation_date", "review_score"]], on="order_id", how="left"
    )

    shipped_at = handover_moment(df["order_approved_at"], df["order_delivered_carrier_date"])
    limit_known = df["shipping_limit"].notna()
    reviewed = df["review_creation_date"].notna()

    # A review cannot be known before the parcel even left: 317 orders carry a
    # review dated at or before their own handover, 30 of them before their own
    # t0. Read literally, those dates would let an order's own outcome into its
    # own aggregate - the one thing this whole construction exists to prevent.
    created = df["review_creation_date"]
    known_at = created.where(created >= shipped_at, shipped_at) + LABEL_DELAY

    return pd.DataFrame(
        {
            "order_id": df["order_id"],
            "seller_id": df["seller_id"],
            "shipped_at": shipped_at,
            "late": (shipped_at > df["shipping_limit"]) & limit_known,
            "limit_known": limit_known,
            "label_known_at": known_at.where(reviewed),
            "y": (df["review_score"] <= 2).astype(float).where(reviewed),
        }
    ).dropna(subset=["seller_id"])


def seller_history_state(base: pd.DataFrame) -> pd.DataFrame:
    """The event table: two rows per order, one per clock, with running totals.

    A past order hands over its information at two different moments, and rule 3
    of CLAUDE.md only covers the first:

    - the operational clock, at the handover - the seller shipped, on time or
      late. Knowable the instant it happens, no review involved.
    - the label clock, a median of 10 days later - the customer wrote a review.

    Splitting them is the point. Counting an order as shipped the moment it
    ships is correct; counting its star rating before the customer has written
    it is using the future.
    """
    shipments = pd.DataFrame(
        {
            "seller_id": base["seller_id"],
            "time": base["shipped_at"],
            "shipment_time": base["shipped_at"],
            "d_orders": 1.0,
            "d_late": base["late"].astype(float),
            "d_limit_known": base["limit_known"].astype(float),
            "d_reviews": 0.0,
            "d_negatives": 0.0,
        }
    )
    labelled = base[base["label_known_at"].notna()]
    labels = pd.DataFrame(
        {
            "seller_id": labelled["seller_id"],
            "time": labelled["label_known_at"],
            "d_orders": 0.0,
            "d_late": 0.0,
            "d_limit_known": 0.0,
            "d_reviews": 1.0,
            "d_negatives": labelled["y"],
        }
    )

    events = pd.concat([shipments, labels], ignore_index=True).sort_values("time", kind="stable")
    deltas = ["d_orders", "d_late", "d_limit_known", "d_reviews", "d_negatives"]

    seller = events.groupby("seller_id", sort=False)[deltas].cumsum()
    events["seller_prior_orders"] = seller["d_orders"]
    events["seller_late_orders"] = seller["d_late"]
    events["seller_limit_known"] = seller["d_limit_known"]
    events["seller_prior_reviews"] = seller["d_reviews"]
    events["seller_negatives"] = seller["d_negatives"]
    # cummin leaves NaT on the label rows, since those carry no shipment time.
    # Carrying the value forward is what makes the answer "the seller's first
    # shipment so far" instead of "so far, if the last event happened to be one".
    by_seller = events.groupby("seller_id", sort=False)["shipment_time"]
    events["seller_first_shipment"] = by_seller.cummin().groupby(events["seller_id"]).ffill()

    # The marketplace running totals, over every seller, for the shrinkage target.
    market = events[deltas].cumsum()
    events["market_late"] = market["d_late"]
    events["market_limit_known"] = market["d_limit_known"]
    events["market_reviews"] = market["d_reviews"]
    events["market_negatives"] = market["d_negatives"]

    return events.drop(columns=[*deltas, "shipment_time"])


def seller_history(spine: pd.DataFrame, tables: data.OlistTables) -> pd.DataFrame:
    """Seller aggregates as of t0, one row per spine order, in spine order.

    Every value answers the same question: scoring this order at t0, what did
    the marketplace already know about this seller? All five land in
    T0_FEATURES - the few days between t0 and t1 add too little history to be
    worth a second version, and keeping them at t0 leaves the t0-versus-t1
    comparison measuring the shipping signal alone.
    """
    state = seller_history_state(seller_history_base(tables))
    seller_columns = [c for c in state.columns if c.startswith("seller_")]
    market_columns = [c for c in state.columns if c.startswith("market_")]

    query = spine[["order_id", "t0"]].copy()
    query["seller_id"] = query["order_id"].map(dominant_item(tables.order_items)["seller_id"])
    query = query.sort_values("t0", kind="stable")

    # merge_asof is a join on "the last row before this one in time", the sorted
    # equivalent of a LEFT JOIN ... ON time < t0. allow_exact_matches=False is
    # what keeps an order out of its own aggregate: its own shipment event sits
    # at t1, which is never earlier than its t0, so an exact hit is itself.
    asof = dict(left_on="t0", right_on="time", direction="backward", allow_exact_matches=False)
    seen = pd.merge_asof(query, state[["time", *seller_columns]], by="seller_id", **asof)
    market = pd.merge_asof(query, state[["time", *market_columns]], **asof)

    # A seller with no history yet has zero prior orders, not an unknown number.
    prior_orders = seen["seller_prior_orders"].fillna(0.0)
    prior_reviews = seen["seller_prior_reviews"].fillna(0.0)
    negatives = seen["seller_negatives"].fillna(0.0)
    late = seen["seller_late_orders"].fillna(0.0)
    limit_known = seen["seller_limit_known"].fillna(0.0)

    # The shrinkage target is backward-looking too: the marketplace rate as it
    # stood at t0, never the rate of the training block. The second would walk
    # the future in through the back door, in the one place nobody inspects.
    market_negative = market["market_negatives"] / market["market_reviews"].replace(0.0, np.nan)
    market_late = market["market_late"] / market["market_limit_known"].replace(0.0, np.nan)

    history = pd.DataFrame(
        {
            "order_id": seen["order_id"],
            "seller_prior_orders": prior_orders,
            # NaT for a seller's first order: it has no tenure yet, and zero
            # would claim it opened today.
            "seller_tenure_days": (seen["t0"] - seen["seller_first_shipment"]) / _DAY,
            "seller_prior_reviews": prior_reviews,
            # With no reviews of its own a seller lands exactly on the market
            # rate, which is the honest answer - not a 0% negative rate.
            "seller_neg_rate": (negatives + SHRINKAGE_NEG * market_negative)
            / (prior_reviews + SHRINKAGE_NEG),
            "seller_late_handover_rate": (late + SHRINKAGE_LATE * market_late)
            / (limit_known + SHRINKAGE_LATE),
        }
    )
    return history.set_index("order_id").reindex(spine["order_id"]).reset_index()


def build_features(spine: pd.DataFrame, tables: data.OlistTables) -> pd.DataFrame:
    """Attach every candidate feature to the spine, one row per order.

    Both decision moments are built in one pass and kept in one frame; what
    separates them is T0_FEATURES and T1_FEATURES, not two pipelines that could
    drift apart. Nothing here reads order_delivered_customer_date - the actual
    delivery date exists at neither moment.

    Missing values are left missing. LightGBM handles them natively and a
    logistic pipeline needs its own imputation; deciding it here would bury a
    modelling choice inside the feature code.
    """
    orders = tables.orders
    items = order_item_attributes(tables.order_items, tables.products)
    payments = payment_attributes(tables.order_payments)
    coordinates = zip_coordinates(tables.geolocation)

    customers = tables.customers.merge(
        coordinates, left_on="customer_zip_code_prefix", right_index=True, how="left"
    ).rename(columns={"geolocation_lat": "customer_lat", "geolocation_lng": "customer_lng"})
    sellers = tables.sellers.merge(
        coordinates, left_on="seller_zip_code_prefix", right_index=True, how="left"
    ).rename(columns={"geolocation_lat": "seller_lat", "geolocation_lng": "seller_lng"})

    df = (
        spine.merge(
            orders[["order_id", "customer_id", "order_estimated_delivery_date"]],
            on="order_id",
            how="left",
        )
        .merge(items, on="order_id", how="left")
        .merge(payments, on="order_id", how="left")
        .merge(
            customers[["customer_id", "customer_state", "customer_lat", "customer_lng"]],
            on="customer_id",
            how="left",
        )
        .merge(
            sellers[["seller_id", "seller_state", "seller_lat", "seller_lng"]],
            on="seller_id",
            how="left",
        )
    ).merge(seller_history(spine, tables), on="order_id", how="left")
    if len(df) != len(spine):
        raise ValueError(f"feature join changed the row count: {len(spine)} -> {len(df)}")

    purchase = df["order_purchase_timestamp"]
    promised = df["order_estimated_delivery_date"]

    # --- t0: everything known once the payment is approved -------------------
    df["promised_days"] = (promised - purchase) / _DAY
    df["approval_latency_h"] = (df["t0"] - purchase) / pd.Timedelta(hours=1)
    df["seller_margin_days"] = (df["shipping_limit"] - df["t0"]) / _DAY
    df["distance_km"] = haversine_km(
        df["customer_lat"], df["customer_lng"], df["seller_lat"], df["seller_lng"]
    )
    df["same_state"] = df["customer_state"] == df["seller_state"]
    # Guarding against division by zero, not imputing: a zero-priced order has
    # no freight ratio to speak of.
    df["freight_ratio"] = df["freight_total"] / df["price_total"].replace(0, np.nan)

    # --- t1: what the handover to the carrier adds ---------------------------
    df["handover_days"] = (df["t1"] - df["t0"]) / _DAY
    df["handover_vs_limit_days"] = (df["t1"] - df["shipping_limit"]) / _DAY
    df["handover_late"] = df["handover_vs_limit_days"] > 0
    df["remaining_days_at_t1"] = (promised - df["t1"]) / _DAY
    df["elapsed_share"] = (df["t1"] - purchase) / (promised - purchase)

    for column in CATEGORICAL:
        df[column] = df[column].astype("category")

    keep = list(spine.columns) + T0_FEATURES + T1_FEATURES
    return df[keep]


def feature_columns(moment: str) -> list[str]:
    """Columns a model is allowed to see at a given decision moment."""
    if moment == "t0":
        return list(T0_FEATURES)
    if moment == "t1":
        return T0_FEATURES + T1_FEATURES
    raise ValueError(f"unknown moment {moment!r}; expected 't0' or 't1'")


def write_features() -> Path:
    """Build the feature matrix from data/raw and store it in data/processed."""
    features = build_features(load_spine(), data.load_all())
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)

    print(f"\nfeatures: {len(features):,} rows x {len(T0_FEATURES) + len(T1_FEATURES)} features")
    print(f"  {'t0':4s} {len(T0_FEATURES):>2} columns")
    print(f"  {'t1':4s} {len(T1_FEATURES):>2} more, {len(feature_columns('t1')):>2} in total")

    missing = features[T0_FEATURES + T1_FEATURES].isna().mean()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing):
        print("\nmissing values (left unimputed on purpose)")
        for column, share in missing.items():
            print(f"  {column:24s} {share:>7.2%}")
    print(f"\nwrote {FEATURES_PATH.relative_to(config.ROOT)}")
    return FEATURES_PATH


def write_spine() -> Path:
    """Build the spine from data/raw and store it in data/interim."""
    spine = build_spine(
        data.load_table("orders"),
        data.load_table("order_reviews"),
        data.load_table("customers"),
    )

    config.DATA_INTERIM.mkdir(parents=True, exist_ok=True)
    spine.to_parquet(SPINE_PATH, index=False)

    print("exclusion funnel")
    for step, count in spine.attrs["funnel"].items():
        print(f"  {step:30s} {count:>8,}")
    print()
    print(f"  {'prevalence (y = score <= 2)':30s} {spine['y'].mean():>8.2%}")
    print(f"  {'t1 clamped to t0':30s} {spine['t1_was_inverted'].sum():>8,}")

    print("\ntemporal split")
    summary = split_summary(spine)
    for block, row in summary.iterrows():
        print(
            f"  {block:6s} {row['months']:>2} months  {row['orders']:>7,} orders  "
            f"{row['positives']:>6,} positives  {row['prevalence']:>6.2%}  "
            f"{row['first']:%Y-%m} .. {row['last']:%Y-%m}"
        )
    print(f"\nwrote {SPINE_PATH.relative_to(config.ROOT)}")
    return SPINE_PATH


def load_features() -> pd.DataFrame:
    """Read the feature matrix back, building it first if it is not there yet."""
    if not FEATURES_PATH.exists():
        write_features()
    return pd.read_parquet(FEATURES_PATH)


def load_spine() -> pd.DataFrame:
    """Read the spine back, building it first if it is not there yet."""
    if not SPINE_PATH.exists():
        write_spine()
    return pd.read_parquet(SPINE_PATH)


def main() -> None:
    """Run with `make features` or `python -m src.features`."""
    write_spine()
    write_features()


if __name__ == "__main__":
    main()
