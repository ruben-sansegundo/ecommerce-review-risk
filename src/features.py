"""Feature engineering for the review-risk model.

Every value produced here has to survive one question: if I were scoring this
order at its decision moment, would this value already exist? docs/features.md
answers it feature by feature, and tests/test_features.py enforces it.
"""

from pathlib import Path

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
            # 1,350 orders record the handover before the approval. That reads as
            # a bookkeeping artefact rather than a real sequence, so t1 is clamped
            # to never precede t0 instead of dropping 1.4% of the sample. The flag
            # keeps the affected rows identifiable.
            "t1": carrier.where(carrier >= approved, approved),
            "t1_was_inverted": carrier < approved,
            "review_creation_date": df["review_creation_date"],
            "review_score": df["review_score"].astype("int8"),
            "y": df["review_score"] <= 2,
        }
    ).reset_index(drop=True)

    # A duplicated key here would mean a merge fanned out - silent, and fatal to
    # every count downstream.
    if spine["order_id"].duplicated().any():
        raise ValueError("build_spine produced more than one row for some order")

    spine.attrs["funnel"] = funnel
    return spine


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
    print(f"\nwrote {SPINE_PATH.relative_to(config.ROOT)}")
    return SPINE_PATH


def load_spine() -> pd.DataFrame:
    """Read the spine back, building it first if it is not there yet."""
    if not SPINE_PATH.exists():
        write_spine()
    return pd.read_parquet(SPINE_PATH)


def main() -> None:
    """Run with `make features` or `python -m src.features`."""
    write_spine()


if __name__ == "__main__":
    main()
