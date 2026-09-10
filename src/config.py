"""Project-wide constants: paths, seed, dataset layout and cost parameters."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# Repository root, derived from this file's own location so the project works
# from any clone path. config.py lives in <root>/src/, hence parents[1].
ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
MANIFEST = DATA / "manifest.json"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"

# Fixed everywhere randomness is involved: splits, model training, sampling.
SEED = 42

# Kaggle dataset identifier, as used by the API and the CLI.
KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

# Logical table name -> file name under data/raw. Call sites refer to tables by
# their logical name, so the "olist_..._dataset.csv" convention is stated once.
# File names verified against the Kaggle API listing.
TABLES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "order_reviews": "olist_order_reviews_dataset.csv",
    "products": "olist_products_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
    "product_category_name_translation": "product_category_name_translation.csv",
}

# Columns to parse as datetimes, per logical table. Listed explicitly rather
# than inferred from a name suffix: order_approved_at is a timestamp that no
# *_date / *_timestamp rule would catch, and it is the t0 decision moment.
DATE_COLUMNS = {
    "orders": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": ["shipping_limit_date"],
    "order_reviews": ["review_creation_date", "review_answer_timestamp"],
}

# Analysis window, on order_purchase_timestamp. The dataset spans 2016-09 to
# 2018-10, but the tails are not a business signal: 329 orders across 2016 are a
# pilot, and 2018-09/10 hold 20 orders from a mid-month export cut. Both ends are
# trimmed so the temporal split blocks are not distorted by them. See D-07.
ANALYSIS_START = pd.Timestamp("2017-01-01")
ANALYSIS_END = pd.Timestamp("2018-09-01")  # exclusive

# Fixed rate, documented rather than modelled. See docs/decisiones.md, D-04.
BRL_PER_EUR = 3.6


@dataclass(frozen=True)
class CostParams:
    """Expected-cost parameters for the intervention decision, in EUR.

    These are documented assumptions, not quantities measured from the data.
    See docs/decisiones.md, D-04, for their derivation and the sensitivity plan
    that varies them.
    """

    c_int: float = 3.0
    c_neg: float = 40.0
    effectiveness: float = 0.30

    @property
    def threshold(self) -> float:
        """Predicted risk above which intervening is cheaper in expectation."""
        return self.c_int / (self.effectiveness * self.c_neg)


DEFAULT_COSTS = CostParams()
