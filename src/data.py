"""Download and load the Olist dataset.

The raw CSVs are never committed, so this module is what makes their
acquisition reproducible. Downloading is a convenience: every loader here works
just as well on files fetched by hand, as long as they sit in data/raw.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src import config


class MissingCredentialsError(RuntimeError):
    """No Kaggle API token was available for an automatic download."""


_CREDENTIALS_HELP = """No Kaggle API token found.

Option A - automatic download:
  1. Generate a token at https://www.kaggle.com/settings/api
  2. Copy .env.example to .env and set KAGGLE_API_TOKEN

Option B - manual download:
  1. Get https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
  2. Unzip the nine CSVs into data/raw/

Everything downstream works the same either way."""


def raw_files_present() -> bool:
    """True when all nine expected CSVs already sit in data/raw."""
    return all((config.DATA_RAW / name).exists() for name in config.TABLES.values())


def download_raw(force: bool = False) -> None:
    """Download the dataset into data/raw, skipping the work if it is there.

    Raises MissingCredentialsError with actionable instructions when no token
    is configured, rather than letting the Kaggle client fail on its own terms.
    """
    if raw_files_present() and not force:
        print(f"data/raw already holds the {len(config.TABLES)} tables, skipping download")
        return

    # Anchored to the project root on purpose: load_dotenv() with no argument
    # searches upward from the calling file, which finds nothing when the caller
    # is a notebook or a script living outside the project.
    load_dotenv(config.ROOT / ".env")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        raise MissingCredentialsError(_CREDENTIALS_HELP)

    # Imported here, not at module scope, so that loading data already on disk
    # never depends on the kaggle package or on credentials being present.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)
    print(f"downloading {config.KAGGLE_DATASET} into {config.DATA_RAW} ...")
    api.dataset_download_files(config.KAGGLE_DATASET, path=config.DATA_RAW, unzip=True)

    missing = [n for n in config.TABLES.values() if not (config.DATA_RAW / n).exists()]
    if missing:
        raise RuntimeError(f"download finished but these files are missing: {missing}")
    print(f"done: {len(config.TABLES)} tables in {config.DATA_RAW}")


@dataclass(frozen=True)
class OlistTables:
    """The nine raw tables, loaded and lightly typed.

    frozen only stops these attributes being rebound; the DataFrames themselves
    stay mutable, as pandas requires.
    """

    orders: pd.DataFrame
    order_items: pd.DataFrame
    order_payments: pd.DataFrame
    order_reviews: pd.DataFrame
    products: pd.DataFrame
    sellers: pd.DataFrame
    customers: pd.DataFrame
    geolocation: pd.DataFrame
    product_category_name_translation: pd.DataFrame


def load_table(name: str) -> pd.DataFrame:
    """Load one raw table by logical name, with its date columns parsed.

    Review comment text is loaded untouched: phase 1 does not use it, but
    phase 2 (NLP) needs it to stay reachable from the data pipeline.
    """
    if name not in config.TABLES:
        raise KeyError(f"unknown table {name!r}; expected one of {sorted(config.TABLES)}")

    path = config.DATA_RAW / config.TABLES[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make data` first")

    # utf-8-sig strips the byte-order mark that product_category_name_translation
    # carries, which would otherwise corrupt the name of its first column.
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        parse_dates=config.DATE_COLUMNS.get(name),
    )


def load_all() -> OlistTables:
    """Load all nine tables. Reads ~121 MB, geolocation being half of that."""
    return OlistTables(**{name: load_table(name) for name in config.TABLES})


def _sha256(path: Path) -> str:
    """Hash a file in 1 MB chunks, so the 61 MB table never lands in memory whole."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict[str, dict]:
    """Describe every raw table: file name, byte size, row count and content hash.

    Row counts come from pandas rather than from counting lines: review comments
    contain embedded newlines, so order_reviews spans 104,720 physical lines for
    99,224 actual rows.
    """
    manifest = {}
    for name, filename in config.TABLES.items():
        path = config.DATA_RAW / filename
        manifest[name] = {
            "file": filename,
            "bytes": path.stat().st_size,
            "rows": len(load_table(name)),
            "sha256": _sha256(path),
        }
    return manifest


def write_manifest() -> Path:
    """Build the manifest and store it at data/manifest.json, which is committed."""
    manifest = build_manifest()
    text = json.dumps(manifest, indent=2, sort_keys=True)
    config.MANIFEST.write_text(text + "\n", encoding="utf-8")
    return config.MANIFEST


def check_raw_data() -> None:
    """Verify data/raw against the committed manifest.

    Catches a truncated download, but mainly catches someone working from a
    different revision of the dataset - which would otherwise silently produce
    numbers that disagree with the ones reported in the README.

    The hash is what verifies; the row counts in the manifest are there to be
    read by a human, since a matching hash already implies matching rows.
    """
    if not config.MANIFEST.exists():
        raise FileNotFoundError(f"{config.MANIFEST} not found; generate it with write_manifest()")

    expected = json.loads(config.MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for name, entry in expected.items():
        path = config.DATA_RAW / entry["file"]
        if not path.exists():
            problems.append(f"{name}: {entry['file']} is missing")
            continue
        actual = _sha256(path)
        if actual != entry["sha256"]:
            problems.append(
                f"{name}: content differs from the manifest "
                f"(expected {entry['sha256'][:12]}..., got {actual[:12]}...)"
            )

    if problems:
        detail = "\n  ".join(problems)
        raise RuntimeError(f"data/raw does not match the manifest:\n  {detail}")


def _print_table_sizes(tables: OlistTables) -> None:
    print("rows per table")
    for name in config.TABLES:
        df = getattr(tables, name)
        print(f"  {name:36s} {len(df):>10,} rows  {len(df.columns):>2} cols")


def _print_join_keys(tables: OlistTables) -> None:
    """Derive the join graph instead of asserting it: shared column names."""
    owners: dict[str, list[str]] = {}
    for name in config.TABLES:
        for column in getattr(tables, name).columns:
            owners.setdefault(column, []).append(name)

    print("join keys (columns present in more than one table)")
    for column, names in sorted(owners.items()):
        if len(names) > 1:
            joined = ", ".join(names)
            print(f"  {column:30s} {joined}")


def _print_time_range(tables: OlistTables) -> None:
    purchase = tables.orders["order_purchase_timestamp"]
    print("temporal range")
    print(f"  first purchase   {purchase.min()}")
    print(f"  last purchase    {purchase.max()}")
    print(f"  distinct months  {purchase.dt.to_period('M').nunique()}")


def _print_target(tables: OlistTables) -> None:
    """Report the target as defined in docs/decisiones.md, D-06."""
    orders = tables.orders
    reviews = tables.order_reviews

    # One review per order, the earliest one: a later review would be
    # information from after the moment the model would be applied.
    first = reviews.sort_values("review_creation_date").drop_duplicates("order_id", keep="first")
    scored = orders[["order_id"]].merge(first[["order_id", "review_score"]], on="order_id")

    without = len(orders) - len(scored)
    dropped = len(reviews) - len(first)
    prevalence = (scored["review_score"] <= 2).mean()

    print("target: review_score <= 2")
    print(f"  review rows                {len(reviews):>10,}")
    print(f"  duplicate reviews dropped  {dropped:>10,}")
    print(f"  orders with a review       {len(scored):>10,}")
    print(f"  orders without a review    {without:>10,}  ({without / len(orders):.2%})")
    print(f"  prevalence (per order)     {prevalence:>10.2%}")
    counts = scored["review_score"].value_counts().sort_index()
    for score, count in counts.items():
        print(f"    {score} star {count:>8,}  {count / len(scored):>6.2%}")


def main() -> None:
    """Download, verify and describe the dataset. Run with `python -m src.data`."""
    download_raw()

    if config.MANIFEST.exists():
        check_raw_data()
        print(f"data/raw matches {config.MANIFEST.name}")
    else:
        write_manifest()
        print(f"{config.MANIFEST.name} created; commit it so others can verify their copy")

    tables = load_all()
    for section in (_print_table_sizes, _print_join_keys, _print_time_range, _print_target):
        print()
        section(tables)


if __name__ == "__main__":
    main()
