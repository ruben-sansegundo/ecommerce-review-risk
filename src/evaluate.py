"""Metrics for a rare target whose base rate moves between blocks.

Written before the first model on purpose. If every notebook scored its own
model its own way, the baseline-versus-model comparison that rule 5 of
CLAUDE.md demands would not be a comparison at all.

Three ideas run through the module:

- Accuracy never appears. At a 10.19% base rate, predicting "no negative
  review" for everyone scores 89.81% and is worth nothing.
- Every headline number travels with the prevalence of the block it was
  measured on. Recall@10% means different things against 14.70% (train) and
  10.19% (test), so the row carries both.
- Costs may be scalars or one value per order. The euro figures are not
  reported until the probabilities are calibrated (rule 8), but the shape of
  the code is fixed now so S6 can vary cost per order without a rewrite.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import config

# Fractions of the population the intervention team could plausibly act on.
DEFAULT_KS = (0.05, 0.10, 0.20)


def _as_arrays(y_true, scores) -> tuple[np.ndarray, np.ndarray]:
    """Coerce labels and scores to aligned float/bool arrays, or explain why not."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores).astype(float)

    if y.shape != s.shape:
        raise ValueError(f"y_true and scores have different shapes: {y.shape} vs {s.shape}")
    if y.size == 0:
        raise ValueError("cannot score an empty block")
    if not np.isfinite(s).all():
        raise ValueError("scores contain NaN or infinity; a model must score every row")
    return y, s


def _ranking(scores: np.ndarray, seed: int = config.SEED) -> np.ndarray:
    """Row indices ordered by descending score, with ties broken at random.

    Tie-breaking is not a detail here: the trivial baseline gives every order
    the same score, and a stable sort would then hand it the rows in dataset
    order, which is chronological. That would quietly measure whether negative
    reviews cluster early in the block instead of measuring the rule. Shuffling
    first makes ties arbitrary; seeding the shuffle keeps it reproducible.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(scores.size)
    return shuffled[np.argsort(-scores[shuffled], kind="stable")]


def _top_k_size(n: int, k: float) -> int:
    """How many orders fall in the top k fraction, at least one and at most all."""
    if not 0 < k <= 1:
        raise ValueError(f"k must be a fraction in (0, 1], got {k}")
    return int(min(n, max(1, np.ceil(k * n))))


def prevalence(y_true) -> float:
    """Share of positives. Reported next to every other metric, never alone."""
    return float(np.asarray(y_true).astype(bool).mean())


def pr_auc(y_true, scores) -> float:
    """Area under the precision-recall curve, as average precision.

    PR rather than ROC because the negative class is ~9 times the positive one:
    ROC-AUC looks flattering when false positives are cheap only because there
    are so many true negatives to dilute them.

    Average precision rather than a trapezoid under the PR curve: interpolating
    between PR points is optimistic, since precision does not move linearly
    between thresholds. A model that knows nothing scores the prevalence.
    """
    y, s = _as_arrays(y_true, scores)
    return float(average_precision_score(y, s))


def recall_at_k(y_true, scores, k: float, seed: int = config.SEED) -> float:
    """Share of all positives captured inside the top k fraction of the ranking.

    k is a fraction of the population, not a number of orders, so that blocks of
    different size stay comparable: validation has 13,612 orders and test 18,664,
    and recall@1000 would not mean the same thing on both.
    """
    y, s = _as_arrays(y_true, scores)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")

    top = _ranking(s, seed)[: _top_k_size(y.size, k)]
    return float(y[top].sum() / positives)


def precision_at_k(y_true, scores, k: float, seed: int = config.SEED) -> float:
    """Share of the top k fraction that really ends in a negative review."""
    y, s = _as_arrays(y_true, scores)
    top = _ranking(s, seed)[: _top_k_size(y.size, k)]
    return float(y[top].mean())


def lift_at_k(y_true, scores, k: float, seed: int = config.SEED) -> float:
    """Precision@k divided by the base rate: how many times richer the top slice is.

    This is the number that survives the base-rate drift between blocks. The same
    recall@10% means something different against 14.70% and against 10.19%; the
    lift already has the prevalence in its denominator, so it can be read across
    blocks without remembering which one is which.
    """
    y, s = _as_arrays(y_true, scores)
    base = y.mean()
    if base == 0:
        return float("nan")
    return float(precision_at_k(y, s, k, seed) / base)


def brier(y_true, p) -> float:
    """Mean squared error of the predicted probabilities.

    A calibration metric, not a ranking one, so it is only meaningful when the
    scores really are probabilities. Scores outside [0, 1] - the raw ranking of
    a trivial rule, say - return NaN instead of a number that invites a
    comparison that makes no sense.
    """
    y, s = _as_arrays(y_true, p)
    if s.min() < 0 or s.max() > 1:
        return float("nan")
    return float(np.mean((s - y) ** 2))


def _cost_arrays(costs: config.CostParams) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cost parameters as arrays, so scalars and per-order vectors share one path."""
    return (
        np.asarray(costs.c_int, dtype=float),
        np.asarray(costs.c_neg, dtype=float),
        np.asarray(costs.effectiveness, dtype=float),
    )


def cost_per_order(y_true, act, costs: config.CostParams = config.DEFAULT_COSTS) -> np.ndarray:
    """Cost in EUR of each order under a decision, incremental to doing nothing.

    The matrix of D-04, written once:

                       review OK        negative review
        do nothing     0                c_neg
        intervene      c_int            c_int + (1 - e) * c_neg

    c_int, c_neg and e may each be a scalar or one value per order; the per-order
    form is what S6 needs to make the cost proportional to the order value.
    """
    y = np.asarray(y_true).astype(float)
    a = np.asarray(act).astype(float)
    c_int, c_neg, e = _cost_arrays(costs)
    return a * c_int + (1.0 - a * e) * c_neg * y


def expected_cost(
    y_true,
    p,
    threshold: float | np.ndarray | None = None,
    costs: config.CostParams = config.DEFAULT_COSTS,
) -> float:
    """Total EUR of intervening wherever predicted risk reaches the threshold.

    Defaults to the break-even threshold of D-04 (0.25 with the standing
    parameters). That number is only the right one if the probabilities are
    calibrated, which is why no euro figure leaves S3.
    """
    y, s = _as_arrays(y_true, p)
    cutoff = costs.threshold if threshold is None else threshold
    return float(cost_per_order(y, s >= cutoff, costs).sum())


def do_nothing_cost(y_true, costs: config.CostParams = config.DEFAULT_COSTS) -> float:
    """Total EUR if nobody intervenes: the yardstick every policy is measured against."""
    y = np.asarray(y_true).astype(bool)
    return float(cost_per_order(y, np.zeros(y.size), costs).sum())


def net_savings(
    y_true,
    p,
    threshold: float | np.ndarray | None = None,
    costs: config.CostParams = config.DEFAULT_COSTS,
) -> float:
    """EUR saved against doing nothing. Negative means the policy destroys value."""
    return do_nothing_cost(y_true, costs) - expected_cost(y_true, p, threshold, costs)


def best_threshold(
    y_true,
    p,
    costs: config.CostParams = config.DEFAULT_COSTS,
    seed: int = config.SEED,
) -> tuple[float, float]:
    """Threshold that minimises realised cost on this block, and the EUR it saves.

    Found by sweeping the ranking rather than a grid of probabilities: sorting
    once and walking down the cumulative cost costs O(n log n), where scoring
    every candidate threshold separately would be quadratic.

    This is a diagnostic, not the operating rule. The threshold that ships is the
    analytic c_int / (e * c_neg) of D-04, fixed before seeing any outcome; the
    gap between the two is a symptom of miscalibration, and reading it on test
    would be fitting the decision to the answer.
    """
    y, s = _as_arrays(y_true, p)
    order = _ranking(s, seed)

    # Cost of acting on each order versus leaving it alone, in ranking order.
    acted = cost_per_order(y[order], np.ones(y.size), costs)
    ignored = cost_per_order(y[order], np.zeros(y.size), costs)

    # Acting on the top m orders: m can be 0, hence the leading zero.
    totals = np.concatenate([[0.0], np.cumsum(acted - ignored)]) + ignored.sum()
    m = int(np.argmin(totals))

    # Acting on the top m means a threshold just at the m-th score; above every
    # score when m is 0, so the rule "p >= cutoff" selects nobody.
    cutoff = float(s[order][m - 1]) if m > 0 else float(np.nextafter(s.max(), np.inf))
    return cutoff, float(do_nothing_cost(y, costs) - totals[m])


def score_model(
    name: str,
    moment: str,
    y_true,
    scores,
    ks: tuple[float, ...] = DEFAULT_KS,
    seed: int = config.SEED,
) -> dict:
    """One row of the comparison table: what this model does on this block.

    Prevalence and the number of orders are part of the row rather than the
    caption, so a row can never be quoted without the base rate it belongs to.
    """
    y, s = _as_arrays(y_true, scores)

    row = {
        "model": name,
        "moment": moment,
        "orders": int(y.size),
        "prevalence": prevalence(y),
        "pr_auc": pr_auc(y, s),
    }
    for k in ks:
        row[f"recall@{k:.0%}"] = recall_at_k(y, s, k, seed)
    for k in ks:
        row[f"lift@{k:.0%}"] = lift_at_k(y, s, k, seed)
    row["brier"] = brier(y, s)
    return row


def comparison_table(rows: list[dict]) -> pd.DataFrame:
    """Stack scored rows into the single table S3 is meant to produce."""
    table = pd.DataFrame(rows)
    return table.sort_values(["moment", "pr_auc"], ascending=[True, False]).reset_index(drop=True)


def sensitivity_grid(
    y_true,
    p,
    c_int_values,
    c_neg_values,
    effectiveness_values,
    base: config.CostParams = config.DEFAULT_COSTS,
) -> pd.DataFrame:
    """Net savings across cost assumptions, one row per combination.

    The assumptions of D-04 are invented, so the honest deliverable is not a
    single euro figure but the region where the case still holds. Used in S6.
    """
    rows = []
    for c_int in c_int_values:
        for c_neg in c_neg_values:
            for e in effectiveness_values:
                costs = replace(base, c_int=c_int, c_neg=c_neg, effectiveness=e)
                rows.append(
                    {
                        "c_int": c_int,
                        "c_neg": c_neg,
                        "effectiveness": e,
                        "threshold": costs.threshold,
                        "net_savings": net_savings(y_true, p, costs=costs),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    """Run with `make eval` or `python -m src.evaluate`."""
    raise SystemExit(
        "src.evaluate holds the metrics; it has no models to score yet.\n"
        "Train them first (make train), then this entry point prints the "
        "comparison table."
    )


if __name__ == "__main__":
    main()
