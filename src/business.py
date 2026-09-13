"""From calibrated probabilities to a decision, and from a decision to euros.

The layer nothing else imports: config < features < evaluate < train <
calibrate < business. Everything here rests on probabilities that have been
calibrated (D-14), because the arithmetic below multiplies a probability by a
cost in euros - with the raw scores, which overstated risk by two thirds, every
figure would be fiction.

All costs are the assumptions of D-04, not measurements: 3 EUR to intervene,
40 EUR for a negative review, an intervention that works 30% of the time. The
job of D-16 is to say how much of the conclusion survives changing them.
"""

from dataclasses import replace

import numpy as np
import pandas as pd

from src import calibrate, config, evaluate, features

MOMENTS = ("t0", "t1")


def calibrated_by_moment(matrix: pd.DataFrame, block: str = "test") -> dict[str, np.ndarray]:
    """Calibrated probabilities for both decision moments, one fit each."""
    return {moment: calibrate.calibrated_scores(moment, matrix)[block] for moment in MOMENTS}


def policy(
    y_true,
    p,
    costs: config.CostParams = config.DEFAULT_COSTS,
    threshold: float | None = None,
) -> dict:
    """What acting above a threshold does, in orders and in euros.

    Savings are stated against doing nothing, and also per thousand orders so
    that a two-month block and a year of trading can be compared without
    remembering how big each one was.
    """
    y = np.asarray(y_true).astype(bool)
    scores = np.asarray(p, dtype=float)
    cutoff = costs.threshold if threshold is None else threshold
    acted = scores >= cutoff

    caught = int((acted & y).sum())
    savings = evaluate.net_savings(y, scores, cutoff, costs)
    return {
        "threshold": float(cutoff),
        "flagged": int(acted.sum()),
        "flagged_share": float(acted.mean()),
        "precision": float(y[acted].mean()) if acted.any() else float("nan"),
        "recall": float(caught / y.sum()) if y.any() else float("nan"),
        "caught": caught,
        "missed": int(y.sum() - caught),
        "false_alarms": int((acted & ~y).sum()),
        "savings_eur": savings,
        "savings_per_1000_orders": float(savings / len(y) * 1000),
    }


def blanket_policies(y_true, costs: config.CostParams = config.DEFAULT_COSTS) -> pd.DataFrame:
    """The two strategies that need no model at all, for contrast.

    Doing nothing is the yardstick and scores zero by definition. Intervening on
    everybody is the one worth showing: at a 10.19% base rate it burns 3 EUR per
    order to save 12 EUR on one order in ten, which loses money. That a model is
    needed at all is a claim, and this is the number behind it.
    """
    y = np.asarray(y_true).astype(bool)
    everyone = np.ones(y.size)
    return pd.DataFrame(
        [
            {"policy": "do nothing", "flagged_share": 0.0, "savings_eur": 0.0},
            {
                "policy": "intervene on everyone",
                "flagged_share": 1.0,
                "savings_eur": evaluate.net_savings(y, everyone, 0.5, costs),
            },
        ]
    )


def policy_table(
    matrix: pd.DataFrame | None = None,
    block: str = "test",
    costs: config.CostParams = config.DEFAULT_COSTS,
) -> pd.DataFrame:
    """The operating point of D-04 applied at both moments, side by side."""
    matrix = features.load_features() if matrix is None else matrix
    y = matrix.loc[matrix["split"] == block, "y"].to_numpy()
    scores = calibrated_by_moment(matrix, block)
    return pd.DataFrame(
        [{"moment": moment, **policy(y, scores[moment], costs)} for moment in MOMENTS]
    )


def threshold_comparison(
    matrix: pd.DataFrame | None = None,
    costs: config.CostParams = config.DEFAULT_COSTS,
) -> pd.DataFrame:
    """The fixed threshold against the one that would have been best.

    The operating threshold is 0.25, fixed by D-04 from the cost ratio alone,
    before any outcome was seen. The empirical optimum is what hindsight would
    have picked on each block. The distance between them is not a missed
    opportunity - it is a reading of how much miscalibration is left, since a
    perfectly calibrated model puts its optimum exactly at c_int / (e * C_neg).
    """
    matrix = features.load_features() if matrix is None else matrix
    rows = []
    for moment in MOMENTS:
        calibrated = calibrate.calibrated_scores(moment, matrix)
        for block in ("val", "test"):
            y = matrix.loc[matrix["split"] == block, "y"].to_numpy()
            best, best_savings = evaluate.best_threshold(y, calibrated[block], costs)
            fixed = evaluate.net_savings(y, calibrated[block], costs.threshold, costs)
            rows.append(
                {
                    "moment": moment,
                    "block": block,
                    "fixed_threshold": costs.threshold,
                    "fixed_savings": fixed,
                    "best_threshold": best,
                    "best_savings": best_savings,
                    "left_on_the_table": best_savings - fixed,
                }
            )
    return pd.DataFrame(rows)


def effectiveness_scenarios(
    matrix: pd.DataFrame | None = None,
    block: str = "test",
    t0_effectiveness=(0.30, 0.35, 0.40, 0.45, 0.50, 0.60),
    base: config.CostParams = config.DEFAULT_COSTS,
) -> pd.DataFrame:
    """t1 sees more; t0 can do more about it. Where do the two cancel out?

    D-04 flagged this and deferred it. At t0 nothing has moved yet, so the
    business still has real levers - reroute, change carrier, warn the seller -
    while at t1 the parcel is already travelling and only goodwill is left. The
    fixed matrix compares the two moments on information alone, which is the
    right way to measure signal and the wrong way to decide what to do.

    So t1 is held at the standing 0.30 and t0 is allowed to be more effective.
    The threshold moves with it, since c_int / (e * C_neg) says it must.
    """
    matrix = features.load_features() if matrix is None else matrix
    y = matrix.loc[matrix["split"] == block, "y"].to_numpy()
    scores = calibrated_by_moment(matrix, block)

    reference = evaluate.net_savings(y, scores["t1"], costs=base)
    rows = []
    for effectiveness in t0_effectiveness:
        costs = replace(base, effectiveness=effectiveness)
        savings = evaluate.net_savings(y, scores["t0"], costs=costs)
        rows.append(
            {
                "t0_effectiveness": effectiveness,
                "t0_threshold": costs.threshold,
                "t0_savings": savings,
                "t1_savings_at_0.30": reference,
                "t0_minus_t1": savings - reference,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Run with `python -m src.business`."""
    matrix = features.load_features()
    y = matrix.loc[matrix["split"] == "test", "y"].to_numpy()
    costs = config.DEFAULT_COSTS

    print(f"test block · {len(y):,} orders · base rate {y.mean():.2%}")
    print(f"costs: c_int {costs.c_int} EUR · C_neg {costs.c_neg} EUR · e {costs.effectiveness}")
    print(f"threshold c_int / (e * C_neg) = {costs.threshold:.2f}\n")

    print("no model at all")
    print(blanket_policies(y, costs).round(2).to_string(index=False))

    print("\nacting above the threshold, on calibrated probabilities")
    print(policy_table(matrix).round(4).to_string(index=False))

    print("\nthe fixed threshold against hindsight")
    print(threshold_comparison(matrix).round(3).to_string(index=False))

    print("\nwhat if intervening at t0 works better than at t1?")
    print(effectiveness_scenarios(matrix).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
