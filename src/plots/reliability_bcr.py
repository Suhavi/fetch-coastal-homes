import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.decision.optimise import fetch_a_sow
from src.decision.pareto import batch_strategies_performance
from src.decision.strategy import strategy_grid
from src.decision.visualisation import _tufte_axes
from src.world.house import House


def main():
    sow = fetch_a_sow()
    house = House(
        sqft=1500,
        value_usd=300000,
        first_floor_elevation_ft=1.0,
        lifetime_years=100,
    )
    results = batch_strategies_performance(
        house,
        sow,
        strategy_grid(np.arange(1, 15), [2026]),
    )
    results.sort(key=lambda r: r["delta_h_ft"])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(
        [r["benefit_cost"] for r in results],
        [100 * r["reliability"] for r in results],
        marker="o",
        color="tab:blue",
    )

    for r in results:
        ax.annotate(
            f'{r["delta_h_ft"]} ft',
            (r["benefit_cost"], 100 * r["reliability"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )

    ax.set_xlabel("Benefit-cost ratio")
    ax.set_ylabel("Reliability (%)")
    ax.set_title("2026 Elevation Strategies")
    _tufte_axes(ax)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
