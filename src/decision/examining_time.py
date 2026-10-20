import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.decision.optimise import fetch_a_sow
from src.decision.pareto import batch_strategies_performance, pareto
from src.decision.strategy import strategy_grid
from src.decision.visualisation import _tufte_axes
from src.world.house import House


def results_for_heights_over_time(
    house,
    sow,
    heights,
    first_floor_elevation_ft,
):
    action_years = sow.years[:sow.lifetime_years]
    strategies = strategy_grid(heights, action_years)
    return batch_strategies_performance(
        house,
        sow,
        strategies,
        first_floor_elevation_ft,
    )


def _results_by_height(results, height):
    rows = [r for r in results if np.isclose(r["delta_h_ft"], height)]
    return sorted(rows, key=lambda r: r["action_year"])


def print_time_table(results, heights):
    for height in heights:
        print(f"\nHeight: {height} ft")
        print("year, upfront_cost, damages, total_cost, reliability, benefit_cost")
        for r in _results_by_height(results, height):
            print(
                f"{r['action_year']}, "
                f"{r['upfront_cost']:.2f}, "
                f"{r['lifetime_expected_damage']:.2f}, "
                f"{r['total_cost']:.2f}, "
                f"{r['reliability']:.6f}, "
                f"{r['benefit_cost']:.6f}"
            )


def print_pareto_times(label, front):
    print(f"\n{label}")
    print("height_ft, action_year")
    for r in sorted(front, key=lambda row: (row["action_year"], row["delta_h_ft"])):
        print(f"{r['delta_h_ft']}, {r['action_year']}")


def plot_time_diagnostics(results, heights):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
    ax_cost, ax_damage, ax_reliability, ax_bcr = axes

    for height in heights:
        rows = _results_by_height(results, height)
        years = np.array([r["action_year"] for r in rows])

        ax_cost.plot(
            years,
            np.array([r["total_cost"] for r in rows]) / 1000,
            lw=1.5,
            label=f"{height} ft",
        )
        ax_damage.plot(
            years,
            np.array([r["lifetime_expected_damage"] for r in rows]) / 1000,
            lw=1.5,
            label=f"{height} ft",
        )
        ax_reliability.plot(
            years,
            100 * np.array([r["reliability"] for r in rows]),
            lw=1.5,
            label=f"{height} ft",
        )
        ax_bcr.plot(
            years,
            np.array([r["benefit_cost"] for r in rows]),
            lw=1.5,
            label=f"{height} ft",
        )

    ax_cost.set_ylabel("Total cost\n(1,000 USD)")
    ax_damage.set_ylabel("Damages\n(1,000 USD)")
    ax_reliability.set_ylabel("Reliability\n(%)")
    ax_bcr.set_ylabel("BCR")
    ax_bcr.set_xlabel("Action year")

    for ax in axes:
        _tufte_axes(ax)

    ax_cost.legend(frameon=False, ncol=min(len(heights), 4), fontsize=8)
    fig.suptitle("Timing Diagnostics")
    fig.tight_layout()

    return fig, axes


def plot_height_time_pareto(results_now, front_now, results_all, front_all):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(
        [r["delta_h_ft"] for r in results_all],
        [r["action_year"] for r in results_all],
        color="0.82",
        s=12,
        label="All strategies",
    )
    ax.scatter(
        [r["delta_h_ft"] for r in front_all],
        [r["action_year"] for r in front_all],
        color="tab:blue",
        s=45,
        label="Pareto, with time flexibility",
    )
    ax.scatter(
        [r["delta_h_ft"] for r in front_now],
        [r["action_year"] for r in front_now],
        color="tab:red",
        marker="^",
        s=60,
        label="Pareto, 2026",
    )

    ax.set_xlabel("Elevation height (ft)")
    ax.set_ylabel("Action year")
    _tufte_axes(ax)
    ax.legend(frameon=False, loc="upper right")
    # fig.suptitle("Height-Time Pareto Diagnostic")
    fig.tight_layout()

    return fig, ax


if __name__ == "__main__":
    sow = fetch_a_sow()
    house = House(
        sqft=1500,
        value_usd=300000,
        lifetime_years=100,
    )

    first_floor_elevation_ft = 5.0
    heights = np.arange(1, 15)
    results = results_for_heights_over_time(
        house,
        sow,
        heights,
        first_floor_elevation_ft,
    )
    front_all = pareto(results)

    strategies_now = strategy_grid(heights, [2026, 2027, 2028, 2029, 2030])
    results_now = batch_strategies_performance(
        house,
        sow,
        strategies_now,
        first_floor_elevation_ft,
    )
    front_now = pareto(results_now)

    print_pareto_times("Pareto front, with time flexibility", front_all)
    print_pareto_times("Pareto front, 2026-2030", front_now)

    # print_time_table(results, heights)

    fig, axes = plot_time_diagnostics(results, heights)
    fig, ax = plot_height_time_pareto(results_now, front_now, results, front_all)

    import matplotlib.pyplot as plt

    plt.show()
