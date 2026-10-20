import sys
import os
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from src.world.house import House
from src.world.states_of_world import StateOfWorld
from src.decision.strategy import ElevationStrategy, strategy_grid
from src.vulnerabilities.getfunctions_depthdamage import load_depth_damage_curve
from src.economics.discounting import fixed_discount_factors
from src.hazards.StormSurge.surge_model import projected_water_level_distributions
from src.hazards.SeaLevel.getfunctions_projections import get_mean_sea_level_rise_path
from src.decision.pareto import batch_strategies_performance, pareto # TODO- only one script ideally should be explosed from each package
from src.decision.visualisation import plot_height_tradeoffs, plot_pareto_cost_reliability


def fetch_a_sow(
    uncertainty=False,
    start_year=2026,
    lifetime_years=100,
    slr_model="BRICK-Slow",
    slr_rcp="rcp6.0",
    c=1.5,
    discount_rate=0.04,
    random_seed=1,
):
    dd = load_depth_damage_curve("hazus")
    slr = get_mean_sea_level_rise_path(
        model=slr_model,
        rcp=slr_rcp,
        syear=start_year,
        eyear=start_year + lifetime_years - 1,
        random_seed=random_seed,
    )
    water_level_dists = projected_water_level_distributions(slr, c=c)

    return StateOfWorld(
        uncertainty=uncertainty,
        water_level_dists=water_level_dists,
        years=slr["years"],
        sea_level_rise_ft=slr["sea_level_rise_ft"],
        factors=fixed_discount_factors(lifetime_years - 1, discount_rate),
        lifetime_years=lifetime_years,
        depth_ft=dd["depth_ft"],
        damage_frac=dd["damage_frac"],
    )

# if __name__ == "__main__":
#     sow = fetch_a_sow()

#     house = House(
#         sqft=1500,
#         value_usd=300000,
#         lifetime_years=100,
#     )

#     strategies = strategy_grid([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], [2026])
#     results = batch_strategies_performance(house, sow, strategies)
#     front = pareto(results)
#     pprint(front, sort_dicts=False)

#     os.makedirs("plots/decision", exist_ok=True)

#     try:
#         fig, ax = plot_height_tradeoffs(results, front, action_year=2026)
#         fig.savefig("plots/decision/height_tradeoffs.png", dpi=300)
#         plt.show()

#         fig, ax = plot_pareto_cost_reliability(results, front)
#         fig.savefig("plots/decision/pareto_cost_reliability.png", dpi=300)
#         plt.show()
#     except ModuleNotFoundError as exc:
#         if exc.name != "matplotlib":
#             raise
#         print("Install matplotlib to save decision plots.")

if __name__ == "__main__":
    sow = fetch_a_sow()

    house = House(
        sqft=1500,
        value_usd=300000,
        lifetime_years=100,
    )

    first_floor_elevation_ft = 5.0
    height_values = np.arange(1, 15)

    # Case 1: elevate now only
    strategies_now = strategy_grid(height_values, [2026])
    results_now = batch_strategies_performance(
        house,
        sow,
        strategies_now,
        first_floor_elevation_ft,
    )
    front_now = pareto(results_now)

    # Case 2: elevate in any year in the 100-year horizon
    action_years = sow.years[:sow.lifetime_years]
    strategies_timed = strategy_grid(height_values, action_years)
    results_timed = batch_strategies_performance(
        house,
        sow,
        strategies_timed,
        first_floor_elevation_ft,
    )
    for r in results_timed:
        if r["delta_h_ft"] == 8:
            print(
                r["action_year"],
                r["upfront_cost"],
                r["lifetime_expected_damage"],
                r["total_cost"],
                r["reliability"],
                r["benefit_cost"],
            )
    front_timed = pareto(results_timed)

    print("##########################")
    print("now")
    print("##########################")
    pprint(front_now, sort_dicts=False)
    print("##########################")
    print("timed")
    print("##########################")
    pprint(front_timed, sort_dicts=False)
