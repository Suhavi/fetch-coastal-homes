import sys
import os
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
from scipy.stats import genextreme
from src.world.house import House
from src.world.states_of_world import StateOfWorld
from src.decision.strategy import ElevationStrategy, strategy_grid
from src.vulnerabilities.depth_damage import load_depth_damage_curve
from src.economics.discounting import fixed_discount_factors
from src.decision.pareto import batch_strategies_performance, pareto # TODO- only one script ideally should be explosed from each package
from src.decision.visualisation import plot_height_tradeoffs, plot_pareto_cost_reliability


def fetch_a_sow():
    dd = load_depth_damage_curve("hazus")

    return StateOfWorld(
        water_level_dist=genextreme(c=-0.05, loc=5.0, scale=1.0),
        factors=fixed_discount_factors(30, 0.04),
        lifetime_years=30,
        depth_ft=dd["depth_ft"],
        damage_frac=dd["damage_frac"],
    )

if __name__ == "__main__":
    sow = fetch_a_sow()

    house = House(
        sqft=1500,
        value_usd=300000,
        first_floor_elevation_ft=2.0,
        lifetime_years=100,
    )

    strategies = strategy_grid([0, 3, 4, 5, 6, 7, 8], [2026])
    results = batch_strategies_performance(house, sow, strategies)
    front = pareto(results)
    pprint(front, sort_dicts=False)

    os.makedirs("plots/decision", exist_ok=True)

    try:
        fig, ax = plot_height_tradeoffs(results, front, action_year=2026)
        # fig.savefig("plots/decision/height_tradeoffs.png", dpi=300)
        plt.show()

        fig, ax = plot_pareto_cost_reliability(results, front)
        # fig.savefig("plots/decision/pareto_cost_reliability.png", dpi=300)
        plt.show()
    except ModuleNotFoundError as exc:
        if exc.name != "matplotlib":
            raise
        print("Install matplotlib to save decision plots.")
