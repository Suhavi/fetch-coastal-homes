import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.objectives.total_cost import total_cost_breakdown_timed
from src.objectives.reliability import lifetime_prob_no_flooding_timed
from src.objectives.benefit_cost import benefit_cost_ratio_timed

OBJECTIVE_DIRECTIONS = {
    "total_cost": "min",
    "upfront_cost": "min",
    "lifetime_expected_damage": "min",
    "reliability": "max",
    "benefit_cost": "max",
}

DEFAULT_OBJECTIVES = ("total_cost", "reliability", "benefit_cost")

def strategy_performance(house, sow, strategy, first_floor_elevation_ft: float):
    costs = total_cost_breakdown_timed(
        house,
        sow,
        strategy,
        first_floor_elevation_ft,
    )

    return{
        "first_floor_elevation_ft": first_floor_elevation_ft,
        "delta_h_ft": strategy.delta_h_ft,
        "action_year": strategy.action_year,
        "upfront_cost": costs["upfront_cost_usd"],
        "undiscounted_upfront_cost": costs["undiscounted_upfront_cost_usd"],
        "lifetime_expected_damage": costs["lifetime_expected_damage_usd"],
        "total_cost": costs["total_cost_usd"],
        "reliability": lifetime_prob_no_flooding_timed(
            sow,
            strategy,
            first_floor_elevation_ft,
        ),
        "benefit_cost": benefit_cost_ratio_timed(
            house,
            sow,
            strategy,
            first_floor_elevation_ft,
        )
    }

def batch_strategies_performance(
    house,
    sow,
    strategies,
    first_floor_elevation_ft: float,
):
    return[
        strategy_performance(
            house,
            sow,
            strategy,
            first_floor_elevation_ft,
        )
        for strategy in strategies
    ]


def batch_floor_heights_performance(house, sow, strategies, floor_heights):
    return [
        strategy_performance(house, sow, strategy, floor_height)
        for floor_height in floor_heights
        for strategy in strategies
    ]


def objective_vector(strategy_performance, objectives=DEFAULT_OBJECTIVES):
    values = []

    for objective in objectives:
        direction = OBJECTIVE_DIRECTIONS[objective]
        value = strategy_performance[objective]

        if np.isnan(value):
            value = -np.inf if direction == "max" else np.inf

        values.append(value if direction == "min" else -value)

    return np.array(values)


def is_dominated(a,b): # read as a is dominated by b, i.e. b is the dominant strategy
    return np.all(b<=a) and np.any(b<a)


def pareto(strategies_performance, objectives=DEFAULT_OBJECTIVES):
    values = np.array([
        objective_vector(stp, objectives)
        for stp in strategies_performance
    ])
        
    keep = []

    for i,row in enumerate(values):
        dominated = False
        for j, other in enumerate(values):
            if i == j:
                continue

            if is_dominated(row, other):
                dominated = True
                break

        if not dominated:
            keep.append(strategies_performance[i])
    
    return keep
