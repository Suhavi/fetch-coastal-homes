import numpy as np
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.objectives.total_cost import total_cost, total_cost_breakdown
from src.objectives.reliability import lifetime_prob_no_flooding
from src.objectives.benefit_cost import benefit_cost_ratio

# action_year is not used yet; timing-aware objectives come next

def strategy_performance(house, sow, strategy):
    costs = total_cost_breakdown(house, sow, strategy.delta_h_ft)

    return{
        "delta_h_ft": strategy.delta_h_ft,
        "action_year": strategy.action_year,
        "upfront_cost": costs["upfront_cost_usd"],
        "lifetime_expected_damage": costs["lifetime_expected_damage_usd"],
        "total_cost": costs["total_cost_usd"],
        "reliability": lifetime_prob_no_flooding(house, sow, strategy.delta_h_ft),
        "benefit_cost": benefit_cost_ratio(house, sow, strategy.delta_h_ft)
    }

def batch_strategies_performance(house, sow, strategies):
    return[
        strategy_performance(house, sow, strategy)
        for strategy in strategies
    ]

"""
A MINIMISING FUNCTION
objectives = [
    total_cost,
    -reliability,
    -benefit_cost,
]
"""

def is_dominated(a,b): # read as a is dominated by b, i.e. b is the dominant strategy
    return np.all(b<=a) and np.any(b<a)


def pareto(strategies_perfomance):
    values = np.array([
                    [
                        stp["total_cost"], 
                       -stp["reliability"], 
                       -(-np.inf if np.isnan(stp["benefit_cost"]) else stp["benefit_cost"]) 
                    ] 
                for stp in strategies_perfomance
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
            keep.append(strategies_perfomance[i])
    
    return keep
