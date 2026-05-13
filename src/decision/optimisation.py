import numpy as np
from objectives.total_cost import total_cost
from objectives.reliability import lifetime_prob_no_flooding
from objectives.benefit_cost import benefit_cost_ratio


def strategy_performance(house, sow, strategy):
    return{
        "delta_h_ft": strategy.delta_h_ft,
        "action_year": strategy.action_year,
        "total_cost": total_cost(house, sow, strategy.delta_h_ft),
        "reliability": lifetime_prob_no_flooding(house, sow, strategy.delta_h_ft),
        "benefit_cost": benefit_cost_ratio(house, sow, strategy.delta_h_ft)
    }

def batch_strategies_performance(house, sow, strategies):
    return[
        strategy_performance(house, sow, stratetgy)
        for strategy in strategies
    ]

def pareto():
    # minimise total cost
    # maximise probability of not flooding (reliability)
    # maximise benefit_cost
    # negative signs and minimise all

    return

