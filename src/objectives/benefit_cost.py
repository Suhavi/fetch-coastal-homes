import numpy as np

from src.economics.elevation_cost import elevation_cost
from src.objectives.damages import lifetime_expected_damage, lifetime_expected_damage_timed
from src.objectives.total_cost import action_year_index


def avoided_damages(house, sow, delta_h_ft: float):
    baseline = lifetime_expected_damage(house, sow, delta_h_ft=0.0)
    elevated = lifetime_expected_damage(house, sow, delta_h_ft=delta_h_ft)

    return baseline - elevated


def benefit_cost_ratio(house, sow, delta_h_ft: float):
    cost = elevation_cost(delta_h_ft, house.sqft)

    if np.isclose(cost, 0) or np.isnan(cost):
        return np.nan

    return avoided_damages(house, sow, delta_h_ft) / cost


def avoided_damages_timed(house, sow, strategy):
    baseline = lifetime_expected_damage(house, sow, delta_h_ft=0.0)
    elevated = lifetime_expected_damage_timed(house, sow, strategy)

    return baseline - elevated


def benefit_cost_ratio_timed(house, sow, strategy):
    idx = action_year_index(sow, strategy.action_year)
    cost = elevation_cost(strategy.delta_h_ft, house.sqft) * sow.factors[idx]

    if np.isclose(cost, 0) or np.isnan(cost):
        return np.nan

    return avoided_damages_timed(house, sow, strategy) / cost

