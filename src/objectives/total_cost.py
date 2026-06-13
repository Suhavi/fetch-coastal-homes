from src.economics.elevation_cost import elevation_cost
from src.objectives.damages import lifetime_expected_damage, lifetime_expected_damage_timed
import numpy as np

def total_cost(house, sow, delta_h_ft: float):
    upfront = elevation_cost(delta_h_ft, house.sqft)
    damages = lifetime_expected_damage(house, sow, delta_h_ft)

    return upfront + damages


def total_cost_breakdown(house, sow, delta_h_ft: float):
    upfront = elevation_cost(delta_h_ft, house.sqft)
    damages = lifetime_expected_damage(house, sow, delta_h_ft)

    return {
        "upfront_cost_usd": upfront,
        "lifetime_expected_damage_usd": damages,
        "total_cost_usd": upfront + damages,
    }

def action_year_index(sow, action_year: int):
    idx = np.where(sow.years == action_year)[0]

    if len(idx) == 0:
        raise ValueError(f"Action year {action_year} is not in sow.years.")

    return idx[0]

def total_cost_breakdown_timed(house, sow, strategy):
    idx = action_year_index(sow, strategy.action_year)

    upfront = elevation_cost(strategy.delta_h_ft, house.sqft)
    discounted_upfront = upfront * sow.factors[idx]

    damages = lifetime_expected_damage_timed(house, sow, strategy)

    return {
        "upfront_cost_usd": discounted_upfront,
        "undiscounted_upfront_cost_usd": upfront,
        "lifetime_expected_damage_usd": damages,
        "total_cost_usd": discounted_upfront + damages,
    }
