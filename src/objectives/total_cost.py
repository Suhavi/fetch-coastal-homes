from src.economics.elevation_cost import elevation_cost
from src.objectives.damages import lifetime_expected_damage


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
