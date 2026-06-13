import numpy as np


def raised_house_elevation(house, delta_h_ft: float):
    return house.first_floor_elevation_ft + delta_h_ft


def annual_no_flood_probability(house, sow, delta_h_ft: float = 0.0):
    floor_elev = raised_house_elevation(house, delta_h_ft)
    water_level_dist = sow.water_level_dist

    return water_level_dist.cdf(floor_elev)


def lifetime_prob_no_flooding(house, sow, delta_h_ft: float = 0.0):
    annual_prob = annual_no_flood_probability(house, sow, delta_h_ft)
    lifetime_years = getattr(sow, "lifetime_years", house.lifetime_years)

    return annual_prob ** lifetime_years


def annual_no_flood_probability_timed(house, sow, strategy):
    annual_probs = []

    for year in sow.years[:sow.lifetime_years]:
        if year < strategy.action_year:
            delta_h = 0.0
        else:
            delta_h = strategy.delta_h_ft

        annual_probs.append(annual_no_flood_probability(house, sow, delta_h))

    return np.array(annual_probs)


def lifetime_prob_no_flooding_timed(house, sow, strategy):
    annual_probs = annual_no_flood_probability_timed(house, sow, strategy)

    return np.prod(annual_probs)


def lifetime_flood_probability(house, sow, delta_h_ft: float = 0.0):
    return 1 - lifetime_prob_no_flooding(house, sow, delta_h_ft)


def lifetime_flood_probability_timed(house, sow, strategy):
    return 1 - lifetime_prob_no_flooding_timed(house, sow, strategy)
