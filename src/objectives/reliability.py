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


def lifetime_flood_probability(house, sow, delta_h_ft: float = 0.0):
    return 1 - lifetime_prob_no_flooding(house, sow, delta_h_ft)
