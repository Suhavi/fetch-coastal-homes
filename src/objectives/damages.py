import numpy as np
from src.vulnerabilities.depth_damage import load_depth_damage_curve
from src.economics.discounting import fixed_discount_factors


def raised_house_elevation(house, delta_h_ft: float):
    return house.first_floor_elevation_ft + delta_h_ft


def expected_annual_damage(house, sow, delta_h_ft = 0.0):

    floor_elev = house.first_floor_elevation_ft + delta_h_ft
    damage_usd = sow.damage_frac * house.value_usd
    thresholds = floor_elev + sow.depth_ft

    exceed_probs = 1 - sow.water_level_dist.cdf(thresholds)

    ead = 0.0
    for i in range(len(thresholds)):
        if i == len(thresholds) - 1:
            prob = exceed_probs[i]
        else:
            prob = exceed_probs[i] - exceed_probs[i+1]

        ead += prob* damage_usd[i]
    
    return ead
    
    
def lifetime_expected_damage(house, sow, delta_h_ft=0.0):

    ead = expected_annual_damage(house, sow, delta_h_ft)
    factors = sow.factors[:sow.lifetime_years]

    return ead * np.sum(factors)
