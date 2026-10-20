import numpy as np


def raised_house_elevation(first_floor_elevation_ft: float, delta_h_ft: float):
    return first_floor_elevation_ft + delta_h_ft


def expected_annual_damage(
    house,
    sow,
    year: int,
    first_floor_elevation_ft: float,
    delta_h_ft: float = 0.0,
):

    floor_elev = raised_house_elevation(first_floor_elevation_ft, delta_h_ft)
    damage_usd = sow.damage_frac * house.value_usd
    thresholds = floor_elev + sow.depth_ft

    water_level_dist = sow.distribution_for_year(year)
    exceed_probs = water_level_dist.sf(thresholds)

    ead = 0.0
    for i in range(len(thresholds)):
        if i == len(thresholds) - 1:
            prob = exceed_probs[i]
        else:
            prob = exceed_probs[i] - exceed_probs[i+1]

        ead += prob* damage_usd[i]
    
    return ead


def annual_expected_damages(
    house,
    sow,
    first_floor_elevation_ft: float,
    delta_h_ft: float = 0.0,
):
    years = sow.years[:sow.lifetime_years]
    return np.array([
        expected_annual_damage(
            house,
            sow,
            year,
            first_floor_elevation_ft,
            delta_h_ft,
        )
        for year in years
    ])


def lifetime_expected_damage(
    house,
    sow,
    first_floor_elevation_ft: float,
    delta_h_ft: float = 0.0,
):
    ead = annual_expected_damages(
        house,
        sow,
        first_floor_elevation_ft,
        delta_h_ft,
    )
    factors = sow.factors[:len(ead)]

    return np.sum(ead * factors)


def lifetime_expected_damage_timed(
    house,
    sow,
    strategy,
    first_floor_elevation_ft: float,
):

    total = 0.0
    for i, year in enumerate(sow.years[:sow.lifetime_years]):
        if year < strategy.action_year:
            delta_h = 0.0
        else:
            delta_h = strategy.delta_h_ft

        ead = expected_annual_damage(
            house,
            sow,
            year,
            first_floor_elevation_ft,
            delta_h,
        )
        total += ead * sow.factors[i] # discount factors

    return total
