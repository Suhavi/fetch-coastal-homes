import numpy as np


def raised_house_elevation(first_floor_elevation_ft: float, delta_h_ft: float):
    return first_floor_elevation_ft + delta_h_ft


def annual_no_flood_probability(
    sow,
    year: int,
    first_floor_elevation_ft: float,
    delta_h_ft: float = 0.0,
):
    floor_elev = raised_house_elevation(first_floor_elevation_ft, delta_h_ft)
    water_level_dist = sow.distribution_for_year(year)

    return water_level_dist.cdf(floor_elev)


def lifetime_prob_no_flooding(
    sow,
    first_floor_elevation_ft: float,
    delta_h_ft: float = 0.0,
):
    annual_probs = [
        annual_no_flood_probability(
            sow,
            year,
            first_floor_elevation_ft,
            delta_h_ft,
        )
        for year in sow.years[:sow.lifetime_years]
    ]

    return np.prod(annual_probs)


def annual_no_flood_probability_timed(
    sow,
    strategy,
    first_floor_elevation_ft: float,
):
    annual_probs = []

    for year in sow.years[:sow.lifetime_years]:
        if year < strategy.action_year:
            delta_h = 0.0
        else:
            delta_h = strategy.delta_h_ft

        annual_probs.append(
            annual_no_flood_probability(
                sow,
                year,
                first_floor_elevation_ft,
                delta_h,
            )
        )

    return np.array(annual_probs)


def lifetime_prob_no_flooding_timed(
    sow,
    strategy,
    first_floor_elevation_ft: float,
):
    annual_probs = annual_no_flood_probability_timed(
        sow,
        strategy,
        first_floor_elevation_ft,
    )

    return np.prod(annual_probs)


def lifetime_flood_probability(
    sow,
    first_floor_elevation_ft: float,
    delta_h_ft: float = 0.0,
):
    return 1 - lifetime_prob_no_flooding(
        sow,
        first_floor_elevation_ft,
        delta_h_ft,
    )


def lifetime_flood_probability_timed(
    sow,
    strategy,
    first_floor_elevation_ft: float,
):
    return 1 - lifetime_prob_no_flooding_timed(
        sow,
        strategy,
        first_floor_elevation_ft,
    )
