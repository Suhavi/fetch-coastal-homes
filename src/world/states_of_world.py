from dataclasses import dataclass
import numpy as np

# uncertain future assumptions

@dataclass
class StateOfWorld:
    uncertainty: bool
    """hazard"""
    years: np.ndarray
    water_level_dists: list
    sea_level_rise_ft: np.ndarray
    """economics"""
    depth_ft: np.ndarray # depth-damage curve x-values
    damage_frac: np.ndarray # sampled damage curve y-values
    factors: np.ndarray   # one discount-factor path
    """life of decision"""
    lifetime_years: int  # sampled/fixed-house lifetime span

    def __post_init__(self):
        self.years = np.asarray(self.years)
        self.sea_level_rise_ft = np.asarray(self.sea_level_rise_ft)
        self.depth_ft = np.asarray(self.depth_ft)
        self.damage_frac = np.asarray(self.damage_frac)
        self.factors = np.asarray(self.factors)

        n_years = len(self.years)
        if len(self.water_level_dists) != n_years:
            raise ValueError("water_level_dists must contain one distribution per year.")
        if len(self.sea_level_rise_ft) != n_years:
            raise ValueError("sea_level_rise_ft must contain one value per year.")
        if len(self.factors) < self.lifetime_years:
            raise ValueError("factors must cover the full house lifetime.")
        if self.lifetime_years > n_years:
            raise ValueError("lifetime_years cannot exceed the SOW time horizon.")

    def year_index(self, year: int):
        idx = np.where(self.years == year)[0]
        if len(idx) == 0:
            raise ValueError(f"Year {year} is not in sow.years.")
        return int(idx[0])

    def distribution_for_year(self, year: int):
        return self.water_level_dists[self.year_index(year)]
    
