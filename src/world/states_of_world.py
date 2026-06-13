from dataclasses import dataclass
import numpy as np

# uncertain future assumptions

@dataclass
class StateOfWorld:
    uncertainty: bool
    """hazard"""
    water_level_dist: object #storm-surge # DISTRIBUTION
    years: np.ndarray
    sea_level_rise_ft: np.ndarray
    """economics"""
    depth_ft: np.ndarray # depth-damage curve x-values
    damage_frac: np.ndarray # sampled damage curve y-values
    factors: np.ndarray   # one discount-factor path
    """life of decision"""
    lifetime_years: int  # samled/fixed-house lifetime span
    
