from dataclasses import dataclass
import numpy as np

# uncertain future assumptions

@dataclass
class StateOfWorld:
    water_level_dist: object #storm-surge
    discount_rate: np.ndarray   # one discount-path
    lifetime_years: int  # samled/fixed-house lifetime span
    depth_ft: np.ndarray # depth-damage curve x-values
    damage_frac: np.ndarray # sampled damage curve y-values
    # sea_level_path_ft
    # years