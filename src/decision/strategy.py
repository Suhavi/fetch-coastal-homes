from dataclasses import dataclass
import numpy as np
from itertools import product

@dataclass
class ElevationStrategy:
    delta_h_ft: float
    action_year: int
    


def strategy_grid(delta_h_values: np.ndarray, action_years):
    strategies = [ElevationStrategy(h, year) 
                  for h, year 
                  in product(delta_h_values, action_years)]
    
    return strategies


delta_h_values = [0, 3, 4, 5, 6, 7, 8]
action_years = [2026, 2030, 2040, 2050] # np.arange(2026, 2026+house.lifetime)
