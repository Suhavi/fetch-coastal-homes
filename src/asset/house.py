from dataclasses import dataclass

@dataclass
class House:
    sqft: float
    value_usd: float
    first_floor_elevation_ft: float
    lifetime_years: int = 30
    # extendable factors
    location: str = ""
    foundation_type: str = ""
    contents_value_usd: float = 0.0
    current_elevation_ft: float = 0.0
