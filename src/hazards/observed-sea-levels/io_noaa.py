import requests
import numpy as np
from datetime import datetime
from dataclasses import dataclass

@dataclass
class WaterLevelStation:
    datum: str
    gage_id: str
    gage_name: str
    units: str
    time_zone: str


NOAA_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

SEWELLS = WaterLevelStation(
            datum     =  "NAVD", 
            gage_id   =  "8638610",
            gage_name =  "Sewells Point, VA",
            units     =  "english",
            time_zone =  "GMT"
            )

def water_level_params(station, year):
    return {
    "product": "hourly_height",
    "application": "FETCH",
    "begin_date": f"{year}0101",
    "end_date": f"{year}1231",
    "datum": station.datum,
    "station": station.gage_id,
    "time_zone": station.time_zone,
    "units": station.units,
    "format": "json"
    }

def fetch_hourly_water_levels(station, year):
    params = water_level_params(station, year)
    response = requests.get(NOAA_BASE_URL, params=params)
    response.raise_for_status()

    raw = response.json()["data"]
    times = np.array([
        datetime.strptime(row["t"], "%Y-%m-%d %H:%M")
        for row in raw
    ])
    water_level_ft = np.array([
        float(row["v"])
        for row in raw
    ])

    return {
        "station": station,
        "year": year,
        "time": times,
        "water_level_ft": water_level_ft,
    }


def fetch_hourly_all_years(station, years):
    all_times = []
    all_water_levels = []

    for year in years:
        data = fetch_hourly_water_levels(station, year)
        all_times.append(data["time"])
        all_water_levels.append(data["water_level_ft"])

    return {
        "station": station,
        "years": np.array(list(years)),
        "time": np.concatenate(all_times),
        "water_level_ft": np.concatenate(all_water_levels),
    }


if __name__ == "__main__":
    print(fetch_hourly_all_years(SEWELLS, [2020,2021,2022]))