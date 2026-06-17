import requests
import numpy as np
import os
from datetime import datetime
from dataclasses import dataclass

NOAA_BASE_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
CACHE_DIR = "data/raw/noaa"

@dataclass
class WaterLevelStation:
    datum: str
    gage_id: str
    gage_name: str
    units: str
    time_zone: str

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
    "application": "FETCH", # project title
    "begin_date": f"{year}0101",
    "end_date": f"{year}1231",
    "datum": station.datum,
    "station": station.gage_id,
    "time_zone": station.time_zone,
    "units": station.units,
    "format": "json"
    }


def cache_file(station, year):
    return os.path.join(CACHE_DIR, f"{station.gage_id}_{year}_{station.datum}_{station.units}.npz")


def fetch_hourly_water_levels(station, year):
    os.makedirs(CACHE_DIR, exist_ok=True)
    fname = cache_file(station, year)

    if os.path.exists(fname):
        cached = np.load(fname, allow_pickle=True)
        return {
            "station": station,
            "year": year,
            "time": cached["time"],
            "water_level_ft": cached["water_level_ft"],
        }

    params = water_level_params(station, year)
    response = requests.get(NOAA_BASE_URL, params=params)
    response.raise_for_status()

    response_data = response.json()
    if "error" in response_data:
        raise ValueError(response_data["error"]["message"])

    raw = response_data["data"]
    times = []
    water_level_ft = []

    for row in raw:
        try:
            value = float(row["v"])
        except ValueError:
            continue

        times.append(datetime.strptime(row["t"], "%Y-%m-%d %H:%M"))
        water_level_ft.append(value)

    times = np.array(times)
    water_level_ft = np.array(water_level_ft)
    np.savez(fname, time=times, water_level_ft=water_level_ft)

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
