import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.hazards.SeaLevel.io_noaa import SEWELLS, fetch_hourly_all_years

plt.style.use("seaborn-v0_8-whitegrid")
KNOWN_STORMS = [
    {"name": "Chesapeake-Potomac", "year": 1933},
    {"name": "Outer Banks hurricane", "year": 1936},
    {"name": "Ash Wednesday", "year": 1962},
    {"name": "Isabel", "year": 2003},
    {"name": "Nor'Ida", "year": 2009},
]


def get_years(times: np.ndarray):
    return np.array([t.year for t in times])


def annual_summary(allyrs_hourly_data):

    """
    Observed water levels are hourly.
    Annual mean sea level is one value per year. 
    Surge can be computed hourly, 
    but the model usually keeps annual maximum surge, 
    which is also one value per year.
    """
    
    times = allyrs_hourly_data["time"]
    water_level_ft = allyrs_hourly_data["water_level_ft"]

    time_years = get_years(times)
    years = np.unique(time_years)

    mean_sl_ft = []
    max_surge_ft = []

    for year in years:
        idx = time_years == year  # gets index of all of the same year, one year at a time.
        levels = water_level_ft[idx]

        mean_level = np.mean(levels)  # mean sea level is comuted yearly
        surge = levels -  mean_level # surge is first defined hourly

        mean_sl_ft.append(mean_level)
        max_surge_ft.append(np.max(surge)) # surge is changed into yearly

    return{

        "station": allyrs_hourly_data["station"],
        "years": years,
        "mean_sl_ft":  np.array(mean_sl_ft),
        "max_surge_ft": np.array(max_surge_ft)

    }


def plot_mean_sl(summary: dict):
    fig, ax = plt.subplots(figsize=(8,4))

    ax.plot(summary["years"], summary["mean_sl_ft"], color="0.45", alpha=0.75)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual mean sea level (ft)")
    ax.grid(True, alpha=0.3)

    return fig, ax


def plot_max_surge(summary):

    fig, ax = plt.subplots(figsize=(8,4))
    ax.scatter(summary["years"], summary["max_surge_ft"])

    ## KNOWN STORMS
    for storm in KNOWN_STORMS:
        year = storm["year"]
        idx = np.where(summary["years"] == year)[0]

        if len(idx) == 0:
            continue

        idx = idx[0]
        x = summary["years"][idx]
        y = summary["max_surge_ft"][idx]

        ax.annotate(
            storm["name"],
            xy=(x, y),
            xytext=(x + 2, y + 0.3),
            arrowprops={"arrowstyle": "-", "color": "red", "linewidth": 0.8},
            fontsize=8,
        )


    ax.set_xlabel("Year")
    ax.set_ylabel("Annual max storm surge (ft)")
    ax.grid(True, alpha=0.3)

    return fig, ax


if __name__ == "__main__":
    print("Computing from Observed Data")

    years = range(1927,2026)

    # fetch the NOAA data
    data = fetch_hourly_all_years(SEWELLS, years)
    print(data)
    
    # comute annual summaries
    summary = annual_summary(data)
    print(summary)

    #make plots

    fig1, ax = plot_mean_sl(summary)
    fig2, ax = plot_max_surge(summary)
    plt.show()

    #save show the plots
    if True:
        os.makedirs("plots/observed-sl", exist_ok=True)
        fig1.savefig("plots/observed-sl/observed_mean_sl.png", dpi=300)
        fig2.savefig("plots/observed-sl/observed_max_surge.png", dpi=300)
