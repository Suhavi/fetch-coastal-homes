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

# can make global?
KNOWN_STORMS = [
    {"name": "Chesapeake-Potomac", "year": 1933},
    {"name": "Outer Banks hurricane", "year": 1936},
    {"name": "Ash Wednesday", "year": 1962},
    {"name": "Isabel", "year": 2003},
    {"name": "Nor'Ida", "year": 2009},
    {"name": "Irene", "year": 2011}
]

DATUM = 2026
# make GLOBAL
YEARS = range(1927,2026)
# take current year and make GLOBAL


def get_years(times: np.ndarray):
    return np.array([t.year for t in times]) # fetches the t from datetime object


def get_annualsummary(years=YEARS, station=SEWELLS):

    """
    Observed water levels are hourly.
    Annual mean sea level is one value per year. 
    Surge can be computed hourly, 
    but the model usually keeps annual maximum surge, 
    which is also one value per year.
    """
    allyrs_hourly_data = fetch_hourly_all_years(station, years)
    
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


# TODO to be removed

# def make_the_plot(summary, save=False):
#     years = summary["years"].astype(int)
#     mean_sl = summary["mean_sl_ft"]
#     max_surge = summary["max_surge_ft"]

#     mean_color = "#2E7D32"    # deep green
#     surge_color = "#1565C0"   # deep blue
#     storm_color = "#6A1B9A"   # dark purple
#     vline_color = "#777777"

#     fig, ax = plt.subplots(figsize=(12, 5.5))

#     ax.plot(years, mean_sl, color=mean_color, lw=2.8)
#     ax.plot(years, max_surge, color=surge_color, lw=2.8)

#     for storm in KNOWN_STORMS:
#         storm_year = storm["year"]

#         if storm_year in years:
#             i = np.where(years == storm_year)[0][0]
#             yval = max_surge[i]

#             ax.scatter(storm_year, yval, color=storm_color, s=42, zorder=5)

#             ax.plot(
#                 [storm_year, storm_year],
#                 [yval, yval + 0.45],
#                 color=vline_color,
#                 lw=0.7,
#                 zorder=4
#             )

#             ax.text(
#                 storm_year,
#                 yval + 0.5,
#                 storm["name"],
#                 color="black",
#                 fontsize=8.5,
#                 ha="center",
#                 va="bottom"
#             )

#     # direct labels instead of legend
#     ax.text(years[-1] + 1, mean_sl[-1], "Annual mean sea level",
#             color=mean_color, fontsize=10, va="center")

#     ax.text(years[-1] + 1, max_surge[-1], "Annual max anomaly",
#             color=surge_color, fontsize=10, va="center")

#     ax.set_xticks(np.arange(1927, 2026, 10))
#     ax.set_xlim(1927, 2031)

#     # ax.set_xlabel("Year")
#     ax.set_ylabel("Water level (ft)")
#     # ax.set_title("Observed Mean Sea Level and Annual Extreme Water-Level Anomalies")

#     ax.grid(False)
#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)

#     plt.tight_layout()

#     if save:
#         fig.savefig(
#             "../figures/01-hazards/annual_mean_sl_and_max_anomaly.pdf",
#             format="pdf",
#             bbox_inches="tight",
#             dpi=300
#         )

#     plt.show()




# def plot_mean_sl(summary: dict):
#     fig, ax = plt.subplots(figsize=(8,4))

#     ax.plot(summary["years"], summary["mean_sl_ft"], color="0.45", alpha=0.75)
#     ax.set_xlabel("Year")
#     ax.set_ylabel("Annual mean sea level (ft)")
#     ax.grid(True, alpha=0.3)

#     return fig, ax


# def plot_max_surge(summary):

#     fig, ax = plt.subplots(figsize=(8,4))
#     ax.scatter(summary["years"], summary["max_surge_ft"])

#     ## KNOWN STORMS
#     for storm in KNOWN_STORMS:
#         year = storm["year"]
#         idx = np.where(summary["years"] == year)[0]

#         if len(idx) == 0:
#             continue

#         idx = idx[0]
#         x = summary["years"][idx]
#         y = summary["max_surge_ft"][idx]

#         ax.annotate(
#             storm["name"],
#             xy=(x, y),
#             xytext=(x + 2, y + 0.3),
#             arrowprops={"arrowstyle": "-", "color": "red", "linewidth": 0.8},
#             fontsize=8,
#         )


#     ax.set_xlabel("Year")
#     ax.set_ylabel("Annual max storm surge (ft)")
#     ax.grid(True, alpha=0.3)

#     return fig, ax


if __name__ == "__main__":
    print("Computing from Observed Data")
    years = range(1927,2026) # 100 years
    # comute annual summaries
    summary = get_annualsummary(years)
    print(summary)

    #make plots

    # fig1, ax = plot_mean_sl(summary)
    # fig2, ax = plot_max_surge(summary)
    # plt.show()
    # make_the_plot(summary,False)

    #save show the plots
    # if True:
    #     os.makedirs("plots/observed-sl", exist_ok=True)
    #     fig1.savefig("plots/observed-sl/observed_mean_sl.png", dpi=300)
    #     fig2.savefig("plots/observed-sl/observed_max_surge.png", dpi=300)
