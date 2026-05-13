import numpy as np
from scipy.stats import genextreme
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.hazards.SeaLevel.observed_noaa import annual_summary
from src.hazards.SeaLevel.io_noaa import SEWELLS, fetch_hourly_all_years


def fit_gev(max_surge_ft):
    shape, loc, scale = genextreme.fit(max_surge_ft)
    return {
        "shape":shape,  # tail behaviour
        "loc": loc,     # roughly the center
        "scale": scale  # controls spread
    }

# definition for a 100-year event, annual exceedance prob = 1/100

def return_levels(return_periods, gev):
    return_periods = np.asarray(return_periods)
    annual_exceedance_prob = 1 - 1/return_periods
    return genextreme.ppf(
        annual_exceedance_prob, 
        gev["shape"], 
        loc= gev["loc"], 
        scale= gev["scale"]
        )

def observed_return_periods(max_surge_ft):
    surge = np.sort(max_surge_ft)
    n = len(surge)

    ranks = np.arange(1, n + 1)
    nonexceedance_prob = ranks / (n + 1)
    exceedance_prob = 1 - nonexceedance_prob

    return_periods = 1 / exceedance_prob

    return return_periods, surge


def plot_return_period(max_surge_ft, gev):
    fig, ax = plt.subplots(figsize=(7, 4))

    obs_periods, obs_surge = observed_return_periods(max_surge_ft)
    ax.scatter(
        obs_periods,
        obs_surge,
        color="0.45",
        s=20,
        label="Observed annual maxima",
    )

    model_periods = np.logspace(0, 3, 200)
    model_surge = return_levels(model_periods, gev)

    ax.plot(
        model_periods,
        model_surge,
        color="tab:red",
        linewidth=2,
        label="GEV fit",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Storm surge (ft)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    return fig, ax

if __name__ == "__main__":
    print("Fetching observed water levels")

    years = range(1927, 2026)

    # Get observed annual max surge
    data = fetch_hourly_all_years(SEWELLS, years)
    summary = annual_summary(data)

    max_surge_ft = summary["max_surge_ft"]

    # Fit an extreme-value distribution
    print("Fitting GEV storm surge model")
    gev = fit_gev(max_surge_ft)
    print(gev)

    # Convert fitted distribution into return levels
    # and
    # plot

    fig, ax = plot_return_period(max_surge_ft, gev)
    os.makedirs("plots", exist_ok=True)
    fig.savefig("plots/storm_surge_return_period.png", dpi=300)

    plt.show()
