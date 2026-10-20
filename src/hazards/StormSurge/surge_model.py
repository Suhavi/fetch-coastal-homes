import numpy as np
from scipy.stats import genextreme
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap, Normalize

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.hazards.SeaLevel.getfunctions_noaa import get_annualsummary
from src.hazards.SeaLevel.io_noaa import SEWELLS

# WITH UNCERTAINTY PARAMETERS
# TODO: MCM AS PER MAGGIE'S CODE.



# WITHOUT UNCERTAINTY PARAMETERS
# FIT THE GEV
def fit_gev(max_surge_ft):
    shape, loc, scale = genextreme.fit(max_surge_ft)
    return {"shape":shape, "loc": loc,  "scale": scale}

def make_gev_distribution(shape, loc, scale):
    # returns gev as an object to be able to call .pdf, .cdf on it
    return genextreme(c=shape, loc=loc, scale=scale)

def observed_surge_distribution(station=SEWELLS, years=range(1927, 2026)):
    summary = get_annualsummary(years=years, station=station)
    gev_params = fit_gev(summary["max_surge_ft"])
    gev_dist = make_gev_distribution(**gev_params)
    return gev_params, gev_dist


def projected_water_level_distributions(slr_path, c=1.5, station=SEWELLS, observed_years=range(1927, 2026)):
    gev_params, _ = observed_surge_distribution(station=station, years=observed_years)

    return [
        make_gev_distribution(
            gev_params["shape"],
            gev_params["loc"] + (1 + c) * slr_ft,
            gev_params["scale"],
        )
        for slr_ft in slr_path["sea_level_rise_ft"]
    ]

# when defining a state of the world
# years
# site
# slr_emsemble choice
# ---> defines the gev distribution

def nonstationary_gev(slr_ensemble, years, station = SEWELLS):
    """
    slr_ensemble = {'years':array(), 'sea_level_rise_ft':array(),'model':,'rcp','source_file'}
    """
    summary = get_annualsummary(years=years, station=station)
    gev_params = fit_gev(summary["max_surge_fit"])

    gev_params_at_t = []
    for i in range(slr_ensemble['years']):
        locshapesscale_t = {"loc": gev_params['loc']+slr_ensemble["sea_level_rise_ft"][i],
                            "shape": gev_params["shape"],
                            "scale": gev_params["scale"]
                            }
        gev_params_at_t.append(locshapesscale_t)

    ## compact, less understandable code: use slr_ensemble.copy() and update key.
    gev_at_time_t = {"years" : slr_ensemble['years'],
                    'gev_t':  gev_params_at_t,
                   "model" : slr_ensemble["model"],
                   "sea_level_rise_ft": slr_ensemble["slr_ensemble"],
                   "rcp": slr_ensemble['rcp'],
                   'source_file': slr_ensemble["source_file"]
                   }

    return gev_at_time_t

# def observed_surge_distribution(station=SEWELLS, years=range(1927, 2026)):
#     # data = fetch_hourly_all_years(station, years)
#     summary = get_annualsummary(years, station)
#     return fit_gev_distribution(summary["max_surge_ft"])


# definition for a 100-year event, annual exceedance prob = 1/100
# FEMA FLOODMAPS USE 100-year floods.
# if return period = 100, probobility in one year = 1/100

# 100-year return level = flood anomaly height exceeded with 1% annual probability
def return_levels(return_periods, gev_params):
    # Given return periods, what flood heights correspond to those probabilities?
    # find all flood heights which classify as a 100-period flood
    return_periods = np.asarray(return_periods)
    annual_non_exceedance_prob = 1 - 1/return_periods
    return genextreme.ppf(
        annual_non_exceedance_prob, 
        gev_params["shape"], 
        loc= gev_params["loc"], 
        scale= gev_params["scale"]
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
    summary = get_annualsummary(years, SEWELLS)
    max_surge_ft = summary["max_surge_ft"]
    # fitting parameters
    print("Fitting GEV storm surge model")
    gev = fit_gev(max_surge_ft)
    print(gev)    
    gev_dist = make_gev_distribution(gev['shape'], gev['loc'], gev['scale'])
    

    # Convert fitted distribution into return levels
    # and
    # plot

    fig, ax = plot_return_period(max_surge_ft, gev)
    os.makedirs("plots", exist_ok=True)
    # fig.savefig("plots/storm_surge_return_period.png", dpi=300)

    plt.show()
