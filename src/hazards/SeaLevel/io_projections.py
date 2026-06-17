import numpy as np
from dataclasses import dataclass
import json
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

CM_TO_FT = 0.0328084
MM_TO_FT = 0.00328084
M_TO_FT = 3.28084
CONFIG_PATH = PROJECT_ROOT / "config" / "model_file_map.json"
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "msl"
with CONFIG_PATH.open() as config_file:
    simMap = json.load(config_file)

# Local Sea Level Simulation

@dataclass
class LSLEnsemble:
    years: np.ndarray
    rcp: float
    model: str
    lsl_ft: np.ndarray
    source_file: str
    
    @property
    def n_sims(self):
        return self.lsl_ft.shape[0]
    

# def load_model_map(CONFIG_PATH):
#     """
#     Fetches model and corresponding rcp data #TODO: understand subsidence
#     """
#     with open(CONFIG_PATH, "r") as f:
#         return json.load(f)


def decadal_to_annual(values: np.ndarray, years_in: np.ndarray, years_out: np.ndarray):
    """
    Convert projection data to annual values with linear extrapolation.

    Values within the source-year range are linearly interpolated. Requested
    years outside that range continue using the slope between the nearest two
    source years instead of holding the boundary value constant.
    """
    years_in = np.asarray(years_in)
    years_out = np.asarray(years_out)
    values = np.atleast_2d(values)

    if years_in.size < 2:
        raise ValueError("At least two source years are required for extrapolation.")

    annual_values = []
    before_source = years_out < years_in[0]
    after_source = years_out > years_in[-1]

    for sim in values:
        annual = np.interp(years_out, years_in, sim)
        start_slope = (sim[1] - sim[0]) / (years_in[1] - years_in[0])
        end_slope = (sim[-1] - sim[-2]) / (years_in[-1] - years_in[-2])

        annual[before_source] = sim[0] + start_slope * (
            years_out[before_source] - years_in[0]
        )
        annual[after_source] = sim[-1] + end_slope * (
            years_out[after_source] - years_in[-1]
        )
        annual_values.append(annual)

    return np.vstack(annual_values)


def convert_to_ft(values, curr_units):
    """
    Converts values to ft
    """
    if curr_units == "cm":
        return values*CM_TO_FT
    elif curr_units == "mm":
        return values*MM_TO_FT
    elif curr_units == "m":
        return values*M_TO_FT
    elif curr_units == "ft":
        return values
    else:
        raise ValueError(f"Unknown sea-level unit: {curr_units}")


def get_k14_background(syear: int, eyear: int) -> np.ndarray:
    """
    Load the K14 background local sea-level component used with BRICK.

    Returns an annualized array in feet with shape ``(n_sims, n_years)``.
    This mirrors James's ``get_K14_subsidence`` helper, using the
    ``LSLProj_bkgd_299_rcp26.csv`` file.
    """
    source_file = os.path.join(DATA_DIR, "LSLProj_bkgd_299_rcp26.csv")
    raw = np.genfromtxt(source_file, delimiter=",", names=True)
    years_in = np.array([int(name) for name in raw.dtype.names])
    values_cm = np.column_stack([raw[name] for name in raw.dtype.names])

    if values_cm.ndim == 1:
        values_cm = values_cm.reshape(1, -1)

    years_out = np.arange(syear, eyear + 1)
    background_cm = decadal_to_annual(values_cm, years_in, years_out)
    return convert_to_ft(background_cm, "cm")


def fetch_simulation(
    model: str,
    rcp: str,
    syear: int | None = None,
    eyear: int | None = None,
    relative_to_start: bool = True,
    apply_brick_background: bool | None = None,
    random_seed: int | None = None,
) -> LSLEnsemble:
    """
    Load one local sea-level-rise ensemble from the model map.

    CSV inputs are expected to have years as column headers and simulations as
    rows. 
    NetCDF inputs are expected to contain the configured variable plus a time
    coordinate, usually ``time_proj`` for the BRICK files. By default, BRICK
    trajectories are adjusted with randomly sampled K14 background trajectories,
    matching James's implementation.

    """

    if model not in simMap:
        raise KeyError(f"Unknown SLR model '{model}'. Options: {list(simMap)}")
    if rcp not in simMap[model]:
        raise KeyError(f"Unknown RCP '{rcp}' for model '{model}'.")

    entry = simMap[model][rcp]
    filename = entry["filename"]
    file_type = entry["file_type"]
    units = entry["units"]
    source_model = entry.get("source_model", model)
    source_file = os.path.join(DATA_DIR, filename)

    if file_type == "csv":
        raw = np.genfromtxt(source_file, delimiter=",", names=True)
        years_in = np.array([int(name) for name in raw.dtype.names])
        values = np.column_stack([raw[name] for name in raw.dtype.names])

        if values.ndim == 1:
            values = values.reshape(1, -1)

        valid_years = ~np.all(np.isnan(values), axis=0)
        years_in = years_in[valid_years]
        values = values[:, valid_years]


    elif file_type == "netcdf":
        try:
            import xarray as xr
        except ImportError as exc:
            raise ImportError(
                "Loading NetCDF SLR data requires xarray. "
                "Install the environment in environment.yml before using BRICK files."
            ) from exc

        variable = entry["variable"]

        with xr.open_dataset(source_file) as ds:
            years_in = (ds["time_proj"].values if "time_proj" in ds else ds["time"].values)
            years_in = years_in.astype(int)
            values_raw = ds[variable].values

        values_raw = np.asarray(values_raw)
        if values_raw.ndim == 1:
            values = values_raw.reshape(1, -1)
        elif values_raw.shape[0] == len(years_in):
            values = values_raw.T
        elif values_raw.shape[-1] == len(years_in):
            values = values_raw.reshape(-1, len(years_in))
        else:
            raise ValueError(f"Cannot align variable '{variable}' with time axis in {filename}.")

    else:
        raise ValueError(f"Unsupported file type '{file_type}' for {model} {rcp}.")


    if syear is None:
        syear = int(np.min(years_in))
    if eyear is None:
        eyear = int(np.max(years_in))

    years_out = np.arange(syear, eyear + 1)
    lsl = decadal_to_annual(values, years_in, years_out)
    lsl_ft = convert_to_ft(lsl, units)

    if apply_brick_background is None:
        apply_brick_background = file_type == "netcdf" and model.startswith("BRICK")

    if apply_brick_background:
        rng = np.random.default_rng(random_seed)
        background_ft = get_k14_background(syear, eyear)
        sample_idx = rng.integers(0, background_ft.shape[0], size=lsl_ft.shape[0])
        lsl_ft = lsl_ft - background_ft[sample_idx, :]

    if relative_to_start:
        lsl_ft = lsl_ft - lsl_ft[:, [0]]

    return LSLEnsemble(
        years=years_out,
        rcp=float(rcp.replace("rcp", "")),
        model=source_model,
        lsl_ft=lsl_ft,
        source_file=source_file,
    )

# WITH UNCERTAINTY 
def fetch_lsl(syear=2026, eyear=2160, models=None, rcps=None):
    ensembles = []
    for model, rcp_map in simMap.items():
        if models is not None and model not in models:
            continue
        for rcp in rcp_map:
            if rcps is not None and rcp not in rcps:
                continue
            ensembles.append(fetch_simulation(model, rcp, syear, eyear))
    return ensembles

# WITHOUT UNCERTAINTY
def mean_sea_level_rise_path(model: str, rcp: str, syear: int, eyear: int):
    ens = fetch_simulation(model, rcp, syear, eyear)
    return {
        "years": ens.years,
        "sea_level_rise_ft": np.mean(ens.lsl_ft, axis=0),
        "model": ens.model,
        "rcp": ens.rcp,
        "source_file": ens.source_file,
    }



if __name__ == "__main__":
    ens = fetch_simulation("BRICK-Slow", "rcp6.0", 2026, 2176)
    print(ens.model, ens.rcp, ens.years[0], ens.years[-1], ens.lsl_ft.shape)
    # print(ens.lsl_ft)

    ens = fetch_simulation("BRICK-Fast","rcp6.0", 2026, 2176)
    print(ens.model, ens.rcp, ens.years[0], ens.years[-1], ens.lsl_ft.shape)

    ens = fetch_simulation("K14","rcp6.0", 2026, 2176)
    print(ens.model, ens.rcp, ens.years[0], ens.years[-1], ens.lsl_ft.shape)

    ens = fetch_simulation("DP16","rcp6.0", 2026, 2176)
    print(ens.model, ens.rcp, ens.years[0], ens.years[-1], ens.lsl_ft.shape)
