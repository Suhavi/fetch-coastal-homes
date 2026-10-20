from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "config" / "model_file_map.json"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "msl"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed" / "msl"

DATUM = 2026
DEFAULT_END_YEAR = 2176
# Reproduce the random pairing of BRICK simulations with K14 background paths.
DEFAULT_RANDOM_SEED = 1
PROCESSING_VERSION = "1"

CM_TO_FT = 0.0328084
MM_TO_FT = 0.00328084
M_TO_FT = 3.28084


def load_model_map(config_path: Path = CONFIG_PATH) -> dict:
    """Load the raw projection file map."""
    with config_path.open() as config_file:
        return json.load(config_file)


MODEL_FILE_MAP = load_model_map()


@dataclass
class LSLEnsemble:
    """Store one processed local sea-level projection ensemble and its metadata."""

    years: np.ndarray
    rcp: float
    model: str
    lsl_ft: np.ndarray
    source_file: str
    datum: int = DATUM
    units: str = "ft"
    apply_brick_background: bool = False
    random_seed: int = DEFAULT_RANDOM_SEED
    processing_version: str = PROCESSING_VERSION
    model_key: str | None = None
    rcp_key: str | None = None

    @property
    def n_sims(self):
        """Return the number of simulations in the ensemble."""
        return self.lsl_ft.shape[0]


def _require_xarray():
    """Import xarray or raise an installation-focused error."""
    try:
        import xarray as xr
    except ImportError as exc:
        raise ImportError(
            "Reading or writing NetCDF projection data requires xarray and "
            "netCDF4. Install the environment in environment.yml."
        ) from exc
    return xr


def _validate_year_range(syear: int, eyear: int, datum: int) -> None:
    """Require an ordered processing range that contains the datum year."""
    if eyear < syear:
        raise ValueError(
            f"eyear ({eyear}) must be greater than or equal to syear ({syear})."
        )
    if datum < 0:
        raise ValueError("datum must be a positive calendar year.")
    if not syear <= datum <= eyear:
        raise ValueError(
            f"Processed year range {syear}-{eyear} must include datum year {datum}."
        )


def decadal_to_annual(
    values: np.ndarray,
    years_in: np.ndarray,
    years_out: np.ndarray,
) -> np.ndarray:
    """Interpolate projection trajectories to annual values and extrapolate linearly."""
    years_in = np.asarray(years_in, dtype=int)
    years_out = np.asarray(years_out, dtype=int)
    values = np.atleast_2d(np.asarray(values, dtype=float))

    if years_in.ndim != 1:
        raise ValueError("Source years must be one-dimensional.")
    if values.shape[1] != years_in.size:
        raise ValueError("Projection columns must align with the source years.")
    if years_in.size < 2:
        raise ValueError("At least two source years are required for extrapolation.")

    order = np.argsort(years_in)
    years_in = years_in[order]
    values = values[:, order]

    if np.any(np.diff(years_in) <= 0):
        raise ValueError("Source years must be unique and strictly increasing.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Raw projection values contain missing or non-finite values.")

    annual_values = []
    before_source = years_out < years_in[0]
    after_source = years_out > years_in[-1]

    for simulation in values:
        annual = np.interp(years_out, years_in, simulation)
        start_slope = (simulation[1] - simulation[0]) / (years_in[1] - years_in[0])
        end_slope = (simulation[-1] - simulation[-2]) / (
            years_in[-1] - years_in[-2]
        )

        annual[before_source] = simulation[0] + start_slope * (
            years_out[before_source] - years_in[0]
        )
        annual[after_source] = simulation[-1] + end_slope * (
            years_out[after_source] - years_in[-1]
        )
        annual_values.append(annual)

    return np.vstack(annual_values)


def convert_to_ft(values: np.ndarray, current_units: str) -> np.ndarray:
    """Convert a projection array to feet."""
    conversion_factors = {
        "cm": CM_TO_FT,
        "mm": MM_TO_FT,
        "m": M_TO_FT,
        "ft": 1.0,
    }
    try:
        factor = conversion_factors[current_units]
    except KeyError as exc:
        raise ValueError(f"Unknown sea-level unit: {current_units}") from exc
    return np.asarray(values, dtype=float) * factor


def _load_csv_projection(source_file: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a CSV ensemble and discard columns containing only missing values."""
    raw = np.genfromtxt(source_file, delimiter=",", names=True)
    years = np.array([int(name) for name in raw.dtype.names], dtype=int)
    values = np.column_stack([raw[name] for name in raw.dtype.names])

    if values.ndim == 1:
        values = values.reshape(1, -1)

    valid_years = ~np.all(np.isnan(values), axis=0)
    return values[:, valid_years], years[valid_years]


def _load_netcdf_projection(
    source_file: Path,
    variable: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one configured projection variable from a NetCDF ensemble."""
    xr = _require_xarray()

    with xr.open_dataset(source_file) as dataset:
        time_name = "time_proj" if "time_proj" in dataset else "time"
        if time_name not in dataset:
            raise ValueError(f"No time coordinate found in {source_file.name}.")
        if variable not in dataset:
            raise KeyError(f"Variable '{variable}' not found in {source_file.name}.")

        years_raw = np.asarray(dataset[time_name].values)
        if not np.allclose(years_raw, years_raw.astype(int)):
            raise ValueError(f"Projection years in {source_file.name} are not integers.")
        years = years_raw.astype(int)

        data = dataset[variable].squeeze(drop=True)
        if time_name not in data.dims:
            raise ValueError(
                f"Variable '{variable}' does not use the '{time_name}' dimension."
            )
        simulation_dims = [dim for dim in data.dims if dim != time_name]
        data = data.transpose(*simulation_dims, time_name)
        values = np.asarray(data.values, dtype=float).reshape(-1, len(years))

    return values, years


def _load_raw_projection(
    model: str,
    rcp: str,
) -> tuple[np.ndarray, np.ndarray, dict, Path]:
    """Load one configured raw projection and return its values and metadata."""
    if model not in MODEL_FILE_MAP:
        raise KeyError(f"Unknown SLR model '{model}'. Options: {list(MODEL_FILE_MAP)}")
    if rcp not in MODEL_FILE_MAP[model]:
        raise KeyError(f"Unknown RCP '{rcp}' for model '{model}'.")

    entry = MODEL_FILE_MAP[model][rcp]
    source_file = RAW_DATA_DIR / entry["filename"]
    if not source_file.exists():
        raise FileNotFoundError(f"Raw projection file not found: {source_file}")

    if entry["file_type"] == "csv":
        values, years = _load_csv_projection(source_file)
    elif entry["file_type"] == "netcdf":
        values, years = _load_netcdf_projection(source_file, entry["variable"])
    else:
        raise ValueError(
            f"Unsupported file type '{entry['file_type']}' for {model} {rcp}."
        )

    return values, years, entry, source_file


def get_k14_background(syear: int, eyear: int) -> np.ndarray:
    """Load, annualize, and convert the K14 background component to feet."""
    source_file = RAW_DATA_DIR / "LSLProj_bkgd_299_rcp26.csv"
    values_cm, years_in = _load_csv_projection(source_file)
    years_out = np.arange(syear, eyear + 1)
    background_cm = decadal_to_annual(values_cm, years_in, years_out)
    return convert_to_ft(background_cm, "cm")


def _uses_brick_background(model: str, entry: dict) -> bool:
    """Return whether a configured projection needs the K14 background adjustment."""
    return entry["file_type"] == "netcdf" and model.startswith("BRICK")


def _normalized_seed(random_seed: int | None) -> int:
    """Replace an omitted random seed with the reproducible project default."""
    return DEFAULT_RANDOM_SEED if random_seed is None else int(random_seed)


def _validate_ensemble(ensemble: LSLEnsemble) -> None:
    """Validate the required shape, units, years, values, and datum normalization."""
    years = np.asarray(ensemble.years)
    values = np.asarray(ensemble.lsl_ft)

    if years.ndim != 1 or years.size == 0:
        raise ValueError("Processed years must be a non-empty one-dimensional array.")
    if values.ndim != 2 or values.shape[1] != years.size:
        raise ValueError("Processed values must have shape (simulations, years).")
    if np.any(np.diff(years) != 1):
        raise ValueError("Processed years must be unique, increasing, and annual.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Processed projections contain missing or non-finite values.")
    if ensemble.units != "ft":
        raise ValueError("Processed projections must use feet.")
    if ensemble.datum not in years:
        raise ValueError(f"Datum year {ensemble.datum} is not present in processed years.")

    datum_index = int(np.where(years == ensemble.datum)[0][0])
    if not np.allclose(values[:, datum_index], 0.0, atol=1e-10):
        raise ValueError("Processed projections must equal zero in the datum year.")


def process_projection(
    model: str,
    rcp: str,
    syear: int = DATUM,
    eyear: int = DEFAULT_END_YEAR,
    datum: int = DATUM,
    apply_brick_background: bool | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> LSLEnsemble:
    """Transform one raw ensemble into annual, datum-relative values in feet."""
    _validate_year_range(syear, eyear, datum)
    values, years_in, entry, source_file = _load_raw_projection(model, rcp)

    if apply_brick_background is None:
        apply_brick_background = _uses_brick_background(model, entry)
    seed = _normalized_seed(random_seed)

    years_out = np.arange(syear, eyear + 1)

    annual_values = decadal_to_annual(values, years_in, years_out)
    lsl_ft = convert_to_ft(annual_values, entry["units"])

    if apply_brick_background:
        background_ft = get_k14_background(syear, eyear)
        rng = np.random.default_rng(seed)
        sample_indices = rng.integers(
            0,
            background_ft.shape[0],
            size=lsl_ft.shape[0],
        )
        lsl_ft = lsl_ft - background_ft[sample_indices, :]

    datum_index = int(np.where(years_out == datum)[0][0])
    lsl_ft = lsl_ft - lsl_ft[:, [datum_index]]

    ensemble = LSLEnsemble(
        years=years_out,
        rcp=float(rcp.removeprefix("rcp")),
        model=entry.get("source_model", model),
        lsl_ft=lsl_ft,
        source_file=source_file.name,
        datum=datum,
        units="ft",
        apply_brick_background=bool(apply_brick_background),
        random_seed=seed,
        processing_version=PROCESSING_VERSION,
        model_key=model,
        rcp_key=rcp,
    )
    _validate_ensemble(ensemble)
    return ensemble


def _filename_token(value: str) -> str:
    """Convert a model or RCP label into a filesystem-safe filename token."""
    return value.lower().replace(" ", "-").replace(".", "p")


def processed_projection_path(
    model: str,
    rcp: str,
    syear: int = DATUM,
    eyear: int = DEFAULT_END_YEAR,
    datum: int = DATUM,
    apply_brick_background: bool | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> Path:
    """Build the unique output path for one set of processing choices."""
    if model not in MODEL_FILE_MAP or rcp not in MODEL_FILE_MAP[model]:
        raise KeyError(f"Unknown model/RCP combination: {model} {rcp}")

    entry = MODEL_FILE_MAP[model][rcp]
    if apply_brick_background is None:
        apply_brick_background = _uses_brick_background(model, entry)
    seed = _normalized_seed(random_seed)
    background = "k14-background" if apply_brick_background else "no-background"

    filename = (
        f"{_filename_token(model)}_{_filename_token(rcp)}_"
        f"{syear}-{eyear}_datum-{datum}_{background}_seed-{seed}.nc"
    )
    return PROCESSED_DATA_DIR / filename


def save_processed_projection(
    ensemble: LSLEnsemble,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Save a validated processed ensemble as a compressed NetCDF file."""
    _validate_ensemble(ensemble)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Processed projection already exists: {output_path}")

    xr = _require_xarray()
    dataset = xr.Dataset(
        data_vars={
            "lsl_ft": (
                ("simulation", "year"),
                ensemble.lsl_ft,
                {"long_name": "local sea-level rise", "units": "ft"},
            )
        },
        coords={
            "simulation": np.arange(ensemble.n_sims),
            "year": ensemble.years,
        },
        attrs={
            "model": ensemble.model,
            "model_key": ensemble.model_key or ensemble.model,
            "rcp": ensemble.rcp,
            "rcp_key": ensemble.rcp_key or str(ensemble.rcp),
            "source_file": ensemble.source_file,
            "units": ensemble.units,
            "datum": ensemble.datum,
            "start_year": int(ensemble.years[0]),
            "end_year": int(ensemble.years[-1]),
            "apply_brick_background": int(ensemble.apply_brick_background),
            "random_seed": ensemble.random_seed,
            "processing_version": ensemble.processing_version,
        },
    )

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    try:
        dataset.to_netcdf(
            temporary_path,
            encoding={"lsl_ft": {"zlib": True, "complevel": 4, "shuffle": True}},
        )
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        dataset.close()

    return output_path


def load_processed_projection(
    model: str,
    rcp: str,
    syear: int = DATUM,
    eyear: int = DEFAULT_END_YEAR,
    datum: int = DATUM,
    apply_brick_background: bool | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> LSLEnsemble:
    """Load one processed projection ensemble from disk."""
    path = processed_projection_path(
        model=model,
        rcp=rcp,
        syear=syear,
        eyear=eyear,
        datum=datum,
        apply_brick_background=apply_brick_background,
        random_seed=random_seed,
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Processed projection not found: {path}. "
            "Run build_processed_projection() or build_all_processed_projections() first."
        )

    xr = _require_xarray()
    with xr.open_dataset(path) as dataset:
        values = np.asarray(
            dataset["lsl_ft"].transpose("simulation", "year").values,
            dtype=float,
        )
        years = np.asarray(dataset["year"].values, dtype=int)
        attrs = dict(dataset.attrs)

    ensemble = LSLEnsemble(
        years=years,
        rcp=float(attrs["rcp"]),
        model=str(attrs["model"]),
        lsl_ft=values,
        source_file=str(attrs["source_file"]),
        datum=int(attrs["datum"]),
        units=str(attrs["units"]),
        apply_brick_background=bool(attrs["apply_brick_background"]),
        random_seed=int(attrs["random_seed"]),
        processing_version=str(attrs["processing_version"]),
        model_key=str(attrs["model_key"]),
        rcp_key=str(attrs["rcp_key"]),
    )
    _validate_ensemble(ensemble)
    return ensemble


def build_processed_projection(
    model: str,
    rcp: str,
    syear: int = DATUM,
    eyear: int = DEFAULT_END_YEAR,
    datum: int = DATUM,
    apply_brick_background: bool | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    force: bool = False,
) -> LSLEnsemble:
    """Process and persist one model/RCP ensemble, reusing it when available."""
    output_path = processed_projection_path(
        model=model,
        rcp=rcp,
        syear=syear,
        eyear=eyear,
        datum=datum,
        apply_brick_background=apply_brick_background,
        random_seed=random_seed,
    )
    if output_path.exists() and not force:
        return load_processed_projection(
            model=model,
            rcp=rcp,
            syear=syear,
            eyear=eyear,
            datum=datum,
            apply_brick_background=apply_brick_background,
            random_seed=random_seed,
        )

    ensemble = process_projection(
        model=model,
        rcp=rcp,
        syear=syear,
        eyear=eyear,
        datum=datum,
        apply_brick_background=apply_brick_background,
        random_seed=random_seed,
    )
    save_processed_projection(ensemble, output_path, overwrite=force)
    return ensemble


def build_all_processed_projections(
    syear: int = DATUM,
    eyear: int = DEFAULT_END_YEAR,
    datum: int = DATUM,
    models: list[str] | None = None,
    rcps: list[str] | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
    force: bool = False,
) -> list[LSLEnsemble]:
    """Process and save all selected model/RCP ensembles."""
    ensembles = []
    for model, rcp_map in MODEL_FILE_MAP.items():
        if models is not None and model not in models:
            continue
        for rcp in rcp_map:
            if rcps is not None and rcp not in rcps:
                continue
            ensembles.append(
                build_processed_projection(
                    model=model,
                    rcp=rcp,
                    syear=syear,
                    eyear=eyear,
                    datum=datum,
                    random_seed=random_seed,
                    force=force,
                )
            )
    return ensembles


def load_all_processed_projections(
    syear: int = DATUM,
    eyear: int = DEFAULT_END_YEAR,
    datum: int = DATUM,
    models: list[str] | None = None,
    rcps: list[str] | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> list[LSLEnsemble]:
    """Load all selected processed model/RCP ensembles."""
    ensembles = []
    for model, rcp_map in MODEL_FILE_MAP.items():
        if models is not None and model not in models:
            continue
        for rcp in rcp_map:
            if rcps is not None and rcp not in rcps:
                continue
            ensembles.append(
                load_processed_projection(
                    model=model,
                    rcp=rcp,
                    syear=syear,
                    eyear=eyear,
                    datum=datum,
                    random_seed=random_seed,
                )
            )
    return ensembles


if __name__ == "__main__":
    built_ensembles = build_all_processed_projections(force=True)
    for built_ensemble in built_ensembles:
        print(
            built_ensemble.model,
            built_ensemble.rcp,
            built_ensemble.years[0],
            built_ensemble.years[-1],
            built_ensemble.lsl_ft.shape,
        )
