from dataclasses import replace

import numpy as np
from scipy.stats import gaussian_kde

from src.hazards.SeaLevel.io_projections import (
    DATUM,
    DEFAULT_END_YEAR,
    DEFAULT_RANDOM_SEED,
    LSLEnsemble,
    load_all_processed_projections,
    load_processed_projection,
)


def _subset_projection(
    ensemble: LSLEnsemble,
    syear: int,
    eyear: int,
) -> LSLEnsemble:
    """Return an ensemble restricted to a complete requested annual range."""
    if eyear < syear:
        raise ValueError(
            f"eyear ({eyear}) must be greater than or equal to syear ({syear})."
        )

    selected = (ensemble.years >= syear) & (ensemble.years <= eyear)
    years = ensemble.years[selected]
    expected_years = np.arange(syear, eyear + 1)
    if not np.array_equal(years, expected_years):
        raise ValueError(
            f"Requested years {syear}-{eyear} are outside the processed "
            f"projection range {ensemble.years[0]}-{ensemble.years[-1]}."
        )

    return replace(
        ensemble,
        years=years.copy(),
        lsl_ft=ensemble.lsl_ft[:, selected].copy(),
    )


def get_projection(
    model: str,
    rcp: str,
    syear: int = DATUM,
    eyear: int = DEFAULT_END_YEAR,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> LSLEnsemble:
    """Load one processed model/RCP ensemble and select the requested years."""
    ensemble = load_processed_projection(
        model=model,
        rcp=rcp,
        syear=DATUM,
        eyear=DEFAULT_END_YEAR,
        datum=DATUM,
        random_seed=random_seed,
    )
    return _subset_projection(ensemble, syear, eyear)


def get_projections(
    syear: int = DATUM,
    eyear: int = DEFAULT_END_YEAR,
    models: list[str] | None = None,
    rcps: list[str] | None = None,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> list[LSLEnsemble]:
    """Load selected processed ensembles and restrict them to requested years."""
    ensembles = load_all_processed_projections(
        syear=DATUM,
        eyear=DEFAULT_END_YEAR,
        datum=DATUM,
        models=models,
        rcps=rcps,
        random_seed=random_seed,
    )
    return [_subset_projection(ensemble, syear, eyear) for ensemble in ensembles]


def get_mean_sea_level_rise_path(
    model: str,
    rcp: str,
    syear: int,
    eyear: int,
    random_seed: int | None = DEFAULT_RANDOM_SEED,
) -> dict:
    """Return the ensemble-mean sea-level-rise path for one model and RCP."""
    ensemble = get_projection(
        model=model,
        rcp=rcp,
        syear=syear,
        eyear=eyear,
        random_seed=random_seed,
    )
    return {
        "years": ensemble.years,
        "sea_level_rise_ft": np.mean(ensemble.lsl_ft, axis=0),
        "model": ensemble.model,
        "rcp": ensemble.rcp,
        "source_file": ensemble.source_file,
    }


def get_projection_summary(ensemble: LSLEnsemble) -> dict:
    """Calculate annual summary statistics for a processed projection ensemble."""
    return {
        "years": ensemble.years.copy(),
        "mean_ft": np.mean(ensemble.lsl_ft, axis=0),
        "median_ft": np.median(ensemble.lsl_ft, axis=0),
        "p05_ft": np.quantile(ensemble.lsl_ft, 0.05, axis=0),
        "p95_ft": np.quantile(ensemble.lsl_ft, 0.95, axis=0),
        "model": ensemble.model,
        "rcp": ensemble.rcp,
        "source_file": ensemble.source_file,
    }


def get_projection_samples(ensemble: LSLEnsemble, year: int) -> np.ndarray:
    """Return all simulation values for one projection year."""
    indices = np.flatnonzero(ensemble.years == year)
    if indices.size == 0:
        raise ValueError(
            f"Year {year} is outside projection range "
            f"{ensemble.years[0]}-{ensemble.years[-1]}."
        )
    return ensemble.lsl_ft[:, indices[0]].copy()


def get_projection_pdf(
    ensemble: LSLEnsemble,
    year: int,
    x_values: np.ndarray | None = None,
    n_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the projection PDF for one year using Gaussian kernel density."""
    samples = get_projection_samples(ensemble, year)
    sample_std = np.std(samples, ddof=1)
    if samples.size < 2 or not np.isfinite(sample_std) or sample_std == 0:
        raise ValueError(
            f"A continuous PDF cannot be estimated for year {year} because "
            "all projection samples are identical."
        )

    if x_values is None:
        if n_points < 2:
            raise ValueError("n_points must be at least 2.")
        x_values = np.linspace(
            np.min(samples) - 3 * sample_std,
            np.max(samples) + 3 * sample_std,
            n_points,
        )
    else:
        x_values = np.asarray(x_values, dtype=float)
        if x_values.ndim != 1 or x_values.size == 0:
            raise ValueError("x_values must be a non-empty one-dimensional array.")

    density = gaussian_kde(samples)(x_values)
    return x_values, density


def get_projection_cdf(
    ensemble: LSLEnsemble,
    year: int,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Calculate empirical probabilities at or below supplied thresholds."""
    samples = get_projection_samples(ensemble, year)
    thresholds = np.atleast_1d(np.asarray(thresholds, dtype=float))
    return np.mean(samples[:, None] <= thresholds[None, :], axis=0)


def get_projection_exceedance(
    ensemble: LSLEnsemble,
    year: int,
    thresholds: np.ndarray,
) -> np.ndarray:
    """Calculate the empirical probability of projection values above thresholds."""
    return 1.0 - get_projection_cdf(ensemble, year, thresholds)

# if __name__ == "__main__":
#     ensembles = get_projections()
