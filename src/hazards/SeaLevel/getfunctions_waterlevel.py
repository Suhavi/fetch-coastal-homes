from dataclasses import dataclass

import numpy as np
from scipy.stats import genextreme

from src.hazards.SeaLevel.getfunctions_projections import get_projection_samples
from src.hazards.SeaLevel.io_projections import LSLEnsemble


@dataclass
class WongBMAStormSurge:
    """Wong (2018) all-covariate storm-surge return levels for 2065."""

    return_periods: np.ndarray
    return_levels_ft: np.ndarray
    year: int = 2065
    station: str = "Sewells Point, Norfolk, VA"
    units: str = "ft above mean sea level"
    model: str = "Wong BMA"
    rcp: float | None = None
    datum: int | None = None
    slr_ft: np.ndarray | None = None

    def __post_init__(self):
        if not np.all(np.isfinite(self.return_levels_ft)):
            raise ValueError("Return levels must be finite.")
        if self.return_levels_ft.ndim != 2:
            raise ValueError("Return levels must be an ensemble-by-period matrix.")
        if self.return_levels_ft.shape[1] != self.return_periods.size:
            raise ValueError("Return periods and return levels do not match.")
        if np.any(np.diff(self.return_periods) <= 0):
            raise ValueError("Return periods must increase.")
        if np.any(np.diff(self.return_levels_ft, axis=1) <= 0):
            raise ValueError("Every ensemble return-level curve must increase.")

    @property
    def exceedance_probabilities(self):
        return 1 / self.return_periods

    @property
    def minimum_height_ft(self):
        """Lowest height supported by every ensemble member (2-year tail)."""
        return float(np.max(self.return_levels_ft[:, 0]))

    def component_sf(self, height):
        """Interpolate one exceedance curve per BMA uncertainty member.

        Ruckert, Srikrishnan & Keller (2019), supporting code
        ReadAnalysis_LocalSeaLevel_StormSurge.R, lines 992-1015. Values above
        a member's 1,000-year return level are assigned zero, as in that code.
        """
        height = np.asarray(height, dtype=float)
        if np.any(height < self.minimum_height_ft):
            raise ValueError(
                f"Heights must be at least {self.minimum_height_ft:.3f} ft; "
                "the published ensemble starts at the 2-year return level."
            )

        flat_height = height.reshape(-1)
        probabilities = np.empty((flat_height.size, self.return_levels_ft.shape[0]))
        for member, return_levels in enumerate(self.return_levels_ft):
            probabilities[:, member] = np.interp(
                flat_height,
                return_levels,
                self.exceedance_probabilities,
                right=0.0,
            )
        return probabilities.reshape(height.shape + (self.return_levels_ft.shape[0],))

    def sf(self, height):
        """Average annual exceedance probability at each height."""
        return np.mean(self.component_sf(height), axis=-1)

    def component_cdf(self, height):
        return 1 - self.component_sf(height)

    def cdf(self, height):
        return 1 - self.sf(height)


@dataclass
class SLRSurgeDistribution:
    """Annual maximum water level relative to 2026 mean sea level, in feet."""

    surge: object
    slr_ft: np.ndarray
    year: int
    model: str
    rcp: float
    datum: int
    units: str

    def component_sf(self, height):
        """Return one shifted surge exceedance curve per SLR realization."""
        height = np.asarray(height, dtype=float)
        return self.surge.sf(height[..., None] - self.slr_ft)

    def sf(self, height):
        return np.mean(self.component_sf(height), axis=-1)

    def component_cdf(self, height):
        """Return one shifted surge CDF per SLR realization."""
        height = np.asarray(height, dtype=float)
        return self.surge.cdf(height[..., None] - self.slr_ft)

    def cdf(self, height):
        return np.mean(self.component_cdf(height), axis=-1)

    def component_pdf(self, height):
        """Return one shifted surge density per SLR realization."""
        height = np.asarray(height, dtype=float)
        return self.surge.pdf(height[..., None] - self.slr_ft)

    def pdf(self, height):
        return np.mean(self.component_pdf(height), axis=-1)


# STEP 1: Shift the stationary annual-maximum GEV by every SLR realization and
# average probabilities at each height. Ruckert, Oddo & Keller (2017), Methods:
# "Estimating current and future flood heights," paragraphs 5-8 and Fig. 3.
def combine_slr_surge(
    projection: LSLEnsemble,
    gev_posterior,
    year: int,
) -> SLRSurgeDistribution:
    """Combine one empirical SLR ensemble with the fitted stationary GEV.

    Assumptions: SLR and residual annual maxima are additive and independent;
    both inputs are in feet; and output elevations use the projection's 2026
    mean-sea-level datum. Following Ruckert et al. (2017), this first version
    uses one baseline surge curve. Its parameters are posterior medians, so GEV
    parameter uncertainty and the BMA pairing in Ruckert et al. (2019), Methods:
    "Combining sea-level and storm surge projections," are not included here.
    Their code samples surge curves without replacement to match the SLR
    ensemble size and pairs them one-to-one; that requires a separate function.
    """
    if projection.units != "ft" or projection.datum != 2026:
        raise ValueError("The SLR projection must be in feet relative to 2026.")

    posterior = gev_posterior.posterior
    location = float(np.median(posterior["location"].values))
    scale = float(np.median(posterior["scale"].values))
    shape = float(np.median(posterior["shape"].values))

    # scipy.stats uses c = -xi for the conventional GEV shape xi.
    surge = genextreme(c=-shape, loc=location, scale=scale)
    slr_ft = get_projection_samples(projection, year)

    return SLRSurgeDistribution(
        surge=surge,
        slr_ft=slr_ft,
        year=year,
        model=projection.model,
        rcp=projection.rcp,
        datum=projection.datum,
        units=projection.units,
    )


############ WITH TONY'S NONSTATIONARY STORM SURGE ############

# STEP 1: Draw SLR realizations without replacement to match Tony's 8,021 BMA
# members. STEP 2: Pair rows one-to-one and shift every return-level curve.
# STEP 3: Preserve every paired curve; WongBMAStormSurge.sf averages their
# exceedance probabilities at each height. Ruckert, Srikrishnan & Keller
# (2019), ReadAnalysis_LocalSeaLevel_StormSurge.R, lines 418-454 and 992-1015.
def combine_slr_wong(
    projections: list[LSLEnsemble],
    storm_surge: WongBMAStormSurge,
    seed: int = 713,
) -> list[WongBMAStormSurge]:
    """Combine Tony's 2065 BMA ensemble with any collection of SLR ensembles."""
    rng = np.random.default_rng(seed)
    n_members = storm_surge.return_levels_ft.shape[0]
    distributions = []

    for projection in projections:
        if projection.units != "ft" or projection.datum != 2026:
            raise ValueError("SLR projections must be in feet relative to 2026.")

        slr_ft = get_projection_samples(projection, storm_surge.year)
        if slr_ft.size < n_members:
            raise ValueError(
                f"{projection.model} RCP {projection.rcp} has {slr_ft.size} "
                f"members; {n_members} are required for pairing without replacement."
            )

        slr_ft = rng.choice(slr_ft, size=n_members, replace=False)
        distributions.append(
            WongBMAStormSurge(
                return_periods=storm_surge.return_periods.copy(),
                return_levels_ft=storm_surge.return_levels_ft + slr_ft[:, None],
                year=storm_surge.year,
                station=storm_surge.station,
                units="ft above 2026 mean sea level",
                model=projection.model,
                rcp=projection.rcp,
                datum=projection.datum,
                slr_ft=slr_ft,
            )
        )

    return distributions
