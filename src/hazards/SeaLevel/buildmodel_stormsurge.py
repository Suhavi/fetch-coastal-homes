import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pathlib import Path

from src.hazards.SeaLevel.getfunctions_waterlevel import WongBMAStormSurge


# Stationary GEV, Doss-Gollin 2022.

# STEPS: Read annual surges, add James's priors and GEV, then sample the posterior.
def stationary_gev(summary, draws=2500, tune=1000, chains=4, seed=713):
    # STEP 1: Read the annual maximum storm-surge anomalies.
    max_surge_ft = np.asarray(summary["max_surge_ft"], dtype=float)

    return_level_priors = [
        (2, 6, 3),
        (10, 8, 4),
        (100, 12, 6),
        (500, 16, 8),
    ]

    with pm.Model() as model:
        # STEP 2: Define James's weak parameter priors.
        location = pm.TruncatedNormal("location", mu=0, sigma=25, lower=0)
        scale = pm.TruncatedNormal("scale", mu=0, sigma=10, lower=0.001)
        shape = pm.TruncatedNormal("shape", mu=0, sigma=2.5, lower=0)

        # STEP 3: Add James's priors on the GEV return levels.
        for period, mean, sd in return_level_priors:
            probability = 1 - 1 / period
            return_level = location + scale / shape * (
                (-np.log(probability)) ** (-shape) - 1
            )
            alpha = 2 + mean**2 / sd**2
            beta = (alpha - 1) * mean
            prior = pm.InverseGamma.dist(alpha=alpha, beta=beta)
            pm.Potential(f"return_level_{period}", pm.logp(prior, return_level))

        # STEP 4: Define the stationary GEV likelihood.
        support = 1 + shape * (max_surge_ft - location) / scale
        logp = (
            -pt.log(scale)
            - (1 + 1 / shape) * pt.log(support)
            - support ** (-1 / shape)
        )
        pm.Potential(
            "observed_surge",
            pt.sum(pt.switch(pt.gt(support, 0), logp, -np.inf)),
        )

        # STEP 5: Draw four NUTS chains and return the fitted model.
        posterior = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=chains,
            init="adapt_diag",
            initvals={"location": 1, "scale": 1, "shape": 1},
            random_seed=seed,
            target_accept=0.95,
        )

    return model, posterior


############

WONG_BMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "data/processed/storm_surge/wong_2018_norfolk_bma_2065.csv.gz"
)


# STEPS: Read Wong's published 2065 return-level ensemble, retain every
# uncertainty member, and expose its annual exceedance curves.
# Wong (2018), Sections 2.2 and 2.4; R/analysis.R, lines 51-149:
# https://doi.org/10.5194/ascmo-4-53-2018
def nonstationary_bma(path=WONG_BMA_PATH):
    """Load Wong's precalibrated all-covariate Norfolk BMA for 2065."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    columns = data.dtype.names[1:]
    return_periods = np.array([int(name.removeprefix("y")) for name in columns])
    return_levels_ft = np.column_stack([data[name] for name in columns]) / 304.8

    return WongBMAStormSurge(
        return_periods=return_periods,
        return_levels_ft=return_levels_ft,
    )

