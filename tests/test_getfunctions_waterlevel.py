from dataclasses import replace
from pathlib import Path
import unittest

import arviz as az
import numpy as np
from scipy.integrate import quad

from src.hazards.SeaLevel.getfunctions_projections import get_projection, get_projections
from src.hazards.SeaLevel.getfunctions_waterlevel import combine_slr_surge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEV_PATH = PROJECT_ROOT / "data/processed/storm_surge/stationary_gev_1928_2025.nc"


class TestSLRSurgeDistribution(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gev = az.from_netcdf(GEV_PATH)
        cls.projection = get_projection("K14", "rcp6.0")

    def test_zero_slr_recovers_stationary_gev(self):
        combined = combine_slr_surge(self.projection, self.gev, 2026)
        height = np.linspace(0, 12, 50)
        np.testing.assert_allclose(combined.sf(height), combined.surge.sf(height))
        np.testing.assert_allclose(combined.cdf(height), combined.surge.cdf(height))
        np.testing.assert_allclose(combined.pdf(height), combined.surge.pdf(height))

    def test_constant_slr_is_a_known_shift(self):
        constant = 1.25
        projection = replace(
            self.projection,
            years=np.array([2100]),
            lsl_ft=np.full((self.projection.n_sims, 1), constant),
        )
        combined = combine_slr_surge(projection, self.gev, 2100)
        height = np.linspace(0, 12, 50)
        np.testing.assert_allclose(
            combined.sf(height),
            combined.surge.sf(height - constant),
        )

    def test_pdf_normalizes_and_survival_is_monotonic(self):
        combined = combine_slr_surge(self.projection, self.gev, 2100)
        integral = quad(combined.pdf, -np.inf, np.inf)[0]
        self.assertAlmostEqual(integral, 1.0, places=6)

        exceedance = combined.sf(np.linspace(-5, 30, 1000))
        self.assertTrue(np.all(np.diff(exceedance) <= 0))

    def test_all_processed_slr_scenarios(self):
        projections = get_projections()
        distributions = [
            combine_slr_surge(projection, self.gev, 2100)
            for projection in projections
        ]
        self.assertEqual(len(distributions), 16)
        for projection, distribution in zip(projections, distributions):
            self.assertEqual(distribution.slr_ft.size, projection.n_sims)
            self.assertEqual(distribution.datum, 2026)
            self.assertEqual(distribution.units, "ft")
            self.assertTrue(0 <= distribution.sf(5.0) <= 1)


if __name__ == "__main__":
    unittest.main()
