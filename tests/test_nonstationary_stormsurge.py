import unittest
from dataclasses import replace

import numpy as np

from src.hazards.SeaLevel.buildmodel_stormsurge import nonstationary_bma
from src.hazards.SeaLevel.getfunctions_projections import get_projection, get_projections
from src.hazards.SeaLevel.getfunctions_waterlevel import combine_slr_wong


class TestWongBMAStormSurge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.surge = nonstationary_bma()

    def test_published_ensemble_shape_and_return_periods(self):
        self.assertEqual(self.surge.return_levels_ft.shape, (8021, 18))
        np.testing.assert_array_equal(
            self.surge.return_periods,
            [2, 3, 5, 10, 20, 30, 50, 80, 100, 150, 200, 250, 300,
             350, 400, 450, 500, 1000],
        )
        self.assertEqual(self.surge.year, 2065)

    def test_reproduces_published_100_year_quantiles(self):
        values_m = self.surge.return_levels_ft[:, 8] * 0.3048
        quantiles = np.quantile(values_m, [0.025, 0.5, 0.95, 0.975])
        np.testing.assert_allclose(
            quantiles,
            [2.115874, 2.358962, 3.074836, 3.709274],
            atol=5e-7,
        )

    def test_exceedance_is_monotonic(self):
        heights = np.linspace(self.surge.minimum_height_ft, 20, 100)
        exceedance = self.surge.sf(heights)
        self.assertTrue(np.all(np.diff(exceedance) <= 0))
        self.assertTrue(np.all((0 <= exceedance) & (exceedance <= 0.5)))

    def test_height_below_published_tail_is_rejected(self):
        with self.assertRaises(ValueError):
            self.surge.sf(self.surge.minimum_height_ft - 0.01)

    def test_upper_tail_matches_authors_zero_rule(self):
        height = np.max(self.surge.return_levels_ft) + 1
        self.assertEqual(float(self.surge.sf(height)), 0.0)
        self.assertEqual(float(self.surge.cdf(height)), 1.0)

    def test_zero_slr_recovers_tony_ensemble(self):
        projection = get_projection("K14", "rcp6.0", 2065, 2065)
        projection = replace(projection, lsl_ft=np.zeros_like(projection.lsl_ft))
        combined = combine_slr_wong([projection], self.surge)[0]
        np.testing.assert_allclose(combined.return_levels_ft, self.surge.return_levels_ft)

    def test_constant_slr_shifts_every_return_level(self):
        projection = get_projection("K14", "rcp6.0", 2065, 2065)
        projection = replace(projection, lsl_ft=np.full_like(projection.lsl_ft, 1.25))
        combined = combine_slr_wong([projection], self.surge)[0]
        np.testing.assert_allclose(
            combined.return_levels_ft,
            self.surge.return_levels_ft + 1.25,
        )

    def test_all_16_slr_projections(self):
        projections = get_projections(2065, 2065)
        first = combine_slr_wong(projections, self.surge, seed=713)
        second = combine_slr_wong(projections, self.surge, seed=713)

        self.assertEqual(len(first), 16)
        for distribution, repeated in zip(first, second):
            self.assertEqual(distribution.return_levels_ft.shape, (8021, 18))
            self.assertEqual(distribution.slr_ft.size, 8021)
            self.assertEqual(distribution.datum, 2026)
            np.testing.assert_array_equal(distribution.slr_ft, repeated.slr_ft)


if __name__ == "__main__":
    unittest.main()
