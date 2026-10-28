"""Small synthetic checks for transformations and inferential helpers.

No replication data or additional test dependencies are required.
"""
import sys
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analysis_utils import contrast

# Pipeline scripts are numbered by step, so they are loaded with import_module.
validate_multinomial = import_module("05_diagnostics").validate_multinomial
build_indices = import_module("04_mechanisms").build_indices
effects = import_module("03_message_effects")
add_companion, econ_vs_pol1_weights, video_terms = (
    effects.add_companion, effects.econ_vs_pol1_weights, effects.video_terms)
dispersion_comparison, range_uncertainty = (
    effects.dispersion_comparison, effects.range_uncertainty)


class CompanionTests(unittest.TestCase):
    def test_both_slots_and_video_types(self):
        data = pd.DataFrame({"video1": ["china_pol2", "china_econ1", "china_econ2"],
                             "video2": ["china_pol1", "china_pol2", "china_pol2"]})
        result = add_companion(data, "china")
        self.assertEqual(result.companion.tolist(), ["china_pol1", "china_econ1", "china_econ2"])
        self.assertEqual(result.companion_econ.tolist(), [0, 1, 1])
        self.assertNotIn("companion", data)

    def test_invalid_pairs_fail(self):
        for pair in [("china_pol1", "china_econ1"), ("china_pol2", "china_pol2"),
                     ("china_pol2", None), ("china_pol2", "us_econ1")]:
            with self.subTest(pair=pair), self.assertRaises(ValueError):
                add_companion(pd.DataFrame([pair], columns=["video1", "video2"]), "china")


class ContrastTests(unittest.TestCase):
    def test_content_weights(self):
        for prefix, column in (("china", "video2"), ("us", "video1")):
            factor = f'C({column}, Treatment("{prefix}_pol2"))'
            expected = {f"{factor}[T.{prefix}_econ1]": 0.5, f"{factor}[T.{prefix}_econ2]": 0.5,
                        f"{factor}[T.{prefix}_pol1]": -1}
            self.assertEqual(econ_vs_pol1_weights(prefix), expected)

    def test_estimate_and_covariance_against_matrix_calculation(self):
        rng = np.random.default_rng(7)
        x = pd.DataFrame(rng.normal(size=(100, 3)), columns=["a", "b", "c"])
        model = sm.OLS(x.a - 2*x.b + rng.normal(size=100), sm.add_constant(x)).fit(cov_type="HC2")
        row = contrast(model, {"a": 0.5, "b": 0.5, "c": -1}, "economic - political")
        weights = np.array([0, 0.5, 0.5, -1])
        self.assertAlmostEqual(row["estimate"], weights @ model.params)
        self.assertAlmostEqual(row["std_error"], np.sqrt(weights @ model.cov_params() @ weights))


def video_models(china, us, variance):
    """Stand-ins for the Competition video models with given coefficients."""
    names = list(video_terms("china")) + list(video_terms("us"))
    params = pd.Series(china + us, index=names)
    covariance = pd.DataFrame(variance * np.eye(6), index=names, columns=names)
    result = SimpleNamespace(params=params, cov_params=lambda: covariance)
    return {"political_model": result, "econ_model": result}


class BootstrapTests(unittest.TestCase):
    def test_precise_estimates_give_observed_range_and_dispersion(self):
        # china effects with pol2 at zero: 0.6, 0.2, -0.4, 0 -> range 1.0;
        # us effects: 0.1, 0.1, 0.1, 0 -> range 0.1.
        china, us = [0.6, 0.2, -0.4], [0.1, 0.1, 0.1]
        models = video_models(china, us, variance=1e-12)
        ranges = range_uncertainty(models, draws=500)
        np.testing.assert_allclose(ranges[["observed_range", "ci_low", "ci_high"]], 1.0,
                                   atol=1e-4)
        dispersion = dispersion_comparison(models, draws=500).set_index(["outcome", "measure"])
        sd = np.std(china + [0], ddof=1) - np.std(us + [0], ddof=1)
        for measure, expected in (("range", 0.9), ("sd", sd)):
            row = dispersion.loc[("political_model", measure)]
            np.testing.assert_allclose(row[["difference", "ci_low", "ci_high"]], expected,
                                       atol=1e-4)

    def test_range_of_equal_effects_is_biased_upward(self):
        models = video_models([0.] * 3, [0.] * 3, variance=0.01)
        row = range_uncertainty(models, draws=2000).iloc[0]
        self.assertEqual(row["observed_range"], 0)
        self.assertGreater(row["bias"], 0.1)
        self.assertAlmostEqual(row["bias_corrected_range"], -row["bias"])


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.control = pd.DataFrame({"a": [-1., -1., 1., 1.], "b": [-1., 1., -1., 1.]})

    def test_control_standardization_and_complete_cases(self):
        sample = pd.DataFrame({"a": [1., 2., np.nan], "b": [1., 0., 1.]}, index=[8, 4, 9])
        result = build_indices(self.control, sample, ["a", "b"])
        expected = 1 / np.sqrt(4/3)
        np.testing.assert_allclose(result.loc[[8, 4], "index_equal"], expected)
        np.testing.assert_allclose(result.loc[[8, 4], "index_inverse_covariance"], expected)
        self.assertTrue(result.loc[9, ["index_equal", "index_inverse_covariance"]].isna().all())
        self.assertNotIn("index_equal", sample)

    def test_inverse_covariance_weights_with_correlated_items(self):
        # Three standardized items have correlations r(a,c)=r(b,c)=1/sqrt(3),
        # r(a,b)=0. Their normalized inverse-covariance weights have a closed form.
        a = np.array([-1., -1., -1., -1., 1., 1., 1., 1.])
        b = np.array([-1., -1., 1., 1., -1., -1., 1., 1.])
        e = np.array([-1., 1., -1., 1., -1., 1., -1., 1.])
        control = pd.DataFrame({"a": a, "b": b, "c": a+b+e})
        sample = pd.DataFrame({"a": [1.], "b": [0.], "c": [0.]})
        result = build_indices(control, sample, ["a", "b", "c"])
        r = 1 / np.sqrt(3)
        expected = (1-r) / (3-4*r) / np.sqrt(8/7)
        self.assertAlmostEqual(result.loc[0, "index_inverse_covariance"], expected)

    def test_degenerate_inputs_fail(self):
        for control in (self.control.assign(a=1.), self.control.assign(b=self.control.a),
                        self.control.iloc[:2]):
            with self.subTest(control=control), self.assertRaises(ValueError):
                build_indices(control, self.control, ["a", "b"])
        with self.assertRaisesRegex(ValueError, "No complete"):
            build_indices(self.control, pd.DataFrame({"a": [np.nan], "b": [1.]}), ["a", "b"])


class ConvergenceTests(unittest.TestCase):
    def test_reject_failed_and_nonfinite_models(self):
        for converged, llf, params in [(False, -10., [1.]), (True, np.nan, [1.]),
                                       (True, -10., [np.inf])]:
            result = SimpleNamespace(mle_retvals={"converged": converged}, llf=llf, params=params)
            with self.subTest(converged=converged, llf=llf), self.assertRaises(ValueError):
                validate_multinomial(result, "test")


if __name__ == "__main__":
    unittest.main()
