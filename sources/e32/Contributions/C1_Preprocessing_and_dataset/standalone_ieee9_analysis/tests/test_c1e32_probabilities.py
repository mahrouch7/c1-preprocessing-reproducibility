"""Synthetic arithmetic tests only; no acquired data and no estimator fitting."""
from decimal import Decimal, localcontext
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PACKAGE / 'scripts'),
    'F:/ResearchRepositories/ieee9bus_research_canonical/Contributions/C1_Preprocessing_and_dataset/R07/scripts']
import c1e32_probabilities as p


class TestPosterior(unittest.TestCase):
    def test_large_common_offset_reproduces_library_cancellation(self):
        from scipy.special import logsumexp
        joint = np.array([[-1e12 - k * .75 for k in range(12)]])
        original = np.exp(joint - logsumexp(joint, axis=1, keepdims=True))
        self.assertGreater(abs(original.sum() - 1), 1e-10)
        stable = p.posterior_from_joint_log_likelihood(joint)
        self.assertLess(abs(stable.sum() - 1), 1e-14)
        with localcontext() as context:
            context.prec = 80
            exact = [Decimal.from_float(float(v)) for v in joint[0]]
            masses = [(v - max(exact)).exp() for v in exact]
            reference = np.array([float(v / sum(masses)) for v in masses])
        np.testing.assert_allclose(stable[0], reference, rtol=1e-14, atol=1e-16)
        self.assertEqual(stable.argmax(), original.argmax())

    def test_common_offset_invariance_and_ties(self):
        relative = np.array([[0., -1024., -2048.] * 4])
        a = p.posterior_from_joint_log_likelihood(relative)
        b = p.posterior_from_joint_log_likelihood(relative - 1e15)
        np.testing.assert_array_equal(a, b)
        uniform = p.posterior_from_joint_log_likelihood(np.full((2, 12), -1e23))
        np.testing.assert_allclose(uniform, 1/12, rtol=0, atol=1e-16)

    def test_finite_likelihoods_required_no_nan_fallback(self):
        for bad in (np.nan, np.inf, -np.inf):
            joint = np.ones((2, 12)); joint[0, 0] = bad
            with self.assertRaises(ValueError): p.posterior_from_joint_log_likelihood(joint)
        with self.assertRaises(ValueError): p.posterior_from_joint_log_likelihood(np.ones((2, 11)))

    def test_other_classifier_delegates_unchanged(self):
        model = object(); x = np.zeros((2, 12)); expected = ('native', 'probability')
        with patch.object(p.base, 'checked_prediction', return_value=expected) as frozen:
            self.assertEqual(p.checked_prediction(model, x), expected)
            frozen.assert_called_once_with(model, x)

    def test_saved_gaussian_state_no_fit(self):
        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.classes_ = np.array(sorted(p.base.LABELS))
        model.n_features_in_ = 2
        model.class_prior_ = np.full(12, 1/12)
        model.theta_ = np.arange(24).reshape(12, 2) * .1
        model.var_ = np.ones((12, 2))
        with patch.object(GaussianNB, 'fit', side_effect=AssertionError('FIT FORBIDDEN')):
            x = np.array([[.1, .2], [2e6, 2e6]])
            native, stable = p.checked_prediction(model, x)
        self.assertEqual(stable.shape, (2, 12))
        np.testing.assert_allclose(stable.sum(axis=1), 1, rtol=0, atol=1e-14)
        np.testing.assert_array_equal(native, model.predict(x))
        self.assertEqual(p.policy(model), p.POLICY)


if __name__ == '__main__': unittest.main()
