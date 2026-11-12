"""Numerically stable GaussianNB posteriors, with the original strict contract.

Only arithmetic changes: softmax of finite joint log likelihoods, subtracting the
row maximum before exponentiation. No clipping, fallback, refit, parameter tuning,
or relaxed tolerance. Other estimators retain the frozen upstream implementation.
"""
from __future__ import annotations
import numpy as np
from scipy.special import softmax
import c1e25 as base

POLICY = 'GaussianNB_joint_log_likelihood_shifted_softmax_v1'


def gaussian_component(model):
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import Pipeline
    candidate = model.steps[-1][1] if isinstance(model, Pipeline) else model
    return candidate if type(candidate) is GaussianNB else None


def posterior_from_joint_log_likelihood(joint):
    joint = np.asarray(joint, dtype=np.float64)
    if joint.ndim != 2 or joint.shape[1] != len(base.LABELS) or not np.isfinite(joint).all():
        raise ValueError('Invalid GaussianNB joint log likelihoods; no repair')
    probabilities = softmax(joint, axis=1)
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0) or
            np.any(probabilities > 1) or not np.allclose(probabilities.sum(axis=1), 1, rtol=0, atol=1e-10)):
        raise ValueError('Invalid stable GaussianNB posterior; no repair')
    return probabilities


def checked_prediction(model, x):
    classifier = gaussian_component(model)
    if classifier is None:
        return base.checked_prediction(model, x)
    from sklearn.pipeline import Pipeline
    native = np.asarray(model.predict(x), dtype=str)
    classes = list(model.classes_)
    if len(classes) != len(base.LABELS) or set(classes) != set(base.LABELS):
        raise ValueError('Model class identity mismatch')
    transformed = model[:-1].transform(x) if isinstance(model, Pipeline) and len(model.steps) > 1 else x
    # Private sklearn API is intentionally version-pinned to 1.9.1 by the worker.
    transformed = classifier._check_X(transformed)
    joint = classifier._joint_log_likelihood(transformed)
    probabilities = posterior_from_joint_log_likelihood(joint)
    if (native.shape != (len(x),) or not set(native).issubset(base.LABELS) or
            probabilities.shape != (len(x), len(base.LABELS)) or
            not np.array_equal(np.asarray(classes)[np.argmax(joint, axis=1)], native)):
        raise ValueError('GaussianNB native prediction identity changed')
    probabilities = probabilities[:, [classes.index(label) for label in base.LABELS]]
    return native, probabilities


def policy(model):
    return POLICY if gaussian_component(model) is not None else 'frozen_library_native_v1'
