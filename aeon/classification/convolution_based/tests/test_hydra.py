"""Tests for Hydra classifiers."""

import numpy as np
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifierCV

from aeon.classification.convolution_based import (
    HydraClassifier,
    MultiRocketHydraClassifier,
)
from aeon.testing.data_generation import make_example_3d_numpy


class _ArrayTransformer(TransformerMixin, BaseEstimator):
    def __init__(
        self, n_kernels=None, n_groups=None, n_jobs=None, random_state=None
    ):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.reshape(X.shape[0], -1)


@pytest.fixture
def mock_hydra_dependencies(monkeypatch):
    """Replace optional torch transforms while testing estimator wiring."""
    monkeypatch.setattr("aeon.base._base._check_estimator_deps", lambda *args: None)
    monkeypatch.setattr(
        "aeon.classification.convolution_based._hydra.HydraTransformer",
        _ArrayTransformer,
    )
    monkeypatch.setattr(
        "aeon.classification.convolution_based._hydra._SparseScaler",
        _ArrayTransformer,
    )
    monkeypatch.setattr(
        "aeon.classification.convolution_based._mr_hydra.HydraTransformer",
        _ArrayTransformer,
    )
    monkeypatch.setattr(
        "aeon.classification.convolution_based._mr_hydra._SparseScaler",
        _ArrayTransformer,
    )
    monkeypatch.setattr(
        "aeon.classification.convolution_based._mr_hydra.MultiRocket",
        _ArrayTransformer,
    )


@pytest.mark.parametrize("classifier", [HydraClassifier, MultiRocketHydraClassifier])
def test_hydra_default_estimator_is_unchanged(classifier, mock_hydra_dependencies):
    """The estimator injection hook must retain RidgeClassifierCV by default."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=16, random_state=0
    )
    clf = classifier(n_kernels=2, n_groups=2, random_state=0).fit(X, y)

    estimator = clf._estimator if classifier is HydraClassifier else clf.classifier
    assert isinstance(estimator, RidgeClassifierCV)
    np.testing.assert_array_equal(estimator.alphas, np.logspace(-3, 3, 10))
    assert estimator.class_weight is None


@pytest.mark.parametrize("classifier", [HydraClassifier, MultiRocketHydraClassifier])
def test_hydra_estimator_injection(classifier, mock_hydra_dependencies):
    """A supplied estimator is cloned, fitted, and used for prediction."""
    X, y = make_example_3d_numpy(
        n_cases=10, n_channels=1, n_timepoints=16, random_state=0
    )
    estimator = DummyClassifier(strategy="most_frequent")
    clf = classifier(
        n_kernels=2, n_groups=2, estimator=estimator, random_state=0
    ).fit(X, y)

    fitted = clf._estimator if classifier is HydraClassifier else clf.classifier
    assert isinstance(fitted, DummyClassifier)
    assert fitted is not estimator
    assert len(clf.predict(X)) == len(y)
