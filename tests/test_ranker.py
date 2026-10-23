"""Cascade ranker: the numpy tree evaluator reproduces scikit-learn's decision function."""
import os

import numpy as np
import pandas as pd

from massivora.ranker import _model, predict

FIXTURE = os.path.join(os.path.dirname(__file__), 'data', 'cascade_ranker_fixture.csv')


def test_matches_sklearn_decision_function():
    fx = pd.read_csv(FIXTURE)
    X = fx[_model()['features']].to_numpy(float)
    np.testing.assert_allclose(predict(X), fx['decision'], rtol=1e-9, atol=1e-9)


def test_missing_features_still_score():
    X = pd.read_csv(FIXTURE)[_model()['features']].to_numpy(float)[:5]
    X[:, 3] = np.nan
    assert np.isfinite(predict(X)).all()
