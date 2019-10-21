from challenge._internal import evaluate_regression
from challenge import DATA_DIR
import pandas as pd
import numpy as np

np.random.seed(3778)

def test_evaluate_regression_perfect():
    path = DATA_DIR / 'answers.csv'
    y_pred = pd.read_csv(path, dtype={'value': float})['value']
    expected = {'explained_variance_score': 1,
                'mean_absolute_error': 0,
                'mean_squared_error': 0,
                'median_absolute_error': 0,
                'r2_score': 1}
    assert evaluate_regression(y_pred) == expected


def test_evaluate_regression_random():
    path = DATA_DIR / 'answers.csv'
    y_pred = (pd.read_csv(path, dtype={'value': float})
              ['value']
              .sample(frac=1.0, replace=False))
    r = evaluate_regression(y_pred)
    
    assert np.isclose(r['explained_variance_score'], -1, rtol=0, atol=1e-2)
    assert np.isclose(r['r2_score'], -1, rtol=0, atol=1e-2)


def test_evaluate_regression_mean():
    path = DATA_DIR / 'answers.csv'
    y_pred = (pd.read_csv(path, dtype={'value': float})
              .assign(value=lambda df: df['value'].mean())
              ['value'])
              
    r = evaluate_regression(y_pred)
    
    assert np.isclose(r['explained_variance_score'], 0)
    assert np.isclose(r['r2_score'], 0)
