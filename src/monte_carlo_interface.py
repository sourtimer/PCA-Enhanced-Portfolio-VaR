"""Gaussian Monte Carlo VaR and its analytical reference."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm
from fe_model import CONFIDENCES


def monte_carlo_var(window_simple: np.ndarray, weights: np.ndarray,
                    *, n_scenarios: int = 10000, seed: int = 2026) -> dict[float, float]:
    """Simulate asset returns from the training-window Gaussian model and return portfolio VaR in decimal loss units."""
    window_simple = np.asarray(window_simple, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if window_simple.ndim != 2 or len(weights) != window_simple.shape[1]:
        raise ValueError('Wrong window or weight shape')
    if len(window_simple) < 2 or n_scenarios < 100:
        raise ValueError('Need at least two observations and 100 simulations')
    rng = np.random.default_rng(seed)
    means = window_simple.mean(axis=0)
    covariance = np.cov(window_simple, rowvar=False, ddof=1)
    scenarios = rng.multivariate_normal(means, covariance, size=n_scenarios,
                                        method='cholesky', check_valid='raise')
    portfolio_scenarios = scenarios @ weights
    return {alpha: float(max(-np.quantile(portfolio_scenarios, 1 - alpha), 0.))
            for alpha in CONFIDENCES}


def normal_analytic_reference(window_simple: np.ndarray, weights: np.ndarray) -> dict[float, float]:
    """Analytic normal VaR: numerical reference for checking the MC implementation."""
    portfolio_returns = np.asarray(window_simple) @ np.asarray(weights)
    mean = portfolio_returns.mean()
    std = portfolio_returns.std(ddof=1)
    return {alpha: float(max(norm.ppf(alpha) * std - mean, 0.)) for alpha in CONFIDENCES}


if __name__ == '__main__':
    from fe_model import load_data, portfolio_weights, WINDOW
    _, data = load_data()
    win = data.iloc[:WINDOW].to_numpy()
    w = portfolio_weights(data)['EqualWeight']
    mc = monte_carlo_var(win, w, n_scenarios=20000)
    ref = normal_analytic_reference(win, w)
    print('Illustrative one-day Monte Carlo interface (not full backtest):')
    for a in CONFIDENCES:
        print(f'{a:.0%}: MC={mc[a]*100:.3f}% analytic={ref[a]*100:.3f}%')
