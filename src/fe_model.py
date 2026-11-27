"""Portfolio returns, covariance estimation, VaR models and coverage tests. Returns and losses are decimal fractions."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"
WINDOW = 500
REBALANCE_EVERY = 21
EWMA_LAMBDA = 0.94
PCA_TARGET_EXPLAINED = 0.90
CONFIDENCES = (0.95, 0.99)
MAIN_METHODS = ("Historical", "Parametric", "RiskMetrics_EWMA", "PCA_Factor")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aligned ETF log and simple returns."""
    raw = pd.read_csv(DATA_DIR / "asset_returns.csv", parse_dates=["Date"]).set_index("Date")
    if not isinstance(raw.index, pd.DatetimeIndex):
        raise ValueError("Date must parse as datetime")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Dates must be unique and increasing")
    if raw.isna().any().any() or not np.isfinite(raw.to_numpy()).all():
        raise ValueError("Missing or invalid values in asset returns")
    if raw.shape[1] < 2 or raw.shape[0] < WINDOW + 100:
        raise ValueError("Not enough assets / observations")
    expected = {"SPY", "AGG"}
    if not expected.issubset(set(raw.columns)):
        raise ValueError("Missing required ETFs: SPY / AGG")
    return raw, np.expm1(raw)


def portfolio_weights(simple_returns: pd.DataFrame) -> dict[str, np.ndarray]:
    cols = list(simple_returns.columns)
    n = len(cols)
    eq = np.full(n, 1.0 / n)
    sixty_forty = np.zeros(n)
    sixty_forty[cols.index("SPY")] = 0.60
    sixty_forty[cols.index("AGG")] = 0.40
    return {"EqualWeight": eq, "6040": sixty_forty}


def min_var_long_only(cov: np.ndarray) -> np.ndarray:
    """Minimize variance (no shorting), with robust scaling and solver check."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("Covariance matrix must be square")
    if not np.isfinite(cov).all():
        raise ValueError("Non-finite covariance")
    scale = max(np.max(np.diag(cov)), 1e-8)
    s = cov / scale
    def objective(w):
        return float(w @ s @ w)
    def gradient(w):
        return 2 * s @ w
    res = minimize(objective, np.ones(n) / n, method="SLSQP", jac=gradient,
                   bounds=[(0., 1.)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.,
                                 "jac": lambda w: np.ones(n)}],
                   options={"maxiter": 2000, "ftol": 1e-12})
    if not res.success:
        raise RuntimeError(f"Minimum variance optimizer failed: {res.message}")
    w = np.asarray(res.x)
    if abs(w.sum() - 1) > 1e-6 or w.min() < -1e-7:
        raise RuntimeError("Invalid optimized weights")
    return np.clip(w, 0, None) / np.clip(w, 0, None).sum()


def factor_covariance(window_simple: np.ndarray, target: float = PCA_TARGET_EXPLAINED,
                      fixed_k: int | None = None) -> tuple[np.ndarray, int, float]:
    """Estimate covariance from rolling standardized simple returns, retaining diagonal residual variance. Full-sample exploratory PCA is not used."""
    v = np.asarray(window_simple, dtype=float)
    sd = v.std(axis=0, ddof=1)
    if (sd < 1e-14).any():
        raise ValueError("Zero-volatility ETF in PCA window")
    z = (v - v.mean(axis=0)) / sd
    corr = np.corrcoef(z, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.)
    eigenvectors = eigenvectors[:, order]
    shares = np.cumsum(eigenvalues) / eigenvalues.sum()
    k = int(np.searchsorted(shares, target) + 1) if fixed_k is None else int(fixed_k)
    if not 1 <= k <= v.shape[1]:
        raise ValueError("Invalid k")
    b = eigenvectors[:, :k]
    common = (b * eigenvalues[:k]) @ b.T
    residual_var = np.maximum(0., 1. - np.diag(common))
    factor_corr = common + np.diag(residual_var)
    factor_corr = (factor_corr + factor_corr.T) / 2
    factor_cov = factor_corr * np.outer(sd, sd)
    return factor_cov, k, float(shares[k - 1])


def ewma_variance(window_portfolio_returns: np.ndarray, lam: float = EWMA_LAMBDA) -> float:
    """Initialize zero-mean EWMA using the training sample variance under current portfolio weights."""
    hist = np.asarray(window_portfolio_returns, dtype=float)
    if hist.ndim != 1 or hist.size < 2:
        raise ValueError("Insufficient portfolio history")
    variance = float(hist.var(ddof=1))
    for value in hist:
        variance = lam * variance + (1 - lam) * float(value * value)
    return variance


def compute_one_day_models(window_simple: np.ndarray, w: np.ndarray,
                           *, fixed_k: int | None = None) -> tuple[dict[float, dict[str, float]], dict]:
    """Return decimal VaR thresholds by confidence and method, plus window diagnostics."""
    window_simple = np.asarray(window_simple, dtype=float)
    w = np.asarray(w, dtype=float)
    if abs(w.sum() - 1) > 1e-6 or np.any(w < -1e-8):
        raise ValueError("Invalid portfolio weights")
    hist = window_simple @ w
    mean = float(hist.mean())
    sd = float(hist.std(ddof=1))
    ewma_sd = float(np.sqrt(ewma_variance(hist)))
    factor_cov, k, explained = factor_covariance(window_simple, fixed_k=fixed_k)
    factor_sd = float(np.sqrt(max(w @ factor_cov @ w, 0.)))
    forecasts = {}
    for confidence in CONFIDENCES:
        z = float(norm.ppf(confidence))
        forecasts[confidence] = {
            "Historical": float(max(-np.quantile(hist, 1 - confidence), 0.)),
            "Parametric": float(max(z * sd - mean, 0.)),
            "RiskMetrics_EWMA": float(max(z * ewma_sd, 0.)),
            "PCA_Factor": float(max(z * factor_sd - mean, 0.)),
        }
    diagnostics = {"pca_components": k, "pca_explained_variance": explained,
                   "portfolio_mu": mean, "portfolio_sd": sd, "ewma_sd": ewma_sd,
                   "pca_factor_sd": factor_sd}
    return forecasts, diagnostics


def kupiec_pof(n: int, breaches: int, expected_rate: float) -> tuple[float, float]:
    """Kupiec (1995) unconditional coverage likelihood ratio, including edge cases."""
    if not (n > 0 and 0 <= breaches <= n and 0 < expected_rate < 1):
        raise ValueError("Invalid Kupiec inputs")
    q = breaches / n
    terms = 0.
    if breaches:
        terms += breaches * np.log(q / expected_rate)
    if breaches < n:
        terms += (n - breaches) * np.log((1 - q) / (1 - expected_rate))
    lr = max(0., float(2 * terms))
    return lr, float(chi2.sf(lr, df=1))


def summarize_daily(daily: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pf, confidence, method), part in daily.groupby(["portfolio", "confidence", "method"], sort=True):
        n = int(len(part)); x = int(part["violation"].sum()); p = 1. - float(confidence)
        lr, pval = kupiec_pof(n, x, p)
        rows.append({"portfolio": pf, "confidence": confidence, "method": method,
                     "backtest_days": n, "breaches": x,
                     "actual_breach_rate": x / n,
                     "expected_rate": p, "expected_breaches": n * p,
                     "Kupiec_LR": lr, "Kupiec_p_value": pval,
                     "reject_at_5pct": pval < 0.05})
    return pd.DataFrame(rows).sort_values(["portfolio", "confidence", "method"]).reset_index(drop=True)


def ensure_results_dirs() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
