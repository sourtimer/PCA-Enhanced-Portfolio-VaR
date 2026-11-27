# PCA-Enhanced Value-at-Risk for Multi-Asset Portfolios

**Group 23 | Topics in Financial Engineering & Technology**

This project compares five one-day VaR estimators for 18 ETFs and examines whether a PCA-based covariance model can simplify common risk representation without materially changing portfolio VaR forecasts.

## Data and methods

- Adjusted daily closing prices: Yahoo Finance, 2020-01-02 to 2026-09-30 (1,695 observations). The input return file contains 1,694 daily log returns. Models convert them to simple returns using `expm1`.
- Portfolios: equal weight (18 ETFs), 60% SPY / 40% AGG, and long-only minimum variance. Minimum-variance target weights are updated every 21 trading days and held constant between updates; the portfolio is assumed to maintain those target weights daily, without transaction costs.
- Forecasts: preceding 500 trading days, one day ahead, 95% and 99% confidence. Evaluation: 2021-12-28 to 2026-09-30 (1,194 trading days).
- Methods: historical simulation, Gaussian parametric VaR, zero-mean RiskMetrics EWMA (0.94), rolling PCA factor VaR (90% cumulative variance with diagonal residual variance), and normal Monte Carlo (20,000 scenarios per forecast date and portfolio; seed = 2026 + zero-based return row).
- Full-sample PCA on standardized log returns is descriptive only. Rolling PCA uses standardized simple returns available before the forecast date.
- All stored returns, losses, VaR and rates are decimals. A breach is recorded when actual loss exceeds VaR. Kupiec unconditional coverage is evaluated at 5% significance.

## Main results

Seven exploratory principal components explain 90.74% of standardized return variance. Rolling estimation retains 6–8 components. For equal-weight 99% VaR, PCA and full-covariance parametric forecasts differ by 0.92 basis points on average in absolute terms. PCA records 21 breaches in 1,194 forecast days (1.76%; Kupiec p = 0.0173). The factor representation is compact, but does not resolve tail-risk calibration in this sample.

## Installation and use

Python 3.12 is recommended. From the project root in Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dashboard.txt
.\.venv\Scripts\python.exe -m streamlit run app/dashboard.py
```

On Linux/macOS, use `python3 -m venv .venv` and replace `.\.venv\Scripts\python.exe` with `./.venv/bin/python`. For numerical analysis without the dashboard, install `requirements.txt` instead. The dashboard reads saved CSV files and does not need a network connection after installation.

Run the following commands using the environment's Python executable:

```bash
python -m unittest discover -s tests -v
python run_all.py --charts-only
```

The default `python run_all.py` also regenerates presentation figures only. To recompute all rolling models, Monte Carlo, combined tables and figures, run `python run_all.py --full`. This is computationally more expensive and overwrites model outputs; use a separate copy to preserve the submitted snapshot. `--base` recalculates only the original four models and leaves Monte Carlo tables unchanged; do not interpret mixed results after a partial rerun.

Upstream data processing and exploratory analysis can be reproduced separately:

```bash
python src/collect_data.py
python src/etf_statistics.py
```

These write to `reproduced/` and leave the submitted data and EDA outputs untouched. `collect_data.py` defaults to the supplied adjusted price snapshot. Optional Yahoo Finance retrieval requires `python -m pip install yfinance` followed by `python src/collect_data.py --download`. Vendor revisions may change a fresh download; use the supplied data for exact comparisons. Exploratory analysis reads the submitted `data/asset_returns.csv`. PyMuPDF is used to produce the exploratory figures.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | Submitted price and log-return inputs; exploratory correlation and PCA tables/figures in `eda_reference/` |
| `src/` | Data processing, exploratory statistics, VaR models, simulation, backtesting and plots |
| `app/` | Streamlit dashboard and Plotly charts |
| `results/tables/` | Saved model outputs, weights, PCA diagnostics and sensitivity checks |
| `results/tables/monte_carlo/` | Monte Carlo and combined five-model results |
| `results/figures/` | Static experiment figures |
| `tests/` | Unit tests and submitted-result integrity checks |
| `run_all.py` | Model and plotting entry point |

The primary combined results are `results/tables/monte_carlo/all_methods_daily_predictions.csv` (35,820 rows) and `all_methods_backtest_summary.csv` in the same directory (30 rows). `data/eda_reference/` contains the original full-sample exploratory results. The supplementary Monte Carlo tables include single-day and equal-weight diagnostics. `results/tables/var_backtest_summary_cn.csv` is a Chinese-language presentation of the original four-model summary, not a separate experiment.

The integrity tests check the submitted results. A full recomputation may produce small numerical differences across library versions; the original baseline hashes should not be overwritten merely to make a test pass.

## References

- Connor, G., & Korajczyk, R. A. (1988). Risk and return in an equilibrium APT: Application of a new test methodology. *Journal of Financial Economics*, 21(2), 255–289.
- Jolliffe, I. T., & Cadima, J. (2016). Principal component analysis: A review and recent developments. *Philosophical Transactions of the Royal Society A*, 374, 20150202.
- J.P. Morgan/Reuters. (1996). *RiskMetrics Technical Document* (4th ed.).
- Kupiec, P. H. (1995). Techniques for verifying the accuracy of risk measurement models. *The Journal of Derivatives*, 3(2), 73–84.
- Yahoo Finance. Historical adjusted ETF prices (https://finance.yahoo.com/).
