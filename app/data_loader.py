"""Read and validate saved result tables. Returns, losses and rates are decimal fractions; positive loss denotes a loss."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

try:  # Streamlit is optional so the validation tests can import this module.
    import streamlit as st

    def _cache_data(func):
        return st.cache_data(show_spinner=False)(func)

except Exception:  # pragma: no cover - exercised only without Streamlit
    st = None  # type: ignore[assignment]

    def _cache_data(func):
        return func


# Paths and canonical enumerations
ROOT = Path(__file__).resolve().parents[1]

PATHS = {
    "predictions": ROOT
    / "results"
    / "tables"
    / "monte_carlo"
    / "all_methods_daily_predictions.csv",
    "summary": ROOT
    / "results"
    / "tables"
    / "monte_carlo"
    / "all_methods_backtest_summary.csv",
    "pca_diagnostics": ROOT / "results" / "tables" / "pca_window_diagnostics.csv",
    "pca_sensitivity_daily": ROOT / "results" / "tables" / "pca_sensitivity_daily.csv",
    "pca_sensitivity_summary": ROOT
    / "results"
    / "tables"
    / "pca_sensitivity_backtest_detail.csv",
    "correlation": ROOT / "data" / "eda_reference" / "ETF-correlation.csv",
    "pca_variance": ROOT / "data" / "eda_reference" / "PCA-explained-variance.csv",
    "pca_loadings": ROOT / "data" / "eda_reference" / "PCA-loadings.csv",
    "weights": ROOT / "results" / "tables" / "portfolio_weights.csv",
    "dynamic_weights": ROOT / "results" / "tables" / "dynamic_portfolio_weights.csv",
    "prices": ROOT / "data" / "asset_prices.csv",
    "returns": ROOT / "data" / "asset_returns.csv",
}

METHODS = [
    "Historical",
    "Parametric",
    "RiskMetrics_EWMA",
    "PCA_Factor",
    "MonteCarlo_Normal",
]
PORTFOLIOS = ["EqualWeight", "6040", "MinVariance"]
CONFIDENCES = [0.95, 0.99]
PCA_SENSITIVITY_METHODS = ["PCA_Factor_k3", "PCA_Factor_k7", "PCA_Factor_k8"]
PCA_SENSITIVITY_K = [3, 7, 8]

PREDICTION_COLUMNS = [
    "Date",
    "portfolio",
    "method",
    "confidence",
    "VaR",
    "actual_return",
    "actual_loss",
    "actual_log_return",
    "violation",
    "pca_components",
    "pca_explained_variance",
    "training_end_date",
]
SUMMARY_COLUMNS = [
    "portfolio",
    "confidence",
    "method",
    "backtest_days",
    "breaches",
    "actual_breach_rate",
    "expected_rate",
    "expected_breaches",
    "Kupiec_LR",
    "Kupiec_p_value",
    "reject_at_5pct",
]

N_FORECASTS = 35_820
N_SUMMARY_ROWS = 30
N_PER_SLICE = 1_194
BACKTEST_START = pd.Timestamp("2021-12-28")
BACKTEST_END = pd.Timestamp("2026-09-30")


class DashboardDataError(RuntimeError):
    """Raised when a required result file is missing or malformed."""


# Low-level loading
def _require(path: Path) -> Path:
    if not path.exists():
        raise DashboardDataError(
            f"Missing data file: {path}. "
            "Run the dashboard from the project root and keep results/ intact."
        )
    return path


def _read_csv(path: Path) -> pd.DataFrame:
    _require(path)
    try:
        return pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - defensive
        raise DashboardDataError(f"Could not read {path.name}: {exc}") from exc


def _check_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DashboardDataError(f"{label} is missing columns: {', '.join(missing)}")


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.normalize()


# Public loaders (cached per Streamlit session)
@_cache_data
def load_predictions() -> pd.DataFrame:
    """35,820 daily out-of-sample forecasts, one row per date/model/slice."""
    df = _read_csv(PATHS["predictions"])
    _check_columns(df, PREDICTION_COLUMNS, "all_methods_daily_predictions.csv")

    df["Date"] = _to_datetime(df["Date"])
    df["training_end_date"] = _to_datetime(df["training_end_date"])
    df["portfolio"] = df["portfolio"].astype(str)
    df["method"] = df["method"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").round(4)
    df["violation"] = pd.to_numeric(df["violation"], errors="coerce").fillna(0).astype("int64")
    for col in ["VaR", "actual_return", "actual_loss", "actual_log_return"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["Date"].isna().any():
        raise DashboardDataError("Forecast dates could not be parsed in the prediction table.")

    return df.sort_values(
        ["portfolio", "confidence", "method", "Date"], kind="mergesort"
    ).reset_index(drop=True)


@_cache_data
def load_summary() -> pd.DataFrame:
    """30 full-period Kupiec/backtest summary rows."""
    df = _read_csv(PATHS["summary"])
    _check_columns(df, SUMMARY_COLUMNS, "all_methods_backtest_summary.csv")

    df["portfolio"] = df["portfolio"].astype(str)
    df["method"] = df["method"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").round(4)
    df["backtest_days"] = pd.to_numeric(df["backtest_days"], errors="coerce").astype("Int64")
    df["breaches"] = pd.to_numeric(df["breaches"], errors="coerce").astype("Int64")
    for col in [
        "actual_breach_rate",
        "expected_rate",
        "expected_breaches",
        "Kupiec_LR",
        "Kupiec_p_value",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["reject_at_5pct"] = (
        df["reject_at_5pct"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    return df


@_cache_data
def load_pca_diagnostics() -> pd.DataFrame:
    """Rolling-PCA window diagnostics (3,582 portfolio/date rows)."""
    df = _read_csv(PATHS["pca_diagnostics"])
    _check_columns(
        df,
        ["Date", "portfolio", "pca_components", "explained_variance", "train_start", "train_end"],
        "pca_window_diagnostics.csv",
    )
    df["Date"] = _to_datetime(df["Date"])
    df["train_start"] = _to_datetime(df["train_start"])
    df["train_end"] = _to_datetime(df["train_end"])
    df["portfolio"] = df["portfolio"].astype(str)
    df["pca_components"] = pd.to_numeric(df["pca_components"], errors="coerce").astype("Int64")
    df["explained_variance"] = pd.to_numeric(df["explained_variance"], errors="coerce")
    return df.sort_values(["portfolio", "Date"], kind="mergesort").reset_index(drop=True)


@_cache_data
def load_pca_sensitivity_daily() -> pd.DataFrame:
    """Pre-computed fixed-k PCA forecasts (k = 3, 7, 8 only)."""
    df = _read_csv(PATHS["pca_sensitivity_daily"])
    _check_columns(
        df,
        [
            "Date",
            "portfolio",
            "method",
            "confidence",
            "VaR",
            "actual_return",
            "actual_loss",
            "violation",
            "pca_components",
            "pca_explained_variance",
            "training_end_date",
        ],
        "pca_sensitivity_daily.csv",
    )
    df["Date"] = _to_datetime(df["Date"])
    df["training_end_date"] = _to_datetime(df["training_end_date"])
    df["portfolio"] = df["portfolio"].astype(str)
    df["method"] = df["method"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").round(4)
    for col in ["VaR", "actual_return", "actual_loss"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["violation"] = pd.to_numeric(df["violation"], errors="coerce").fillna(0).astype("int64")
    return df.sort_values(
        ["portfolio", "confidence", "method", "Date"], kind="mergesort"
    ).reset_index(drop=True)


@_cache_data
def load_pca_sensitivity_summary() -> pd.DataFrame:
    df = _read_csv(PATHS["pca_sensitivity_summary"])
    _check_columns(df, SUMMARY_COLUMNS, "pca_sensitivity_backtest_detail.csv")
    df["portfolio"] = df["portfolio"].astype(str)
    df["method"] = df["method"].astype(str)
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce").round(4)
    for col in [
        "backtest_days",
        "breaches",
        "actual_breach_rate",
        "expected_rate",
        "expected_breaches",
        "Kupiec_LR",
        "Kupiec_p_value",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["reject_at_5pct"] = (
        df["reject_at_5pct"].astype(str).str.lower().isin(["true", "1", "yes"])
    )
    return df


@_cache_data
def load_correlation() -> tuple[pd.DataFrame, list[str]]:
    """18x18 full-sample ETF correlation matrix (rows = ETFs)."""
    df = _read_csv(PATHS["correlation"])
    if df.columns[0] != "ETF":
        raise DashboardDataError("ETF-correlation.csv must start with an 'ETF' column.")
    df = df.set_index("ETF")
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    df = df.apply(pd.to_numeric, errors="coerce")
    if df.shape[0] != df.shape[1]:
        raise DashboardDataError("The correlation matrix is not square.")
    return df, list(df.columns)


@_cache_data
def load_pca_variance() -> pd.DataFrame:
    df = _read_csv(PATHS["pca_variance"])
    _check_columns(df, ["component", "eigenvalue", "explained_ratio", "cumulative_ratio"], "PCA-explained-variance.csv")
    for col in ["eigenvalue", "explained_ratio", "cumulative_ratio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True)


@_cache_data
def load_pca_loadings() -> pd.DataFrame:
    """18 ETFs x PC1..PC18 loadings (correlations with PC scores)."""
    df = _read_csv(PATHS["pca_loadings"])
    if df.columns[0] != "ETF":
        raise DashboardDataError("PCA-loadings.csv must start with an 'ETF' column.")
    df = df.set_index("ETF")
    df.index = df.index.astype(str)
    return df.apply(pd.to_numeric, errors="coerce")


@_cache_data
def load_static_weights() -> pd.DataFrame:
    df = _read_csv(PATHS["weights"])
    _check_columns(df, ["asset", "equal_weight", "eq_6040", "min_variance_first500"], "portfolio_weights.csv")
    return df.set_index("asset").apply(pd.to_numeric, errors="coerce")


@_cache_data
def load_dynamic_weights() -> pd.DataFrame:
    df = _read_csv(PATHS["dynamic_weights"])
    df["Date"] = _to_datetime(df["Date"])
    df["train_end"] = _to_datetime(df["train_end"])
    return df.sort_values(["portfolio", "Date"], kind="mergesort").reset_index(drop=True)


# Statistics (derived, never written back)
def kupiec_test(n: int, breaches: int, confidence: float) -> tuple[float, float]:
    """Kupiec unconditional-coverage LR test.

    Returns (LR statistic, p-value) for a chi-square(1).  The p-value uses
    ``erfc(sqrt(LR/2))``, which is exact for one degree of freedom, so the
    dashboard does not need SciPy at runtime.
    """
    if n <= 0:
        return float("nan"), float("nan")
    p0 = 1.0 - float(confidence)
    x = int(breaches)
    if x <= 0:
        lr = -2.0 * n * math.log(1.0 - p0)
    elif x >= n:
        lr = -2.0 * n * math.log(p0)
    else:
        phat = x / n
        lr = -2.0 * (
            x * math.log(p0 / phat) + (n - x) * math.log((1.0 - p0) / (1.0 - phat))
        )
    lr = max(lr, 0.0)
    p_value = math.erfc(math.sqrt(lr / 2.0))
    return lr, p_value


def slice_predictions(
    predictions: pd.DataFrame,
    portfolio: str,
    confidence: float,
    method: str,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter one portfolio/confidence/method slice, optionally by date."""
    mask = (
        (predictions["portfolio"] == portfolio)
        & (predictions["confidence"].round(4) == round(float(confidence), 4))
        & (predictions["method"] == method)
    )
    out = predictions.loc[mask]
    if start is not None:
        out = out.loc[out["Date"] >= pd.Timestamp(start)]
    if end is not None:
        out = out.loc[out["Date"] <= pd.Timestamp(end)]
    return out.sort_values("Date", kind="mergesort").reset_index(drop=True)


def metrics_for(slice_df: pd.DataFrame, confidence: float) -> dict[str, float | int]:
    """Backtest metrics for an arbitrary (possibly date-filtered) slice."""
    n = int(len(slice_df))
    breaches = int(slice_df["violation"].sum()) if n else 0
    expected_rate = 1.0 - float(confidence)
    actual_rate = breaches / n if n else float("nan")
    lr, p_value = kupiec_test(n, breaches, confidence)
    return {
        "n": n,
        "breaches": breaches,
        "actual_rate": actual_rate,
        "expected_rate": expected_rate,
        "expected_breaches": n * expected_rate,
        "kupiec_lr": lr,
        "kupiec_p": p_value,
        "rejected_5pct": bool(n > 0 and p_value < 0.05),
    }


def summary_row(summary: pd.DataFrame, portfolio: str, confidence: float, method: str) -> pd.Series:
    mask = (
        (summary["portfolio"] == portfolio)
        & (summary["confidence"].round(4) == round(float(confidence), 4))
        & (summary["method"] == method)
    )
    rows = summary.loc[mask]
    if rows.empty:
        raise DashboardDataError(
            f"No summary row for {portfolio} / {confidence:.0%} / {method}."
        )
    return rows.iloc[0]


def comparison_table(summary: pd.DataFrame, portfolio: str, confidence: float) -> pd.DataFrame:
    """Five-model comparison table in canonical method order."""
    from app.theme import METHOD_LABELS

    rows = []
    for method in METHODS:
        mask = (
            (summary["portfolio"] == portfolio)
            & (summary["confidence"].round(4) == round(float(confidence), 4))
            & (summary["method"] == method)
        )
        match = summary.loc[mask]
        if match.empty:
            continue
        r = match.iloc[0]
        rows.append(
            {
                "Method": METHOD_LABELS[method],
                "method": method,
                "Breaches": int(r["breaches"]),
                "Actual breach rate": float(r["actual_breach_rate"]) * 100.0,
                "Expected breach rate": float(r["expected_rate"]) * 100.0,
                "Kupiec p-value": float(r["Kupiec_p_value"]),
                "At 5% level": "Rejected" if bool(r["reject_at_5pct"]) else "Not rejected",
            }
        )
    return pd.DataFrame(rows)


def breach_log(slice_df: pd.DataFrame) -> pd.DataFrame:
    """Table of VaR exceptions for the selected slice."""
    if slice_df.empty:
        return pd.DataFrame(
            columns=["Date", "Actual loss (%)", "VaR (%)", "Excess (bp)"]
        )
    breached = slice_df.loc[slice_df["violation"] == 1].copy()
    breached["Date"] = breached["Date"].dt.strftime("%Y-%m-%d")
    breached["Actual loss (%)"] = breached["actual_loss"] * 100.0
    breached["VaR (%)"] = breached["VaR"] * 100.0
    breached["Excess (bp)"] = (breached["actual_loss"] - breached["VaR"]) * 10_000.0
    return breached[["Date", "Actual loss (%)", "VaR (%)", "Excess (bp)"]].reset_index(drop=True)


def date_bounds(predictions: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    return predictions["Date"].min(), predictions["Date"].max()


def n_unique_dates(predictions: pd.DataFrame) -> int:
    return int(predictions["Date"].nunique())


def rolling_pca_identical(diagnostics: pd.DataFrame) -> bool:
    """True when all portfolios share the same asset-panel PCA diagnostics."""
    if diagnostics.empty:
        return True
    pivot = diagnostics.pivot_table(
        index="Date", columns="portfolio", values="pca_components", aggfunc="first"
    )
    if pivot.shape[1] <= 1:
        return True
    return bool((pivot.nunique(axis=1) == 1).all())


# Integrity report (shown in the sidebar and tested)
def verify_integrity(
    predictions: pd.DataFrame, summary: pd.DataFrame
) -> list[dict[str, object]]:
    """Return the dashboard's data-contract checks as a list of dicts."""
    checks: list[dict[str, object]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    add(
        "Forecast row count",
        len(predictions) == N_FORECASTS,
        f"{len(predictions):,} of {N_FORECASTS:,} rows",
    )
    add(
        "Summary row count",
        len(summary) == N_SUMMARY_ROWS,
        f"{len(summary)} of {N_SUMMARY_ROWS} rows",
    )

    group_sizes = predictions.groupby(["portfolio", "confidence", "method"]).size()
    add(
        "Balanced slices (1,194 days)",
        bool(len(group_sizes) == 30 and (group_sizes == N_PER_SLICE).all()),
        f"{int((group_sizes == N_PER_SLICE).sum())}/{len(group_sizes)} groups at 1,194",
    )

    add(
        "Method vocabulary",
        set(predictions["method"].unique()) == set(METHODS),
        ", ".join(sorted(predictions["method"].unique())),
    )
    add(
        "Portfolio vocabulary",
        set(predictions["portfolio"].unique()) == set(PORTFOLIOS),
        ", ".join(sorted(predictions["portfolio"].unique())),
    )
    add(
        "Confidence vocabulary",
        set(predictions["confidence"].round(4).unique()) == {0.95, 0.99},
        "0.95 / 0.99",
    )

    duplicate_keys = predictions.duplicated(
        ["Date", "portfolio", "confidence", "method"]
    ).sum()
    add("Unique forecast keys", int(duplicate_keys) == 0, f"{int(duplicate_keys)} duplicates")

    key_cols = ["Date", "portfolio", "method", "confidence", "VaR", "actual_loss", "violation"]
    missing = int(predictions[key_cols].isna().sum().sum())
    add("Critical fields populated", missing == 0, f"{missing} missing values")

    violation_mismatch = int(
        (
            predictions["violation"].astype(int)
            != (predictions["actual_loss"] > predictions["VaR"]).astype(int)
        ).sum()
    )
    add(
        "Violation = actual_loss > VaR",
        violation_mismatch == 0,
        f"{violation_mismatch} mismatches",
    )

    lookahead = int(
        (predictions["training_end_date"] >= predictions["Date"]).sum()
    )
    add(
        "No look-ahead in training window",
        lookahead == 0,
        f"{lookahead} rows with training_end_date >= Date",
    )

    # Independent reconciliation of the summary against the daily rows.
    recon_errors = 0
    p_errors = 0
    for _, row in summary.iterrows():
        match = predictions.loc[
            (predictions["portfolio"] == row["portfolio"])
            & (predictions["confidence"].round(4) == round(float(row["confidence"]), 4))
            & (predictions["method"] == row["method"])
        ]
        if match.empty:
            recon_errors += 1
            continue
        n = len(match)
        x = int(match["violation"].sum())
        if (
            n != int(row["backtest_days"])
            or x != int(row["breaches"])
            or abs(x / n - float(row["actual_breach_rate"])) > 1e-6
            or abs(n * (1.0 - float(row["confidence"])) - float(row["expected_breaches"])) > 1e-6
        ):
            recon_errors += 1
        _, p_value = kupiec_test(n, x, float(row["confidence"]))
        if abs(p_value - float(row["Kupiec_p_value"])) > 1e-9:
            p_errors += 1

    add("Summary reconciles to daily rows", recon_errors == 0, f"{recon_errors} mismatched groups")
    add("Kupiec p-values reproducible", p_errors == 0, f"{p_errors} p-value mismatches")

    return checks


def source_label(key: str) -> str:
    """Project-relative path used in captions/footers."""
    return PATHS[key].relative_to(ROOT).as_posix()


def as_percent(value: float, digits: int = 2) -> str:
    return f"{value * 100:.{digits}f}%"


def fmt_p_value(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    if value < 0.0001:
        return "<0.0001"
    return f"{value:.4f}"


def fmt_int(value: int | float) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_bp(value: float, digits: int = 1) -> str:
    return f"{value:+,.{digits}f} bp"


def ensure_known(value: str, allowed: Iterable[str], label: str) -> str:
    allowed = list(allowed)
    if value not in allowed:
        raise DashboardDataError(f"Unknown {label}: {value!r}. Expected one of {allowed}.")
    return value
