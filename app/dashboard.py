"""Streamlit interface for exploring saved VaR and PCA results."""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allows `streamlit run app/dashboard.py`
    sys.path.insert(0, str(ROOT))

from app import charts  # noqa: E402
from app import components as ui  # noqa: E402
from app.data_loader import (  # noqa: E402
    BACKTEST_END,
    BACKTEST_START,
    CONFIDENCES,
    METHODS,
    PORTFOLIOS,
    DashboardDataError,
    breach_log,
    comparison_table,
    date_bounds,
    fmt_int,
    fmt_p_value,
    kupiec_test,
    load_correlation,
    load_pca_diagnostics,
    load_pca_loadings,
    load_pca_sensitivity_daily,
    load_pca_sensitivity_summary,
    load_pca_variance,
    load_predictions,
    load_summary,
    metrics_for,
    n_unique_dates,
    slice_predictions,
    source_label,
    summary_row,
    verify_integrity,
)
from app.theme import (  # noqa: E402
    ACCENT,
    AMBER,
    METHOD_COLORS,
    METHOD_LABELS,
    METHOD_SHORT,
    MUTED,
    MUTED_DIM,
    PCA_K_COLORS,
    PINK,
    PORTFOLIO_LABELS,
    RED,
    TEAL,
    TEXT,
)

# Page setup
st.set_page_config(
    page_title="Portfolio Risk Lab | PCA-Enhanced VaR",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)
ui.inject_theme()

PAGES: list[tuple[str, str]] = [
    ("Overview", "01   Overview"),
    ("Data & PCA", "02   Data & PCA"),
    ("VaR Backtesting", "03   VaR Backtesting"),
    ("Model Comparison", "04   Model Comparison"),
]
PAGE_ALIASES = {
    "overview": "Overview",
    "data": "Data & PCA",
    "pca": "Data & PCA",
    "var": "VaR Backtesting",
    "backtest": "VaR Backtesting",
    "compare": "Model Comparison",
    "comparison": "Model Comparison",
}


def _query_param(name: str, default: str) -> str:
    try:
        value = st.query_params.get(name, default)
    except Exception:  # pragma: no cover - very old Streamlit
        return default
    if isinstance(value, list):
        value = value[0] if value else default
    return str(value)


def _init_state() -> None:
    if "page" not in st.session_state:
        st.session_state.page = PAGE_ALIASES.get(
            _query_param("page", "overview").lower(), "Overview"
        )
    if "ctx_portfolio" not in st.session_state:
        requested = _query_param("portfolio", "EqualWeight")
        st.session_state.ctx_portfolio = (
            requested if requested in PORTFOLIOS else "EqualWeight"
        )
    if "ctx_confidence" not in st.session_state:
        try:
            requested_conf = float(_query_param("confidence", "0.99"))
        except ValueError:
            requested_conf = 0.99
        st.session_state.ctx_confidence = (
            requested_conf if requested_conf in CONFIDENCES else 0.99
        )
    if "ctx_method" not in st.session_state:
        requested_method = _query_param("method", "PCA_Factor")
        st.session_state.ctx_method = (
            requested_method if requested_method in METHODS else "PCA_Factor"
        )


def _apply_preset(portfolio: str, confidence: float, method: str) -> None:
    st.session_state.ctx_portfolio = portfolio
    st.session_state.ctx_confidence = confidence
    st.session_state.ctx_method = method
    st.session_state.page = "VaR Backtesting"


def _range_tuple(start: pd.Timestamp, end: pd.Timestamp) -> tuple:
    return (pd.Timestamp(start).date(), pd.Timestamp(end).date())


def _remember_range() -> None:
    value = st.session_state.get("bt_range")
    if isinstance(value, (tuple, list)) and len(value) == 2:
        st.session_state.bt_saved_range = (value[0], value[1])


def _set_range(start: pd.Timestamp, end: pd.Timestamp) -> None:
    dates = _range_tuple(start, end)
    st.session_state.bt_saved_range = dates
    st.session_state.bt_range = dates
    st.session_state.page = "VaR Backtesting"


def _reset_range(min_date: pd.Timestamp, max_date: pd.Timestamp) -> None:
    dates = _range_tuple(min_date, max_date)
    st.session_state.bt_saved_range = dates
    st.session_state.bt_range = dates


_init_state()


# Data
@st.cache_data(show_spinner="Loading results…")
def _load_bundle() -> dict[str, object]:
    return {
        "predictions": load_predictions(),
        "summary": load_summary(),
        "diagnostics": load_pca_diagnostics(),
        "correlation": load_correlation(),
        "pca_variance": load_pca_variance(),
        "pca_loadings": load_pca_loadings(),
        "sens_daily": load_pca_sensitivity_daily(),
        "sens_summary": load_pca_sensitivity_summary(),
    }


try:
    DATA = _load_bundle()
except DashboardDataError as exc:
    st.error("Missing result file.")
    st.code(str(exc), language="text")
    st.caption(
        "Expected layout: results/tables/monte_carlo/*.csv, results/tables/*.csv and "
        "data/eda_reference/*.csv inside the project root."
    )
    st.stop()
except Exception as exc:  # pragma: no cover - defensive
    st.error("Unexpected error while loading the result tables.")
    st.code(f"{type(exc).__name__}: {exc}", language="text")
    st.stop()

PREDICTIONS: pd.DataFrame = DATA["predictions"]  # type: ignore[assignment]
SUMMARY: pd.DataFrame = DATA["summary"]  # type: ignore[assignment]
DIAGNOSTICS: pd.DataFrame = DATA["diagnostics"]  # type: ignore[assignment]
CORR, ETF_LABELS = DATA["correlation"]  # type: ignore[misc]
PCA_VARIANCE: pd.DataFrame = DATA["pca_variance"]  # type: ignore[assignment]
PCA_LOADINGS: pd.DataFrame = DATA["pca_loadings"]  # type: ignore[assignment]
SENS_DAILY: pd.DataFrame = DATA["sens_daily"]  # type: ignore[assignment]
SENS_SUMMARY: pd.DataFrame = DATA["sens_summary"]  # type: ignore[assignment]

MIN_DATE, MAX_DATE = date_bounds(PREDICTIONS)
FULL_RANGE = (MIN_DATE.date(), MAX_DATE.date())
N_DATES = n_unique_dates(PREDICTIONS)
INTEGRITY = verify_integrity(PREDICTIONS, SUMMARY)
INTEGRITY_OK = all(bool(c["ok"]) for c in INTEGRITY)


# Sidebar
with st.sidebar:
    ui.brand()
    st.markdown('<div style="height:0.55rem"></div>', unsafe_allow_html=True)
    ui.side_label("Workspace")

    for name, label in PAGES:
        active = st.session_state.page == name
        if st.button(
            label,
            key=f"nav_{name.lower().replace(' ', '_').replace('&', 'and')}",
            type="primary" if active else "secondary",
        ):
            st.session_state.page = name
            st.rerun()

    ui.side_rule()
    ui.side_label("Analysis context")

    portfolio = st.selectbox(
        "Portfolio",
        PORTFOLIOS,
        format_func=lambda p: PORTFOLIO_LABELS[p],
        key="ctx_portfolio",
    )
    confidence = st.radio(
        "Confidence level",
        CONFIDENCES,
        horizontal=True,
        format_func=lambda c: f"{c:.0%}",
        key="ctx_confidence",
    )
    method = st.selectbox(
        "VaR method",
        METHODS,
        format_func=lambda m: METHOD_LABELS[m],
        key="ctx_method",
    )

    st.markdown('<div style="height:0.35rem"></div>', unsafe_allow_html=True)
    with st.expander("Presentation presets", expanded=False):
        st.button(
            "Desk default · EqualWeight 99% PCA",
            key="preset_default",
            on_click=_apply_preset,
            args=("EqualWeight", 0.99, "PCA_Factor"),
        )
        st.button(
            "Tail stress · MinVariance 99% PCA",
            key="preset_tail",
            on_click=_apply_preset,
            args=("MinVariance", 0.99, "PCA_Factor"),
        )
        st.button(
            "60/40 · 95% Historical",
            key="preset_6040",
            on_click=_apply_preset,
            args=("6040", 0.95, "Historical"),
        )

    st.caption(f"{len(PREDICTIONS):,} forecasts · {N_DATES:,} days")


# Defensive normalisation in case session values were edited elsewhere.
if portfolio not in PORTFOLIOS:
    portfolio = "EqualWeight"
if confidence not in CONFIDENCES:
    confidence = 0.99
if method not in METHODS:
    method = "PCA_Factor"


if not INTEGRITY_OK:
    failed = [c for c in INTEGRITY if not c["ok"]]
    ui.note(
        "Data check failed: "
        + "; ".join(f"{c['name']} ({c['detail']})" for c in failed),
        "DATA WARNING",
        "danger",
    )


# Shared page helpers
def _page_footer(*sources: str) -> None:
    ui.spacer(8)
    st.markdown(
        '<div style="height:1px;background:rgba(125,160,200,0.16);margin:0.6rem 0 0.7rem 0"></div>',
        unsafe_allow_html=True,
    )
    st.caption("Sources: " + " · ".join(f"`{s}`" for s in sources))


def _kpi_slice_label(base: str, confidence: float, window_label: str) -> str:
    return f"{base} · {confidence:.0%} · {window_label}"


# Page A - Overview
def page_overview() -> None:
    ui.page_header(
        "Financial engineering · research dashboard",
        "Portfolio Risk Lab | PCA-Enhanced Value-at-Risk",
        "18 ETFs · 5 Models · 3 Portfolios · Rolling Backtesting",
        accent_word="Value-at-Risk",
        chips=[
            f"OOS <b>{BACKTEST_START:%d %b %Y} → {BACKTEST_END:%d %b %Y}</b>",
            "<b>1-day</b> forecast horizon",
            f"<b>{len(PREDICTIONS):,}</b> VaR forecasts",
        ],
    )

    slice_df = slice_predictions(PREDICTIONS, portfolio, confidence, method)
    metrics = metrics_for(slice_df, confidence)
    window = "Full period"

    ui.spacer(14)
    ui.kpi_row(
        [
            {
                "label": "Universe",
                "value": "18 <small>ETFs</small>",
                "sub": "US-listed equity, rates, credit, commodity and real-estate sleeves",
                "tone": ACCENT,
            },
            {
                "label": "Out-of-sample days",
                "value": f"{N_DATES:,}",
                "sub": f"{BACKTEST_START:%d %b %Y} → {BACKTEST_END:%d %b %Y} · 500-day rolling window",
                "tone": TEAL,
            },
            {
                "label": "VaR methods",
                "value": "5",
                "sub": "Historical · Parametric · EWMA · PCA factor · Monte Carlo",
                "tone": AMBER,
            },
            {
                "label": _kpi_slice_label("Violation rate", confidence, window),
                "value": f"{metrics['actual_rate']:.2%}",
                "sub": (
                    f"Expected <b>{metrics['expected_rate']:.2%}</b> · "
                    f"<b>{metrics['breaches']}</b> breaches in {metrics['n']:,} days"
                ),
                "tone": ui.tone_for_rate(metrics["actual_rate"], metrics["expected_rate"]),
            },
        ]
    )

    ui.spacer(16)
    ui.intro_grid(
        "Does PCA preserve portfolio risk measurement quality with fewer factors?",
        f"<p>{N_DATES:,} out-of-sample trading days, "
        f"{BACKTEST_START:%d %b %Y} to {BACKTEST_END:%d %b %Y}.</p>"
        "<p>Five VaR models on three portfolios at 95% and 99%.</p>",
        [
            ("Assets", "18 ETFs"),
            ("Forecast rows", f"{len(PREDICTIONS):,}"),
            ("Summary rows", f"{len(SUMMARY)}"),
            ("Backtest window", "2021-12-28 → 2026-09-30"),
            ("Training window", "500 trading days"),
            ("Confidence levels", "95% and 99%"),
        ],
    )

    ui.section(
        "Method workflow",
        "Prices → Returns → EDA → Rolling PCA → VaR → Backtesting",
    )
    ui.workflow(
        [
            ("Prices", "18 adjusted ETF series", ""),
            ("Returns", "Daily log → simple returns", ""),
            ("EDA", "Correlation & full-sample PCA", ""),
            ("Rolling PCA", "500-day window, k to 90%", ""),
            ("VaR", "Five models, 95% / 99%", ""),
            ("Backtesting", "1,194 OOS days + Kupiec", ""),
        ]
    )

    _page_footer(
        source_label("predictions"),
        source_label("summary"),
        source_label("pca_variance"),
    )


# Page B - Data & PCA
def page_data_pca() -> None:
    ui.page_header(
        "Cross-asset diagnostics",
        "Data & PCA",
        "Correlation structure, full-sample variance decomposition and the rolling factor model.",
        chips=[
            "Full-sample EDA reference",
            "Standardized log returns",
            "18 × 18 matrix",
            f"Rolling k ∈ {{6, 7, 8}}",
        ],
    )

    ui.section(
        "01 · Correlation",
        "Cross-asset correlation matrix",
        "Full-sample daily log-return correlation.",
    )
    with st.container(border=True):
        ui.chart(charts.correlation_heatmap(CORR, ETF_LABELS))
        st.caption(
            f"Source: `{source_label('correlation')}` · colour scale −1 … +1 "
            "(red positive, blue negative)."
        )

    pc_options = [pc for pc in ["PC1", "PC2", "PC3", "PC7"] if pc in PCA_LOADINGS.columns]
    head_left, head_right = st.columns([1.18, 1.0], gap="large")
    with head_left:
        ui.section(
            "02 · Full-sample PCA",
            "Variance explained and ETF loadings",
            "Standardized log-return panel; PC7 = 90.74% cumulative variance.",
        )
    with head_right:
        selected_pc = st.selectbox(
            "Principal component",
            pc_options,
            index=0,
            key="pca_component",
        )

    left, right = st.columns([1.18, 1.0], gap="large")
    with left:
        with st.container(border=True):
            ui.chart_head(
                "Variance decomposition",
                "Bars: variance per component · line: cumulative",
                right="PC7 = 90.74%",
            )
            ui.chart(charts.pca_variance_figure(PCA_VARIANCE))

    with right:
        with st.container(border=True):
            ui.chart_head(
                f"ETF loadings · {selected_pc}",
                "Sorted loadings; cyan positive, red negative",
            )
            ui.chart(charts.pca_loadings_figure(PCA_LOADINGS, selected_pc))

    ui.section(
        "03 · Rolling PCA",
        "Factor count through the backtest",
    )

    diag = DIAGNOSTICS.loc[DIAGNOSTICS["portfolio"] == portfolio].sort_values("Date")
    col_a, col_b = st.columns([1.9, 1.0], gap="large")
    with col_a:
        with st.container(border=True):
            ui.chart_head(
                "Retained components over time",
                f"Portfolio view: {PORTFOLIO_LABELS[portfolio]} · step line = model decision",
                right=f"{diag['pca_components'].min():.0f}–{diag['pca_components'].max():.0f} PCs",
            )
            ui.chart(charts.rolling_k_figure(diag))
    with col_b:
        with st.container(border=True):
            ui.chart_head("Distribution of k", "Share of the 1,194 forecast days")
            ui.chart(charts.rolling_k_distribution(diag))

    with st.container(border=True):
        ui.chart_head(
            "Variance captured by the retained components",
            "Each point is the training-window cumulative variance at the selected k · dashed line = 90% target",
        )
        ui.chart(charts.rolling_ev_figure(diag))
        st.caption("Source: `results/tables/pca_window_diagnostics.csv`.")

    _page_footer(
        source_label("correlation"),
        source_label("pca_variance"),
        source_label("pca_loadings"),
        source_label("pca_diagnostics"),
    )


# Page C - VaR Backtesting (primary demo page)
def page_var_backtesting() -> None:
    ui.page_header(
        "Out-of-sample evidence",
        "VaR Backtesting",
        "One-day VaR against the realised portfolio loss. Red × marks every exception.",
        chips=[
            f"<b>{PORTFOLIO_LABELS[portfolio]}</b>",
            f"Confidence <b>{confidence:.0%}</b>",
            f"Method <b>{METHOD_LABELS[method]}</b>",
        ],
    )

    # --- Date window -----------------------------------------------------
    # Keep the selected dates outside the widget key: Streamlit may clean up
    # widget state when the page changes, while bt_saved_range survives.
    if "bt_saved_range" not in st.session_state:
        saved = st.session_state.get("bt_range", FULL_RANGE)
        if isinstance(saved, (tuple, list)) and len(saved) == 2:
            st.session_state.bt_saved_range = (saved[0], saved[1])
        else:
            st.session_state.bt_saved_range = FULL_RANGE
    if "bt_range" not in st.session_state:
        st.session_state.bt_range = st.session_state.bt_saved_range

    with st.expander("Forecast window", expanded=False):
        selected_range = st.slider(
            "Forecast date range",
            min_value=MIN_DATE.date(),
            max_value=MAX_DATE.date(),
            format="YYYY-MM-DD",
            key="bt_range",
            on_change=_remember_range,
        )
        if isinstance(selected_range, (tuple, list)) and len(selected_range) == 2:
            start_date, end_date = selected_range
        else:  # partial drag
            start_date, end_date = FULL_RANGE
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.button(
                "Full period",
                key="win_full",
                on_click=_reset_range,
                args=(MIN_DATE, MAX_DATE),
            )
        with p2:
            st.button(
                "Last 12M",
                key="win_12m",
                on_click=_set_range,
                args=(MAX_DATE - pd.DateOffset(months=12), MAX_DATE),
            )
        with p3:
            st.button(
                "Last 24M",
                key="win_24m",
                on_click=_set_range,
                args=(MAX_DATE - pd.DateOffset(months=24), MAX_DATE),
            )
        with p4:
            st.button(
                "Calendar 2022",
                key="win_2022",
                on_click=_set_range,
                args=(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
            )


    slice_df = slice_predictions(
        PREDICTIONS, portfolio, confidence, method, start_ts, end_ts
    )
    if slice_df.empty:
        ui.empty_state("No forecast dates fall inside the selected window. Widen the date range.")
        return

    metrics = metrics_for(slice_df, confidence)
    official = summary_row(SUMMARY, portfolio, confidence, method)
    is_full = (start_ts <= MIN_DATE) and (end_ts >= MAX_DATE)
    window_label = "Full backtest period" if is_full else "Filtered window"

    # Cross-check the official summary whenever the full window is selected.
    if is_full:
        reconcile_ok = (
            int(official["breaches"]) == metrics["breaches"]
            and int(official["backtest_days"]) == metrics["n"]
            and abs(float(official["Kupiec_p_value"]) - float(metrics["kupiec_p"])) < 1e-9
        )
        if not reconcile_ok:
            ui.note(
                "Full-period metrics do not match the summary row.",
                "CHECK",
                "danger",
            )

    if is_full:
        kupiec_value = fmt_p_value(float(metrics["kupiec_p"]))
        kupiec_sub = (
            f"{window_label} · rejected at 5%"
            if metrics["rejected_5pct"]
            else f"{window_label} · not rejected"
        )
        kupiec_tone = ui.tone_for_p_value(float(metrics["kupiec_p"]))
    else:
        kupiec_value = "—"
        kupiec_sub = "Reported for the full backtest only"
        kupiec_tone = MUTED

    ui.spacer(4)
    ui.kpi_row(
        [
            {
                "label": f"VaR breaches · {confidence:.0%}",
                "value": fmt_int(metrics["breaches"]),
                "sub": f"{window_label} · out of {metrics['n']:,} days",
                "tone": RED if metrics["breaches"] else TEAL,
            },
            {
                "label": f"Actual breach rate · {confidence:.0%}",
                "value": f"{metrics['actual_rate']:.2%}",
                "sub": (
                    f"{window_label} · expected <b>{metrics['expected_rate']:.2%}</b> · "
                    f"{metrics['expected_breaches']:.1f} breaches"
                ),
                "tone": ui.tone_for_rate(metrics["actual_rate"], metrics["expected_rate"]),
            },
            {
                "label": f"Expected rate · {confidence:.0%}",
                "value": f"{metrics['expected_rate']:.2%}",
                "sub": f"{window_label} · theoretical tail probability",
                "tone": ACCENT,
            },
            {
                "label": f"Kupiec p-value · {confidence:.0%}",
                "value": kupiec_value,
                "sub": kupiec_sub,
                "tone": kupiec_tone,
            },
            {
                "label": f"Sample size · {confidence:.0%}",
                "value": fmt_int(metrics["n"]),
                "sub": f"{window_label} · {PORTFOLIO_LABELS[portfolio]}",
                "tone": MUTED,
            },
        ],
        compact=True,
    )
    # --- Main chart ------------------------------------------------------
    ui.spacer(6)
    with st.container(border=True):
        ui.chart_head(
            f"Actual loss vs {confidence:.0%} VaR",
            f"{PORTFOLIO_LABELS[portfolio]} · {METHOD_LABELS[method]} · "
            f"{metrics['breaches']} breaches / {metrics['n']:,} days",
            right=f"{start_ts:%d %b %Y} → {end_ts:%d %b %Y}",
        )
        ui.chart(
            charts.var_backtest_figure(
                slice_df, confidence, METHOD_LABELS[method], PORTFOLIO_LABELS[portfolio]
            )
        )
        st.caption(
            "Y axis: % of portfolio value. Positive actual loss = loss; negative = gain."
        )

    # --- Exception log + methodology ------------------------------------
    ui.spacer(10)
    left, right = st.columns([1.45, 1.0], gap="large")
    log = breach_log(slice_df)
    with left:
        with st.container(border=True):
            ui.chart_head(
                "Exception log",
                f"Every day where actual loss exceeded the {confidence:.0%} VaR threshold",
                right=f"{len(log)} rows",
            )
            if log.empty:
                ui.empty_state("No violations in this window.")
            else:
                ui.dataframe(
                    log,
                    column_config={
                        "Actual loss (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "VaR (%)": st.column_config.NumberColumn(format="%.2f%%"),
                        "Excess (bp)": st.column_config.NumberColumn(format="%+.1f bp"),
                    },
                    height=min(330, 38 + 35 * len(log)),
                    key="breach_table",
                )
        d1, d2 = st.columns(2)
        with d1:
            ui.download_csv(
                log,
                f"var_breaches_{portfolio}_{int(confidence * 100)}_{method}.csv",
                "Download exception log",
                key="dl_breaches",
            )
        with d2:
            export = slice_df[
                ["Date", "VaR", "actual_loss", "violation", "training_end_date"]
            ].copy()
            export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
            export["training_end_date"] = export["training_end_date"].dt.strftime("%Y-%m-%d")
            ui.download_csv(
                export,
                f"var_forecasts_{portfolio}_{int(confidence * 100)}_{method}.csv",
                "Download filtered forecasts",
                key="dl_forecasts",
            )

    with right:
        ui.card(
            "Definitions",
            f"""
            <ul>
                <li><b>VaR</b> — loss threshold at {confidence:.0%} confidence.</li>
                <li><b>Violation</b> — actual loss greater than VaR.</li>
                <li><b>Forecast</b> — 1 day ahead, 500-day training window.</li>
            </ul>
            """,
        )
        if not log.empty:
            ui.spacer(8)
            with st.container(border=True):
                ui.chart_head(
                    "Exceptions by year",
                    "Breach days in the selected window",
                    right=f"{len(log)} days",
                )
                ui.chart(charts.breach_year_figure(log))

    _page_footer(
        source_label("predictions"),
        source_label("summary"),
    )


# Page D - Model Comparison
def page_model_comparison() -> None:
    ui.page_header(
        "Cross-model evidence",
        "Model Comparison",
        "Five models, one portfolio and one confidence level — compared on the same 1,194 out-of-sample days.",
        chips=[
            f"<b>{PORTFOLIO_LABELS[portfolio]}</b>",
            f"Confidence <b>{confidence:.0%}</b>",
            "<b>Full</b> backtest period",
        ],
    )

    comparison = comparison_table(SUMMARY, portfolio, confidence)
    if comparison.empty:
        ui.empty_state("No summary rows for this portfolio / confidence combination.")
        return

    ui.section(
        "01 · Exception rates",
        "Actual breach rate by method",
        f"Bars are measured exception rates; the dashed line is the {1 - confidence:.0%} theoretical rate.",
    )
    with st.container(border=True):
        ui.chart_head(
            f"Breach rate · {PORTFOLIO_LABELS[portfolio]} · {confidence:.0%}",
            "Same 1,194 forecast days for every model",
            right=f"n = {int(SUMMARY['backtest_days'].max()):,}",
        )
        ui.chart(charts.breach_rate_figure(comparison, confidence))

    ui.section(
        "02 · Performance table",
        "Full-period backtest statistics",
    )
    table = comparison[
        [
            "Method",
            "Breaches",
            "Actual breach rate",
            "Expected breach rate",
            "Kupiec p-value",
            "At 5% level",
        ]
    ].copy()
    with st.container(border=True):
        ui.dataframe(
            table,
            column_config={
                "Breaches": st.column_config.NumberColumn(format="%d"),
                "Actual breach rate": st.column_config.NumberColumn(format="%.3f%%"),
                "Expected breach rate": st.column_config.NumberColumn(format="%.3f%%"),
                "Kupiec p-value": st.column_config.NumberColumn(format="%.4f"),
            },
            height=248,
            key="comparison_table",
        )
        ui.download_csv(
            table,
            f"model_comparison_{portfolio}_{int(confidence * 100)}.csv",
            "Download comparison table",
            key="dl_comparison",
        )

    # --- Pairwise diagnostics -------------------------------------------
    ui.section(
        "03 · Pairwise diagnostics",
        "Where the models diverge",
    )
    pair_tabs = st.tabs(
        ["PCA factor vs Parametric", "Monte Carlo vs Parametric"]
    )

    def render_pair(method_a: str, method_b: str, label_a: str, label_b: str) -> None:
        pair = slice_predictions(
            PREDICTIONS.loc[
                PREDICTIONS["method"].isin([method_a, method_b])
            ],
            portfolio,
            confidence,
            method_a,
        )
        pair_b = slice_predictions(
            PREDICTIONS.loc[PREDICTIONS["method"] == method_b],
            portfolio,
            confidence,
            method_b,
        )
        combined = pd.concat([pair, pair_b], ignore_index=True)
        wide = (
            combined.pivot_table(
                index="Date", columns="method", values="VaR", aggfunc="first"
            )
            .sort_index()
        )
        diff_bp = (wide[method_a] - wide[method_b]) * 10_000.0
        mean_abs = float(diff_bp.abs().mean())
        mean_signed = float(diff_bp.mean())

        with st.container(border=True):
            ui.chart_head(
                f"{label_a} vs {label_b} · {confidence:.0%} VaR",
                f"{PORTFOLIO_LABELS[portfolio]} · daily paths, same forecast dates",
                right=f"mean |Δ| = {mean_abs:.1f} bp",
            )
            ui.chart(
                charts.pair_var_figure(combined, method_a, method_b, confidence)
            )
            ui.chart(
                charts.pair_diff_figure(combined, method_a, method_b, confidence)
            )
            st.caption(
                f"Difference = {label_a} − {label_b}. Mean signed difference "
                f"{mean_signed:+.1f} bp; mean absolute difference {mean_abs:.1f} bp. "
                "Positive = the first model has the higher VaR."
            )

    with pair_tabs[0]:
        render_pair("PCA_Factor", "Parametric", "PCA factor", "Parametric")
    with pair_tabs[1]:
        render_pair("MonteCarlo_Normal", "Parametric", "Monte Carlo", "Parametric")

    # --- PCA sensitivity ------------------------------------------------
    ui.section(
        "04 · PCA sensitivity",
        "Fixed-component records (k = 3, 7, 8 only)",
    )
    sens = SENS_SUMMARY.loc[
        (SENS_SUMMARY["portfolio"] == portfolio)
        & (SENS_SUMMARY["confidence"].round(4) == round(confidence, 4))
    ].copy()
    with st.expander("Open factor-count sensitivity", expanded=False):
        if sens.empty:
            ui.empty_state("No sensitivity records for this portfolio / confidence.")
        else:
            sens["k"] = sens["method"].str.extract(r"_k(\d+)$").astype(int)
            sens = sens.sort_values("k")
            col_a, col_b = st.columns([1.0, 1.15], gap="large")
            with col_a:
                with st.container(border=True):
                    ui.chart_head("Breach rate by k", f"{PORTFOLIO_LABELS[portfolio]} · {confidence:.0%}")
                    ui.chart(charts.pca_sensitivity_figure(sens, confidence))
            with col_b:
                disp = sens[
                    ["k", "breaches", "actual_breach_rate", "Kupiec_p_value", "reject_at_5pct"]
                ].copy()
                disp["k"] = disp["k"].map(lambda v: f"k = {int(v)}")
                disp["actual_breach_rate"] *= 100.0
                disp["reject_at_5pct"] = disp["reject_at_5pct"].map(
                    lambda v: "Yes" if bool(v) else "No"
                )
                disp = disp.rename(
                    columns={
                        "k": "Components",
                        "breaches": "Breaches",
                        "actual_breach_rate": "Breach rate (%)",
                        "Kupiec_p_value": "Kupiec p",
                        "reject_at_5pct": "Rejected 5%?",
                    }
                )
                with st.container(border=True):
                    ui.chart_head("Sensitivity table", "By component count")
                    ui.dataframe(
                        disp,
                        column_config={
                            "Breaches": st.column_config.NumberColumn(format="%d"),
                            "Breach rate (%)": st.column_config.NumberColumn(format="%.3f%%"),
                            "Kupiec p": st.column_config.NumberColumn(format="%.4f"),
                        },
                        height=190,
                        key="sens_table",
                    )
            with st.container(border=True):
                ui.chart_head(
                    "Daily VaR paths · fixed k vs production rolling k",
                    "Fixed k = 3, 7, 8 and the rolling-k model",
                )
                sens_daily = SENS_DAILY.loc[
                    (SENS_DAILY["portfolio"] == portfolio)
                    & (SENS_DAILY["confidence"].round(4) == round(confidence, 4))
                ]
                base_daily = slice_predictions(
                    PREDICTIONS, portfolio, confidence, "PCA_Factor"
                )
                ui.chart(
                    charts.sensitivity_var_figure(sens_daily, base_daily, confidence)
                )
                st.caption(
                    "Fixed-k lines come from `pca_sensitivity_daily.csv`; the dotted line is the "
                    "production PCA model whose k is chosen dynamically in each window."
                )

    _page_footer(
        source_label("summary"),
        source_label("predictions"),
        source_label("pca_sensitivity_summary"),
        source_label("pca_sensitivity_daily"),
    )


# Router
PAGE_RENDERERS = {
    "Overview": page_overview,
    "Data & PCA": page_data_pca,
    "VaR Backtesting": page_var_backtesting,
    "Model Comparison": page_model_comparison,
}

render = PAGE_RENDERERS.get(st.session_state.page, page_overview)

try:
    render()
except DashboardDataError as exc:
    st.error("Could not load the result tables.")
    st.code(str(exc), language="text")
except Exception as exc:  # pragma: no cover - show a safe failure state
    st.error("Could not render this view.")
    with st.expander("Technical details", expanded=False):
        st.code(f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}", language="text")
