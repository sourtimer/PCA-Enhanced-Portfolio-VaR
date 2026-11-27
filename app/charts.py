"""Plotly figures with consistent model colors and external legends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from app.theme import (
    ACCENT,
    ACCENT_SOFT,
    AMBER,
    BG,
    FONT_STACK,
    LINE,
    LINE_STRONG,
    METHOD_COLORS,
    METHOD_SHORT,
    MUTED,
    MUTED_DIM,
    PANEL_2,
    PCA_K_COLORS,
    PINK,
    RED,
    SLATE,
    TEAL,
    TEXT,
    VIOLET,
)

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "toggleSpikelines",
        "hoverClosestCartesian",
        "hoverCompareCartesian",
    ],
}


def _legend_top(fig: go.Figure) -> None:
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.012,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(family=FONT_STACK, size=11, color=MUTED),
            itemsizing="constant",
            tracegroupgap=8,
        )
    )


def _base(
    fig: go.Figure,
    height: int,
    *,
    legend: bool = True,
    top: int = 58,
    left: int = 6,
    right: int = 18,
    bottom: int = 10,
    hovermode: str = "closest",
) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=left, r=right, t=top, b=bottom),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=11.5, color=TEXT),
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor=PANEL_2,
            bordercolor=LINE_STRONG,
            font=dict(family=FONT_STACK, size=11.5, color=TEXT),
        ),
        showlegend=legend,
        dragmode="zoom",
        modebar=dict(
            bgcolor="rgba(0,0,0,0)",
            color=MUTED_DIM,
            activecolor=ACCENT,
        ),
    )
    if legend:
        _legend_top(fig)
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=LINE,
        linewidth=1,
        ticks="outside",
        tickcolor=LINE,
        ticklen=4,
        tickfont=dict(family=FONT_STACK, size=10.5, color=MUTED),
        zeroline=False,
        automargin=True,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(148, 163, 184, 0.095)",
        gridwidth=1,
        zeroline=False,
        showline=False,
        ticks="",
        tickfont=dict(family=FONT_STACK, size=10.5, color=MUTED),
        title_font=dict(family=FONT_STACK, size=10.5, color=MUTED_DIM),
        automargin=True,
    )
    return fig


# VaR backtesting
def var_backtest_figure(
    slice_df: pd.DataFrame,
    confidence: float,
    method_label: str,
    portfolio_label: str,
) -> go.Figure:
    """Actual loss vs VaR with a distinct layer for every breach."""
    df = slice_df.sort_values("Date")
    breaches = df.loc[df["violation"] == 1].copy()

    # customdata keeps the tooltip complete even for the marker layer
    df = df.assign(status=np.where(df["violation"] == 1, "YES", "No"))
    custom = df[["actual_loss", "status"]].to_numpy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["actual_loss"],
            name="Actual loss",
            mode="lines",
            line=dict(color=SLATE, width=1.15),
            fill="tozeroy",
            fillcolor="rgba(159, 179, 200, 0.045)",
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "Actual loss: <b>%{y:.2%}</b><br>"
                "Violation: %{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["Date"],
            y=df["VaR"],
            name=f"VaR ({confidence:.0%})",
            mode="lines",
            line=dict(color=ACCENT, width=1.55),
            customdata=custom,
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                "VaR: <b>%{y:.2%}</b><br>"
                f"Method: {method_label}<br>"
                "Portfolio: " + portfolio_label + "<br>"
                "Violation: %{customdata[1]}<extra></extra>"
            ),
        )
    )

    if not breaches.empty:
        breach_custom = breaches[["VaR", "actual_loss"]].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=breaches["Date"],
                y=breaches["actual_loss"],
                name=f"VaR breach ({len(breaches)})",
                mode="markers",
                marker=dict(
                    symbol="x",
                    size=10.5,
                    color=RED,
                    line=dict(width=1.6, color="#FFE4E8"),
                ),
                customdata=breach_custom,
                hovertemplate=(
                    "<b>VaR breach</b><br>"
                    "Date: %{x|%Y-%m-%d}<br>"
                    "Actual loss: <b>%{y:.2%}</b><br>"
                    "VaR: %{customdata[0]:.2%}<br>"
                    f"Method: {method_label}<br>"
                    "Violation: <b>YES</b><extra></extra>"
                ),
            )
        )

    fig.add_hline(
        y=0,
        line=dict(color="rgba(226, 232, 240, 0.30)", width=1, dash="dot"),
        layer="below",
    )
    fig.update_xaxes(
        tickformat="%b %Y",
        nticks=9,
        hoverformat="%Y-%m-%d",
        title_text=None,
    )
    fig.update_yaxes(
        tickformat=".1%",
        title_text="Return / loss (% of portfolio value)",
        ticksuffix="",
    )
    _base(fig, 360, legend=True, top=62, left=10, right=16, bottom=8)
    return fig


# Model comparison
def breach_rate_figure(comparison: pd.DataFrame, confidence: float) -> go.Figure:
    """Horizontal breach-rate bars with a theoretical expected-rate line."""
    comparison = comparison.iloc[::-1]  # first method ends up at the top
    labels = [METHOD_SHORT[m] for m in comparison["method"]]
    rates = comparison["Actual breach rate"].to_numpy() / 100.0
    colors = [METHOD_COLORS[m] for m in comparison["method"]]
    expected = 1.0 - confidence

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=rates,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{r:.2%}" for r in rates],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(family=FONT_STACK, size=11, color=TEXT),
            showlegend=False,
            hovertemplate="%{y}<br>Actual breach rate: <b>%{x:.2%}</b><extra></extra>",
        )
    )
    # A transparent trace supplies the legend entry for the reference line.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"Expected rate ({expected:.0%})",
            line=dict(color="#E2E8F0", width=1.2, dash="dash"),
            hoverinfo="skip",
            showlegend=True,
        )
    )
    fig.add_vline(
        x=expected,
        line=dict(color="#E2E8F0", width=1.2, dash="dash"),
        layer="below",
    )
    top = float(max(rates.max() if len(rates) else 0.0, expected))
    fig.update_xaxes(
        tickformat=".1%",
        range=[0, top * 1.32],
        title_text="Actual breach rate (% of forecast days)",
    )
    fig.update_yaxes(
        type="category",
        categoryorder="array",
        categoryarray=list(comparison["method"].map(METHOD_SHORT)),
        tickfont=dict(size=11.5, color=TEXT),
    )
    _base(fig, 400, legend=True, top=60, left=6, right=24, bottom=6)
    return fig


def breach_year_figure(log: pd.DataFrame) -> go.Figure:
    """Small bar chart: exception days by calendar year."""
    dates = pd.to_datetime(log["Date"])
    years = dates.dt.year
    span = range(int(years.min()), int(years.max()) + 1)
    counts = years.value_counts().reindex(span, fill_value=0).sort_index()
    fig = go.Figure(
        go.Bar(
            x=[str(y) for y in counts.index],
            y=counts.to_numpy(),
            marker=dict(color=RED, line=dict(width=0)),
            text=[str(v) for v in counts.to_numpy()],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(family=FONT_STACK, size=11, color=TEXT),
            hovertemplate="%{x}: <b>%{y}</b> breach days<extra></extra>",
        )
    )
    _base(fig, 210, legend=False, top=24, left=6, right=8, bottom=6)
    fig.update_xaxes(type="category", tickfont=dict(size=11, color=TEXT))
    fig.update_yaxes(
        title_text="Breach days",
        dtick=2,
        rangemode="tozero",
        tickfont=dict(size=10.5, color=MUTED),
    )
    return fig


def pair_var_figure(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    confidence: float,
) -> go.Figure:
    """VaR paths for two methods (long-format frame with a method column)."""
    fig = go.Figure()
    for method, width, dash in ((method_a, 1.55, "solid"), (method_b, 1.25, "solid")):
        part = df.loc[df["method"] == method].sort_values("Date")
        fig.add_trace(
            go.Scatter(
                x=part["Date"],
                y=part["VaR"],
                name=METHOD_SHORT[method],
                mode="lines",
                line=dict(color=METHOD_COLORS[method], width=width, dash=dash),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    f"{METHOD_SHORT[method]} VaR ({confidence:.0%}): "
                    "<b>%{y:.3%}</b><extra></extra>"
                ),
            )
        )
    fig.update_xaxes(tickformat="%b %Y", nticks=8, hoverformat="%Y-%m-%d")
    fig.update_yaxes(tickformat=".2%", title_text="VaR (% of portfolio value)")
    _base(fig, 372, legend=True, top=58, left=10, right=14, bottom=6, hovermode="x unified")
    return fig


def pair_diff_figure(
    df: pd.DataFrame,
    method_a: str,
    method_b: str,
    confidence: float,
) -> go.Figure:
    """Daily VaR difference in basis points, on its own panel."""
    wide = (
        df.loc[df["method"].isin([method_a, method_b]), ["Date", "method", "VaR"]]
        .pivot_table(index="Date", columns="method", values="VaR", aggfunc="first")
        .sort_index()
    )
    diff_bp = (wide[method_a] - wide[method_b]) * 10_000.0
    color = METHOD_COLORS[method_a]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=diff_bp.index,
            y=diff_bp.to_numpy(),
            mode="lines",
            name="Difference",
            line=dict(color=color, width=1.3),
            fill="tozeroy",
            fillcolor=_rgba(color, 0.13),
            hovertemplate=(
                "%{x|%Y-%m-%d}<br>"
                f"{METHOD_SHORT[method_a]} − {METHOD_SHORT[method_b]}: "
                "<b>%{y:+.1f} bp</b><extra></extra>"
            ),
            showlegend=False,
        )
    )
    fig.add_hline(
        y=0,
        line=dict(color="rgba(226, 232, 240, 0.34)", width=1, dash="dot"),
        layer="below",
    )
    fig.update_xaxes(tickformat="%b %Y", nticks=8, hoverformat="%Y-%m-%d")
    fig.update_yaxes(
        ticksuffix=" bp",
        title_text=f"VaR difference ({confidence:.0%})",
        zeroline=False,
    )
    _base(fig, 235, legend=False, top=22, left=10, right=14, bottom=6)
    return fig


def pca_sensitivity_figure(
    sens_summary: pd.DataFrame, confidence: float
) -> go.Figure:
    """Breach rate for the three pre-computed fixed-k PCA models only."""
    df = sens_summary.copy()
    df["k"] = df["method"].str.extract(r"_k(\d+)$").astype(int)
    df = df.sort_values("k")
    labels = [f"k = {k}" for k in df["k"]]
    rates = df["actual_breach_rate"].to_numpy()
    colors = [PCA_K_COLORS.get(int(k), AMBER) for k in df["k"]]
    expected = 1.0 - confidence

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=rates,
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{r:.2%}" for r in rates],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(family=FONT_STACK, size=11, color=TEXT),
            showlegend=False,
            hovertemplate="%{x}<br>Breach rate: <b>%{y:.2%}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=f"Expected rate ({expected:.0%})",
            line=dict(color="#E2E8F0", width=1.2, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_hline(
        y=expected,
        line=dict(color="#E2E8F0", width=1.2, dash="dash"),
        layer="below",
    )
    top = float(max(rates.max() if len(rates) else 0.0, expected))
    fig.update_yaxes(tickformat=".1%", range=[0, top * 1.38], title_text="Breach rate")
    fig.update_xaxes(type="category", title_text="Retained principal components")
    _base(fig, 250, legend=True, top=52, left=8, right=12, bottom=6)
    return fig


def sensitivity_var_figure(
    sens_daily: pd.DataFrame,
    base_daily: pd.DataFrame,
    confidence: float,
) -> go.Figure:
    """Daily VaR for the fixed-k PCA records plus the production rolling-k model."""
    fig = go.Figure()
    ordered = (
        sens_daily[["method", "pca_components"]]
        .drop_duplicates()
        .assign(k=lambda d: d["method"].str.extract(r"_k(\d+)$").astype(int))
        .sort_values("k")
    )
    for _, row in ordered.iterrows():
        part = sens_daily.loc[sens_daily["method"] == row["method"]].sort_values("Date")
        color = PCA_K_COLORS.get(int(row["k"]), AMBER)
        fig.add_trace(
            go.Scatter(
                x=part["Date"],
                y=part["VaR"],
                name=f"Fixed k = {int(row['k'])}",
                mode="lines",
                line=dict(color=color, width=1.2),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    f"Fixed k = {int(row['k'])} VaR ({confidence:.0%}): "
                    "<b>%{y:.3%}</b><extra></extra>"
                ),
            )
        )
    if base_daily is not None and not base_daily.empty:
        fig.add_trace(
            go.Scatter(
                x=base_daily["Date"],
                y=base_daily["VaR"],
                name="Rolling k (production)",
                mode="lines",
                line=dict(color=ACCENT, width=1.6, dash="dot"),
                hovertemplate=(
                    "%{x|%Y-%m-%d}<br>"
                    f"Rolling-k VaR ({confidence:.0%}): <b>%{{y:.3%}}</b><extra></extra>"
                ),
            )
        )
    fig.update_xaxes(tickformat="%b %Y", nticks=7, hoverformat="%Y-%m-%d")
    fig.update_yaxes(tickformat=".2%", title_text="VaR (% of portfolio value)")
    _base(fig, 330, legend=True, top=58, left=8, right=12, bottom=6, hovermode="x unified")
    return fig


# Data & PCA
def correlation_heatmap(corr: pd.DataFrame, labels: list[str]) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.loc[labels, labels].to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            zmid=0,
            colorscale=[
                [0.0, "#1D4ED8"],
                [0.22, "#2563EB"],
                [0.5, "#12233A"],
                [0.78, "#EA580C"],
                [1.0, "#F97316"],
            ],
            colorbar=dict(
                title=dict(text="ρ", side="top", font=dict(size=11, color=MUTED)),
                thickness=9,
                len=0.72,
                tickfont=dict(size=9.5, color=MUTED),
                outlinewidth=0,
                bgcolor="rgba(0,0,0,0)",
            ),
            hovertemplate="%{y} × %{x}<br>Correlation: <b>%{z:.3f}</b><extra></extra>",
            xgap=1.5,
            ygap=1.5,
        )
    )
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        tickfont=dict(size=9.5, color=MUTED),
    )
    fig.update_xaxes(
        showgrid=False,
        tickangle=0,
        tickfont=dict(size=9.5, color=MUTED),
    )
    fig.update_layout(
        height=618,
        # ETF tick labels need room on the left and below the matrix.
        margin=dict(l=58, r=10, t=10, b=46),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_STACK, size=10, color=MUTED),
        showlegend=False,
        hoverlabel=dict(
            bgcolor=PANEL_2,
            bordercolor=LINE_STRONG,
            font=dict(color=TEXT, size=11.5),
        ),
    )
    return fig


def pca_variance_figure(var_df: pd.DataFrame) -> go.Figure:
    labels = var_df["component"].tolist()
    explained = var_df["explained_ratio"].to_numpy() * 100.0
    cumulative = var_df["cumulative_ratio"].to_numpy() * 100.0

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=explained,
            name="Explained variance",
            marker=dict(color=_rgba(ACCENT, 0.55), line=dict(width=0)),
            hovertemplate="%{x}<br>Explained: <b>%{y:.2f}%</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels,
            y=cumulative,
            name="Cumulative",
            mode="lines+markers",
            line=dict(color=AMBER, width=2),
            marker=dict(size=5, color=AMBER, line=dict(width=0)),
            hovertemplate="%{x}<br>Cumulative: <b>%{y:.2f}%</b><extra></extra>",
        )
    )
    fig.add_hline(
        y=90,
        line=dict(color=AMBER, width=1, dash="dash"),
        opacity=0.55,
        layer="below",
    )

    if len(var_df) >= 7:
        k7 = var_df.iloc[6]
        fig.add_trace(
            go.Scatter(
                x=[k7["component"]],
                y=[float(k7["cumulative_ratio"]) * 100.0],
                mode="markers",
                marker=dict(size=10, color=BG, line=dict(color=AMBER, width=2)),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_annotation(
            x=k7["component"],
            y=float(k7["cumulative_ratio"]) * 100.0,
            text=f"PC7 · {float(k7['cumulative_ratio']):.2%}",
            showarrow=False,
            yshift=18,
            font=dict(family=FONT_STACK, size=10, color=AMBER),
        )

    fig.add_annotation(
        x=0.015,
        y=91.5,
        xref="paper",
        yref="y",
        text="90% target",
        showarrow=False,
        xanchor="left",
        font=dict(family=FONT_STACK, size=9.5, color=AMBER),
        opacity=0.85,
    )
    fig.update_yaxes(tickformat=".0f", ticksuffix="%", range=[0, 106], title_text="Variance explained")
    fig.update_xaxes(type="category")
    _base(fig, 500, legend=True, top=60, left=6, right=12, bottom=6)
    return fig


def pca_loadings_figure(loadings: pd.DataFrame, component: str) -> go.Figure:
    series = loadings[component].sort_values(ascending=False)
    values = series.to_numpy()
    colors = [ACCENT if v >= 0 else RED for v in values]
    span = float(np.nanmax(np.abs(values))) if len(values) else 1.0

    fig = go.Figure(
        go.Bar(
            x=values,
            y=series.index.tolist(),
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{v:+.2f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(family=FONT_STACK, size=9, color=MUTED),
            hovertemplate="%{y} · " + component + "<br>Loading: <b>%{x:.3f}</b><extra></extra>",
        )
    )
    fig.add_vline(
        x=0,
        line=dict(color="rgba(226, 232, 240, 0.36)", width=1),
        layer="below",
    )
    fig.update_xaxes(
        range=[-span * 1.32, span * 1.32],
        zeroline=False,
        title_text="Loading (correlation with PC score)",
    )
    fig.update_yaxes(
        autorange="reversed",
        tickfont=dict(size=9.5, color=TEXT),
        showgrid=False,
    )
    _base(fig, 500, legend=False, top=18, left=6, right=20, bottom=6)
    return fig


def rolling_k_figure(diag: pd.DataFrame) -> go.Figure:
    df = diag.sort_values("Date")
    fig = go.Figure(
        go.Scatter(
            x=df["Date"],
            y=df["pca_components"].astype(float),
            mode="lines+markers",
            line=dict(color=ACCENT, width=1.7, shape="hv"),
            marker=dict(size=3.2, color=ACCENT, opacity=0.45),
            hovertemplate="%{x|%Y-%m-%d}<br>Retained PCs: <b>%{y:.0f}</b><extra></extra>",
            showlegend=False,
        )
    )
    fig.update_xaxes(tickformat="%b %Y", nticks=7, hoverformat="%Y-%m-%d")
    fig.update_yaxes(
        tickmode="array",
        tickvals=[6, 7, 8],
        range=[5.55, 8.45],
        title_text="Retained PCs",
    )
    _base(fig, 326, legend=False, top=18, left=6, right=12, bottom=6)
    return fig


def rolling_k_distribution(diag: pd.DataFrame) -> go.Figure:
    counts = diag["pca_components"].value_counts().sort_index()
    total = int(counts.sum())
    labels = [f"k = {int(k)}" for k in counts.index]
    values = counts.to_numpy()
    colors = [PCA_K_COLORS.get(int(k), AMBER) for k in counts.index]

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=colors, line=dict(width=0)),
            text=[f"{int(v):,} · {v / total:.0%}" for v in values],
            textposition="outside",
            cliponaxis=False,
            textfont=dict(family=FONT_STACK, size=10, color=TEXT),
            hovertemplate="%{y}<br>Days: <b>%{x:,}</b><extra></extra>",
        )
    )
    fig.update_xaxes(range=[0, float(values.max()) * 1.45], title_text="Forecast days")
    fig.update_yaxes(type="category", tickfont=dict(size=11, color=TEXT), showgrid=False)
    _base(fig, 326, legend=False, top=18, left=6, right=16, bottom=6)
    return fig


def rolling_ev_figure(diag: pd.DataFrame) -> go.Figure:
    df = diag.sort_values("Date")
    values = df["explained_variance"].to_numpy() * 100.0
    lo = max(85.0, float(np.nanmin(values)) - 0.8)
    hi = min(96.0, float(np.nanmax(values)) + 0.8)

    fig = go.Figure(
        go.Scatter(
            x=df["Date"],
            y=values,
            mode="lines",
            line=dict(color=TEAL, width=1.6),
            fill="tozeroy",
            fillcolor=_rgba(TEAL, 0.06),
            hovertemplate="%{x|%Y-%m-%d}<br>Explained variance: <b>%{y:.2f}%</b><extra></extra>",
            showlegend=False,
        )
    )
    fig.add_hline(
        y=90,
        line=dict(color=AMBER, width=1, dash="dash"),
        opacity=0.55,
        layer="below",
    )
    fig.update_xaxes(tickformat="%b %Y", nticks=9, hoverformat="%Y-%m-%d")
    fig.update_yaxes(
        tickformat=".1f",
        ticksuffix="%",
        range=[lo, hi],
        title_text="Variance captured by retained PCs",
    )
    _base(fig, 288, legend=False, top=20, left=6, right=12, bottom=6)
    return fig


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) == 3:
        hex_color = "".join(c * 2 for c in hex_color)
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha:.3f})"
