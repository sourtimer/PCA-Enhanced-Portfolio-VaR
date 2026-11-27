"""Reusable UI primitives for the Portfolio Risk Lab dashboard."""

from __future__ import annotations

import html
import inspect
from typing import Any, Iterable, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.charts import PLOTLY_CONFIG
from app.theme import (
    ACCENT,
    AMBER,
    CSS,
    METHOD_LABELS,
    MUTED,
    MUTED_DIM,
    PORTFOLIO_LABELS,
    RED,
    TEAL,
    TEXT,
)


# Bootstrap
def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _stretch_kwargs(func: Any) -> dict[str, Any]:
    """Support both the old use_container_width and the new width API."""
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return {}
    if "width" in params:
        return {"width": "stretch"}
    if "use_container_width" in params:
        return {"use_container_width": True}
    return {}


def chart(fig: go.Figure, *, key: str | None = None) -> None:
    kwargs = _stretch_kwargs(st.plotly_chart)
    if key is not None:
        kwargs["key"] = key
    st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, **kwargs)


def dataframe(
    df: pd.DataFrame,
    *,
    column_config: dict[str, Any] | None = None,
    height: int | None = None,
    key: str | None = None,
) -> None:
    kwargs = _stretch_kwargs(st.dataframe)
    if height is not None:
        kwargs["height"] = height
    if key is not None:
        kwargs["key"] = key
    st.dataframe(
        df,
        hide_index=True,
        column_config=column_config,
        **kwargs,
    )


def download_csv(
    df: pd.DataFrame, filename: str, label: str = "Download CSV", *, key: str | None = None
) -> None:
    st.download_button(
        label=label,
        data=df.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


# Text / HTML primitives
def esc(value: Any) -> str:
    return html.escape(str(value))


def spacer(height: int = 12) -> None:
    st.markdown(f'<div style="height:{int(height)}px"></div>', unsafe_allow_html=True)


def side_label(text: str) -> None:
    st.markdown(f'<div class="prl-side-label">{esc(text)}</div>', unsafe_allow_html=True)


def side_rule() -> None:
    st.markdown('<div class="prl-side-rule"></div>', unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        """
        <div class="prl-brand">
            <div class="prl-monogram">PRL</div>
            <div>
                <div class="prl-brand-name">Portfolio Risk Lab</div>
                <div class="prl-brand-sub">PCA-ENHANCED VaR</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_header(
    eyebrow: str,
    title: str,
    subtitle: str,
    *,
    chips: Sequence[str] = (),
    accent_word: str | None = None,
) -> None:
    if accent_word and accent_word in title:
        title = title.replace(accent_word, f'<span class="accent">{accent_word}</span>', 1)
    badge_html = ""
    if chips:
        badge_html = (
            '<div class="prl-badges">'
            + "".join(f'<span class="prl-badge">{c}</span>' for c in chips)
            + "</div>"
        )
    st.markdown(
        f"""
        <div class="prl-hero">
            <div class="prl-eyebrow">{esc(eyebrow)}</div>
            <div class="prl-title">{title}</div>
            <div class="prl-subtitle">{esc(subtitle)}</div>
            {badge_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(kicker: str, title: str, note: str | None = None) -> None:
    note_html = f'<div class="prl-section-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="prl-section">
            <div class="prl-section-kicker">{esc(kicker)}</div>
            <div class="prl-section-title">{esc(title)}</div>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def chart_head(
    title: str,
    subtitle: str | None = None,
    right: str | None = None,
    right_kind: str = "ok",
) -> None:
    right_html = f'<div class="prl-pill {right_kind}">{right}</div>' if right else ""
    sub_html = f'<div class="prl-chart-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="prl-chart-head">
            <div>
                <div class="prl-chart-title">{esc(title)}</div>
                {sub_html}
            </div>
            {right_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def note(text: str, tag: str = "NOTE", kind: str = "info") -> None:
    st.markdown(
        f'<div class="prl-note {kind}"><span class="tag">{esc(tag)}</span>'
        f'<span>{text}</span></div>',
        unsafe_allow_html=True,
    )


def pill(text: str, kind: str = "") -> None:
    st.markdown(f'<span class="prl-pill {kind}">{esc(text)}</span>', unsafe_allow_html=True)


def pill_html(text: str, kind: str = "") -> str:
    return f'<span class="prl-pill {kind}">{esc(text)}</span>'


# KPI cards
def _kpi_html(label: str, value: str, sub: str, tone: str) -> str:
    # Keep this on one line: blank lines inside an HTML block make Markdown
    # treat the following cards as indented code blocks.
    return (
        f'<div class="prl-kpi" style="--tone:{tone}">'
        f'<div class="prl-kpi-label">{esc(label)}</div>'
        f'<div class="prl-kpi-value">{value}</div>'
        f'<div class="prl-kpi-sub">{sub}</div>'
        "</div>"
    )


def kpi_card(
    label: str,
    value: str,
    sub: str = "",
    *,
    tone: str = ACCENT,
) -> None:
    st.markdown(_kpi_html(label, value, sub, tone), unsafe_allow_html=True)


def kpi_row(items: Iterable[dict[str, str]], *, compact: bool = False) -> None:
    """Responsive CSS grid so five cards wrap instead of squeezing."""
    cards = "".join(
        _kpi_html(
            item.get("label", ""),
            item.get("value", ""),
            item.get("sub", ""),
            item.get("tone", ACCENT),
        )
        for item in items
    )
    modifier = " compact" if compact else ""
    st.markdown(
        f'<div class="prl-kpi-grid{modifier}">{cards}</div>',
        unsafe_allow_html=True,
    )


# Composite HTML blocks
def workflow(steps: Sequence[tuple[str, str, str]]) -> None:
    blocks: list[str] = []
    for idx, (name, desc, _icon) in enumerate(steps):
        blocks.append(
            f"""
            <div class="prl-flow-step">
                <div class="num">0{idx + 1}</div>
                <div class="name">{esc(name)}</div>
                <div class="desc">{esc(desc)}</div>
            </div>
            """
        )
        if idx < len(steps) - 1:
            blocks.append('<div class="prl-flow-arrow">→</div>')
    st.markdown(f'<div class="prl-flow">{"".join(blocks)}</div>', unsafe_allow_html=True)


def question(text: str, label: str = "Research question") -> None:
    st.markdown(
        f"""
        <div class="prl-question">
            <div class="q-label">{esc(label)}</div>
            <div class="q-text">{esc(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def intro_grid(
    question_text: str,
    sample_html: str,
    facts: Sequence[tuple[str, str]],
    label: str = "Research question",
    sample_title: str = "Sample",
    facts_title: str = "Dataset snapshot",
) -> None:
    """Question + sample on the left, dataset facts stretched to the same height."""
    st.markdown(
        f"""
        <div class="prl-intro-grid">
            <div class="prl-intro-left">
                <div class="prl-question">
                    <div class="q-label">{esc(label)}</div>
                    <div class="q-text">{esc(question_text)}</div>
                </div>
                <div class="prl-card"><div class="prl-card-title">{esc(sample_title)}</div>{sample_html}</div>
            </div>
            <div class="prl-card prl-intro-dataset">
                <div class="prl-card-title">{esc(facts_title)}</div>{facts_html(facts)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def facts_html(items: Sequence[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div class="prl-fact"><div class="k">{esc(k)}</div>'
        f'<div class="v">{esc(v)}</div></div>'
        for k, v in items
    )
    return f'<div class="prl-facts">{cells}</div>'


def facts(items: Sequence[tuple[str, str]]) -> None:
    st.markdown(facts_html(items), unsafe_allow_html=True)


def card(title: str, body_html: str) -> None:
    st.markdown(
        f'<div class="prl-card"><div class="prl-card-title">{esc(title)}</div>'
        f"{body_html}</div>",
        unsafe_allow_html=True,
    )


def context_chips(portfolio: str, confidence: float, method: str) -> None:
    st.markdown(
        f'<div class="prl-badges" style="margin-top:0.15rem">'
        f'<span class="prl-badge"><span class="dot"></span><b>{esc(PORTFOLIO_LABELS[portfolio])}</b></span>'
        f'<span class="prl-badge">Confidence <b>{confidence:.0%}</b></span>'
        f'<span class="prl-badge">Method <b>{esc(METHOD_LABELS[method])}</b></span>'
        f"</div>",
        unsafe_allow_html=True,
    )


def error_state(title: str, detail: str) -> None:
    st.error(title)
    st.code(detail, language="text")


def empty_state(message: str) -> None:
    st.markdown(
        f'<div class="prl-note warn"><span class="tag">EMPTY</span><span>{esc(message)}</span></div>',
        unsafe_allow_html=True,
    )


def method_options() -> list[str]:
    return list(METHOD_LABELS)


def tone_for_p_value(p_value: float) -> str:
    if p_value != p_value:
        return MUTED_DIM
    return RED if p_value < 0.05 else TEAL


def tone_for_rate(actual: float, expected: float) -> str:
    if actual != actual:
        return MUTED_DIM
    gap = actual - expected
    if abs(gap) <= expected * 0.25:
        return TEAL
    if gap > 0:
        return AMBER if gap <= expected else RED
    return ACCENT
