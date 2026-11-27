"""Dashboard colors, labels and local CSS."""

from __future__ import annotations

# Palette
BG = "#070E1A"
BG_DEEP = "#050B14"
PANEL = "#0B192B"
PANEL_2 = "#0F2138"
LINE = "rgba(125, 160, 200, 0.16)"
LINE_STRONG = "rgba(125, 160, 200, 0.30)"
TEXT = "#E8EEF9"
MUTED = "#8FA5C2"
MUTED_DIM = "#6B819E"
ACCENT = "#38BDF8"
ACCENT_SOFT = "#7DD3FC"
TEAL = "#2DD4BF"
AMBER = "#F5B942"
RED = "#FF5D73"
RED_SOFT = "#FF8FA3"
VIOLET = "#A78BFA"
PINK = "#F472B6"
SLATE = "#9FB3C8"

FONT_STACK = '"Segoe UI Variable Text", "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif'
MONO_STACK = '"Cascadia Mono", "Consolas", "SFMono-Regular", monospace'

# Canonical domain labels (raw CSV values are never changed)
METHODS = [
    "Historical",
    "Parametric",
    "RiskMetrics_EWMA",
    "PCA_Factor",
    "MonteCarlo_Normal",
]
PORTFOLIOS = ["EqualWeight", "6040", "MinVariance"]
CONFIDENCES = [0.99, 0.95]

METHOD_LABELS = {
    "Historical": "Historical simulation",
    "Parametric": "Parametric normal",
    "RiskMetrics_EWMA": "RiskMetrics EWMA",
    "PCA_Factor": "PCA factor model",
    "MonteCarlo_Normal": "Monte Carlo (normal)",
}

METHOD_SHORT = {
    "Historical": "Historical",
    "Parametric": "Parametric",
    "RiskMetrics_EWMA": "RiskMetrics EWMA",
    "PCA_Factor": "PCA factor",
    "MonteCarlo_Normal": "Monte Carlo",
}

METHOD_COLORS = {
    "Historical": VIOLET,
    "Parametric": ACCENT,
    "RiskMetrics_EWMA": TEAL,
    "PCA_Factor": AMBER,
    "MonteCarlo_Normal": PINK,
}

PORTFOLIO_LABELS = {
    "EqualWeight": "Equal-weight (18 ETFs)",
    "6040": "60/40 (SPY 60%, AGG 40%)",
    "MinVariance": "Rolling Minimum Variance",
}

PCA_K_COLORS = {
    3: "#FB923C",
    7: AMBER,
    8: "#FDE68A",
}

# Stylesheet
CSS = f"""
<style>
:root {{
    --prl-bg: {BG};
    --prl-panel: {PANEL};
    --prl-panel-2: {PANEL_2};
    --prl-line: {LINE};
    --prl-text: {TEXT};
    --prl-muted: {MUTED};
    --prl-accent: {ACCENT};
    --prl-teal: {TEAL};
    --prl-amber: {AMBER};
    --prl-red: {RED};
}}

html, body, .stApp {{
    background: {BG};
    color: {TEXT};
    font-family: {FONT_STACK};
    -webkit-font-smoothing: antialiased;
}}

[data-testid="stAppViewContainer"] {{
    background:
        radial-gradient(1100px 520px at 12% -12%, rgba(56, 189, 248, 0.075), transparent 62%),
        radial-gradient(900px 480px at 96% -6%, rgba(45, 212, 191, 0.045), transparent 58%),
        {BG};
}}

[data-testid="stHeader"] {{
    background: transparent;
    height: 2.2rem;
}}
/* Keep the toolbar only when it carries the expand-sidebar control. Hiding   */
/* the whole toolbar leaves a collapsed sidebar with no way to reopen it.    */
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stToolbar"]:has([data-testid="stExpandSidebarButton"]) {{
    display: flex;
    align-items: center;
    background: transparent;
    height: 2.2rem;
    padding: 0;
}}
[data-testid="stToolbar"] [data-testid="stToolbarActions"],
[data-testid="stToolbar"] [data-testid="stAppDeployButton"],
[data-testid="stToolbar"] [data-testid="stStatusWidget"] {{ display: none; }}
[data-testid="stExpandSidebarButton"] {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 0.24rem;
    min-width: 2rem;
    min-height: 1.8rem;
    padding: 0 0.55rem;
    margin: 0 0 0 0.45rem;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid {LINE};
    border-radius: 8px;
    color: {TEXT};
    cursor: pointer;
    transition: background 120ms ease, border-color 120ms ease;
}}
[data-testid="stExpandSidebarButton"]::after {{
    content: "Menu";
    font-family: {FONT_STACK};
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    line-height: 1;
}}
[data-testid="stExpandSidebarButton"]:hover {{
    background: rgba(56, 189, 248, 0.18);
    border-color: rgba(56, 189, 248, 0.48);
}}
[data-testid="stExpandSidebarButton"] span {{ color: {TEXT} !important; }}
#MainMenu, footer, [data-testid="stStatusWidget"] {{ visibility: hidden; }}

.block-container {{
    max-width: 1460px;
    padding: 1.15rem 2.1rem 3.2rem 2.1rem;
}}

/* ------------------------------------------------------------------ */
/* Sidebar                                                             */
/* ------------------------------------------------------------------ */
[data-testid="stSidebar"] {{
    background: {BG_DEEP};
    border-right: 1px solid {LINE};
    min-width: 292px;
    max-width: 292px;
}}
[data-testid="stSidebar"][data-collapsed="true"] {{
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
    border-right: 0;
}}
[data-testid="stSidebar"] > div:first-child {{
    padding-top: 0.4rem;
}}
[data-testid="stSidebar"] .block-container {{
    padding: 0.9rem 1.0rem 1.2rem 1.0rem;
}}
[data-testid="stSidebar"] .stButton > button {{
    width: 100%;
    justify-content: flex-start;
    text-align: left;
    background: transparent;
    border: 1px solid transparent;
    border-radius: 9px;
    color: {MUTED};
    font-size: 0.86rem;
    font-weight: 500;
    padding: 0.42rem 0.7rem;
    min-height: 2.15rem;
    box-shadow: none;
    transition: background 120ms ease, color 120ms ease, border-color 120ms ease;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(56, 189, 248, 0.07);
    color: {TEXT};
    border-color: rgba(56, 189, 248, 0.18);
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
    background: rgba(56, 189, 248, 0.13);
    border: 1px solid rgba(56, 189, 248, 0.42);
    color: {TEXT};
    font-weight: 600;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
    background: rgba(56, 189, 248, 0.18);
    border-color: rgba(56, 189, 248, 0.55);
}}

[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p {{
    font-size: 0.66rem;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: {MUTED_DIM};
    font-weight: 650;
}}

/* Nav / section micro labels */
.prl-side-label {{
    font-size: 0.63rem;
    letter-spacing: 0.17em;
    text-transform: uppercase;
    color: {MUTED_DIM};
    font-weight: 650;
    margin: 0.35rem 0 0.45rem 0.2rem;
}}
.prl-side-rule {{
    height: 1px;
    background: {LINE};
    margin: 0.85rem 0 0.85rem 0;
}}

/* Brand */
.prl-brand {{
    display: flex;
    align-items: center;
    gap: 0.68rem;
    padding: 0.15rem 0.15rem 0.35rem 0.15rem;
}}
.prl-monogram {{
    width: 2.25rem;
    height: 2.25rem;
    border-radius: 10px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, {ACCENT} 0%, {TEAL} 100%);
    color: #05111F;
    font-weight: 800;
    font-size: 0.72rem;
    letter-spacing: 0.02em;
    box-shadow: 0 0 24px rgba(56, 189, 248, 0.22);
}}
.prl-brand-name {{
    font-size: 0.95rem;
    font-weight: 650;
    color: {TEXT};
    line-height: 1.15;
}}
.prl-brand-sub {{
    font-size: 0.67rem;
    color: {MUTED_DIM};
    letter-spacing: 0.06em;
    margin-top: 0.12rem;
}}

/* Sidebar footer */
.prl-side-foot {{
    margin-top: 0.9rem;
    padding-top: 0.8rem;
    border-top: 1px solid {LINE};
    font-size: 0.68rem;
    color: {MUTED_DIM};
    line-height: 1.55;
}}
.prl-side-foot b {{ color: {MUTED}; font-weight: 600; }}

/* ------------------------------------------------------------------ */
/* Hero / page headers                                                 */
/* ------------------------------------------------------------------ */
.prl-hero {{
    padding: 0.55rem 0 0.4rem 0;
}}
.prl-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {ACCENT_SOFT};
    font-weight: 650;
}}
.prl-eyebrow::before {{
    content: "";
    width: 1.35rem;
    height: 1px;
    background: {ACCENT};
    display: inline-block;
}}
.prl-title {{
    font-family: {FONT_STACK};
    font-size: 2.25rem;
    line-height: 1.12;
    letter-spacing: -0.028em;
    font-weight: 680;
    color: {TEXT};
    margin: 0.42rem 0 0.32rem 0;
}}
.prl-title .accent {{
    color: {ACCENT_SOFT};
    /* Keep "Value-at-Risk" together on narrow title wraps. */
    white-space: nowrap;
}}
.prl-subtitle {{
    font-size: 0.98rem;
    color: {MUTED};
    max-width: 62rem;
    line-height: 1.55;
}}
.prl-badges {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.42rem;
    margin-top: 0.85rem;
}}
.prl-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.38rem;
    font-size: 0.7rem;
    color: #B9CDE6;
    border: 1px solid {LINE};
    background: rgba(56, 189, 248, 0.045);
    border-radius: 999px;
    padding: 0.26rem 0.62rem;
    white-space: nowrap;
}}
.prl-badge b {{ color: {TEXT}; font-weight: 620; }}
.prl-badge .dot {{
    width: 0.36rem; height: 0.36rem; border-radius: 50%;
    background: {TEAL};
    box-shadow: 0 0 8px rgba(45, 212, 191, 0.7);
}}

/* Section titles */
.prl-section {{
    margin: 1.55rem 0 0.65rem 0;
}}
.prl-section-kicker {{
    font-size: 0.63rem;
    letter-spacing: 0.17em;
    text-transform: uppercase;
    color: {MUTED_DIM};
    font-weight: 650;
    margin-bottom: 0.18rem;
}}
.prl-section-title {{
    font-size: 1.12rem;
    font-weight: 640;
    color: {TEXT};
    letter-spacing: -0.012em;
    line-height: 1.3;
}}
.prl-section-note {{
    font-size: 0.8rem;
    color: {MUTED};
    margin-top: 0.22rem;
    line-height: 1.5;
}}

/* ------------------------------------------------------------------ */
/* KPI cards                                                           */
/* ------------------------------------------------------------------ */
.prl-kpi-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(168px, 1fr));
    gap: 0.6rem;
    align-items: stretch;
}}
.prl-kpi {{
    position: relative;
    background: linear-gradient(180deg, rgba(15, 33, 56, 0.92) 0%, rgba(10, 23, 41, 0.92) 100%);
    border: 1px solid {LINE};
    border-radius: 14px;
    padding: 0.85rem 0.95rem 0.8rem 0.95rem;
    min-height: 6.1rem;
    overflow: hidden;
    box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
}}
.prl-kpi::before {{
    content: "";
    position: absolute;
    left: 0; top: 0; right: 0;
    height: 2px;
    background: var(--tone, {ACCENT});
    opacity: 0.9;
}}
.prl-kpi-label {{
    font-size: 0.64rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {MUTED_DIM};
    font-weight: 650;
    line-height: 1.3;
    /* Reserve two lines so values align when a label wraps at 1280px. */
    display: flex;
    align-items: flex-end;
    min-height: 2.6em;
}}
.prl-kpi-value {{
    font-size: 1.68rem;
    font-weight: 660;
    letter-spacing: -0.03em;
    color: {TEXT};
    margin-top: 0.3rem;
    line-height: 1.08;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
}}
.prl-kpi-value small {{
    font-size: 0.82rem;
    font-weight: 500;
    color: {MUTED};
    letter-spacing: 0;
    margin-left: 0.18rem;
}}
.prl-kpi-sub {{
    font-size: 0.69rem;
    color: {MUTED};
    margin-top: 0.34rem;
    line-height: 1.4;
}}
.prl-kpi-sub b {{ color: #C9D8EA; font-weight: 600; }}
.prl-kpi-grid.compact .prl-kpi {{
    padding: 0.6rem 0.8rem 0.55rem 0.8rem;
    min-height: 5.1rem;
}}
.prl-kpi-grid.compact .prl-kpi-label {{
    min-height: 2.25em;
    font-size: 0.6rem;
}}
.prl-kpi-grid.compact .prl-kpi-value {{
    font-size: 1.5rem;
    margin-top: 0.18rem;
}}
.prl-kpi-grid.compact .prl-kpi-sub {{
    font-size: 0.66rem;
    margin-top: 0.2rem;
    line-height: 1.35;
}}

/* ------------------------------------------------------------------ */
/* Cards, notes, pills                                                 */
/* ------------------------------------------------------------------ */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {{
    background: rgba(10, 23, 41, 0.55);
    border: 1px solid {LINE};
    border-radius: 16px;
    padding: 0.2rem 0.55rem 0.35rem 0.55rem;
}}
.prl-card {{
    background: linear-gradient(180deg, rgba(15, 33, 56, 0.72) 0%, rgba(10, 23, 41, 0.72) 100%);
    border: 1px solid {LINE};
    border-radius: 15px;
    padding: 1rem 1.15rem;
}}
.prl-card-title {{
    font-size: 0.93rem;
    font-weight: 620;
    color: {TEXT};
    margin-bottom: 0.5rem;
}}
.prl-card p, .prl-card li {{
    font-size: 0.82rem;
    color: #B7C8DD;
    line-height: 1.62;
}}
.prl-card ul {{ margin: 0.2rem 0 0 1.05rem; padding: 0; }}
.prl-card li {{ margin-bottom: 0.38rem; }}
.prl-card li::marker {{ color: {ACCENT}; }}
.prl-card b {{ color: {TEXT}; font-weight: 620; }}
.prl-intro-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1.55fr) minmax(0, 1fr);
    gap: 1.6rem;
    align-items: stretch;
}}
.prl-intro-left {{
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    min-width: 0;
}}
.prl-intro-grid > .prl-card {{ height: 100%; }}
.prl-intro-dataset {{ display: flex; flex-direction: column; }}
.prl-intro-dataset .prl-facts {{ flex: 1; align-content: space-between; }}

.prl-note {{
    display: flex;
    gap: 0.6rem;
    align-items: flex-start;
    padding: 0.62rem 0.8rem;
    border-left: 2px solid {ACCENT};
    background: rgba(56, 189, 248, 0.055);
    border-radius: 0 9px 9px 0;
    color: #AEC3DB;
    font-size: 0.78rem;
    line-height: 1.55;
    margin: 0.45rem 0 0.6rem 0;
}}
.prl-note .tag {{
    font-size: 0.62rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: {ACCENT_SOFT};
    font-weight: 700;
    white-space: nowrap;
    padding-top: 0.08rem;
}}
.prl-note.warn {{ border-left-color: {AMBER}; background: rgba(245, 185, 66, 0.055); }}
.prl-note.warn .tag {{ color: {AMBER}; }}
.prl-note.danger {{ border-left-color: {RED}; background: rgba(255, 93, 115, 0.055); }}
.prl-note.danger .tag {{ color: {RED_SOFT}; }}
.prl-note.ok {{ border-left-color: {TEAL}; background: rgba(45, 212, 191, 0.05); }}
.prl-note.ok .tag {{ color: {TEAL}; }}
.prl-note code {{
    font-family: {MONO_STACK};
    font-size: 0.72rem;
    color: #C9D8EA;
    background: rgba(125, 160, 200, 0.1);
    padding: 0.05rem 0.28rem;
    border-radius: 4px;
}}

.prl-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.36rem;
    font-size: 0.71rem;
    font-weight: 600;
    padding: 0.26rem 0.6rem;
    border-radius: 999px;
    border: 1px solid {LINE};
    color: {MUTED};
    background: rgba(125, 160, 200, 0.06);
    white-space: nowrap;
}}
.prl-pill::before {{
    content: "";
    width: 0.36rem; height: 0.36rem; border-radius: 50%;
    background: {MUTED_DIM};
}}
.prl-pill.ok {{ color: #8FF0E2; border-color: rgba(45, 212, 191, 0.32); background: rgba(45, 212, 191, 0.08); }}
.prl-pill.ok::before {{ background: {TEAL}; box-shadow: 0 0 8px rgba(45, 212, 191, 0.8); }}
.prl-pill.bad {{ color: {RED_SOFT}; border-color: rgba(255, 93, 115, 0.32); background: rgba(255, 93, 115, 0.08); }}
.prl-pill.bad::before {{ background: {RED}; box-shadow: 0 0 8px rgba(255, 93, 115, 0.7); }}
.prl-pill.warn {{ color: #F7CE7A; border-color: rgba(245, 185, 66, 0.32); background: rgba(245, 185, 66, 0.08); }}
.prl-pill.warn::before {{ background: {AMBER}; }}

/* Chart heading block */
.prl-chart-head {{
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.55rem 0.35rem 0 0.35rem;
}}
.prl-chart-title {{
    font-size: 0.98rem;
    font-weight: 620;
    color: {TEXT};
    letter-spacing: -0.008em;
}}
.prl-chart-sub {{
    font-size: 0.74rem;
    color: {MUTED};
    margin-top: 0.16rem;
    line-height: 1.5;
}}

/* ------------------------------------------------------------------ */
/* Workflow strip                                                      */
/* ------------------------------------------------------------------ */
.prl-flow {{
    display: flex;
    align-items: stretch;
    gap: 0;
    flex-wrap: wrap;
    margin-top: 0.2rem;
}}
.prl-flow-step {{
    flex: 1 1 0;
    min-width: 7.4rem;
    background: rgba(15, 33, 56, 0.66);
    border: 1px solid {LINE};
    border-radius: 11px;
    padding: 0.62rem 0.7rem;
    position: relative;
}}
.prl-flow-step .num {{
    font-family: {MONO_STACK};
    font-size: 0.61rem;
    color: {ACCENT};
    letter-spacing: 0.08em;
}}
.prl-flow-step .name {{
    font-size: 0.83rem;
    color: {TEXT};
    font-weight: 600;
    margin-top: 0.16rem;
}}
.prl-flow-step .desc {{
    font-size: 0.66rem;
    color: {MUTED_DIM};
    margin-top: 0.12rem;
    line-height: 1.35;
}}
.prl-flow-arrow {{
    flex: 0 0 auto;
    align-self: center;
    color: #38536F;
    padding: 0 0.32rem;
    font-size: 0.8rem;
}}

/* Research question */
.prl-question {{
    border: 1px solid rgba(56, 189, 248, 0.24);
    background: linear-gradient(135deg, rgba(56, 189, 248, 0.085), rgba(45, 212, 191, 0.045));
    border-radius: 15px;
    padding: 1rem 1.2rem;
    margin: 0.35rem 0 0.2rem 0;
}}
.prl-question .q-label {{
    font-size: 0.63rem;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: {ACCENT_SOFT};
    font-weight: 650;
}}
.prl-question .q-text {{
    font-size: 1.18rem;
    color: {TEXT};
    font-weight: 580;
    letter-spacing: -0.012em;
    margin-top: 0.32rem;
    line-height: 1.42;
}}

/* Fact grid */
.prl-facts {{
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 0.55rem 1rem;
}}
.prl-fact .k {{
    font-size: 0.63rem;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: {MUTED_DIM};
    font-weight: 650;
}}
.prl-fact .v {{
    font-size: 0.82rem;
    color: #C7D6E8;
    margin-top: 0.08rem;
}}

/* ------------------------------------------------------------------ */
/* Widgets                                                             */
/* ------------------------------------------------------------------ */
label[data-testid="stWidgetLabel"] p {{
    font-size: 0.68rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {MUTED_DIM};
    font-weight: 650;
}}
div[data-baseweb="select"] > div {{
    background: rgba(15, 33, 56, 0.85);
    border-color: {LINE};
    border-radius: 9px;
    min-height: 2.2rem;
    font-size: 0.84rem;
}}
div[data-baseweb="select"] svg {{ color: {MUTED}; }}
div[data-baseweb="popover"] li {{ font-size: 0.84rem; }}

[data-testid="stSidebar"] div[role="radiogroup"] {{
    display: flex;
    gap: 0.4rem;
    background: rgba(15, 33, 56, 0.7);
    border: 1px solid {LINE};
    padding: 0.2rem;
    border-radius: 10px;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label {{
    flex: 1;
    justify-content: center;
    padding: 0.26rem 0.2rem;
    border-radius: 7px;
    margin: 0;
    background: transparent;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
    background: rgba(56, 189, 248, 0.16);
    box-shadow: inset 0 0 0 1px rgba(56, 189, 248, 0.35);
}}
[data-testid="stSidebar"] div[role="radiogroup"] label p {{
    font-size: 0.8rem;
    color: {TEXT};
    text-transform: none;
    letter-spacing: 0;
}}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {{ display: none; }}

div[data-testid="stSlider"] {{ padding-top: 0.15rem; }}
div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

.stButton > button {{
    border-radius: 9px;
    font-size: 0.82rem;
    font-weight: 550;
}}
.stDownloadButton > button {{
    border-radius: 9px;
    border: 1px solid {LINE_STRONG};
    background: rgba(15, 33, 56, 0.8);
    color: #C9D8EA;
    font-size: 0.78rem;
}}
.stDownloadButton > button:hover {{
    border-color: rgba(56, 189, 248, 0.5);
    color: {TEXT};
}}

[data-testid="stDataFrame"] {{
    border: 1px solid {LINE};
    border-radius: 11px;
    overflow: hidden;
}}
[data-testid="stDataFrame"] [role="columnheader"] {{
    font-size: 0.68rem !important;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}}

details[data-testid="stExpander"] {{
    border: 1px solid {LINE};
    border-radius: 12px;
    background: rgba(10, 23, 41, 0.45);
}}
details[data-testid="stExpander"] summary p {{
    font-size: 0.82rem;
    color: #C7D6E8;
    font-weight: 550;
}}

.stTabs [data-baseweb="tab-list"] {{
    gap: 0.3rem;
    border-bottom: 1px solid {LINE};
}}
.stTabs [data-baseweb="tab"] {{
    height: 2.35rem;
    padding: 0 0.95rem;
    border-radius: 9px 9px 0 0;
    color: {MUTED};
    font-size: 0.82rem;
    font-weight: 550;
}}
.stTabs [aria-selected="true"] {{
    color: {TEXT};
    background: rgba(56, 189, 248, 0.08);
}}
.stTabs [data-baseweb="tab-highlight"] {{ background: {ACCENT}; }}

[data-testid="stAlert"] {{
    border-radius: 11px;
    font-size: 0.82rem;
}}

/* Plotly container: let our transparent template show through */
[data-testid="stPlotlyChart"] {{
    background: transparent;
}}
.js-plotly-plot .plotly .modebar {{ opacity: 0.45; }}

/* Scrollbar */
::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: {BG_DEEP}; }}
::-webkit-scrollbar-thumb {{ background: #1B3350; border-radius: 6px; }}
::-webkit-scrollbar-thumb:hover {{ background: #274766; }}

/* Keep the sensitivity chart and table cards level. */
[data-testid="stHorizontalBlock"]:has(.js-plotly-plot):has([data-testid="stDataFrame"]) > [data-testid="stColumn"] {{
    display: flex;
    flex-direction: column;
}}
[data-testid="stHorizontalBlock"]:has(.js-plotly-plot):has([data-testid="stDataFrame"]) > [data-testid="stColumn"] > [data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"]:has(.js-plotly-plot):has([data-testid="stDataFrame"]) [data-testid="stLayoutWrapper"],
[data-testid="stHorizontalBlock"]:has(.js-plotly-plot):has([data-testid="stDataFrame"]) [data-testid="stLayoutWrapper"] > [data-testid="stVerticalBlock"] {{
    flex: 1 1 auto;
}}
[data-testid="stHorizontalBlock"]:has(.js-plotly-plot):has([data-testid="stDataFrame"]) [data-testid="stLayoutWrapper"] {{
    display: flex;
    flex-direction: column;
}}

@media (max-height: 760px) {{
    /* 1280x720 demo screens: give the chart the vertical space. */
    .prl-eyebrow,
    .prl-subtitle {{ display: none; }}
    .prl-hero {{ padding: 0.12rem 0 0.15rem 0; }}
    .prl-title {{ font-size: 1.7rem; margin: 0.05rem 0 0.1rem 0; }}
    .prl-badges {{ margin-top: 0.4rem; }}
    .prl-kpi-grid.compact .prl-kpi {{
        padding: 0.45rem 0.7rem 0.4rem 0.7rem;
        min-height: 4.45rem;
    }}
    .prl-kpi-grid.compact .prl-kpi-label {{ min-height: 2em; font-size: 0.57rem; }}
    .prl-kpi-grid.compact .prl-kpi-value {{ font-size: 1.4rem; margin-top: 0.08rem; }}
    .prl-kpi-grid.compact .prl-kpi-sub {{ font-size: 0.63rem; margin-top: 0.14rem; }}
}}

@media (max-width: 1100px) {{
    .block-container {{ padding-left: 1.2rem; padding-right: 1.2rem; }}
    .prl-title {{ font-size: 1.9rem; }}
    .prl-kpi-value {{ font-size: 1.45rem; }}

    /* Stack paired charts, data tables, and overview fact cards on narrow screens. */
    [data-testid="stHorizontalBlock"]:has([data-testid="stDataFrame"]),
    [data-testid="stHorizontalBlock"]:has(.js-plotly-plot),
    [data-testid="stHorizontalBlock"]:has(.prl-facts),
    [data-testid="stHorizontalBlock"]:has(.prl-card) {{
        flex-wrap: wrap;
    }}
    [data-testid="stHorizontalBlock"]:has([data-testid="stDataFrame"]) > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has(.js-plotly-plot) > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has(.prl-facts) > [data-testid="stColumn"],
    [data-testid="stHorizontalBlock"]:has(.prl-card) > [data-testid="stColumn"] {{
        flex: 1 1 100% !important;
        width: 100% !important;
        min-width: 100% !important;
    }}
    .prl-flow-arrow {{ display: none; }}
    .prl-flow-step {{ flex: 1 1 30%; }}
    .prl-intro-grid {{ grid-template-columns: 1fr; }}
    .prl-intro-grid > .prl-card {{ height: auto; }}
}}
</style>
"""
