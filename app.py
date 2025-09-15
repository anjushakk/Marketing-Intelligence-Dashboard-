import os
import io
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st


# ---------------------------
# Page / Theme Configuration
# ---------------------------
st.set_page_config(
    page_title="Marketing Intelligence Dashboard",
    layout="wide",
    page_icon="📈",
    initial_sidebar_state="collapsed",
)

# Subtle, modern styling inspired by premium BI dashboards
# Force dark theme for this app (no toggle)
dark = True

# Indigo/teal dark theme inspired by the reference image
TEXT_COLOR = "#E6EAF2" if dark else "#1F2937"
HEADING_COLOR = "#FFFFFF" if dark else "#111827"
MUTED_COLOR = "#A6B0C3" if dark else "#6B7280"
CHART_BG = "#17223A" if dark else "#FFFFFF"      # panel/chart background
PANEL_COLOR = "#1D2A48" if dark else "#F8FAFC"    # card surface
GRID_COLOR = "#2A3A5F" if dark else "#E5E7EB"
APP_BG_GRADIENT = (
    "linear-gradient(180deg, #1C2C57 0%, #162443 45%, #0F1B33 100%)" if dark else "#FFFFFF"
)
ACCENT_CYAN = "#5AE3FF"
ACCENT_PURPLE = "#7C8CFF"

st.markdown(
    f"""
    <style>
    :root {{
        --primary: #2E6BFF;
        --text: {TEXT_COLOR};
        --heading: {HEADING_COLOR};
        --muted: {MUTED_COLOR};
        --bg: {CHART_BG};
        --panel: {PANEL_COLOR};
        --good: #10B981;
        --bad: #EF4444;
        --app-bg: {APP_BG_GRADIENT};
    }}
    .stApp {{ background: var(--app-bg); }}
    .block-container {{padding-top: 2rem; padding-bottom: 2rem;}}
    h1, h2, h3 {{ color: var(--heading) !important; font-weight: 900 !important; }}
    .metric-card {{ background: var(--panel); padding: 16px; border-radius: 14px; border: 1px solid rgba(148,163,184,0.25); box-shadow: 0 2px 12px rgba(0,0,0,0.25); }}
    .kpi-label {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 6px; letter-spacing: .2px; }}
    .kpi-value {{ font-size: 1.9rem; font-weight: 800; color: var(--heading); }}
    .kpi-delta {{ font-size: 0.95rem; margin-top: 4px; }}
    .kpi-delta.pos {{ color: var(--good); }}
    .kpi-delta.neg {{ color: var(--bad); }}
    .caption {{ color: var(--muted); }}
    .help-hint {{ color: var(--muted); font-size: 0.85rem; }}
    /* Make Plotly charts fully transparent within KPI containers */
    .stPlotlyChart, .stPlotlyChart > div {{ background: transparent !important; }}
    .js-plotly-plot .plotly .bg {{ fill: transparent !important; }}
    .kpi-headline {{ font-size: 1.15rem; font-weight: 800; color: var(--heading); display: flex; align-items: center; gap: 6px; margin-bottom: 6px; }}
    .info-icon {{ display:inline-flex; width:18px; height:18px; border-radius: 50%; align-items:center; justify-content:center; background: rgba(255,255,255,0.08); color: var(--muted); font-size: 12px; cursor: help; }}
    .info-icon:hover {{ color: var(--heading); }}
    /* Date input styling */
    div[data-testid="stDateInput"] {{ width: 260px !important; }}
    div[data-testid="stDateInput"] label {{ color: var(--muted) !important; }}
    div[data-testid="stDateInput"] input {{
        background-color: var(--panel) !important;
        color: var(--heading) !important;
        border: 1px solid rgba(148,163,184,0.25) !important;
        border-radius: 10px !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Constants / Config
# ---------------------------
DATA_DIR = "data"
BUSINESS_CSV = os.path.join(DATA_DIR, "business.csv")
CHANNEL_FILES = [
    os.path.join(DATA_DIR, "Facebook.csv"),
    os.path.join(DATA_DIR, "Google.csv"),
    os.path.join(DATA_DIR, "TikTok.csv"),
]

BUSINESS_REQUIRED_COLS = [
    "date",
    "# of orders",
    "# of new orders",
    "new customers",
    "total revenue",
    "gross profit",
    "COGS",
]

BUSINESS_RENAME_MAP = {
    "date": "date",
    "# of orders": "orders",
    "# of new orders": "new_orders",
    "new customers": "new_customers",
    "total revenue": "total_revenue",
    "gross profit": "gross_profit",
    "COGS": "cogs",
}

CHANNEL_REQUIRED_COLS = [
    "date",
    "tactic",
    "state",
    "campaign",
    "impression",
    "clicks",
    "spend",
    "attributed revenue",
]

CHANNEL_RENAME_MAP = {
    "date": "date",
    "tactic": "tactic",
    "state": "state",
    "campaign": "campaign",
    "impression": "impressions",  # normalize to plural
    "clicks": "clicks",
    "spend": "spend",
    "attributed revenue": "attributed_revenue",
}

CHANNEL_COLORS = {
    "Facebook": "#1877F2",
    "Google": "#34A853",
    "TikTok": "#0F0F0F",
}

# Consistent palette for tactics across charts
TACTIC_PALETTE = (
    px.colors.qualitative.Set2
    + px.colors.qualitative.Set3
    + px.colors.qualitative.Pastel
)

# Plotly defaults for consistent style
px.defaults.template = "plotly_dark" if dark else "simple_white"
px.defaults.color_discrete_sequence = [ACCENT_CYAN, ACCENT_PURPLE, "#10B981", "#F59E0B", "#EF4444", "#06B6D4"]


def apply_plot_style(fig: go.Figure, height: int | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_COLOR),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    # Ensure no stray 'undefined' title appears
    try:
        title_text = fig.layout.title.text if fig.layout.title else None
        if title_text is None:
            fig.update_layout(title_text="")
    except Exception:
        pass
    if height:
        fig.update_layout(height=height)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


# ---------------------------
# Utilities
# ---------------------------
def safe_divide(numerator: float, denominator: float) -> float:
    if denominator is None or denominator == 0 or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def moving_avg(series: pd.Series, window: int = 7) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()


def z_scores(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) != 0 else 1.0)


def wow_change(current: float, previous: float) -> float:
    if previous in [0, None] or pd.isna(previous):
        return np.nan
    return (current - previous) / previous


def _to_date(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, errors="coerce")
    return dt.dt.date


def _coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(",", "", regex=False)
                .str.replace("$", "", regex=False)
                .str.replace("€", "", regex=False)
                .str.replace("£", "", regex=False)
                .str.replace("%", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _validate_columns(df: pd.DataFrame, required: List[str], file_label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            f"{file_label} is missing required columns: {', '.join(missing)}.\n"
            f"Exact headers required. Please fix the CSV headers."
        )
        st.stop()


# ---------------------------
# Data Loading (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def load_business() -> pd.DataFrame:
    if not os.path.exists(BUSINESS_CSV):
        st.error(f"Missing file: {BUSINESS_CSV}")
        st.stop()
    df = pd.read_csv(BUSINESS_CSV)
    _validate_columns(df, BUSINESS_REQUIRED_COLS, "business.csv")

    # Normalize
    df = df.rename(columns=BUSINESS_RENAME_MAP)
    df["date"] = _to_date(df["date"])  # parse as date only
    df = _coerce_numeric(
        df,
        [
            "orders",
            "new_orders",
            "new_customers",
            "total_revenue",
            "gross_profit",
            "cogs",
        ],
    )
    return df


@st.cache_data(show_spinner=False)
def load_channel_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"Missing file: {path}")
        st.stop()
    df = pd.read_csv(path)
    _validate_columns(df, CHANNEL_REQUIRED_COLS, os.path.basename(path))
    df = df.rename(columns=CHANNEL_RENAME_MAP)
    df["date"] = _to_date(df["date"])  # parse as date only

    channel_name = os.path.splitext(os.path.basename(path))[0]
    df["channel"] = channel_name

    df = _coerce_numeric(df, ["impressions", "clicks", "spend", "attributed_revenue"])
    return df


@st.cache_data(show_spinner=False)
def load_channels() -> pd.DataFrame:
    frames = [load_channel_file(p) for p in CHANNEL_FILES]
    df = pd.concat(frames, ignore_index=True)
    return df


@st.cache_data(show_spinner=False)
def data_refresh_info() -> Dict[str, str]:
    # Return last modified times for each CSV
    info = {}
    for path in [BUSINESS_CSV] + CHANNEL_FILES:
        if os.path.exists(path):
            ts = datetime.fromtimestamp(os.path.getmtime(path))
            info[os.path.basename(path)] = ts.strftime("%Y-%m-%d %H:%M:%S")
        else:
            info[os.path.basename(path)] = "missing"
    info["loaded_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return info


# ---------------------------
# Aggregations / Derived (cached)
# ---------------------------
@st.cache_data(show_spinner=False)
def build_daily_joins(business: pd.DataFrame, channels: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Daily business
    biz_daily = (
        business.groupby("date", as_index=False)
        .agg(
            orders=("orders", "sum"),
            new_orders=("new_orders", "sum"),
            new_customers=("new_customers", "sum"),
            total_revenue=("total_revenue", "sum"),
            gross_profit=("gross_profit", "sum"),
            cogs=("cogs", "sum"),
        )
        .sort_values("date")
    )

    # Daily channels overall
    ch_daily = (
        channels.groupby("date", as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            spend=("spend", "sum"),
            attributed_revenue=("attributed_revenue", "sum"),
        )
        .sort_values("date")
    )

    # Daily by channel (for small multiples)
    ch_daily_by_channel = (
        channels.groupby(["date", "channel"], as_index=False)
        .agg(
            impressions=("impressions", "sum"),
            clicks=("clicks", "sum"),
            spend=("spend", "sum"),
            attributed_revenue=("attributed_revenue", "sum"),
        )
        .sort_values(["date", "channel"])
    )

    # Join
    daily = pd.merge(ch_daily, biz_daily, on="date", how="outer").sort_values("date")

    # Derived daily blended metrics
    daily["ctr"] = daily.apply(lambda r: safe_divide(r.get("clicks", np.nan), r.get("impressions", np.nan)), axis=1)
    daily["cpc"] = daily.apply(lambda r: safe_divide(r.get("spend", np.nan), r.get("clicks", np.nan)), axis=1)
    daily["cpm"] = daily.apply(lambda r: safe_divide(1000 * r.get("spend", np.nan), r.get("impressions", np.nan)), axis=1)
    daily["roas"] = daily.apply(
        lambda r: safe_divide(r.get("attributed_revenue", np.nan), r.get("spend", np.nan)), axis=1
    )
    daily["aov"] = daily.apply(lambda r: safe_divide(r.get("total_revenue", np.nan), r.get("orders", np.nan)), axis=1)
    daily["gross_margin"] = daily.apply(
        lambda r: safe_divide(r.get("gross_profit", np.nan), r.get("total_revenue", np.nan)), axis=1
    )

    return daily, biz_daily, ch_daily_by_channel


# ---------------------------
# Filters
# ---------------------------
def _ensure_all_is_exclusive(key: str, all_label: str = "All") -> None:
    """Keep 'All' mutually exclusive with specific selections based on intent.

    Logic:
    - If 'All' was previously selected and user selects another option → drop 'All'.
    - If 'All' was not selected and user selects 'All' → keep only 'All'.
    """
    try:
        sel = st.session_state.get(key, [])
        prev = st.session_state.get(f"{key}__prev", [])
        if not isinstance(sel, list):
            return

        has_all_now = all_label in sel
        has_all_prev = isinstance(prev, list) and (all_label in prev)

        # Only need to resolve when 'All' appears with others
        if has_all_now and len(sel) > 1:
            if has_all_prev:
                # Case: previously 'All' (likely just 'All'), user added specifics → remove 'All'
                st.session_state[key] = [v for v in sel if v != all_label]
            else:
                # Case: previously specifics, user added 'All' → keep only 'All'
                st.session_state[key] = [all_label]

        # Sync prev to the resolved selection
        st.session_state[f"{key}__prev"] = st.session_state.get(key, sel)
    except Exception:
        # Fail-safe: do not block UI if session state isn't set as expected
        pass


def build_filters(channels: pd.DataFrame, business: pd.DataFrame):
    # Date bounds from union of both
    min_date = min(pd.to_datetime(channels["date"]).min(), pd.to_datetime(business["date"]).min()).date()
    max_date = max(pd.to_datetime(channels["date"]).max(), pd.to_datetime(business["date"]).max()).date()

    # Top-of-page filter bar (no sidebar)
    with st.container():
        left, right = st.columns([1, 4])
        with left:
            st.markdown("<div style='font-weight:600; font-size:14px;'>Date range</div>", unsafe_allow_html=True)
        with right:
            date_range = st.date_input(
                "",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date,
                label_visibility="collapsed",
            )
    # Guard against partial selection (streamlit may provide a single date while picking)
    if isinstance(date_range, (list, tuple)) and len(date_range) == 2 and all(date_range):
        start_date, end_date = date_range
        if start_date > end_date:
            start_date, end_date = end_date, start_date
    else:
        st.info("Select a start and end date to update the dashboard.")
        st.stop()

    return (start_date, end_date)


def apply_filters(
    business: pd.DataFrame,
    channels: pd.DataFrame,
    date_range: Tuple[datetime, datetime],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_date, end_date = date_range
    # Filter channels
    ch = channels.copy()
    ch = ch[(pd.to_datetime(ch["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(ch["date"]) <= pd.to_datetime(end_date))]

    # Filter business only by date (other dims do not apply)
    biz = business.copy()
    biz = biz[(pd.to_datetime(biz["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(biz["date"]) <= pd.to_datetime(end_date))]

    return biz, ch


# ---------------------------
# KPI Helpers
# ---------------------------
def _humanize_number(x: float, decimals: int = 1) -> str:
    try:
        n = float(x)
    except Exception:
        return "–"
    if pd.isna(n):
        return "–"
    abs_n = abs(n)
    if abs_n >= 1_000_000_000:
        return f"{n/1_000_000_000:.{decimals}f}B"
    if abs_n >= 1_000_000:
        return f"{n/1_000_000:.{decimals}f}M"
    if abs_n >= 1_000:
        return f"{n/1_000:.{decimals}f}k"
    return f"{n:,.0f}"


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return "–"
    return "$" + _humanize_number(x, decimals=2)


def fmt_int(x: float) -> str:
    if pd.isna(x):
        return "–"
    # Show compact form for large counts
    return _humanize_number(x, decimals=1)


def fmt_pct(x: float, digits: int = 1) -> str:
    if pd.isna(x):
        return "–"
    return f"{x*100:.{digits}f}%"


def _period_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())


def compute_kpis(biz: pd.DataFrame, ch: pd.DataFrame, date_range: Tuple[datetime, datetime]):
    # Current period totals
    total_revenue = _period_sum(biz, "total_revenue")
    gross_profit = _period_sum(biz, "gross_profit")
    orders = _period_sum(biz, "orders")
    new_customers = _period_sum(biz, "new_customers")

    spend = _period_sum(ch, "spend")
    attributed_revenue = _period_sum(ch, "attributed_revenue")
    clicks_total = _period_sum(ch, "clicks")
    impressions_total = _period_sum(ch, "impressions")

    blended_roas = safe_divide(attributed_revenue, spend)
    blended_cac = safe_divide(spend, new_customers)
    aov = safe_divide(total_revenue, orders)
    gross_margin = safe_divide(gross_profit, total_revenue)

    # WoW: compute based on last 14 days relative to end_date, applying same non-date filters (already applied)
    start_date, end_date = date_range
    anchor = pd.to_datetime(end_date)
    curr_start = (anchor - timedelta(days=6)).date()  # last 7 days inclusive
    prev_start = (anchor - timedelta(days=13)).date()
    prev_end = (anchor - timedelta(days=7)).date()

    biz_curr = biz[(pd.to_datetime(biz["date"]) >= pd.to_datetime(curr_start))]
    biz_curr = biz_curr[(pd.to_datetime(biz_curr["date"]) <= pd.to_datetime(end_date))]
    biz_prev = biz[(pd.to_datetime(biz["date"]) >= pd.to_datetime(prev_start))]
    biz_prev = biz_prev[(pd.to_datetime(biz_prev["date"]) <= pd.to_datetime(prev_end))]

    ch_curr = ch[(pd.to_datetime(ch["date"]) >= pd.to_datetime(curr_start))]
    ch_curr = ch_curr[(pd.to_datetime(ch_curr["date"]) <= pd.to_datetime(end_date))]
    ch_prev = ch[(pd.to_datetime(ch["date"]) >= pd.to_datetime(prev_start))]
    ch_prev = ch_prev[(pd.to_datetime(ch_prev["date"]) <= pd.to_datetime(prev_end))]

    def _wow_pair(curr_df, prev_df, col):
        return _period_sum(curr_df, col), _period_sum(prev_df, col)

    tr_c, tr_p = _wow_pair(biz_curr, biz_prev, "total_revenue")
    gp_c, gp_p = _wow_pair(biz_curr, biz_prev, "gross_profit")
    ord_c, ord_p = _wow_pair(biz_curr, biz_prev, "orders")
    nc_c, nc_p = _wow_pair(biz_curr, biz_prev, "new_customers")
    sp_c, sp_p = _wow_pair(ch_curr, ch_prev, "spend")
    ar_c, ar_p = _wow_pair(ch_curr, ch_prev, "attributed_revenue")
    clk_c, clk_p = _wow_pair(ch_curr, ch_prev, "clicks")
    imp_c, imp_p = _wow_pair(ch_curr, ch_prev, "impressions")
    roas_c = safe_divide(ar_c, sp_c)
    roas_p = safe_divide(ar_p, sp_p)
    cac_c = safe_divide(sp_c, nc_c)
    cac_p = safe_divide(sp_p, nc_p)
    aov_c = safe_divide(tr_c, ord_c)
    aov_p = safe_divide(tr_p, ord_p)
    gm_c = safe_divide(gp_c, tr_c)
    gm_p = safe_divide(gp_p, tr_p)
    ctr_c = safe_divide(clk_c, imp_c)
    ctr_p = safe_divide(clk_p, imp_p)

    kpis = {
        "Total Revenue": {
            "value": total_revenue,
            "delta": wow_change(tr_c, tr_p),
            "format": fmt_money,
            "help": "Sum of total revenue (business.csv) over current filters.",
        },
        "Attributed Revenue": {
            "value": attributed_revenue,
            "delta": wow_change(ar_c, ar_p),
            "format": fmt_money,
            "help": "Revenue attributed to ads across all platforms.",
        },
        "Gross Profit": {
            "value": gross_profit,
            "delta": wow_change(gp_c, gp_p),
            "format": fmt_money,
            "help": "Sum of gross profit (business.csv).",
        },
        "Orders": {
            "value": orders,
            "delta": wow_change(ord_c, ord_p),
            "format": fmt_int,
            "help": "Number of orders.",
        },
        "New Customers": {
            "value": new_customers,
            "delta": wow_change(nc_c, nc_p),
            "format": fmt_int,
            "help": "Unique new customers acquired.",
        },
        "Total Spend": {
            "value": spend,
            "delta": wow_change(sp_c, sp_p),
            "format": fmt_money,
            "help": "Sum of ad spend across channels.",
        },
        # ROAS and Gross Margin are shown as gauges; omit their cards
        "Blended CAC": {
            "value": blended_cac,
            "delta": wow_change(cac_c, cac_p),
            "format": fmt_money,
            "help": "Spend / new customers.",
        },
        "AOV": {
            "value": aov,
            "delta": wow_change(aov_c, aov_p),
            "format": fmt_money,
            "help": "Average order value: total revenue / orders.",
        },
        "Clicks": {
            "value": clicks_total,
            "delta": wow_change(clk_c, clk_p),
            "format": fmt_int,
            "help": "Total ad clicks across platforms.",
        },
        # Keep values available for gauges, but not displayed as KPI cards
        "Blended ROAS": {"value": blended_roas, "delta": wow_change(roas_c, roas_p), "format": lambda x: (f"{x:.2f}x" if not pd.isna(x) else "–"), "help": "(Gauge) ROAS."},
        "Gross Margin": {"value": gross_margin, "delta": wow_change(gm_c, gm_p), "format": lambda x: fmt_pct(x, 1), "help": "(Gauge) Gross margin."},
    }

    return kpis


def kpi_sparklines(daily: pd.DataFrame, start_date, end_date):
    # Create small time-series for key KPIs within selected range
    dd = daily[(pd.to_datetime(daily["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(daily["date"]) <= pd.to_datetime(end_date))]
    figs = {}

    # Combined Revenue vs Spend (dual y-axes)
    fig_rs = make_subplots(specs=[[{"secondary_y": True}]])
    fig_rs.add_trace(
        go.Scatter(
            x=dd["date"],
            y=dd["total_revenue"],
            mode="lines",
            name="Total Revenue",
            line=dict(color=ACCENT_CYAN, width=2),
        ),
        secondary_y=False,
    )
    fig_rs.add_trace(
        go.Scatter(
            x=dd["date"],
            y=dd["spend"],
            mode="lines",
            name="Total Spend",
            line=dict(color=ACCENT_PURPLE, width=2),
        ),
        secondary_y=True,
    )
    fig_rs.update_layout(
        margin=dict(l=40, r=10, t=10, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
    )
    fig_rs.update_yaxes(title_text="Total Revenue", tickformat=",.0f", tickprefix="$", showgrid=True, secondary_y=False)
    fig_rs.update_yaxes(title_text="Total Spend", tickformat=",.0f", tickprefix="$", showgrid=True, secondary_y=True)
    figs["Revenue vs Spend"] = fig_rs

    # ROAS
    fig_roas = px.area(dd, x="date", y="roas")
    fig_roas.update_traces(line_color="#059669", fillcolor="rgba(5,150,105,0.15)")
    fig_roas.update_layout(
        margin=dict(l=40, r=10, t=10, b=40), showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis_title="Date",
        yaxis_title="Blended ROAS",
    )
    fig_roas.update_yaxes(tickformat=",.2f", ticksuffix="x", showgrid=True)
    figs["Blended ROAS"] = fig_roas

    return figs


# ---------------------------
# Charts
# ---------------------------
def chart_trends(daily: pd.DataFrame, start_date, end_date, show_anomalies: bool = True) -> go.Figure:
    dd = daily[(pd.to_datetime(daily["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(daily["date"]) <= pd.to_datetime(end_date))]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=dd["date"], y=dd["total_revenue"], mode="lines", name="Total Revenue", line=dict(color=ACCENT_CYAN, width=3)
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=dd["date"], y=dd["spend"], mode="lines", name="Spend", line=dict(color=ACCENT_PURPLE, width=2)
        ),
        secondary_y=True,
    )

    # Anomalies via z-score
    if show_anomalies and len(dd) > 3:
        for col, color in [("total_revenue", "#EF4444"), ("spend", "#F59E0B"), ("attributed_revenue", "#14B8A6")]:
            z = z_scores(dd[col].fillna(0))
            mask = z.abs() >= 2
            if mask.any():
                fig.add_trace(
                    go.Scatter(
                        x=dd.loc[mask, "date"],
                        y=dd.loc[mask, col],
                        mode="markers",
                        name=f"Anomaly: {col}",
                        marker=dict(color=color, size=8, symbol="x"),
                        hovertemplate=f"%{{x}}<br>{col}: %{{y:,.0f}}<extra></extra>",
                    ),
                    secondary_y=(col == "spend"),
                )

    fig.update_layout(
        title={
            "x": 0.0,
            "xanchor": "left",
            "font": {"size": 18, "color": HEADING_COLOR},
        },
        margin=dict(l=10, r=10, t=10, b=0),
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="Revenue", showgrid=True, gridcolor=GRID_COLOR, secondary_y=False)
    fig.update_yaxes(title_text="Spend", showgrid=True, gridcolor=GRID_COLOR, secondary_y=True)
    fig = apply_plot_style(fig)
    return fig


def chart_blended_roas(daily: pd.DataFrame, start_date, end_date) -> go.Figure:
    dd = daily[(pd.to_datetime(daily["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(daily["date"]) <= pd.to_datetime(end_date))]
    dd = dd.copy()
    dd["roas_ma7"] = moving_avg(dd["roas"], 7)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dd["date"], y=dd["roas"], mode="lines", name="ROAS", line=dict(color=ACCENT_CYAN, width=3)))
    fig.add_trace(
        go.Scatter(
            x=dd["date"], y=dd["roas_ma7"], mode="lines", name="ROAS MA(7)", line=dict(color="#7DD3FC", width=2, dash="dot")
        )
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=0),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_tickformat=",.2%",
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig = apply_plot_style(fig)
    return fig


def chart_platform_roas(ch: pd.DataFrame) -> go.Figure:
    """Compare ROAS time series across platforms (channels).

    Expects channel-level rows with columns: date, channel, spend, attributed_revenue.
    """
    if ch.empty:
        return apply_plot_style(go.Figure())

    df = (
        ch.groupby(["date", "channel"], as_index=False)
        .agg(spend=("spend", "sum"), revenue=("attributed_revenue", "sum"))
    )
    df["roas"] = df.apply(lambda r: safe_divide(r["revenue"], r["spend"]), axis=1)

    # Keep only channels present; map colors where available
    color_map = {k: v for k, v in CHANNEL_COLORS.items() if k in df["channel"].unique()}

    fig = px.line(
        df,
        x="date",
        y="roas",
        color="channel",
        color_discrete_map=color_map,
        labels={"roas": "ROAS", "date": "Date", "channel": "Platform"},
    )
    fig.update_traces(mode="lines", line=dict(width=2))
    fig.update_yaxes(tickformat=",.2f", ticksuffix="x")
    fig.update_layout(
        title_text="Daily ROAS by Platform",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=300,
        margin=dict(l=10, r=10, t=10, b=0),
    )
    return apply_plot_style(fig)


def chart_platform_roas_agg(ch: pd.DataFrame) -> go.Figure:
    """Aggregate ROAS per platform across the filtered range."""
    if ch.empty:
        return apply_plot_style(go.Figure())
    df = ch.groupby("channel", as_index=False).agg(
        spend=("spend", "sum"), revenue=("attributed_revenue", "sum")
    )
    df["roas"] = df.apply(lambda r: safe_divide(r["revenue"], r["spend"]), axis=1)
    df = df.sort_values("roas", ascending=False)
    color_map = {k: v for k, v in CHANNEL_COLORS.items() if k in df["channel"].unique()}
    fig = px.bar(
        df,
        x="channel",
        y="roas",
        color="channel",
        color_discrete_map=color_map,
        labels={"channel": "Platform", "roas": "ROAS"},
    )
    fig.update_traces(texttemplate="%{y:.2f}x", textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=",.2f", ticksuffix="x")
    # Apply base styling first, then set a larger top margin and explicit title placement
    fig = apply_plot_style(fig)
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=10, r=10, t=48, b=0),
        title=dict(
            text="Aggregated ROAS",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=16, color=HEADING_COLOR),
        ),
    )
    return fig


def chart_platform_revenue_timeseries(ch: pd.DataFrame) -> go.Figure:
    """Daily attributed revenue by platform (stacked area)."""
    if ch.empty:
        return apply_plot_style(go.Figure())
    df = ch.groupby(["date", "channel"], as_index=False).agg(revenue=("attributed_revenue", "sum"))
    color_map = {k: v for k, v in CHANNEL_COLORS.items() if k in df["channel"].unique()}
    fig = px.area(
        df,
        x="date",
        y="revenue",
        color="channel",
        color_discrete_map=color_map,
        labels={"revenue": "Attributed Revenue", "channel": "Platform", "date": "Date"},
        title="Attributed Revenue by Platform (Daily)",
    )
    fig.update_yaxes(tickformat=",.0f", tickprefix="$")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=300, margin=dict(l=10, r=10, t=10, b=0))
    return apply_plot_style(fig)


def chart_platform_revenue_agg(ch: pd.DataFrame) -> go.Figure:
    """Aggregated attributed revenue by platform for the current filters."""
    if ch.empty:
        return apply_plot_style(go.Figure())
    df = ch.groupby("channel", as_index=False).agg(revenue=("attributed_revenue", "sum"))
    df = df.sort_values("revenue", ascending=False)
    color_map = {k: v for k, v in CHANNEL_COLORS.items() if k in df["channel"].unique()}
    fig = px.bar(
        df,
        x="channel",
        y="revenue",
        color="channel",
        color_discrete_map=color_map,
        labels={"channel": "Platform", "revenue": "Attributed Revenue"},
    )
    fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=",.0f", tickprefix="$")
    fig = apply_plot_style(fig)
    fig.update_layout(
        showlegend=False,
        height=300,
        margin=dict(l=10, r=10, t=48, b=0),
        title=dict(
            text="Aggregated Attributed Revenue",
            x=0.0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=16, color=HEADING_COLOR),
        ),
    )
    return fig


def chart_rev_spend_agg(daily: pd.DataFrame) -> go.Figure:
    """Aggregated totals for revenue vs spend within the selected period."""
    if daily.empty:
        return apply_plot_style(go.Figure())
    total_rev = pd.to_numeric(daily.get("total_revenue"), errors="coerce").fillna(0).sum()
    total_sp = pd.to_numeric(daily.get("spend"), errors="coerce").fillna(0).sum()
    df = pd.DataFrame({
        "metric": ["Total Revenue", "Total Spend"],
        "value": [total_rev, total_sp],
    })
    fig = px.bar(
        df,
        x="metric",
        y="value",
        color="metric",
        color_discrete_map={"Total Revenue": ACCENT_CYAN, "Total Spend": ACCENT_PURPLE},
        labels={"metric": "", "value": "Amount"},
        title="Aggregated Revenue vs Spend",
    )
    fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=",.0f", tickprefix="$")
    fig = apply_plot_style(fig)
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=48, b=0), title=dict(text="Aggregated Revenue vs Spend", x=0.0, xanchor="left", y=0.98, yanchor="top", font=dict(size=16, color=HEADING_COLOR)))
    return fig


# ---------------------------
# Tactics Analysis helpers
# ---------------------------
def chart_tactic_roas_timeseries(ch: pd.DataFrame) -> go.Figure:
    """ROAS by tactic over time."""
    if ch.empty or "tactic" not in ch.columns:
        return apply_plot_style(go.Figure())
    df = (
        ch.groupby(["date", "tactic"], as_index=False)
        .agg(spend=("spend", "sum"), revenue=("attributed_revenue", "sum"))
    )
    df["roas"] = df.apply(lambda r: safe_divide(r["revenue"], r["spend"]), axis=1)
    # Build a stable color map for tactics present
    tactics = df["tactic"].dropna().unique().tolist()
    color_map = {t: TACTIC_PALETTE[i % len(TACTIC_PALETTE)] for i, t in enumerate(sorted(tactics))}
    fig = px.line(
        df,
        x="date",
        y="roas",
        color="tactic",
        color_discrete_map=color_map,
        labels={"roas": "ROAS", "tactic": "Tactic", "date": "Date"},
        title="ROAS by Tactic (Daily)",
    )
    fig.update_traces(line=dict(width=2))
    fig.update_yaxes(tickformat=",.2f", ticksuffix="x")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=300, margin=dict(l=10, r=10, t=10, b=0))
    return apply_plot_style(fig)


def chart_tactic_roas_agg(ch: pd.DataFrame) -> go.Figure:
    """Aggregated ROAS by tactic for current filters."""
    if ch.empty or "tactic" not in ch.columns:
        return apply_plot_style(go.Figure())
    df = ch.groupby("tactic", as_index=False).agg(spend=("spend", "sum"), revenue=("attributed_revenue", "sum"))
    df["roas"] = df.apply(lambda r: safe_divide(r["revenue"], r["spend"]), axis=1)
    df = df.sort_values("roas", ascending=False)
    tactics = df["tactic"].dropna().unique().tolist()
    color_map = {t: TACTIC_PALETTE[i % len(TACTIC_PALETTE)] for i, t in enumerate(sorted(tactics))}
    fig = px.bar(
        df,
        x="tactic",
        y="roas",
        color="tactic",
        color_discrete_map=color_map,
        labels={"tactic": "Tactic", "roas": "ROAS"},
        title="Aggregated ROAS by Tactic",
    )
    fig.update_traces(texttemplate="%{y:.2f}x", textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=",.2f", ticksuffix="x")
    fig = apply_plot_style(fig)
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=48, b=0), title=dict(text="Aggregated ROAS by Tactic", x=0.0, xanchor="left", y=0.98, yanchor="top", font=dict(size=16, color=HEADING_COLOR)))
    return fig

def chart_platform_tactic_sankey(ch: pd.DataFrame) -> go.Figure:
    """Sankey mapping of Platforms (channels) to Tactics weighted by spend (or count)."""
    if ch.empty or not {"channel", "tactic"}.issubset(ch.columns):
        return apply_plot_style(go.Figure())

    df = ch.copy()
    has_spend = "spend" in df.columns
    df = df[["channel", "tactic", "spend"]] if has_spend else df[["channel", "tactic"]]
    df = df.dropna(subset=["channel", "tactic"]).copy()

    if has_spend:
        links = df.groupby(["channel", "tactic"], as_index=False).agg(value=("spend", "sum"))
        metric_name = "Spend"
    else:
        links = df.groupby(["channel", "tactic"], as_index=False).size().rename(columns={"size": "value"})
        metric_name = "Count"

    channels = sorted(links["channel"].unique().tolist())
    tactics = sorted(links["tactic"].unique().tolist())
    labels = channels + tactics
    idx = {name: i for i, name in enumerate(labels)}

    sources = [idx[row.channel] for row in links.itertuples(index=False)]
    targets = [idx[row.tactic] for row in links.itertuples(index=False)]
    values = links["value"].astype(float).tolist()

    ch_color = "#2E6BFF"
    tact_color = "#10B981"
    node_colors = [ch_color] * len(channels) + [tact_color] * len(tactics)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=12,
                    thickness=16,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    label=labels,
                    color=node_colors,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color="rgba(46,107,255,0.28)",
                    label=[f"{row.channel} → {row.tactic}" for row in links.itertuples(index=False)],
                    hovertemplate=(
                        "%{source.label} → %{target.label}<br>" +
                        f"Metric: {metric_name}<br>" +
                        ("$%{value:,.0f}" if has_spend else "%{value}") +
                        "<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
    # Add a small caption explaining link width metric
    fig.add_annotation(
        text=("Link width = total spend" if has_spend else "Link width = count of rows"),
        x=0.0,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color=HEADING_COLOR),
        align="left",
    )
    return apply_plot_style(fig)

def _platform_metric_timeseries(ch: pd.DataFrame, metric: str, title: str, yfmt: str = None, yprefix: str | None = None, ysuffix: str | None = None) -> go.Figure:
    if ch.empty:
        return apply_plot_style(go.Figure())

    g = ch.groupby(["date", "channel"], as_index=False).agg(
        impressions=("impressions", "sum"), clicks=("clicks", "sum"), spend=("spend", "sum")
    )
    if metric == "ctr":
        g["value"] = g.apply(lambda r: safe_divide(r["clicks"], r["impressions"]), axis=1)
        y_label = "CTR"
    elif metric == "cpc":
        g["value"] = g.apply(lambda r: safe_divide(r["spend"], r["clicks"]), axis=1)
        y_label = "CPC"
    elif metric == "cpm":
        g["value"] = g.apply(lambda r: safe_divide(1000 * r["spend"], r["impressions"]), axis=1)
        y_label = "CPM"
    else:
        raise ValueError("Unsupported metric for platform timeseries")

    color_map = {k: v for k, v in CHANNEL_COLORS.items() if k in g["channel"].unique()}
    fig = px.line(g, x="date", y="value", color="channel", color_discrete_map=color_map, labels={"value": y_label, "channel": "Platform"}, title=title)
    fig.update_traces(mode="lines", line=dict(width=2))
    if yfmt is not None:
        fig.update_yaxes(tickformat=yfmt)
    if yprefix is not None:
        fig.update_yaxes(tickprefix=yprefix)
    if ysuffix is not None:
        fig.update_yaxes(ticksuffix=ysuffix)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=300, margin=dict(l=10, r=10, t=10, b=0))
    return apply_plot_style(fig)


def _platform_metric_agg(ch: pd.DataFrame, metric: str, title: str, yfmt: str = None, yprefix: str | None = None, ysuffix: str | None = None, higher_is_better: bool = True) -> go.Figure:
    if ch.empty:
        return apply_plot_style(go.Figure())

    g = ch.groupby("channel", as_index=False).agg(impressions=("impressions", "sum"), clicks=("clicks", "sum"), spend=("spend", "sum"))
    if metric == "ctr":
        g["value"] = g.apply(lambda r: safe_divide(r["clicks"], r["impressions"]), axis=1)
        y_label = "CTR"
        sort_ascending = not higher_is_better
    elif metric == "cpc":
        g["value"] = g.apply(lambda r: safe_divide(r["spend"], r["clicks"]), axis=1)
        y_label = "CPC"
        sort_ascending = higher_is_better is False  # lower is better
    elif metric == "cpm":
        g["value"] = g.apply(lambda r: safe_divide(1000 * r["spend"], r["impressions"]), axis=1)
        y_label = "CPM"
        sort_ascending = True  # lower is better by default
    else:
        raise ValueError("Unsupported metric for platform aggregate")

    # Order bars by value (descending if higher is better)
    g = g.sort_values("value", ascending=sort_ascending)
    color_map = {k: v for k, v in CHANNEL_COLORS.items() if k in g["channel"].unique()}
    fig = px.bar(g, x="channel", y="value", color="channel", color_discrete_map=color_map, labels={"channel": "Platform", "value": y_label}, title=title)
    if yfmt is not None:
        fig.update_yaxes(tickformat=yfmt)
    if yprefix is not None:
        fig.update_yaxes(tickprefix=yprefix)
    if ysuffix is not None:
        fig.update_yaxes(ticksuffix=ysuffix)
    # Metric-specific value labels
    if metric == "ctr":
        fig.update_traces(texttemplate="%{y:.1%}")
    elif metric in ("cpc", "cpm"):
        fig.update_traces(texttemplate="$%{y:,.2f}")
    else:
        fig.update_traces(texttemplate="%{y:,.2f}")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig = apply_plot_style(fig)
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=48, b=0), title=dict(text=title, x=0.0, xanchor="left", y=0.98, yanchor="top", font=dict(size=16, color=HEADING_COLOR)))
    return fig


def chart_platform_kpi_radar(ch: pd.DataFrame) -> go.Figure:
    """Spider chart where each KPI axis is normalized against the best platform.

    Higher-is-better KPIs (Revenue, ROAS, CTR): score = value / max(value).
    Lower-is-better KPIs (CPC, CPM): score = min(value) / value.
    """
    if ch.empty:
        return apply_plot_style(go.Figure())

    g = ch.groupby("channel", as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        spend=("spend", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
    )
    if g.empty:
        return apply_plot_style(go.Figure())

    # Compute KPIs per platform
    g["revenue"] = pd.to_numeric(g["attributed_revenue"], errors="coerce").fillna(0.0)
    g["roas"] = g.apply(lambda r: safe_divide(r["revenue"], r["spend"]), axis=1)
    g["ctr"] = g.apply(lambda r: safe_divide(r["clicks"], r["impressions"]), axis=1)
    g["cpc"] = g.apply(lambda r: safe_divide(r["spend"], r["clicks"]), axis=1)
    g["cpm"] = g.apply(lambda r: safe_divide(1000.0 * r["spend"], r["impressions"]), axis=1)

    metrics = [
        ("Attributed Revenue", "revenue", "money", True),  # higher better
        ("ROAS", "roas", "roas", True),
        ("CTR", "ctr", "pct", True),
        ("CPC", "cpc", "money", False),  # lower better
        ("CPM", "cpm", "money", False),  # lower better
    ]

    # Compute best per KPI (max for higher, min for lower)
    best: Dict[str, float] = {}
    for _label, col, _fmt, higher in metrics:
        series = pd.to_numeric(g[col], errors="coerce")
        if not series.notna().any():
            best[col] = 0.0
        else:
            best[col] = float(series.max() if higher else series.min())

    # Hover formatting helper
    def _fmt_value(val, kind: str) -> str:
        if pd.isna(val):
            return "–"
        if kind == "money":
            return f"$ {val:,.2f}"
        if kind == "pct":
            return f"{val*100:.2f}%"
        if kind == "roas":
            return f"{val:.2f}x"
        return f"{val:,.2f}"

    color_map = {k: v for k, v in CHANNEL_COLORS.items() if k in g["channel"].unique()}
    fig = go.Figure()

    thetas = [m[0] for m in metrics]

    for row in g.itertuples(index=False):
        channel = row.channel
        r_vals: List[float] = []
        hover_lines: List[str] = []
        for label, col, kind, higher in metrics:
            val = getattr(row, col)
            b = best[col]
            # compute score vs best
            if pd.isna(val):
                score = 0.0
            else:
                if higher:
                    if b <= 0:
                        score = 1.0 if val <= 0 else 0.0
                    else:
                        score = float(val) / b
                else:
                    if val <= 0:
                        score = 1.0 if b <= 0 else min(1.0, b / 0.0000001)
                    else:
                        score = b / float(val)
            # clamp to [0,1]
            score = max(0.0, min(1.0, score))
            r_vals.append(score)
            hover_lines.append(f"{label}: {_fmt_value(val, kind)} • Best: {_fmt_value(b, kind)}")

        fig.add_trace(
            go.Scatterpolar(
                r=r_vals,
                theta=thetas,
                name=channel,
                fill="toself",
                line=dict(width=2),
                marker=dict(color=color_map.get(channel)),
                customdata=hover_lines,
                hovertemplate=(
                    "Platform: %{fullData.name}<br>" +
                    "%{theta}<br>" +
                    "%{customdata}<br>" +
                    "Score vs best: %{r:.2f}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(text="Platform KPI Radar (best-normalized)", x=0.0, xanchor="left", font=dict(size=16, color=HEADING_COLOR)),
        polar=dict(
            radialaxis=dict(range=[0, 1], showticklabels=True, tickvals=[0, 0.25, 0.5, 0.75, 1.0], tickformat=",.2f"),
            angularaxis=dict(rotation=90, direction="clockwise", tickfont=dict(size=12, color=TEXT_COLOR)),
            bgcolor="rgba(0,0,0,0)",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
        margin=dict(l=10, r=10, t=48, b=36),
    )
    
    return apply_plot_style(fig)

def chart_aov_timeseries(daily: pd.DataFrame) -> go.Figure:
    dd = daily.copy()
    if dd.empty:
        return apply_plot_style(go.Figure())
    # Ensure AOV exists
    if "aov" not in dd.columns:
        dd["aov"] = dd.apply(lambda r: safe_divide(r.get("total_revenue", np.nan), r.get("orders", np.nan)), axis=1)
    fig = px.line(dd, x="date", y="aov", labels={"aov": "AOV", "date": "Date"}, title="AOV Over Time (Store-wide)")
    fig.update_traces(line=dict(color=ACCENT_CYAN, width=2))
    fig.update_yaxes(tickformat=",.0f", tickprefix="$")
    fig.update_layout(legend=dict(orientation="h"), height=300, margin=dict(l=10, r=10, t=10, b=0))
    return apply_plot_style(fig)


def chart_aov_agg(daily: pd.DataFrame) -> go.Figure:
    dd = daily.copy()
    if dd.empty:
        return apply_plot_style(go.Figure())
    rev = dd["total_revenue"].sum()
    orders = dd["orders"].sum()
    aov_val = safe_divide(rev, orders)
    fig = px.bar(x=["Overall"], y=[aov_val], labels={"x": "", "y": "AOV"}, title="Aggregated AOV")
    fig.update_traces(marker_color=ACCENT_PURPLE, texttemplate="$%{y:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=",.0f", tickprefix="$")
    fig = apply_plot_style(fig)
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=48, b=0), title=dict(text="Aggregated AOV", x=0.0, xanchor="left", y=0.98, yanchor="top", font=dict(size=16, color=HEADING_COLOR)))
    return fig


# ---------------------------
# Region Analysis (CA vs NY)
# ---------------------------
def _filter_ca_ny(ch: pd.DataFrame) -> pd.DataFrame:
    if ch.empty or "state" not in ch.columns:
        return ch.iloc[0:0]
    return ch[ch["state"].astype(str).str.upper().isin(["CA", "NY"])]


def chart_region_roas_timeseries(ch: pd.DataFrame) -> go.Figure:
    df = _filter_ca_ny(ch)
    if df.empty:
        return apply_plot_style(go.Figure())
    g = (
        df.groupby(["date", "state"], as_index=False)
        .agg(spend=("spend", "sum"), revenue=("attributed_revenue", "sum"))
    )
    g["roas"] = g.apply(lambda r: safe_divide(r["revenue"], r["spend"]), axis=1)
    color_map = {"CA": ACCENT_CYAN, "NY": ACCENT_PURPLE}
    fig = px.line(
        g,
        x="date",
        y="roas",
        color="state",
        color_discrete_map=color_map,
        labels={"state": "Region", "roas": "ROAS"},
        title="ROAS Over Time (CA vs NY)",
    )
    fig.update_traces(line=dict(width=2))
    fig.update_yaxes(tickformat=",.2f", ticksuffix="x")
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=300, margin=dict(l=10, r=10, t=10, b=0))
    return apply_plot_style(fig)


def chart_region_metric_agg(ch: pd.DataFrame, metric: str, title: str, yfmt: str = None, yprefix: str | None = None, ysuffix: str | None = None, higher_is_better: bool = True) -> go.Figure:
    df = _filter_ca_ny(ch)
    if df.empty:
        return apply_plot_style(go.Figure())
    g = df.groupby("state", as_index=False).agg(impressions=("impressions", "sum"), clicks=("clicks", "sum"), spend=("spend", "sum"), revenue=("attributed_revenue", "sum"))
    if metric == "roas":
        g["value"] = g.apply(lambda r: safe_divide(r["revenue"], r["spend"]), axis=1)
        y_label = "ROAS"
        sort_ascending = False
    elif metric == "ctr":
        g["value"] = g.apply(lambda r: safe_divide(r["clicks"], r["impressions"]), axis=1)
        y_label = "CTR"
        sort_ascending = False
    elif metric == "cpc":
        g["value"] = g.apply(lambda r: safe_divide(r["spend"], r["clicks"]), axis=1)
        y_label = "CPC"
        sort_ascending = True
    elif metric == "cpm":
        g["value"] = g.apply(lambda r: safe_divide(1000 * r["spend"], r["impressions"]), axis=1)
        y_label = "CPM"
        sort_ascending = True
    else:
        raise ValueError("Unsupported region metric")

    g = g.sort_values("value", ascending=sort_ascending)
    fig = px.bar(g, x="state", y="value", color="state", color_discrete_map={"CA": ACCENT_CYAN, "NY": ACCENT_PURPLE}, labels={"state": "Region", "value": y_label}, title=title)
    if yfmt is not None:
        fig.update_yaxes(tickformat=yfmt)
    if yprefix is not None:
        fig.update_yaxes(tickprefix=yprefix)
    if ysuffix is not None:
        fig.update_yaxes(ticksuffix=ysuffix)
    # Labels
    if metric == "ctr":
        fig.update_traces(texttemplate="%{y:.1%}")
    elif metric in ("cpc", "cpm"):
        fig.update_traces(texttemplate="$%{y:,.2f}")
    else:
        fig.update_traces(texttemplate="%{y:.2f}x")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig = apply_plot_style(fig)
    fig.update_layout(showlegend=False, height=300, margin=dict(l=10, r=10, t=48, b=0), title=dict(text=title, x=0.0, xanchor="left", y=0.98, yanchor="top", font=dict(size=16, color=HEADING_COLOR)))
    return fig


def chart_region_spend_mix(ch: pd.DataFrame) -> go.Figure:
    df = _filter_ca_ny(ch)
    if df.empty:
        return apply_plot_style(go.Figure())
    g = df.groupby(["state", "channel"], as_index=False).agg(spend=("spend", "sum"))
    color_map = {k: v for k, v in CHANNEL_COLORS.items() if k in g["channel"].unique()}
    fig = px.bar(
        g,
        x="state",
        y="spend",
        color="channel",
        color_discrete_map=color_map,
        labels={"state": "Region", "spend": "Spend", "channel": "Platform"},
        title="Spend Mix by Platform (CA vs NY)",
    )
    fig.update_layout(barmode="stack", barnorm="fraction", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), height=300, margin=dict(l=10, r=10, t=48, b=10), title=dict(text="Spend Mix by Platform (CA vs NY)", x=0.0, xanchor="left", y=0.98, yanchor="top", font=dict(size=16, color=HEADING_COLOR)))
    fig.update_yaxes(tickformat=".0%")
    return apply_plot_style(fig)


def chart_correlation_heatmap(daily: pd.DataFrame) -> go.Figure:
    # Build Pearson correlation across selected numeric metrics
    metric_cols = [
        "spend",
        "impressions",
        "clicks",
        "attributed_revenue",
        "orders",
        "total_revenue",
        "new_customers",
    ]
    cols = [c for c in metric_cols if c in daily.columns]
    df = daily[cols].replace([np.inf, -np.inf], np.nan)
    # Require at least some non-na values; fill remaining NAs with 0 for corr stability
    if df.dropna(how="all").empty or len(cols) < 2:
        # Create an empty figure if insufficient data
        fig = go.Figure()
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        return apply_plot_style(fig)
    corr = df.corr(numeric_only=True).round(2)
    fig = px.imshow(
        corr,
        color_continuous_scale=px.colors.sequential.Blues,
        zmin=-1,
        zmax=1,
        text_auto=True,
        aspect="auto",  # allow the heatmap to expand to container width
    )
    fig.update_layout(
        autosize=True,
        height=420,
        margin=dict(l=0, r=0, t=10, b=10),
        coloraxis_colorbar=dict(title="", thickness=12),
    )
    fig.update_xaxes(side="bottom", automargin=True, constrain="domain")
    fig.update_yaxes(automargin=True, constrain="domain")
    return apply_plot_style(fig)

def chart_channel_contribution(ch: pd.DataFrame, metric: str = "spend") -> go.Figure:
    agg = ch.groupby("channel", as_index=False).agg(value=(metric, "sum"))
    agg = agg.sort_values("value", ascending=False)
    fig = px.bar(agg, x="channel", y="value", color="channel", color_discrete_map=CHANNEL_COLORS)
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        yaxis_title=metric.replace("_", " ").title(),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig = apply_plot_style(fig)
    return fig


def chart_tactic_stack(ch: pd.DataFrame, group_by: str = "channel") -> go.Figure:
    if group_by == "channel":
        agg = ch.groupby(["channel", "tactic"], as_index=False).agg(spend=("spend", "sum"))
        fig = px.bar(
            agg,
            x="channel",
            y="spend",
            color="tactic",
            barmode="stack",
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_yaxes(title_text="Spend", showgrid=True, gridcolor=GRID_COLOR)
        fig = apply_plot_style(fig)
        fig.update_xaxes(showgrid=False)
        return fig
    else:  # over_time
        agg = ch.groupby(["date", "tactic"], as_index=False).agg(spend=("spend", "sum"))
        fig = px.area(agg, x="date", y="spend", color="tactic", color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10))
        fig.update_yaxes(title_text="Spend", showgrid=True, gridcolor=GRID_COLOR)
        fig = apply_plot_style(fig)
        return fig


def chart_small_multiples(ch_daily_by_channel: pd.DataFrame, metric: str = "spend") -> go.Figure:
    fig = px.line(
        ch_daily_by_channel,
        x="date",
        y=metric,
        facet_col="channel",
        color="channel",
        color_discrete_map=CHANNEL_COLORS,
        facet_col_wrap=3,
    )
    # Increase top margin and move facet titles down so they are fully visible
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
    for a in fig.layout.annotations:
        a.text = a.text.split("=")[-1]
        a.yshift = -16  # move labels slightly into the subplot to avoid clipping
        a.font = dict(size=14, color=HEADING_COLOR)
    fig.update_yaxes(matches=None, showgrid=True, gridcolor=GRID_COLOR)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig = apply_plot_style(fig)
    return fig


def chart_campaign_scatter(camp: pd.DataFrame) -> go.Figure:
    # Expect columns: spend, attributed_revenue, clicks, channel, roas, cpc, ctr, cpm
    fig = px.scatter(
        camp,
        x="spend",
        y="attributed_revenue",
        size="clicks",
        color="channel",
        color_discrete_map=CHANNEL_COLORS,
        hover_data={
            "clicks": ":,d",
            "roas": ":.2f",
            "cpc": ":.2f",
            "ctr": ":.2%",
            "cpm": ":.2f",
            "campaign": True,
            "channel": True,
        },
        labels={
            "spend": "Spend",
            "attributed_revenue": "Attributed Revenue",
            "clicks": "Clicks (bubble size)",
            "roas": "ROAS",
            "cpc": "CPC",
            "ctr": "CTR",
            "cpm": "CPM",
        },
    )
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig = apply_plot_style(fig)
    return fig


# ---------------------------
# Geography helpers
# ---------------------------
STATE_ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO",
    "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
    "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
    "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def build_geo_tables(ch: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    geo = ch.groupby("state", as_index=False).agg(
        spend=("spend", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
    )
    geo["roas"] = geo.apply(lambda r: safe_divide(r["attributed_revenue"], r["spend"]), axis=1)
    total_spend = geo["spend"].sum()
    geo["share"] = geo["spend"].apply(lambda x: safe_divide(x, total_spend))

    # Attempt to map to US state codes
    def to_code(s):
        if pd.isna(s):
            return None
        s = str(s)
        if len(s) == 2 and s.upper() in STATE_ABBR.values():
            return s.upper()
        return STATE_ABBR.get(s, None)

    geo["state_code"] = geo["state"].apply(to_code)
    geo_map = geo.dropna(subset=["state_code"]).copy()
    return geo, geo_map


def chart_geo_choropleth(geo_map: pd.DataFrame) -> go.Figure:
    fig = px.choropleth(
        geo_map,
        locations="state_code",
        locationmode="USA-states",
        color="roas",
        scope="usa",
        color_continuous_scale=px.colors.sequential.Viridis,
        hover_name="state",
        hover_data={"spend": ":,.0f", "attributed_revenue": ":,.0f", "roas": ":.2%", "state_code": False},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    fig = apply_plot_style(fig)
    return fig


# ---------------------------
# Campaign leaderboard
# ---------------------------
def build_campaign_leaderboard(ch: pd.DataFrame) -> pd.DataFrame:
    camp = ch.groupby(["channel", "campaign"], as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        spend=("spend", "sum"),
        attributed_revenue=("attributed_revenue", "sum"),
    )
    camp["ctr"] = camp.apply(lambda r: safe_divide(r["clicks"], r["impressions"]), axis=1)
    camp["cpc"] = camp.apply(lambda r: safe_divide(r["spend"], r["clicks"]), axis=1)
    camp["cpm"] = camp.apply(lambda r: safe_divide(1000 * r["spend"], r["impressions"]), axis=1)
    camp["roas"] = camp.apply(lambda r: safe_divide(r["attributed_revenue"], r["spend"]), axis=1)
    camp = camp.sort_values(["roas", "attributed_revenue"], ascending=[False, False])
    return camp


def style_campaign_table(df: pd.DataFrame):
    # Use pandas Styler for conditional formatting
    def green_scale(v):
        return f"background-color: rgba(16,185,129,{0.15 + 0.35 * (0 if pd.isna(v) else min(max(float(v), 0), 1))});"  # for pct columns

    def red_rev_scale(v, vmax):
        # Lower is better for cost metrics (CPC, CPM)
        try:
            val = float(v)
            # Normalize inversely: lower -> stronger green
            norm = 1 - (val / vmax) if vmax and vmax > 0 else 0
            alpha = 0.15 + 0.35 * max(0, min(norm, 1))
            return f"background-color: rgba(16,185,129,{alpha});"
        except Exception:
            return ""

    vmax_cpc = df["cpc"].replace([np.inf, -np.inf], np.nan).dropna().max() or 0
    vmax_cpm = df["cpm"].replace([np.inf, -np.inf], np.nan).dropna().max() or 0

    styler = (
        df.style.format(
            {
                "impressions": "{:,}",
                "clicks": "{:,}",
                "spend": "${:,.0f}",
                "attributed_revenue": "${:,.0f}",
                "ctr": "{:.2%}",
                "cpc": "${:,.2f}",
                "cpm": "${:,.2f}",
                "roas": "{:.2f}",
            }
        )
        .map(lambda v: green_scale(v), subset=["ctr", "roas"])  # Styler.map replaces deprecated applymap
        .map(lambda v: red_rev_scale(v, vmax_cpc), subset=["cpc"]) 
        .map(lambda v: red_rev_scale(v, vmax_cpm), subset=["cpm"]) 
    )
    return styler


# ---------------------------
# Insights
# ---------------------------
def build_insights(ch: pd.DataFrame, biz: pd.DataFrame, date_range: Tuple[datetime, datetime]) -> List[str]:
    insights = []

    # Top/bottom channels by ROAS
    ch_agg = ch.groupby("channel", as_index=False).agg(spend=("spend", "sum"), revenue=("attributed_revenue", "sum"))
    ch_agg["roas"] = ch_agg.apply(lambda r: safe_divide(r["revenue"], r["spend"]), axis=1)
    ch_agg = ch_agg.sort_values("roas", ascending=False)
    if not ch_agg.empty:
        top = ch_agg.iloc[0]
        bot = ch_agg.iloc[-1]
        insights.append(
            f"Top channel by ROAS: {top['channel']} ({top['roas']:.2f}x). Lowest: {bot['channel']} ({(bot['roas'] if not pd.isna(bot['roas']) else 0):.2f}x)."
        )

    # Campaigns extremes by ROAS (with spend threshold)
    camp = build_campaign_leaderboard(ch)
    if not camp.empty:
        thresh = max(100.0, camp["spend"].median() * 0.2)  # ignore tiny spenders
        camp_f = camp[camp["spend"] >= thresh]
        if not camp_f.empty:
            top_c = camp_f.iloc[0]
            bottom_c = camp_f.iloc[-1]
            insights.append(
                f"Campaign winner: {top_c['campaign']} ({top_c['channel']}) at {top_c['roas']:.2f}x ROAS. Underperformer: {bottom_c['campaign']} ({bottom_c['channel']}) at {bottom_c['roas']:.2f}x."
            )

    # WoW movers on spend
    start_date, end_date = date_range
    anchor = pd.to_datetime(end_date)
    curr_start = (anchor - timedelta(days=6)).date()
    prev_start = (anchor - timedelta(days=13)).date()
    prev_end = (anchor - timedelta(days=7)).date()

    ch_curr = ch[(pd.to_datetime(ch["date"]) >= pd.to_datetime(curr_start)) & (pd.to_datetime(ch["date"]) <= pd.to_datetime(end_date))]
    ch_prev = ch[(pd.to_datetime(ch["date"]) >= pd.to_datetime(prev_start)) & (pd.to_datetime(ch["date"]) <= pd.to_datetime(prev_end))]

    mov = (
        ch_curr.groupby("channel", as_index=False).agg(curr_spend=("spend", "sum"))
        .merge(ch_prev.groupby("channel", as_index=False).agg(prev_spend=("spend", "sum")), on="channel", how="outer")
        .fillna(0)
    )
    mov["delta"] = mov.apply(lambda r: wow_change(r["curr_spend"], r["prev_spend"]), axis=1)
    mov = mov.replace([np.inf, -np.inf], np.nan)
    mov = mov.dropna(subset=["delta"]) if not mov.empty else mov
    if not mov.empty:
        mov = mov.sort_values("delta", ascending=False)
        up = mov.iloc[0]
        down = mov.iloc[-1]
        if not pd.isna(up["delta"]) and not pd.isna(down["delta"]) and len(mov) > 1:
            insights.append(
                f"Biggest WoW spend mover: {up['channel']} ({up['delta']*100:.1f}% up). Biggest decline: {down['channel']} ({down['delta']*100:.1f}% down)."
            )

    # # Anomaly summary
    # daily, _, _ = build_daily_joins(biz, ch)
    # dd = daily[(pd.to_datetime(daily["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(daily["date"]) <= pd.to_datetime(end_date))]
    # if not dd.empty:
    #     z_spend = (z_scores(dd["spend"].fillna(0)).abs() >= 2).sum()
    #     z_revenue = (z_scores(dd["total_revenue"].fillna(0)).abs() >= 2).sum()
    #     if z_spend or z_revenue:
    #         insights.append(f"Detected {int(z_spend)} spend and {int(z_revenue)} revenue anomaly days (|z| ≥ 2).")

    return insights


# ---------------------------
# Gauges (semicircle meters)
# ---------------------------
def chart_semicircle_gauge(
    value: float,
    min_value: float,
    max_value: float,
    title: str,
    target: float | None = None,
    suffix: str = "",
    color: str = "#2E6BFF",
) -> go.Figure:
    try:
        val = float(value) if not pd.isna(value) else 0.0
    except Exception:
        val = 0.0

    # Build dynamic steps so the blue gradient represents the value (fill),
    # and the remainder of the arc is muted (purple-gray), matching the reference.
    rng = max_value - min_value if max_value is not None else 1
    if rng <= 0:
        rng = 1
    # Clamp the value within the gauge range for coloring purposes
    try:
        fill_end = min(max(float(val), float(min_value)), float(max_value))
    except Exception:
        fill_end = val

    gradient_steps = []
    grad_colors = [
        "#A7E8FF",  # light cyan
        "#7DD3FF",
        "#4FC7FF",
        "#2EBBFF",
        "#19B0FF",
        "#0FA4FF",
        "#0A96FF",  # deeper blue
    ]
    n = len(grad_colors)
    # Only color up to the current value
    if fill_end > min_value:
        for i, c in enumerate(grad_colors):
            start = min_value + (fill_end - min_value) * (i / n)
            end = min_value + (fill_end - min_value) * ((i + 1) / n)
            if end <= start:
                continue
            gradient_steps.append({"range": [start, end], "color": c})
    # Remainder from value to max in muted purple-gray
    if fill_end < max_value:
        gradient_steps.append({"range": [fill_end, max_value], "color": "#6D5560"})

    # No white needle — the gradient represents the value. Keep threshold unset here.

    # Show only extreme tick labels to reduce clutter and avoid clipping
    def _fmt_tick(x: float) -> str:
        try:
            xv = float(x)
        except Exception:
            return str(x)
        if suffix == "%":
            return f"{xv:,.0f}%"
        if suffix == "x":
            s = f"{xv:,.2f}x"
            return s.rstrip("0").rstrip(".")
        return f"{xv:,.0f}"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix": suffix, "font": {"size": 32, "color": HEADING_COLOR}},
            # Empty built-in title; we'll place a custom header as an annotation
            title={"text": "", "font": {"size": 14, "color": "#6B7280"}},
            gauge={
                "axis": {
                    "range": [min_value, max_value],
                    "tickcolor": MUTED_COLOR,
                    "tickmode": "array",
                    "tickvals": [min_value, max_value],
                    "ticktext": [_fmt_tick(min_value), _fmt_tick(max_value)],
                    "tickfont": {"color": MUTED_COLOR, "size": 12},
                },
                # Hide fill and rely on a needle for the value
                "bar": {"color": "rgba(0,0,0,0)", "thickness": 0.5},
                # Subtle inner ring
                "bgcolor": "rgba(255,255,255,0.08)",
                # Gradient background arc and tail
                "steps": gradient_steps,
                # No needle: gradient fill indicates the value extent
                "threshold": None,
            },
            # Inset the gauge horizontally to avoid clipping extreme tick labels
            domain={"x": [0.03, 0.97], "y": [0, 1]},
        )
    )
    # Target markers removed per request; target parameter is ignored for rendering
    # Add heading above the gauge so it doesn't overlap the center value
    fig.add_annotation(
        text=title,
        showarrow=False,
        x=0.5,
        y=1.12,
        xref="paper",
        yref="paper",
        font=dict(size=16, color=HEADING_COLOR),
        align="center",
    )
    # Target pill removed per request
    fig.update_layout(
        margin=dict(l=42, r=42, t=30, b=6),
        height=230,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------
# Main App
# ---------------------------
def main():
    st.title("Marketing Intelligence Dashboard")
    info = data_refresh_info()

    # Load data
    business = load_business()
    channels = load_channels()

    # Precompute joins
    daily_all, biz_daily_all, ch_daily_by_channel_all = build_daily_joins(business, channels)

    # Filters
    date_range = build_filters(channels, business)
    biz_f, ch_f = apply_filters(business, channels, date_range)

    # Update joins for filtered data
    daily, biz_daily, ch_daily_by_channel = build_daily_joins(biz_f, ch_f)

    # Header meta
    start_date, end_date = date_range
    st.caption(f"Active date range: {start_date} → {end_date}")

    # KPIs
    kpis = compute_kpis(biz_f, ch_f, date_range)
    spark = kpi_sparklines(daily, start_date, end_date)

    kpi_order = [
        "Total Revenue",
        "Attributed Revenue",
        "Gross Profit",
        "Orders",
        "New Customers",
        "Total Spend",
        "Clicks",
        "Blended CAC",
        "AOV",
    ]
    cols = st.columns(3)
    cols2 = st.columns(3)
    cols3 = st.columns(3)
    all_cols = cols + cols2 + cols3
    for i, key in enumerate(kpi_order):
        with all_cols[i]:
            card = kpis[key]
            val = card["value"]
            delta = card["delta"]
            fmt = card["format"]
            help_text = card.get("help") or ""
            tooltip = help_text.replace('"', '&quot;')

            # KPI heading with hover tooltip (outside the box)
            st.markdown(
                f"<div class='kpi-headline'>{key}<span class='info-icon' title=\"{tooltip}\">i</span></div>",
                unsafe_allow_html=True,
            )

            # KPI value only inside the bordered box
            with st.container(border=True):
                st.markdown(f"<div class='kpi-value'>{fmt(val)}</div>", unsafe_allow_html=True)

            # Delta and sparkline outside the box
            if not pd.isna(delta):
                cls = "pos" if delta >= 0 else "neg"
                st.markdown(
                    f"<div class='kpi-delta {cls}'>{'▲' if delta >= 0 else '▼'} {abs(delta)*100:.1f}% WoW</div>",
                    unsafe_allow_html=True,
                )
            # Removed sparkline from KPI section — relocated to bottom tab

    # Semicircle meters (hero gauges) right after KPIs
    g1, gutter1, g2, gutter2, g3 = st.columns([1, 0.05, 1, 0.05, 1])
    current_roas = kpis["Blended ROAS"]["value"]
    current_gm = kpis["Gross Margin"]["value"]
    ctr_val = np.nan
    if not ch_f.empty:
        total_clicks = ch_f["clicks"].sum()
        total_impr = ch_f["impressions"].sum()
        ctr_val = safe_divide(total_clicks, total_impr)

    with g1:
        st.plotly_chart(
            chart_semicircle_gauge(
                value=current_roas if not pd.isna(current_roas) else 0.0,
                min_value=0.0,
                max_value=4.0,
                title="Blended ROAS",
                suffix="x",
                color="#2E6BFF",
            ),
            width='stretch',
            config={"displayModeBar": False},
        )

    with g2:
        st.plotly_chart(
            chart_semicircle_gauge(
                value=(float(current_gm) * 100.0) if not pd.isna(current_gm) else 0.0,
                min_value=0.0,
                max_value=100.0,
                title="Gross Margin",
                suffix="%",
                color="#10B981",
            ),
            width='stretch',
            config={"displayModeBar": False},
        )

    with g3:
        st.plotly_chart(
            chart_semicircle_gauge(
                value=(float(ctr_val) * 100.0) if not pd.isna(ctr_val) else 0.0,
                min_value=0.0,
                max_value=10.0,
                title="CTR",
                suffix="%",
                color="#F59E0B",
            ),
            width='stretch',
            config={"displayModeBar": False},
        )

    st.divider()

    # Heading for platform metric sections
    st.subheader("Platform KPI Analysis")
    # Spider chart comparing platforms to the best per KPI
    if not ch_f.empty:
        st.plotly_chart(chart_platform_kpi_radar(ch_f), width='stretch', config={"displayModeBar": False})

    # Attributed revenue analysis (move to top of analysis sections)
    with st.expander("Attributed Revenue Analysis", expanded=False):
        st.caption("Compare attributed revenue across platforms over time and in total.")
        if not ch_f.empty:
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(chart_platform_revenue_timeseries(ch_f), width='stretch', config={"displayModeBar": False})
            with right:
                st.plotly_chart(chart_platform_revenue_agg(ch_f), width='stretch', config={"displayModeBar": False})
        else:
            st.info("No channel data for revenue analysis.")

    # New: ROAS comparison across platforms (before gauges)
    with st.expander("ROAS Analysis", expanded=False):
        st.caption("Daily ROAS per platform with an aggregated comparison.")
        if not ch_f.empty:
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(chart_platform_roas(ch_f), width='stretch', config={"displayModeBar": False})
            with right:
                st.plotly_chart(chart_platform_roas_agg(ch_f), width='stretch', config={"displayModeBar": False})
        else:
            st.info("No channel data for current filters.")

    # CTR analysis
    with st.expander("CTR Analysis", expanded=False):
        st.caption("Click-through rate indicates creative/message resonance. Higher is better.")
        if not ch_f.empty:
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(_platform_metric_timeseries(ch_f, "ctr", "CTR by Platform", yfmt=".1%"), width='stretch', config={"displayModeBar": False})
            with right:
                st.plotly_chart(_platform_metric_agg(ch_f, "ctr", "Aggregated CTR", yfmt=".1%"), width='stretch', config={"displayModeBar": False})
        else:
            st.info("No channel data for CTR analysis.")

    # CPC analysis
    with st.expander("CPC Analysis", expanded=False):
        st.caption("Cost per click highlights traffic efficiency. Lower is better.")
        if not ch_f.empty:
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(_platform_metric_timeseries(ch_f, "cpc", "CPC by Platform", yfmt=",.2f", yprefix="$"), width='stretch', config={"displayModeBar": False})
            with right:
                st.plotly_chart(_platform_metric_agg(ch_f, "cpc", "Aggregated CPC", yfmt=",.2f", yprefix="$", higher_is_better=False), width='stretch', config={"displayModeBar": False})
        else:
            st.info("No channel data for CPC analysis.")

    # CPM analysis
    with st.expander("CPM Analysis", expanded=False):
        st.caption("Cost per thousand impressions reflects media buying efficiency. Lower is better.")
        if not ch_f.empty:
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(_platform_metric_timeseries(ch_f, "cpm", "CPM by Platform", yfmt=",.2f", yprefix="$"), width='stretch', config={"displayModeBar": False})
            with right:
                st.plotly_chart(_platform_metric_agg(ch_f, "cpm", "Aggregated CPM", yfmt=",.2f", yprefix="$", higher_is_better=False), width='stretch', config={"displayModeBar": False})
        else:
            st.info("No channel data for CPM analysis.")

    # Tactics Analysis
    with st.expander("Tactics Analysis", expanded=False):
        st.caption("Explore how platforms connect to tactics and their performance.")
        # Replace static text with a Sankey visualization of Platform → Tactic
        if not ch_f.empty and {"channel", "tactic"}.issubset(ch_f.columns):
            st.plotly_chart(chart_platform_tactic_sankey(ch_f), width='stretch', config={"displayModeBar": False})
        if not ch_f.empty:
            left, right = st.columns([2, 1])
            with left:
                st.plotly_chart(chart_tactic_roas_timeseries(ch_f), width='stretch', config={"displayModeBar": False})
            with right:
                st.plotly_chart(chart_tactic_roas_agg(ch_f), width='stretch', config={"displayModeBar": False})
        else:
            st.info("No channel data for tactics analysis.")

    # Other analysis heading
    st.subheader("Other Analysis")

    # Revenue vs Spend
    with st.expander("Revenue vs Spend", expanded=False):
        st.caption("Overall business revenue vs total ad spend over time.")
        left, right = st.columns([2, 1])
        with left:
            fig = chart_trends(daily, start_date, end_date, show_anomalies=False)
            st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        with right:
            st.plotly_chart(chart_rev_spend_agg(daily), width='stretch', config={"displayModeBar": False})

    # Region comparison: CA vs NY
    with st.expander("Region Comparison: CA vs NY", expanded=False):
        st.caption("Compare efficiency and mix across California and New York.")
        ch_states = ch_f[ch_f["state"].astype(str).str.upper().isin(["CA", "NY"])] if not ch_f.empty else ch_f
        if not ch_states.empty:
            row1_left, row1_right = st.columns([2, 1])
            with row1_left:
                st.plotly_chart(chart_region_roas_timeseries(ch_states), width='stretch', config={"displayModeBar": False})
            with row1_right:
                st.plotly_chart(chart_region_metric_agg(ch_states, "roas", "Aggregated ROAS (CA vs NY)", yfmt=",.2f", ysuffix="x"), width='stretch', config={"displayModeBar": False})

            c1, c2, c3 = st.columns(3)
            with c1:
                st.plotly_chart(chart_region_metric_agg(ch_states, "ctr", "CTR (CA vs NY)", yfmt=".1%"), width='stretch', config={"displayModeBar": False})
            with c2:
                st.plotly_chart(chart_region_metric_agg(ch_states, "cpc", "CPC (CA vs NY)", yfmt=",.2f", yprefix="$", higher_is_better=False), width='stretch', config={"displayModeBar": False})
            with c3:
                st.plotly_chart(chart_region_metric_agg(ch_states, "cpm", "CPM (CA vs NY)", yfmt=",.2f", yprefix="$", higher_is_better=False), width='stretch', config={"displayModeBar": False})

            st.plotly_chart(chart_region_spend_mix(ch_states), width='stretch', config={"displayModeBar": False})
        else:
            st.info("No CA/NY data available for current filters.")

    # Campaign performance
    with st.expander("Campaign Performance", expanded=False):
        camp = build_campaign_leaderboard(ch_f)
        if not camp.empty:
            st.plotly_chart(chart_campaign_scatter(camp), width='stretch')
            st.caption("Bubble radius indicates click volume per campaign.")

            st.caption("Leaderboard (current filters)")
            styler = style_campaign_table(camp)
            st.dataframe(styler, width='stretch')

            csv_bytes = camp.to_csv(index=False).encode("utf-8")
            st.download_button("Download Campaign CSV", data=csv_bytes, file_name="campaign_leaderboard.csv", mime="text/csv")
        else:
            st.info("No campaign data for current filters.")

    # Correlation analysis
    with st.expander("Correlation Matrix", expanded=False):
        st.caption("Pearson correlations on daily aggregates within the selected date range.")
        dd = daily[(pd.to_datetime(daily["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(daily["date"]) <= pd.to_datetime(end_date))]
        if not dd.empty:
            st.plotly_chart(chart_correlation_heatmap(dd), width='stretch', config={"displayModeBar": False})
        else:
            st.info("No data available for the selected range to compute correlations.")

    st.subheader("Insights")
    bullets = build_insights(ch_f, biz_f, date_range)
    if bullets:
        for b in bullets:
            st.markdown(f"- {b}")
    else:
        st.info("No standout insights for current filters.")


if __name__ == "__main__":
    main()
