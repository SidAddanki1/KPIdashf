# ===========================================
# C-Store KPI Insights Suite (Executive Edition)
# - One GLOBAL filter: Department (compact sidebar)
# - LOCAL per-tab filters (Store/Category/Brand/Product/Payment/Date/Granularity)
# - Visual polish: Plotly template, KPI tiles, consistent tooltips
# - Metric definitions everywhere (hover + glossary)
# - Forecasts: conservative ETS/naive with capped horizons
# - No CSV uploader (reads DATA_PATH only)
# ===========================================

import os
from typing import Dict, Tuple, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import streamlit as st

from itertools import combinations
from collections import Counter
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ----------------------------
# App & Theme
# ----------------------------
st.set_page_config(
    page_title="C-Store KPI Insights Suite",
    layout="wide",
    initial_sidebar_state="collapsed",  # compact—only one global filter lives here
)

# Plotly template (clean, brandable)
pio.templates["sa_clean"] = pio.templates["plotly_white"]
pio.templates["sa_clean"].layout.update(
    font=dict(family="Inter, system-ui, -apple-system, Segoe UI", size=12),
    margin=dict(l=8, r=8, t=36, b=8),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", title=None),
    yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False),
    colorway=["#4F46E5","#0EA5E9","#22C55E","#F59E0B","#EF4444","#8B5CF6"],
)
pio.templates.default = "sa_clean"

# ----------------------------
# Configuration
# ----------------------------
DATA_PATH = "cstorereal.csv"  # <— set to your file path

# ----------------------------
# Metric Glossary (global + reused in tooltips)
# ----------------------------
GLOSSARY = {
    "Total_Sale": "Gross revenue (post-discount if provided). Sum of Total_Sale.",
    "Quantity": "Units sold. Sum of Quantity.",
    "Transactions": "Count of unique Transaction_ID.",
    "Spend per Basket": "Total Sales ÷ Transactions; average ticket.",
    "ASP": "Average selling price = Total Sales ÷ Units.",
    "Median_Unit_Price": "Median of Unit_Price observations in scope.",
    "Velocity_$/SKU/Store/Week": "SKU sales normalized by Active Stores and Weeks in range.",
    "% of Category Sales": "SKU sales ÷ Category sales × 100.",
    "Velocity_Rank_in_Category": "Dense rank of SKU by Velocity within Category (1 = fastest).",
    "Lift": "Co-purchase frequency vs independence baseline (>1 positive association).",
    "Confidence (%)": "P(B|A): share of A baskets that also contain B.",
    "Support (A,B)": "Share of total baskets that contain both A and B.",
    "Avg Co-Basket Spend": "Average value of baskets where both A and B are present.",
    "Index_vs_Overall": "100 × Geo_Share ÷ Overall_Share ( >100 over-index ).",
    "Geo_Share": "Geo’s sales share for the item/brand.",
    "Overall_Share": "Overall sales share for the item/brand across all geos.",
}

# ----------------------------
# Utilities
# ----------------------------
def _to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

@st.cache_data(show_spinner=False)
def load_and_normalize(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        st.error(f"CSV not found at: {os.path.abspath(path)}")
        st.stop()

    df = pd.read_csv(path)

    # Canonicalize column names when common alternates are present
    rename_map = {
        "Brand": "Brand_Name",
        "Units_Sold": "Quantity",
        "Total_Sales": "Total_Sale",
        "Payment_Type": "Payment_Method",
        "StoreID": "Store_ID",
        "TransactionID": "Transaction_ID",
        "UnitPrice": "Unit_Price",
        "Product": "Item",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = {"Date", "Transaction_ID", "Store_ID", "Category", "Item"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Transaction_ID"])
    for c in ["Transaction_ID", "Store_ID", "Item", "Brand_Name", "Category", "Payment_Method"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if "Unit_Price" in df.columns:
        df["Unit_Price"] = _to_num(df["Unit_Price"]).fillna(0.0)
    if "Quantity" in df.columns:
        df["Quantity"] = _to_num(df["Quantity"]).fillna(0.0)
    if "Total_Sale" in df.columns:
        df["Total_Sale"] = _to_num(df["Total_Sale"]).fillna(0.0)

    # Compute Total_Sale when missing/zero and we have Unit_Price × Quantity
    if {"Quantity", "Unit_Price"} <= set(df.columns):
        if "Total_Sale" not in df.columns:
            df["Total_Sale"] = (df["Quantity"] * df["Unit_Price"]).round(2)
        else:
            need = (df["Total_Sale"].isna()) | (df["Total_Sale"] == 0)
            df.loc[need, "Total_Sale"] = (df.loc[need, "Quantity"] * df.loc[need, "Unit_Price"]).round(2)

    return df

RAW = load_and_normalize(DATA_PATH)

# ---------------------------------
# GLOBAL: Department filter (compact)
# ---------------------------------
st.sidebar.markdown("### Department")
DEPT_OPTIONS = [
    "Strategy & Finance",
    "Merchandising & Pricing",
    "Store Ops & Supply Chain",
    "Marketing & Loyalty",
    "All Departments"
]
department = st.sidebar.selectbox("Select Department", DEPT_OPTIONS, index=0, key="global_department")

# Which tabs show for which department
DEPT_TABS = {
    "Strategy & Finance": ["📊 KPI Overview", "📈 KPI Trends"],
    "Merchandising & Pricing": ["🏆 Top-N Views", "💲 Price Ladder", "📦 Assortment & Space Optimization"],
    "Store Ops & Supply Chain": ["🗺️ Store Map", "📊 KPI Overview"],
    "Marketing & Loyalty": ["🧺 Basket Affinity", "🏆 Top-N Views"],
    "All Departments": [
        "📊 KPI Overview", "📈 KPI Trends", "🏆 Top-N Views",
        "🧺 Basket Affinity", "💲 Price Ladder", "🗺️ Store Map",
        "📦 Assortment & Space Optimization"
    ]
}

# ----------------------------
# Local Filter Builder (per tab)
# ----------------------------
RULE_MAP = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}

def local_filters_block(df: pd.DataFrame, prefix: str):
    """
    Per-tab filter block; keys isolated by prefix to avoid collisions.
    Returns (filtered_df, rule, start_date, end_date, selections_dict)
    """
    st.markdown(
        f"""
        <div style="display:flex;flex-wrap:wrap;gap:10px;margin:-6px 0 6px 0;">
          <span style="font-weight:600;">Filters:</span>
        </div>
        """, unsafe_allow_html=True
    )

    min_d = df["Date"].min().date()
    max_d = df["Date"].max().date()

    col1, col2, col3, col4, col5, col6 = st.columns([1.0,1.0,1.0,1.0,1.0,1.0])

    with col1:
        stores = sorted(df["Store_ID"].unique().tolist())
        sel_stores = st.multiselect("Store(s)", stores, default=stores, key=f"{prefix}_stores")

    with col2:
        cats = sorted(df["Category"].unique().tolist())
        sel_cats = st.multiselect("Category", cats, default=cats, key=f"{prefix}_cats")

    with col3:
        brands = sorted(df["Brand_Name"].unique().tolist()) if "Brand_Name" in df.columns else []
        sel_brands = st.multiselect("Brand", brands, default=brands, key=f"{prefix}_brands")

    with col4:
        prods = sorted(df["Item"].unique().tolist())
        sel_items = st.multiselect("Product", prods, default=prods, key=f"{prefix}_items")

    with col5:
        pays = sorted(df["Payment_Method"].unique().tolist()) if "Payment_Method" in df.columns else []
        sel_pay = st.multiselect("Payment Method", pays, default=pays, key=f"{prefix}_pays")

    with col6:
        gran = st.radio("Time", ["Daily","Weekly","Monthly"], horizontal=True, index=1, key=f"{prefix}_gran")
        rule = RULE_MAP[gran]

    c7, c8 = st.columns([1,1])
    with c7:
        start_date = st.date_input("Start", value=min_d, key=f"{prefix}_start")
    with c8:
        end_date = st.date_input("End", value=max_d, key=f"{prefix}_end")

    # Apply local filters only to this tab
    scope = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))].copy()
    if sel_stores: scope = scope[scope["Store_ID"].isin(sel_stores)]
    if sel_cats: scope = scope[scope["Category"].isin(sel_cats)]
    if sel_brands and "Brand_Name" in scope.columns:
        scope = scope[scope["Brand_Name"].isin(sel_brands)]
    if sel_items: scope = scope[scope["Item"].isin(sel_items)]
    if sel_pay and "Payment_Method" in scope.columns:
        scope = scope[scope["Payment_Method"].isin(sel_pay)]

    selections = {
        "stores": sel_stores, "categories": sel_cats, "brands": sel_brands,
        "items": sel_items, "payments": sel_pay, "granularity": gran
    }
    return scope, rule, start_date, end_date, selections

# ----------------------------
# KPI helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def kpis(df: pd.DataFrame) -> Dict[str, float]:
    ts = df["Total_Sale"].sum() if "Total_Sale" in df.columns else 0.0
    qty = df["Quantity"].sum() if "Quantity" in df.columns else 0.0
    tx = df["Transaction_ID"].nunique() if "Transaction_ID" in df.columns else 0
    spb = ts/tx if tx else 0.0
    asp = ts/qty if qty else 0.0
    return dict(total_sales=ts, total_qty=qty, tx=tx, spend_per_basket=spb, asp=asp)

@st.cache_data(show_spinner=False)
def trends(df: pd.DataFrame, rule: str, group_dim: Optional[str]) -> pd.DataFrame:
    group_cols = [pd.Grouper(key="Date", freq=rule)]
    if group_dim and group_dim in df.columns: group_cols.append(group_dim)
    t = (df.groupby(group_cols, dropna=False)
           .agg(Total_Sale=("Total_Sale","sum"),
                Quantity=("Quantity","sum"),
                Transactions=("Transaction_ID","nunique"))
           .reset_index().sort_values("Date"))
    t["Spend per Basket"] = np.where(t["Transactions"]>0, t["Total_Sale"]/t["Transactions"], 0.0)
    return t

# ----------------------------
# Forecast helpers (conservative)
# ----------------------------
def _forecast_meta(rule: str):
    if rule == "D": return 30, 7, "D", "next 30 days"
    if rule.startswith("W-"): return 4, 52, rule, "next 4 weeks"
    if rule == "MS": return 1, 12, "MS", "next month"
    return 30, 7, "D", "next 30 days"

def _series(df: pd.DataFrame, rule: str, metric: str) -> pd.Series:
    s = (df.set_index("Date")[metric].sort_index().resample(rule).sum().astype(float).fillna(0.0))
    s.index.name = "Date"
    return s

def _fit_ets(ts: pd.Series, rule: str, seasonal_periods: int):
    use_fixed = rule.startswith("W-")
    use_season = seasonal_periods and (len(ts) >= 2*seasonal_periods)
    try:
        trend_type = 'add' if use_fixed else None
        fit_params = {'smoothing_level':0.1,'smoothing_trend':0.01,'optimized':False} if use_fixed else {'optimized':True}
        model = ExponentialSmoothing(
            ts, trend=trend_type, damped_trend=False,
            seasonal="add" if use_season else None,
            seasonal_periods=seasonal_periods if use_season else None,
            initialization_method="estimated"
        ).fit(**fit_params)
        fitted = model.fittedvalues.reindex(ts.index)
        resid = (ts - fitted).to_numpy()
        return model, float(np.nanstd(resid))
    except Exception:
        return None, None

def _seasonal_naive(ts: pd.Series, steps: int, seasonal_periods: int) -> np.ndarray:
    if seasonal_periods and len(ts) >= seasonal_periods:
        rep = np.tile(ts.iloc[-seasonal_periods:].to_numpy(), int(np.ceil(steps/seasonal_periods)))[:steps]
        return rep.astype(float)
    return np.full(steps, float(ts.iloc[-1]) if len(ts) else 0.0)

def _choose_model(ts: pd.Series, rule: str, seasonal_periods: int):
    n = len(ts)
    if n < max(12, seasonal_periods+4): return "naive", None, None
    h = max(6, int(round(n*0.12)))
    train, hold = ts.iloc[:-h], ts.iloc[-h:]
    ets_model, resid_std = _fit_ets(train, rule, seasonal_periods)
    naive_fc = _seasonal_naive(train, h, seasonal_periods)

    def smape(a, f):
        a, f = a.astype(float), f.astype(float)
        d = (np.abs(a)+np.abs(f)); d[d==0] = 1.0
        return float(np.mean(2*np.abs(a-f)/d))

    a = hold.to_numpy()
    ets_fc = ets_model.forecast(h).to_numpy() if ets_model is not None else None
    if ets_fc is None or smape(a, ets_fc) >= smape(a, naive_fc):
        return "naive", None, None
    return "ets", ets_model, resid_std

def forecast_conservative(df: pd.DataFrame, rule: str, metric: str, alpha=0.05) -> pd.DataFrame:
    steps, seasonal_periods, freq, _ = _forecast_meta(rule)
    ts = _series(df, rule, metric).clip(lower=0)
    kind, ets_model, resid_std = _choose_model(ts, rule, seasonal_periods)
    if kind == "ets" and ets_model is not None:
        mean = ets_model.forecast(steps).to_numpy()
        z = 1.96 if alpha==0.05 else 1.64
        band = (resid_std or 0.0) * z
        lo, hi = np.maximum(0.0, mean-band), np.maximum(0.0, mean+band)
    else:
        mean = _seasonal_naive(ts, steps, seasonal_periods)
        lo, hi = np.maximum(0.0, mean*0.9), mean*1.1

    look = min(len(ts), 28 if rule=="D" else (12 if rule=="MS" else 12))
    recent = float(ts.iloc[-look:].mean()) if look>0 else 0.0
    cap_lo, cap_hi = recent*steps*0.7, recent*steps*1.3
    horizon_sum = float(mean.sum())
    if cap_hi>0 and horizon_sum>cap_hi:
        s = cap_hi/horizon_sum; mean*=s; lo*=s; hi*=s
    elif horizon_sum<cap_lo and horizon_sum>0:
        s = cap_lo/horizon_sum; mean*=s; lo*=s; hi*=s

    idx = pd.date_range(ts.index[-1], periods=steps+1, freq=freq)[1:]
    return pd.DataFrame({"Date": idx, "yhat": mean, "yhat_lower": lo, "yhat_upper": hi})

# ----------------------------
# Affinity rules
# ----------------------------
@st.cache_data(show_spinner=False)
def affinity_rules(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    scope = df.dropna(subset=["Transaction_ID", key_col]).copy()
    scope["Transaction_ID"] = scope["Transaction_ID"].astype(str)
    scope[key_col] = scope[key_col].astype(str)

    tx_n = scope["Transaction_ID"].nunique()
    if tx_n == 0: return pd.DataFrame()

    basket_sales = scope.groupby("Transaction_ID")["Total_Sale"].sum()
    tx_keys = scope.groupby("Transaction_ID")[key_col].apply(lambda s: tuple(sorted(set(s))))

    item_counts = Counter(); pair_counts = Counter()
    for keys in tx_keys:
        for k in keys: item_counts[k]+=1
        for a,b in combinations(keys,2): pair_counts[tuple(sorted((a,b)))] += 1
    if not pair_counts: return pd.DataFrame()

    item_txids_map = scope.groupby(key_col)["Transaction_ID"].apply(set)

    def qty_in_txids(item: str, tx_ids: set) -> float:
        if "Quantity" not in scope.columns or not tx_ids: return 0.0
        v = scope[(scope["Transaction_ID"].isin(tx_ids)) & (scope[key_col]==item)]
        return float(v["Quantity"].sum())

    rows=[]
    for (a,b), n_ab in pair_counts.items():
        ca, cb = item_counts[a], item_counts[b]
        supp = n_ab/tx_n
        lift = (supp/((ca/tx_n)*(cb/tx_n))) if (ca and cb) else 0.0

        tx_a, tx_b = item_txids_map.get(a,set()), item_txids_map.get(b,set())
        co_tx = tx_a & tx_b
        total_co = float(basket_sales.loc[list(co_tx)].sum()) if co_tx else 0.0
        avg_co = (total_co/n_ab) if n_ab else 0.0

        qa = qty_in_txids(a, co_tx); qb = qty_in_txids(b, co_tx)

        rows += [
            {"Antecedent":a,"Consequent":b,"Total Co-Baskets":n_ab,
             "Support (A,B)":supp,"Confidence (A->B)":(n_ab/ca if ca else 0.0),
             "Lift (A,B)":lift,"Total_Ante_Qty":qa,"Avg_Ante_Qty":(qa/n_ab if n_ab else 0.0),
             "Total_CoBasket_Sales_Value":total_co,"Avg_CoBasket_Spend":avg_co},
            {"Antecedent":b,"Consequent":a,"Total Co-Baskets":n_ab,
             "Support (A,B)":supp,"Confidence (A->B)":(n_ab/cb if cb else 0.0),
             "Lift (A,B)":lift,"Total_Ante_Qty":qb,"Avg_Ante_Qty":(qb/n_ab if n_ab else 0.0),
             "Total_CoBasket_Sales_Value":total_co,"Avg_CoBasket_Spend":avg_co}
        ]
    out = pd.DataFrame(rows)
    return out.sort_values(["Lift (A,B)","Confidence (A->B)"], ascending=[False,False]).reset_index(drop=True)

# ----------------------------
# Reusable UI blocks
# ----------------------------
def kpi_tile(label: str, value: str, delta: Optional[float]=None, help_text: Optional[str]=None):
    col = st.container(border=True)
    with col:
        left, right = st.columns([1,1])
        with left:
            st.markdown(f"<span style='font-size:12px;color:#64748B' title='{help_text or ''}'>{label}</span>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:34px;font-weight:600;line-height:1.1'>{value}</div>", unsafe_allow_html=True)
        with right:
            if delta is not None:
                color = "#22C55E" if delta >= 0 else "#EF4444"
                arrow = "▲" if delta >= 0 else "▼"
                st.markdown(
                    f"<div style='text-align:right;color:{color};margin-top:6px' title='Period-over-period % change'>{arrow} {abs(delta):.1f}%</div>",
                    unsafe_allow_html=True
                )

def glossary_expander():
    with st.expander("ℹ️ Metric Definitions", expanded=False):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}** — {v}")

# ----------------------------
# Displays
# ----------------------------
def view_kpi_overview(df: pd.DataFrame, prefix: str):
    scope, rule, start, end, _ = local_filters_block(df, prefix)
    st.subheader("📊 KPI Overview")
    glossary_expander()

    if scope.empty:
        st.info("No data for selected filters.")
        return

    # current period
    now = kpis(scope)

    # prior period (same length)
    span = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
    prior_start = pd.to_datetime(start) - pd.Timedelta(days=span)
    prior_end = pd.to_datetime(start) - pd.Timedelta(days=1)
    prior_scope = df[(df["Date"]>=prior_start)&(df["Date"]<=prior_end)]
    prev = kpis(prior_scope) if not prior_scope.empty else dict(total_sales=0,total_qty=0,tx=0,spend_per_basket=0,asp=0)

    def pct_delta(a,b):
        if b == 0: return None
        return (a-b)/b*100

    c1,c2,c3,c4,c5 = st.columns(5)
    with c1: kpi_tile("Total Sales", f"${now['total_sales']:,.0f}", pct_delta(now['total_sales'], prev['total_sales']), GLOSSARY["Total_Sale"])
    with c2: kpi_tile("Transactions", f"{now['tx']:,}", pct_delta(now['tx'], prev['tx']), GLOSSARY["Transactions"])
    with c3: kpi_tile("Units", f"{int(now['total_qty']):,}", pct_delta(now['total_qty'], prev['total_qty']), GLOSSARY["Quantity"])
    with c4: kpi_tile("Spend/Basket", f"${now['spend_per_basket']:,.2f}", pct_delta(now['spend_per_basket'], prev['spend_per_basket']), GLOSSARY["Spend per Basket"])
    with c5: kpi_tile("ASP", f"${now['asp']:,.2f}", pct_delta(now['asp'], prev['asp']), GLOSSARY["ASP"])

def view_kpi_trends(df: pd.DataFrame, prefix: str):
    scope, rule, *_ = local_filters_block(df, prefix)
    st.subheader("📈 KPI Trends")
    glossary_expander()
    if scope.empty:
        st.info("No data for selected filters.")
        return

    metric = st.selectbox("Metric", ["Total_Sale","Quantity","Spend per Basket","Transactions"], index=0, key=f"{prefix}_trend_metric")
    # auto pick a grouping if any filter has multiple selections
    group_dim = None
    for dim, sel in {
        "Store_ID": st.session_state.get(f"{prefix}_stores", []),
        "Category": st.session_state.get(f"{prefix}_cats", []),
        "Brand_Name": st.session_state.get(f"{prefix}_brands", []),
        "Item": st.session_state.get(f"{prefix}_items", []),
    }.items():
        if isinstance(sel, list) and len(sel) > 1:
            group_dim = dim; break

    t = trends(scope, rule, group_dim)
    if t.empty or metric not in t.columns or t[metric].dropna().empty:
        st.info("No trend data for current filters and metric.")
        return

    fig = px.line(t, x="Date", y=metric, color=group_dim, title=f"{metric} Over Time" + (f" by {group_dim}" if group_dim else ""))
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Aggregate forecast only (not split)
    if (group_dim is None) and metric in ["Total_Sale","Quantity"]:
        steps,_,_, horizon = _forecast_meta(rule)
        show = st.checkbox(f"Show {metric} forecast ({horizon})", value=True, key=f"{prefix}_fc_show")
        if show:
            fc = forecast_conservative(scope, rule, metric)
            fig2 = go.Figure()
            # history
            s_hist = _series(scope, rule, metric)
            fig2.add_trace(go.Scatter(x=s_hist.index, y=s_hist.values, mode="lines", name="History"))
            # forecast + band
            fig2.add_trace(go.Scatter(x=fc["Date"], y=fc["yhat"], mode="lines", name="Forecast", line=dict(dash="dash")))
            fig2.add_traces([
                go.Scatter(x=pd.concat([fc["Date"], fc["Date"][::-1]]),
                           y=pd.concat([fc["yhat_upper"], fc["yhat_lower"][::-1]]),
                           fill="toself", fillcolor="rgba(99,110,250,0.15)",
                           line=dict(color="rgba(255,255,255,0)"),
                           hoverinfo="skip", name="95% interval")
            ])
            fig2.update_layout(title=f"{metric} Forecast — {horizon}", hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(f"**Forecast summary** — projected {metric.replace('_',' ').title()}: **{fc['yhat'].sum():,.0f}** "
                       f"(95% CI: {fc['yhat_lower'].sum():,.0f} – {fc['yhat_upper'].sum():,.0f})")

def view_topn(df: pd.DataFrame, prefix: str):
    scope, *_ = local_filters_block(df, prefix)
    st.subheader("🏆 Top-N Views")
    glossary_expander()
    if scope.empty:
        st.info("No data for selected filters.")
        return

    dims = [d for d in ["Category","Brand_Name","Store_ID","Item"] if d in scope.columns]
    c1, c2 = st.columns([1,1])
    with c1:
        dim = st.selectbox("Dimension", dims, index=0, key=f"{prefix}_top_dim")
    with c2:
        n = st.slider("N", 5, 30, 10, key=f"{prefix}_top_n")

    top_df = (scope.groupby(dim)["Total_Sale"].sum().sort_values(ascending=False).head(n).reset_index())
    fig = px.bar(top_df, x=dim, y="Total_Sale", text_auto=".2s", title=f"Top {n} {dim} by Total Sales")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        top_df.rename(columns={"Total_Sale":"Total Sales"}),
        use_container_width=True, hide_index=True,
        column_config={"Total Sales": st.column_config.NumberColumn("Total Sales", format="$%.0f", help=GLOSSARY["Total_Sale"])}
    )

def view_affinity(df: pd.DataFrame, prefix: str):
    scope, *_ = local_filters_block(df, prefix)
    st.subheader("🧺 Targeted Basket Affinity")
    st.markdown("Higher **Lift** and **Confidence** indicate stronger historical co-purchase. Evaluate with business context.")
    glossary_expander()

    if scope.empty:
        st.info("No data for selected filters.")
        return

    grains = [g for g in ["Item","Brand_Name","Category"] if g in scope.columns]
    gcol = st.radio("Granularity", grains, horizontal=True, index=0, key=f"{prefix}_aff_grain")
    universe = sorted(scope[gcol].astype(str).unique().tolist())
    target = st.selectbox("Target", universe, index=0 if universe else None, key=f"{prefix}_aff_target")
    if not target:
        st.info("Select a target.")
        return

    rules = affinity_rules(scope, gcol)
    if rules.empty:
        st.info("No co-basket pairs found.")
        return

    out = rules[rules["Antecedent"]==str(target)].copy()
    if out.empty:
        st.info(f"No associations for {target}.")
        return

    out["Associated Item"] = out["Consequent"]
    out["Confidence (%)"] = (out["Confidence (A->B)"]*100).round(2)
    out = (out.rename(columns={"Lift (A,B)":"Lift",
                               "Total_CoBasket_Sales_Value":"Total Co-Basket Sales",
                               "Avg_CoBasket_Spend":"Avg Co-Basket Spend",
                               "Total_Ante_Qty":f"Total Qty of {target}",
                               "Avg_Ante_Qty":f"Avg Qty of {target}"})
              [["Associated Item","Lift","Confidence (%)","Total Co-Baskets",
                f"Total Qty of {target}", f"Avg Qty of {target}",
                "Total Co-Basket Sales","Avg Co-Basket Spend"]])

    out = out.sort_values(["Lift","Confidence (%)"], ascending=[False,False]).reset_index(drop=True)
    out.insert(0,"Rank", range(1,1+len(out)))

    colconf = {
        "Rank": st.column_config.NumberColumn("Rank", format="%d"),
        "Associated Item": st.column_config.TextColumn("Associated"),
        "Lift": st.column_config.NumberColumn("Lift", format="%.2f", help=GLOSSARY["Lift"]),
        "Confidence (%)": st.column_config.NumberColumn("Confidence (%)", format="%.2f", help=GLOSSARY["Confidence (%)"]),
        "Total Co-Baskets": st.column_config.NumberColumn("Total Co-Baskets", format="%d", help="Count of baskets where both items co-occur."),
        f"Total Qty of {target}": st.column_config.NumberColumn(f"Total Qty of {target}", format="%d"),
        f"Avg Qty of {target}": st.column_config.NumberColumn(f"Avg Qty of {target}", format="%.2f"),
        "Total Co-Basket Sales": st.column_config.NumberColumn("Total Co-Basket Sales", format="$%.0f", help=GLOSSARY["Avg Co-Basket Spend"]),
        "Avg Co-Basket Spend": st.column_config.NumberColumn("Avg Co-Basket Spend", format="$%.2f", help=GLOSSARY["Avg Co-Basket Spend"]),
    }
    st.dataframe(out, hide_index=True, use_container_width=True, column_config=colconf)

def view_price_ladder(df: pd.DataFrame, prefix: str):
    scope, *_ = local_filters_block(df, prefix)
    st.subheader("💲 Price Ladder")
    glossary_expander()
    if scope.empty:
        st.info("No data for selected filters.")
        return

    if "Unit_Price" not in scope.columns:
        st.info("No unit price data present.")
        return

    levels = [l for l in ["Item","Brand_Name","Category"] if l in scope.columns]
    c1, c2 = st.columns([1,1])
    with c1: lvl = st.selectbox("Level", levels, index=0, key=f"{prefix}_pl_level")
    with c2: sort_by = st.selectbox("Sort by", ["Median Price","Average Price","Count"], index=0, key=f"{prefix}_pl_sort")

    agg = (scope.groupby(lvl)
                .agg(Avg_Price=("Unit_Price","mean"),
                     Median_Price=("Unit_Price","median"),
                     Count=("Unit_Price","size"))
                .reset_index())

    sort_col = {"Median Price":"Median_Price","Average Price":"Avg_Price","Count":"Count"}[sort_by]
    agg = agg.sort_values(sort_col, ascending=False)

    fig = px.bar(agg, x=lvl, y="Median_Price", text_auto=".2f", title=f"Median Price by {lvl}",
                 hover_data={"Avg_Price":":.2f","Median_Price":":.2f","Count":":,"})
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        agg.rename(columns={"Avg_Price":"Average Price","Median_Price":"Median Price"}),
        use_container_width=True, hide_index=True,
        column_config={
            lvl: st.column_config.TextColumn(lvl.replace("_"," ")),
            "Median Price": st.column_config.NumberColumn("Median Price", format="$%.2f", help=GLOSSARY["Median_Unit_Price"]),
            "Average Price": st.column_config.NumberColumn("Average Price", format="$%.2f"),
            "Count": st.column_config.NumberColumn("Observations", format="%d")
        }
    )

def view_store_map(df: pd.DataFrame, prefix: str):
    scope, *_ = local_filters_block(df, prefix)
    st.subheader("🗺️ Store Map")
    st.caption("Size = Total Sales | Color = ASP or Spend/Basket. Hover markers for full KPIs.")
    glossary_expander()

    need = {"Store_ID","Store_Latitude","Store_Longitude"}
    if (scope.empty) or (not need.issubset(scope.columns)):
        st.info("Map requires Store_ID, Store_Latitude, Store_Longitude and some data.")
        return

    m = (scope.groupby("Store_ID", as_index=False)
               .agg(Total_Sale=("Total_Sale","sum"),
                    Quantity=("Quantity","sum"),
                    Transactions=("Transaction_ID","nunique")))
    m["Spend_per_Basket"] = np.where(m["Transactions"]>0, m["Total_Sale"]/m["Transactions"], 0.0)
    m["ASP"] = np.where(m["Quantity"]>0, m["Total_Sale"]/m["Quantity"], 0.0)

    loc_cols = [c for c in ["Store_ID","Store_City","Store_State","Store_Latitude","Store_Longitude"] if c in scope.columns]
    locs = scope[loc_cols].drop_duplicates(subset=["Store_ID"])
    store = m.merge(locs, on="Store_ID", how="left")
    if {"Store_City","Store_State"}.issubset(store.columns):
        store["Store_Label"] = store["Store_ID"].astype(str)+" — "+store["Store_City"].astype(str)+", "+store["Store_State"].astype(str)
    else:
        store["Store_Label"] = store["Store_ID"].astype(str)

    c1,c2 = st.columns([1,1])
    with c1:
        size_metric = st.selectbox("Bubble Size", ["Total_Sale","Transactions","Quantity"], index=0, key=f"{prefix}_map_size")
    with c2:
        color_metric = st.selectbox("Bubble Color", ["Total_Sale","Spend_per_Basket","ASP","Transactions","Quantity"], index=1, key=f"{prefix}_map_color")

    fig = px.scatter_mapbox(
        store, lat="Store_Latitude", lon="Store_Longitude",
        size=size_metric, color=color_metric, size_max=28,
        zoom=3, center={"lat":39.5,"lon":-98.35}, mapbox_style="open-street-map",
        hover_name="Store_Label",
        custom_data=np.stack([
            store["Total_Sale"].values, store["Transactions"].values, store["Quantity"].values,
            store["Spend_per_Basket"].values, store["ASP"].values
        ], axis=-1)
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>" +
            "Total Sales: $%{customdata[0]:,.0f}<br>" +
            "Transactions: %{customdata[1]:,}<br>" +
            "Units: %{customdata[2]:,}<br>" +
            "Spend/Basket: $%{customdata[3]:,.2f}<br>" +
            "ASP: $%{customdata[4]:,.2f}<br>" +
            "<extra></extra>"
        )
    )
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=560)
    st.plotly_chart(fig, use_container_width=True)

def view_assortment_space(df: pd.DataFrame, prefix: str):
    scope, rule, start, end, _ = local_filters_block(df, prefix)
    st.subheader("📦 Assortment & Space Optimization")
    st.markdown(
        """
        *How to use:* Start with **SKU Productivity** to see velocity & distribution;  
        use **Rationalization** to flag low-velocity, low-share, or price-tier duplicates;  
        scan **Opportunity Map** for over/under-index by geo; finish with **New Item Tracker** vs benchmarks.
        """
    )
    glossary_expander()

    if scope.empty:
        st.info("No data for selected filters.")
        return

    # --- Local helpers
    def weeks_in_range(sdf):
        try:
            return max(int(sdf.set_index("Date").resample("W-SUN")["Total_Sale"].sum().shape[0]),1)
        except Exception:
            span=(pd.to_datetime(end)-pd.to_datetime(start)).days
            return max(int(np.ceil(span/7.0)),1)

    n_weeks = weeks_in_range(scope)
    price_col = "Unit_Price" if "Unit_Price" in scope.columns else None

    # A) Productivity
    st.markdown("### A) SKU Productivity")
    dim_opts = [c for c in ["Item","Brand_Name"] if c in scope.columns and scope[c].notna().any()]
    prod_dim = st.selectbox("Granularity", dim_opts, index=0, key=f"{prefix}_prod_dim")
    show_price = st.checkbox("Show median Unit Price", value=True if price_col else False, disabled=not price_col, key=f"{prefix}_prod_price")

    sku_store = (scope.groupby([prod_dim,"Store_ID"], dropna=False)["Total_Sale"].sum().reset_index())
    active_stores = sku_store.groupby(prod_dim)["Store_ID"].nunique().rename("Active_Stores")
    if "Category" not in scope.columns:
        st.info("Category missing.")
        return

    sku_sales = scope.groupby([prod_dim,"Category"], dropna=False)["Total_Sale"].sum().rename("SKU_Sales")
    cat_sales = scope.groupby("Category", dropna=False)["Total_Sale"].sum().rename("Category_Sales")

    prod = sku_sales.reset_index().merge(active_stores, on=prod_dim, how="left")
    prod["Active_Stores"] = prod["Active_Stores"].fillna(0).astype(int)
    prod["Weeks"] = n_weeks
    prod["Velocity_$/SKU/Store/Week"] = np.where(
        (prod["Active_Stores"]>0)&(prod["Weeks"]>0),
        prod["SKU_Sales"]/(prod["Active_Stores"]*prod["Weeks"]), 0.0
    )
    prod = prod.merge(cat_sales.reset_index(), on="Category", how="left")
    prod["% of Category Sales"] = np.where(prod["Category_Sales"]>0, 100*prod["SKU_Sales"]/prod["Category_Sales"], 0.0)
    if not prod.empty:
        prod["Velocity_Rank_in_Category"] = (
            prod.groupby("Category")["Velocity_$/SKU/Store/Week"].rank(ascending=False, method="dense").astype(int)
        )

    if price_col and show_price:
        med_price = scope.groupby(prod_dim)[price_col].median().rename("Median_Unit_Price")
        prod = prod.merge(med_price.reset_index(), on=prod_dim, how="left")

    # attach complementary label
    if prod_dim=="Item" and "Brand_Name" in scope.columns:
        lab = (scope.groupby(["Item","Brand_Name"])["Total_Sale"].sum().reset_index()
                   .sort_values(["Item","Total_Sale"], ascending=[True,False])
                   .drop_duplicates(subset=["Item"])[["Item","Brand_Name"]])
        prod = prod.merge(lab, on="Item", how="left")
    elif prod_dim=="Brand_Name" and "Item" in scope.columns:
        lab = (scope.groupby(["Brand_Name","Item"])["Total_Sale"].sum().reset_index()
                   .sort_values(["Brand_Name","Total_Sale"], ascending=[True,False])
                   .drop_duplicates(subset=["Brand_Name"])[["Brand_Name","Item"]])
        prod = prod.merge(lab, on="Brand_Name", how="left")

    # show
    order_cols = [c for c in [prod_dim, "Brand_Name" if prod_dim=="Item" else "Item", "Category",
                              "Active_Stores","Weeks","Velocity_$/SKU/Store/Week","SKU_Sales",
                              "Category_Sales","% of Category Sales","Velocity_Rank_in_Category",
                              "Median_Unit_Price"] if c in prod.columns]
    prod_view = prod[order_cols].sort_values(["Category","Velocity_$/SKU/Store/Week"], ascending=[True,False]).reset_index(drop=True)

    colcfg = {
        prod_dim: st.column_config.TextColumn(prod_dim.replace("_"," "), help="SKU grain used in this view."),
        "Brand_Name": st.column_config.TextColumn("Brand", help="Brand most associated with the item."),
        "Item": st.column_config.TextColumn("Item", help="Top-selling item within brand."),
        "Category": st.column_config.TextColumn("Category", help="Product category."),
        "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d", help="Stores where SKU sold ≥1 time."),
        "Weeks": st.column_config.NumberColumn("Weeks", format="%d", help="Anchored weekly buckets in range."),
        "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f", help=GLOSSARY["Velocity_$/SKU/Store/Week"]),
        "SKU_Sales": st.column_config.NumberColumn("SKU Sales", format="$%.0f", help="Sales of this SKU in range."),
        "Category_Sales": st.column_config.NumberColumn("Category Sales", format="$%.0f", help="Total category sales in range."),
        "% of Category Sales": st.column_config.NumberColumn("% of Category Sales", format="%.2f", help=GLOSSARY["% of Category Sales"]),
        "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d", help=GLOSSARY["Velocity_Rank_in_Category"]),
        "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f", help=GLOSSARY["Median_Unit_Price"]),
    }
    st.dataframe(prod_view, hide_index=True, use_container_width=True, column_config=colcfg)

    st.divider()

    # B) Rationalization
    st.markdown("### B) SKU Rationalization")
    st.caption("Flagging rules: Low Velocity (percentile), Low Share (min %), Similar Price-Tier Duplicate within Brand+Category.")
    p1,p2,p3,p4 = st.columns([1,1,1,1])
    with p1:
        perc = st.slider("Low Velocity percentile", 5, 50, 25, step=5, key=f"{prefix}_rat_pct")
    with p2:
        share_min = st.number_input("Min Category Share (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.5, key=f"{prefix}_rat_share")
    with p3:
        price_band = st.slider("Similar Price Tier (±%)", 1, 30, 10, step=1, key=f"{prefix}_rat_band")
    with p4:
        only_flagged = st.checkbox("Show only flagged", value=True, key=f"{prefix}_rat_only")

    rat = prod.copy()
    rat["Low_Velocity_Flag"] = False
    for cat, grp in rat.groupby("Category"):
        cutoff = np.percentile(grp["Velocity_$/SKU/Store/Week"], perc) if len(grp) else 0.0
        idx = rat["Category"]==cat
        rat.loc[idx,"Low_Velocity_Flag"] = rat.loc[idx,"Velocity_$/SKU/Store/Week"] <= cutoff

    rat["Low_Share_Flag"] = rat["% of Category Sales"] < share_min

    if price_col and "Median_Unit_Price" in rat.columns and "Brand_Name" in rat.columns:
        base = rat.copy()
        base["Group"] = base["Brand_Name"].astype(str)+" | "+base["Category"].astype(str)
        def _mark(g):
            med = g["Median_Unit_Price"].median()
            lo, hi = med*(1-price_band/100), med*(1+price_band/100)
            g["Similar_Price_Tier"] = g["Median_Unit_Price"].between(lo,hi)
            return g
        base = base.groupby("Group", group_keys=False).apply(_mark)
        grp_size = base.groupby("Group")["SKU_Sales"].transform("size")
        base["Redundant_Candidate"] = (grp_size>1) & base["Similar_Price_Tier"] & base["Low_Share_Flag"]
        rat = base
    else:
        rat["Similar_Price_Tier"] = False
        rat["Redundant_Candidate"] = False

    rat["Rationalize?"] = rat["Low_Velocity_Flag"] | rat["Low_Share_Flag"] | rat["Redundant_Candidate"]
    view = rat[rat["Rationalize?"]] if only_flagged else rat

    keep = [c for c in [
        prod_dim, "Brand_Name","Item","Category","Median_Unit_Price",
        "Velocity_$/SKU/Store/Week","% of Category Sales","Velocity_Rank_in_Category",
        "Low_Velocity_Flag","Low_Share_Flag","Similar_Price_Tier","Redundant_Candidate","Rationalize?"
    ] if c in view.columns]
    view = view[keep].loc[:, ~pd.Index(keep).duplicated()].sort_values(
        ["Rationalize?","Category","Velocity_$/SKU/Store/Week"], ascending=[False,True,True]
    )

    colcfg2 = {
        prod_dim: st.column_config.TextColumn(prod_dim, help="SKU grain used here."),
        "Brand_Name": st.column_config.TextColumn("Brand"),
        "Item": st.column_config.TextColumn("Item"),
        "Category": st.column_config.TextColumn("Category"),
        "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f", help=GLOSSARY["Median_Unit_Price"]),
        "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f", help=GLOSSARY["Velocity_$/SKU/Store/Week"]),
        "% of Category Sales": st.column_config.NumberColumn("% of Category Sales", format="%.2f", help=GLOSSARY["% of Category Sales"]),
        "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d", help=GLOSSARY["Velocity_Rank_in_Category"]),
        "Low_Velocity_Flag": st.column_config.CheckboxColumn("Low Velocity"),
        "Low_Share_Flag": st.column_config.CheckboxColumn("Low Share"),
        "Similar_Price_Tier": st.column_config.CheckboxColumn("Similar Price Tier"),
        "Redundant_Candidate": st.column_config.CheckboxColumn("Price-Tier Duplicate"),
        "Rationalize?": st.column_config.CheckboxColumn("Flag"),
    }
    st.dataframe(view, hide_index=True, use_container_width=True, column_config=colcfg2)

    st.divider()

    # C) Opportunity Map
    st.markdown("### C) Assortment Opportunity Map")
    geo_cols = [c for c in ["Store_State","Store_City","Store_ID"] if c in scope.columns]
    if not geo_cols:
        st.info("No geography columns found.")
    else:
        geo = st.selectbox("Geo", geo_cols, index=0, key=f"{prefix}_opp_geo")
        dim = st.selectbox("Analyze by", [c for c in ["Item","Brand_Name"] if c in scope.columns], index=0, key=f"{prefix}_opp_dim")
        topn = st.slider("Top-N per geo", 3, 15, 5, key=f"{prefix}_opp_n")

        overall = scope.groupby(dim)["Total_Sale"].sum().rename("Overall_Sales")
        overall_total = overall.sum()
        overall_share = (overall/overall_total).rename("Overall_Share").reset_index()

        geo_sales = scope.groupby([geo, dim])["Total_Sale"].sum().rename("Geo_Sales").reset_index()
        geo_tot = scope.groupby(geo)["Total_Sale"].sum().rename("Geo_Total").reset_index()

        ge = geo_sales.merge(geo_tot, on=geo, how="left")
        ge["Geo_Share"] = np.where(ge["Geo_Total"]>0, ge["Geo_Sales"]/ge["Geo_Total"], 0.0)
        ge = ge.merge(overall_share, on=dim, how="left")
        ge["Index_vs_Overall"] = np.where(ge["Overall_Share"]>0, 100*ge["Geo_Share"]/ge["Overall_Share"], 0.0)

        top = (ge.sort_values("Index_vs_Overall", ascending=False)
                 .groupby(geo).head(topn).reset_index(drop=True))

        colcfg3 = {
            geo: st.column_config.TextColumn(geo.replace("_"," "), help="Selected geography level."),
            dim: st.column_config.TextColumn(dim.replace("_"," "), help="Item or Brand."),
            "Index_vs_Overall": st.column_config.NumberColumn("Index vs Overall", format="%.1f", help=GLOSSARY["Index_vs_Overall"]),
            "Geo_Share": st.column_config.NumberColumn("Geo Share", format="%.2%", help=GLOSSARY["Geo_Share"]),
            "Overall_Share": st.column_config.NumberColumn("Overall Share", format="%.2%", help=GLOSSARY["Overall_Share"]),
            "Geo_Sales": st.column_config.NumberColumn("Geo Sales", format="$%.0f", help="Sales for this geo and item/brand."),
        }
        st.dataframe(top[[geo, dim, "Index_vs_Overall","Geo_Share","Overall_Share","Geo_Sales"]],
                     hide_index=True, use_container_width=True, column_config=colcfg3)

    st.divider()

    # D) New Item Tracker
    st.markdown("### D) New Item Tracker")
    if "Item" not in scope.columns:
        st.info("Item column required.")
        return

    new_window = st.select_slider("Define 'New' as first sold within last (days)", options=[30,60,90,120,180], value=90, key=f"{prefix}_new_days")
    tiers = st.selectbox("Price Tiers (within Category)", options=[3,4,5], index=0, key=f"{prefix}_new_tiers")

    # new items based on RAW (first ever sale)
    all_first = RAW.groupby("Item")["Date"].min().rename("First_Sale_Date").reset_index()
    cutoff = pd.to_datetime(end) - pd.Timedelta(days=int(new_window))
    all_first["Is_New"] = all_first["First_Sale_Date"] >= cutoff

    f = scope.merge(all_first, on="Item", how="left")
    new_items = f[f["Is_New"]==True].copy()
    if new_items.empty:
        st.info(f"No items first sold within last {new_window} days.")
        return

    price_meta = None
    if price_col:
        med_item = f.groupby("Item")[price_col].median().rename("Median_Unit_Price")
        cat_item = f.groupby("Item")["Category"].agg(lambda x: x.mode().iloc[0] if len(x) else None).rename("Category_Assigned")
        price_meta = pd.concat([med_item, cat_item], axis=1).reset_index()

        def _tier(g):
            g = g.sort_values("Median_Unit_Price")
            g["Price_Tier"] = pd.qcut(g["Median_Unit_Price"].rank(method="first"), q=tiers, labels=[f"T{i+1}" for i in range(tiers)])
            return g
        ref = price_meta.groupby("Category_Assigned", group_keys=True).apply(_tier).reset_index(drop=True)
        new_items = new_items.merge(price_meta, on="Item", how="left").merge(ref[["Item","Price_Tier"]], on="Item", how="left")
    else:
        new_items["Median_Unit_Price"] = np.nan
        new_items["Category_Assigned"] = new_items["Category"]
        new_items["Price_Tier"] = "NA"

    perf = (f.groupby(["Item","Category"], dropna=False)["Total_Sale"].sum().rename("Item_Sales").reset_index()
              .merge(f.groupby("Item")["Store_ID"].nunique().rename("Active_Stores").reset_index(), on="Item", how="left"))
    perf["Weeks"] = n_weeks
    perf["Velocity_$/SKU/Store/Week"] = np.where((perf["Active_Stores"]>0)&(perf["Weeks"]>0),
                                                 perf["Item_Sales"]/(perf["Active_Stores"]*perf["Weeks"]), 0.0)

    new_perf = new_items[["Item","Category","Price_Tier"]].drop_duplicates().merge(perf, on=["Item","Category"], how="left")

    bench = f.merge(all_first[all_first["Is_New"]==False][["Item"]].assign(Not_New=True), on="Item", how="inner")
    if price_meta is not None: bench = bench.merge(price_meta, on="Item", how="left")
    else:
        bench["Category_Assigned"] = bench["Category"]; bench["Price_Tier"] = "NA"

    bperf = (bench.groupby(["Item","Category_Assigned","Price_Tier"], dropna=False)["Total_Sale"].sum().rename("Item_Sales").reset_index()
                 .merge(bench.groupby("Item")["Store_ID"].nunique().rename("Active_Stores").reset_index(), on="Item", how="left"))
    bperf["Weeks"] = n_weeks
    bperf["Velocity_$/SKU/Store/Week"] = np.where((bperf["Active_Stores"]>0)&(bperf["Weeks"]>0),
                                                 bperf["Item_Sales"]/(bperf["Active_Stores"]*bperf["Weeks"]), 0.0)

    if "Price_Tier" in bperf.columns and bperf["Price_Tier"].notna().any():
        bavg = bperf.groupby(["Category_Assigned","Price_Tier"])["Velocity_$/SKU/Store/Week"].mean().rename("Benchmark_Velocity").reset_index()
        new_perf = new_perf.merge(bavg, left_on=["Category","Price_Tier"], right_on=["Category_Assigned","Price_Tier"], how="left").drop(columns=["Category_Assigned"])
    else:
        bavg = bperf.groupby(["Category_Assigned"])["Velocity_$/SKU/Store/Week"].mean().rename("Benchmark_Velocity").reset_index()
        new_perf = new_perf.merge(bavg, left_on=["Category"], right_on=["Category_Assigned"], how="left").drop(columns=["Category_Assigned"])

    new_perf["Benchmark_Velocity"] = new_perf["Benchmark_Velocity"].fillna(0.0)
    new_perf["Velocity_vs_Benchmark_%"] = np.where(new_perf["Benchmark_Velocity"]>0,
                                                   100*new_perf["Velocity_$/SKU/Store/Week"]/new_perf["Benchmark_Velocity"], np.nan)
    if price_meta is not None:
        new_perf = new_perf.merge(price_meta[["Item","Median_Unit_Price"]], on="Item", how="left")

    show_cols = [c for c in ["Item","Category","Price_Tier","Median_Unit_Price","Active_Stores","Weeks",
                             "Velocity_$/SKU/Store/Week","Benchmark_Velocity","Velocity_vs_Benchmark_%"] if c in new_perf.columns]
    st.dataframe(
        new_perf[show_cols].sort_values(["Category","Velocity_vs_Benchmark_%"], ascending=[True,False]).reset_index(drop=True),
        hide_index=True, use_container_width=True,
        column_config={
            "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f", help=GLOSSARY["Median_Unit_Price"]),
            "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d"),
            "Weeks": st.column_config.NumberColumn("Weeks", format="%d"),
            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f", help=GLOSSARY["Velocity_$/SKU/Store/Week"]),
            "Benchmark_Velocity": st.column_config.NumberColumn("Benchmark Velocity", format="$%.2f"),
            "Velocity_vs_Benchmark_%": st.column_config.NumberColumn("Velocity vs Benchmark (%)", format="%.1f"),
        }
    )

# ----------------------------
# Main: Tabs based on Department
# ----------------------------
st.markdown("<h2 style='margin-top:0'>C-Store KPI Insights Suite</h2>", unsafe_allow_html=True)

visible_tabs = DEPT_TABS.get(department, DEPT_TABS["All Departments"])

# Build tabs dynamically in requested order
tab_objs = st.tabs(visible_tabs)

tab_map = {name: obj for name, obj in zip(visible_tabs, tab_objs)}

if "📊 KPI Overview" in tab_map:
    with tab_map["📊 KPI Overview"]:
        view_kpi_overview(RAW, prefix="kpi")

if "📈 KPI Trends" in tab_map:
    with tab_map["📈 KPI Trends"]:
        view_kpi_trends(RAW, prefix="trend")

if "🏆 Top-N Views" in tab_map:
    with tab_map["🏆 Top-N Views"]:
        view_topn(RAW, prefix="topn")

if "🧺 Basket Affinity" in tab_map:
    with tab_map["🧺 Basket Affinity"]:
        view_affinity(RAW, prefix="aff")

if "💲 Price Ladder" in tab_map:
    with tab_map["💲 Price Ladder"]:
        view_price_ladder(RAW, prefix="pl")

if "🗺️ Store Map" in tab_map:
    with tab_map["🗺️ Store Map"]:
        view_store_map(RAW, prefix="map")

if "📦 Assortment & Space Optimization" in tab_map:
    with tab_map["📦 Assortment & Space Optimization"]:
        view_assortment_space(RAW, prefix="aso")
