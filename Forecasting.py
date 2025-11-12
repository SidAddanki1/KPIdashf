# C-Store KPI Dashboard (Fast, MDL-enabled, Department Global + Per-Tab Local Filters)
# - Single compact global Department filter (sidebar)
# - All other filters are local per tab
# - Upload CSV removed (reads cstorereal.csv)
# - Materialized Data Layer (MDL) to precompute heavy tables once per frequency
# - Tabs: KPI Overview, KPI Trends (with conservative forecast), Top-N, Basket Affinity,
#         Price Ladder, Store Map, Assortment & Space Optimization (Productivity, Rationalization, Opportunity, New Items)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from collections import Counter
from itertools import combinations
from typing import Dict, Tuple, List

# Forecasting
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="C-Store KPI Dashboard", layout="wide", initial_sidebar_state="collapsed")
pio.templates.default = "plotly_white"

# ---------------------------------
# Helpers (generic)
# ---------------------------------
def to_num_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

@st.cache_data(show_spinner=False)
def load_and_normalize(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
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

    required_min = {"Date", "Transaction_ID", "Store_ID", "Category", "Item"}
    missing = [c for c in required_min if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Transaction_ID"])

    for c in ["Transaction_ID", "Store_ID", "Item", "Brand_Name", "Category", "Payment_Method"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    if "Unit_Price" in df.columns:
        df["Unit_Price"] = to_num_clean(df["Unit_Price"]).fillna(0.0)
    if "Quantity" in df.columns:
        df["Quantity"] = to_num_clean(df["Quantity"]).fillna(0.0)
    if "Total_Sale" in df.columns:
        df["Total_Sale"] = to_num_clean(df["Total_Sale"]).fillna(0.0)

    if {"Quantity", "Unit_Price"} <= set(df.columns):
        if "Total_Sale" not in df.columns:
            df["Total_Sale"] = (df["Quantity"] * df["Unit_Price"]).round(2)
        else:
            needs_impute = (df["Total_Sale"] == 0) | (df["Total_Sale"].isna())
            df.loc[needs_impute, "Total_Sale"] = (df.loc[needs_impute, "Quantity"] * df["Unit_Price"]).round(2)
    return df

# ---------------------------------
# Materialized Data Layer (precompute heavy tables ONCE per frequency)
# ---------------------------------
@st.cache_data(show_spinner="Building materialized tables…", ttl=3600)
def build_mdl(raw_df: pd.DataFrame, rule: str):
    """
    Precompute heavy aggregates across the FULL dataset for the given re-sample rule.
    Local per-tab filters (date/store/category/etc.) will slice these tables quickly.
    """
    df = raw_df.copy()

    # Resample helper
    def _anchored(dfX, by=None):
        groupers = [pd.Grouper(key="Date", freq=rule)]
        if by is not None:
            groupers.append(by)
        g = (dfX
             .groupby(groupers, dropna=False)
             .agg(Total_Sale=("Total_Sale","sum"),
                  Quantity=("Quantity","sum"),
                  Transactions=("Transaction_ID","nunique"))
             .reset_index()
             .sort_values("Date"))
        g["Spend per Basket"] = np.where(g["Transactions"]>0, g["Total_Sale"]/g["Transactions"], 0.0)
        return g

    trends_agg     = _anchored(df, by=None)
    trends_by_store= _anchored(df, by="Store_ID") if "Store_ID" in df.columns else pd.DataFrame()
    trends_by_cat  = _anchored(df, by="Category") if "Category" in df.columns else pd.DataFrame()
    trends_by_brand= _anchored(df, by="Brand_Name") if "Brand_Name" in df.columns else pd.DataFrame()
    trends_by_item = _anchored(df, by="Item") if "Item" in df.columns else pd.DataFrame()

    # Price ladder bases
    def _price_stats(level):
        if "Unit_Price" not in df.columns or level not in df.columns:
            return pd.DataFrame()
        return (df.groupby(level)
                  .agg(Avg_Price=("Unit_Price","mean"),
                       Median_Price=("Unit_Price","median"),
                       Count=("Unit_Price","size"))
                  .reset_index())
    price_item  = _price_stats("Item")
    price_brand = _price_stats("Brand_Name")
    price_cat   = _price_stats("Category")

    # Store KPI base (for map)
    store_kpis = pd.DataFrame()
    if {"Store_ID","Total_Sale","Transaction_ID","Quantity"}.issubset(df.columns):
        store_kpis = (df.groupby("Store_ID", as_index=False)
                        .agg(Total_Sale=("Total_Sale","sum"),
                             Quantity=("Quantity","sum"),
                             Transactions=("Transaction_ID","nunique")))
        store_kpis["Spend_per_Basket"] = np.where(store_kpis["Transactions"]>0,
                                                  store_kpis["Total_Sale"]/store_kpis["Transactions"], 0.0)
        store_kpis["ASP"] = np.where(store_kpis["Quantity"]>0,
                                     store_kpis["Total_Sale"]/store_kpis["Quantity"], 0.0)
        meta_cols = [c for c in ["Store_City","Store_State","Store_Latitude","Store_Longitude"] if c in df.columns]
        if meta_cols:
            meta = df[["Store_ID"]+meta_cols].drop_duplicates("Store_ID")
            store_kpis = store_kpis.merge(meta, on="Store_ID", how="left")

    # Affinity (Item/Brand/Category)
    def _affinity_for(key_col: str):
        if key_col not in df.columns: return pd.DataFrame()
        scope = df.dropna(subset=["Transaction_ID", key_col]).copy()
        scope["Transaction_ID"] = scope["Transaction_ID"].astype(str)
        scope[key_col] = scope[key_col].astype(str)
        tx_count_total = scope["Transaction_ID"].nunique()
        if tx_count_total == 0: return pd.DataFrame()
        basket_sales = scope.groupby("Transaction_ID")["Total_Sale"].sum()
        tx_keys = scope.groupby("Transaction_ID")[key_col].apply(lambda s: tuple(sorted(set(s))))
        item_counts, pair_counts = Counter(), Counter()
        for keys in tx_keys:
            for k in keys: item_counts[k] += 1
            for a,b in combinations(keys,2): pair_counts[tuple(sorted((a,b)))] += 1
        if not pair_counts: return pd.DataFrame()
        item_txids_map = scope.groupby(key_col)["Transaction_ID"].apply(set)
        def qty_in_txids(item, tx_ids):
            if "Quantity" not in scope.columns or not tx_ids: return 0.0
            v = scope[(scope["Transaction_ID"].isin(tx_ids)) & (scope[key_col]==item)]
            return float(v["Quantity"].sum())
        rows=[]
        for (a,b), n_ab in pair_counts.items():
            count_a, count_b = item_counts[a], item_counts[b]
            support_ab = n_ab/tx_count_total
            lift = (support_ab/((count_a/tx_count_total)*(count_b/tx_count_total))) if (count_a and count_b) else 0.0
            co_tx = item_txids_map.get(a,set()) & item_txids_map.get(b,set())
            total_co_spend = float(basket_sales.loc[list(co_tx)].sum()) if co_tx else 0.0
            avg_co_spend = (total_co_spend/n_ab) if n_ab else 0.0
            qa = qty_in_txids(a,co_tx); qb = qty_in_txids(b,co_tx)
            rows.append({"Key":key_col,"Antecedent":a,"Consequent":b,"Total Co-Baskets":n_ab,
                         "Support (A,B)":support_ab, "Confidence (A->B)":(n_ab/count_a) if count_a else 0.0,
                         "Lift (A,B)":lift, "Total_Antecedent_Qty_in_CoBasket":qa,
                         "Avg_Antecedent_Qty_in_CoBasket":(qa/n_ab) if n_ab else 0.0,
                         "Total_CoBasket_Sales_Value":total_co_spend, "Avg_CoBasket_Spend":avg_co_spend})
            rows.append({"Key":key_col,"Antecedent":b,"Consequent":a,"Total Co-Baskets":n_ab,
                         "Support (A,B)":support_ab, "Confidence (A->B)":(n_ab/count_b) if count_b else 0.0,
                         "Lift (A,B)":lift, "Total_Antecedent_Qty_in_CoBasket":qb,
                         "Avg_Antecedent_Qty_in_CoBasket":(qb/n_ab) if n_ab else 0.0,
                         "Total_CoBasket_Sales_Value":total_co_spend, "Avg_CoBasket_Spend":avg_co_spend})
        out = pd.DataFrame(rows)
        if not out.empty:
            out = out.sort_values(["Lift (A,B)","Confidence (A->B)"], ascending=[False,False]).reset_index(drop=True)
        return out

    affinity_item  = _affinity_for("Item")
    affinity_brand = _affinity_for("Brand_Name")
    affinity_cat   = _affinity_for("Category")

    # SKU Productivity bases (Item/Brand)
    # Use full span for Active_Stores; n_weeks based on anchored weekly buckets over full data
    def _weeks_full():
        try:
            return max(int(df.set_index("Date").resample("W-SUN")["Total_Sale"].sum().shape[0]), 1)
        except Exception:
            return 1

    n_weeks = _weeks_full()
    sku_bases = {}
    for grain in ["Item","Brand_Name"]:
        if grain not in df.columns:
            sku_bases[grain] = pd.DataFrame(); continue
        sku_store = (df.groupby([grain,"Store_ID"], dropna=False)["Total_Sale"].sum().reset_index())
        active_stores = sku_store.groupby(grain)["Store_ID"].nunique().rename("Active_Stores")
        if "Category" not in df.columns:
            sku_bases[grain] = pd.DataFrame(); continue
        sku_sales = df.groupby([grain,"Category"], dropna=False)["Total_Sale"].sum().rename("SKU_Sales")
        cat_sales = df.groupby("Category", dropna=False)["Total_Sale"].sum().rename("Category_Sales")
        base = sku_sales.reset_index().merge(active_stores, on=grain, how="left")
        base["Active_Stores"] = base["Active_Stores"].fillna(0).astype(int)
        base["Weeks"] = n_weeks
        base["Velocity_$/SKU/Store/Week"] = np.where(
            (base["Active_Stores"]>0) & (base["Weeks"]>0),
            base["SKU_Sales"]/(base["Active_Stores"]*base["Weeks"]), 0.0
        )
        base = base.merge(cat_sales.reset_index(), on="Category", how="left")
        base["% of Category Sales"] = np.where(base["Category_Sales"]>0,
                                               100.0*base["SKU_Sales"]/base["Category_Sales"], 0.0)
        base["Velocity_Rank_in_Category"] = (base.groupby("Category")["Velocity_$/SKU/Store/Week"]
                                                  .rank(ascending=False, method="dense").astype(int))
        if "Unit_Price" in df.columns:
            med_price = df.groupby(grain)["Unit_Price"].median().rename("Median_Unit_Price").reset_index()
            base = base.merge(med_price, on=grain, how="left")
        if grain=="Item" and "Brand_Name" in df.columns:
            top_brand = (df.groupby(["Item","Brand_Name"])["Total_Sale"].sum().reset_index()
                           .sort_values(["Item","Total_Sale"], ascending=[True,False])
                           .drop_duplicates("Item")[["Item","Brand_Name"]])
            base = base.merge(top_brand, on="Item", how="left")
        if grain=="Brand_Name" and "Item" in df.columns:
            top_item = (df.groupby(["Brand_Name","Item"])["Total_Sale"].sum().reset_index()
                          .sort_values(["Brand_Name","Total_Sale"], ascending=[True,False])
                          .drop_duplicates("Brand_Name")[["Brand_Name","Item"]])
            base = base.merge(top_item, on="Brand_Name", how="left")
        sku_bases[grain] = base

    # First sale (for new item tracker)
    first_sale = df.groupby("Item")["Date"].min().rename("First_Sale_Date").reset_index() if "Item" in df.columns else pd.DataFrame()

    # Static Top bases (full span)
    def _sum_by(col):
        return (df.groupby(col)["Total_Sale"].sum().reset_index().sort_values("Total_Sale", ascending=False)) if col in df.columns else pd.DataFrame()
    top_category = _sum_by("Category")
    top_brand    = _sum_by("Brand_Name")
    top_store    = _sum_by("Store_ID")
    top_item     = _sum_by("Item")

    # Geo share base
    geo_base = {}
    for geo in ["Store_State","Store_City","Store_ID"]:
        geo_base[geo] = {}
        if geo not in df.columns: 
            continue
        for dim in ["Item","Brand_Name"]:
            if dim not in df.columns: 
                geo_base[geo][dim] = (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()); continue
            overall_sales = df.groupby(dim)["Total_Sale"].sum().rename("Overall_Sales")
            overall_total = overall_sales.sum()
            overall_share = (overall_sales/overall_total).rename("Overall_Share").reset_index()
            geo_sales = df.groupby([geo, dim])["Total_Sale"].sum().rename("Geo_Sales").reset_index()
            geo_totals = df.groupby(geo)["Total_Sale"].sum().rename("Geo_Total").reset_index()
            share = geo_sales.merge(geo_totals, on=geo, how="left")
            share["Geo_Share"] = np.where(share["Geo_Total"]>0, share["Geo_Sales"]/share["Geo_Total"], 0.0)
            share = share.merge(overall_share, on=dim, how="left")
            share["Index_vs_Overall"] = np.where(share["Overall_Share"]>0, 100.0*share["Geo_Share"]/share["Overall_Share"], 0.0)
            geo_base[geo][dim] = (overall_share, geo_totals, share)

    mdl = {
        "trends_agg": trends_agg,
        "trends_by_store": trends_by_store,
        "trends_by_cat": trends_by_cat,
        "trends_by_brand": trends_by_brand,
        "trends_by_item": trends_by_item,
        "price_item": price_item,
        "price_brand": price_brand,
        "price_cat": price_cat,
        "store_kpis": store_kpis,
        "affinity_item": affinity_item,
        "affinity_brand": affinity_brand,
        "affinity_cat": affinity_cat,
        "sku_item": sku_bases.get("Item", pd.DataFrame()),
        "sku_brand": sku_bases.get("Brand_Name", pd.DataFrame()),
        "first_sale": first_sale,
        "top_category": top_category,
        "top_brand": top_brand,
        "top_store": top_store,
        "top_item": top_item,
        "geo_base": geo_base,
    }
    return mdl

# ---------------------------------
# Forecast helpers (unchanged, works off pre-aggregated series)
# ---------------------------------
def _forecast_freq_meta(rule: str):
    if rule == "D":
        return 30, 7, "D", "next 30 days"
    if rule.startswith("W-"):
        return 4, 52, rule, "next 4 weeks"
    if rule == "MS":
        return 1, 12, "MS", "next month"
    return 30, 7, "D", "next 30 days"

def _fit_ets_safe(ts: pd.Series, rule: str, seasonal_periods: int):
    use_fixed_smoothing = rule.startswith("W-")
    use_season = seasonal_periods and (len(ts) >= 2 * seasonal_periods)
    try:
        if use_fixed_smoothing:
            trend_type = 'add'
            fit_params = {'smoothing_level': 0.1, 'smoothing_trend': 0.01, 'optimized': False}
        else:
            trend_type = None
            fit_params = {'optimized': True}
        model = ExponentialSmoothing(
            ts, trend=trend_type, damped_trend=False,
            seasonal="add" if use_season else None,
            seasonal_periods=seasonal_periods if use_season else None,
            initialization_method="estimated"
        ).fit(**fit_params)
        fitted = model.fittedvalues.reindex(ts.index)
        resid = (ts - fitted).to_numpy()
        resid_std = float(np.nanstd(resid))
        return model, resid_std
    except Exception:
        return None, None

def _seasonal_naive(ts: pd.Series, steps: int, seasonal_periods: int) -> np.ndarray:
    if seasonal_periods and len(ts) >= seasonal_periods:
        rep = np.tile(ts.iloc[-seasonal_periods:].to_numpy(), int(np.ceil(steps/seasonal_periods)))[:steps]
        return rep.astype(float)
    return np.full(steps, float(ts.iloc[-1]) if len(ts) else 0.0)

def _choose_model(ts: pd.Series, rule: str, seasonal_periods: int):
    n = len(ts)
    if n < max(12, seasonal_periods + 4):
        return "naive", None, None
    h = max(6, int(round(n * 0.12)))
    train, hold = ts.iloc[:-h], ts.iloc[-h:]
    ets_model, resid_std = _fit_ets_safe(train, rule, seasonal_periods)
    naive_fc = _seasonal_naive(train, h, seasonal_periods)
    ets_fc = ets_model.forecast(h).to_numpy() if ets_model is not None else None
    def smape(a, f):
        a = a.astype(float); f = f.astype(float)
        denom = (np.abs(a) + np.abs(f)); denom[denom == 0] = 1.0
        return float(np.mean(2.0 * np.abs(a - f) / denom))
    a = hold.to_numpy()
    naive_err = smape(a, naive_fc)
    ets_err = smape(a, ets_fc) if ets_fc is not None else np.inf
    return ("ets", ets_model, resid_std) if ets_err < naive_err else ("naive", None, None)

def forecast_from_series(ts: pd.Series, rule: str, alpha: float = 0.05) -> pd.DataFrame:
    steps, seasonal_periods, freq, _ = _forecast_freq_meta(rule)
    ts = ts.astype(float).clip(lower=0)
    model_type, ets_model, resid_std = _choose_model(ts, rule, seasonal_periods)
    if model_type == "ets" and ets_model is not None:
        mean_fc = ets_model.forecast(steps).to_numpy()
        z = 1.96 if alpha == 0.05 else 1.64
        band = (resid_std or 0.0) * z
        lower = np.maximum(0.0, mean_fc - band)
        upper = np.maximum(0.0, mean_fc + band)
    else:
        mean_fc = _seasonal_naive(ts, steps, seasonal_periods)
        lower = np.maximum(0.0, mean_fc * 0.9)
        upper = mean_fc * 1.1
    idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.DataFrame({"Date": idx, "yhat": mean_fc, "yhat_lower": lower, "yhat_upper": upper})

# ---------------------------------
# Data & Global (Department-only) Sidebar
# ---------------------------------
DATA_PATH = "cstorereal.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"CSV not found at: {os.path.abspath(DATA_PATH)}"); st.stop()
raw_df = load_and_normalize(DATA_PATH)

# Compact global sidebar (Department only)
with st.sidebar:
    st.markdown("### 🧭 Department")
    dept = st.selectbox(
        "Choose a department",
        [
            "Strategy & Finance",
            "Sales & Category",
            "Marketing & Insights",
            "Supply Chain & Ops",
            "Merchandising & Pricing",
            "Store & Field",
            "Executive"
        ],
        index=0,
        help="This controls which tabs are most relevant. All analytics compute from the same dataset."
    )
    with st.expander("About this dashboard", expanded=False):
        st.caption(
            "This app uses a **materialized data layer**—heavy calculations are precomputed once per frequency for fast interactions."
        )

# Department → recommended tabs (we still show all tabs; we’ll highlight relevance)
dept_tabs_map = {
    "Strategy & Finance": {"📊 KPI Overview", "📈 KPI Trends"},
    "Sales & Category": {"🏆 Top-N Views", "🧺 Basket Affinity", "📦 Assortment & Space Optimization"},
    "Marketing & Insights": {"📈 KPI Trends", "🏆 Top-N Views", "🧺 Basket Affinity"},
    "Supply Chain & Ops": {"🗺️ Store Map", "📊 KPI Overview"},
    "Merchandising & Pricing": {"💲 Price Ladder", "📦 Assortment & Space Optimization"},
    "Store & Field": {"🗺️ Store Map", "🏆 Top-N Views"},
    "Executive": {"📊 KPI Overview", "📈 KPI Trends", "🗺️ Store Map"},
}

# ---------------------------------
# Frequency control (shared) + Build MDL
# ---------------------------------
RULE_MAP = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}
freq_choice = st.radio(
    "Time Granularity",
    ["Daily","Weekly","Monthly"],
    index=1,
    horizontal=True,
    help="Changes the anchoring used in time-series tables & trends."
)
rule = RULE_MAP[freq_choice]
MDL = build_mdl(raw_df, rule)

# ---------------------------------
# Local Filter Builder (per tab)
# ---------------------------------
def local_filter_controls(key_prefix: str):
    """Render local filters and return a filter dict + date range."""
    cols = st.columns([1,1,1,1,1,1])
    min_d, max_d = raw_df["Date"].min().date(), raw_df["Date"].max().date()
    with cols[0]:
        start_d = st.date_input("Start", value=min_d, key=f"{key_prefix}_start")
    with cols[1]:
        end_d = st.date_input("End", value=max_d, key=f"{key_prefix}_end")
    with cols[2]:
        stores = sorted(raw_df["Store_ID"].unique().tolist())
        sel_stores = st.multiselect("Store", stores, key=f"{key_prefix}_stores")
    with cols[3]:
        cats = sorted(raw_df["Category"].unique().tolist())
        sel_cats = st.multiselect("Category", cats, key=f"{key_prefix}_cats")
    with cols[4]:
        brands = sorted(raw_df["Brand_Name"].unique().tolist()) if "Brand_Name" in raw_df.columns else []
        sel_brands = st.multiselect("Brand", brands, key=f"{key_prefix}_brands")
    with cols[5]:
        items = sorted(raw_df["Item"].unique().tolist())
        sel_items = st.multiselect("Item", items, key=f"{key_prefix}_items")

    pays = sorted(raw_df["Payment_Method"].unique().tolist()) if "Payment_Method" in raw_df.columns else []
    sel_pays = st.multiselect("Payment", pays, key=f"{key_prefix}_pays")

    filt = {
        "Store_ID": sel_stores,
        "Category": sel_cats,
        "Brand_Name": sel_brands,
        "Item": sel_items,
        "Payment_Method": sel_pays
    }
    return (pd.to_datetime(start_d), pd.to_datetime(end_d)), filt

def apply_local_filters_df(df: pd.DataFrame, date_range, filt: dict) -> pd.DataFrame:
    if df.empty:
        return df
    start, end = date_range
    view = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
    for col, vals in filt.items():
        if vals and col in view.columns:
            view = view[view[col].isin(vals)]
    return view

# ---------------------------------
# Displays (Tabs)
# ---------------------------------
def display_kpi_overview_tab():
    st.header("📊 KPI Overview" + (" — Focus" if "📊 KPI Overview" in dept_tabs_map.get(dept,set()) else ""))
    date_range, filt = local_filter_controls("kpi")
    view = apply_local_filters_df(raw_df, date_range, filt)

    if view.empty:
        st.info("No data for selected filters."); return

    total_sales = view["Total_Sale"].sum()
    qty = view["Quantity"].sum() if "Quantity" in view.columns else 0.0
    tx = view["Transaction_ID"].nunique()
    spend_basket = (total_sales/tx) if tx else 0.0
    asp = (total_sales/qty) if qty else 0.0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Sales", f"${total_sales:,.0f}")
    c2.metric("Transactions", f"{tx:,}")
    c3.metric("Units", f"{int(qty):,}")
    c4.metric("Spend/Basket", f"${spend_basket:,.2f}")
    c5.metric("ASP", f"${asp:,.2f}")

def display_kpi_trends_tab():
    st.header("📈 KPI Trends" + (" — Focus" if "📈 KPI Trends" in dept_tabs_map.get(dept,set()) else ""))
    date_range, filt = local_filter_controls("trends")

    metric = st.selectbox("Metric", ["Total_Sale","Quantity","Spend per Basket","Transactions"], index=0, key="trends_metric")
    # choose grouping automatically if multiple selected, else aggregate
    group_dim = st.selectbox("Split by (optional)", [None,"Store_ID","Category","Brand_Name","Item"], index=0, key="trends_group")

    # Select MDL base
    if group_dim is None:
        base = MDL["trends_agg"].copy()
    elif group_dim == "Store_ID":
        base = MDL["trends_by_store"].copy()
    elif group_dim == "Category":
        base = MDL["trends_by_cat"].copy()
    elif group_dim == "Brand_Name":
        base = MDL["trends_by_brand"].copy()
    else:
        base = MDL["trends_by_item"].copy()

    # Apply local filters to pre-aggregated trends
    start, end = date_range
    base = base[(base["Date"] >= start) & (base["Date"] <= end)]
    for col, vals in filt.items():
        if vals and col in base.columns:
            base = base[base[col].isin(vals)]

    if base.empty or metric not in base.columns:
        st.info("No trend data for current selection."); return

    fig = px.line(
        base, x="Date", y=metric,
        color=group_dim if (group_dim and group_dim in base.columns) else None,
        title=f"{metric} Over Time" + (f" by {group_dim}" if group_dim else "")
    )
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Conservative forecast only for aggregate (no split)
    if group_dim is None and metric in ["Total_Sale","Quantity"]:
        show_fc = st.checkbox("Show forecast", value=True, key="trends_show_fc")
        if show_fc:
            ts = (base.set_index("Date")[metric].astype(float).fillna(0.0))
            fc_df = forecast_from_series(ts, rule=rule, alpha=0.05)
            fig2 = px.line(base, x="Date", y=metric, title=f"{metric} + Forecast")
            fig2.add_scatter(x=fc_df["Date"], y=fc_df["yhat"], mode="lines", name="forecast", line=dict(dash="dash"))
            fig2.add_traces([
                dict(
                    x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
                    y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="95% CI"
                )
            ])
            st.plotly_chart(fig2, use_container_width=True)
            st.caption(
                f"Projected {metric.replace('_',' ').title()} sum: "
                f"**{fc_df['yhat'].sum():,.0f}** (95%: {fc_df['yhat_lower'].sum():,.0f}–{fc_df['yhat_upper'].sum():,.0f})"
            )

def display_top_n_tab():
    st.header("🏆 Top-N Views" + (" — Focus" if "🏆 Top-N Views" in dept_tabs_map.get(dept,set()) else ""))
    date_range, filt = local_filter_controls("topn")
    dim = st.selectbox("Top-N by", [d for d in ["Category","Brand_Name","Store_ID","Item"] if d in raw_df.columns], index=0, key="topn_dim")
    n = st.slider("N", 5, 30, 10, key="topn_n")

    view = apply_local_filters_df(raw_df, date_range, filt)
    if view.empty:
        st.info("No data for selected filters."); return
    top_df = (view.groupby(dim)["Total_Sale"].sum().reset_index().sort_values("Total_Sale", ascending=False).head(n))
    fig = px.bar(top_df, x=dim, y="Total_Sale", title=f"Top {n} {dim} by Sales", text_auto=".2s")
    st.plotly_chart(fig, use_container_width=True)

def display_basket_affinity_tab():
    st.header("🧺 Basket Affinity" + (" — Focus" if "🧺 Basket Affinity" in dept_tabs_map.get(dept,set()) else ""))
    date_range, filt = local_filter_controls("aff")
    key_col = st.radio("Granularity", [g for g in ["Item","Brand_Name","Category"] if g in raw_df.columns], index=0, horizontal=True, key="aff_key")
    target = st.selectbox("Target", sorted(raw_df[key_col].unique().tolist()), key="aff_target")

    # Precomputed rules
    key_map = {"Item":"affinity_item","Brand_Name":"affinity_brand","Category":"affinity_cat"}
    rules = MDL.get(key_map.get(key_col,""), pd.DataFrame()).copy()
    if rules.empty:
        st.info("No co-basket pairs found."); return

    # Local filter on consequent if user narrowed product sets (lightweight)
    out = rules[rules["Antecedent"] == str(target)].copy()
    # Optional: filter by category/brand/item lists (if present)
    for col, vals in filt.items():
        if vals and col in out.columns:
            out = out[out[col].isin(vals)]

    if out.empty:
        st.info("No associations for the selected target."); return

    out["Associated Item"] = out["Consequent"]
    out["Confidence (%)"] = (out["Confidence (A->B)"]*100).round(2)
    out = (out.rename(columns={"Lift (A,B)":"Lift"})
            .sort_values(["Lift","Confidence (%)"], ascending=[False,False])
            .reset_index(drop=True))
    out.insert(0,"Rank", range(1, len(out)+1))

    st.dataframe(
        out[["Rank","Associated Item","Lift","Confidence (%)","Total Co-Baskets",
             "Total_CoBasket_Sales_Value","Avg_CoBasket_Spend"]],
        hide_index=True, use_container_width=True,
        column_config={
            "Lift": st.column_config.NumberColumn("Lift", format="%.2f", help=">1 = items appear together more than random."),
            "Confidence (%)": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
            "Total_CoBasket_Sales_Value": st.column_config.NumberColumn("Total Co-Basket Sales", format="$%.0f"),
            "Avg_CoBasket_Spend": st.column_config.NumberColumn("Avg Co-Basket Spend", format="$%.2f"),
        }
    )

def display_price_ladder_tab():
    st.header("💲 Price Ladder" + (" — Focus" if "💲 Price Ladder" in dept_tabs_map.get(dept,set()) else ""))
    date_range, filt = local_filter_controls("pl")
    level = st.selectbox("Price Level", [l for l in ["Item","Brand_Name","Category"] if l in raw_df.columns], index=0, key="pl_level")
    sort_metric = st.selectbox("Sort by", ["Median Price","Average Price","Count"], index=0, key="pl_sort")

    key = {"Item":"price_item","Brand_Name":"price_brand","Category":"price_cat"}[level]
    agg = MDL.get(key, pd.DataFrame()).copy()
    if agg.empty:
        st.info("No price data available."); return

    sort_col = {"Median Price":"Median_Price","Average Price":"Avg_Price","Count":"Count"}[sort_metric]
    agg = agg.sort_values(sort_col, ascending=False)
    fig = px.bar(
        agg, x=level, y="Median_Price",
        title=f"Price Ladder (Median) by {level}",
        text_auto=".2f", hover_data={"Avg_Price":":.2f","Median_Price":":.2f","Count":":,"}
    )
    fig.update_layout(xaxis_title=level, yaxis_title="Median Price")
    st.plotly_chart(fig, use_container_width=True)

def display_store_map_tab():
    st.header("🗺️ Store Map" + (" — Focus" if "🗺️ Store Map" in dept_tabs_map.get(dept,set()) else ""))
    date_range, filt = local_filter_controls("map")
    stores = MDL["store_kpis"].copy()
    if stores.empty or not {"Store_Latitude","Store_Longitude"}.issubset(stores.columns):
        st.info("Store location columns not found."); return

    size_metric = st.selectbox("Bubble Size", ["Total_Sale","Transactions","Quantity"], index=0, key="map_size")
    color_metric = st.selectbox("Bubble Color", ["Total_Sale","Spend_per_Basket","ASP","Transactions","Quantity"], index=0, key="map_color")

    fig = px.scatter_mapbox(
        stores,
        lat="Store_Latitude", lon="Store_Longitude",
        size=size_metric, color=color_metric,
        size_max=28, zoom=3, center={"lat":39.5,"lon":-98.35},
        hover_name="Store_ID",
        custom_data=np.stack([
            stores["Total_Sale"].values,
            stores["Transactions"].values,
            stores["Quantity"].values,
            stores["Spend_per_Basket"].values,
            stores["ASP"].values,
        ], axis=-1),
        mapbox_style="open-street-map"
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
    st.plotly_chart(fig, use_container_width=True)

# -------- Assortment & Space Optimization (uses MDL bases) --------
def display_assortment_space_optimization_tab():
    st.header("📦 Assortment & Space Optimization" + (" — Focus" if "📦 Assortment & Space Optimization" in dept_tabs_map.get(dept,set()) else ""))

    # Local filters shared across subsections
    date_range, filt = local_filter_controls("aso")

    st.subheader("A) SKU Productivity")
    grain = st.radio("Granularity", [g for g in ["Item","Brand_Name"] if g in raw_df.columns], index=0, horizontal=True, key="aso_prod_grain")
    base = MDL["sku_item"].copy() if grain=="Item" else MDL["sku_brand"].copy()
    if base.empty:
        st.info("No SKU base available."); return

    # Light local slicing by category/brand/item if provided
    view = base.copy()
    for col, vals in filt.items():
        if vals and col in view.columns:
            view = view[view[col].isin(vals)]
    if view.empty:
        st.info("No rows after current filters."); return

    # Show table with explanations
    column_config = {
        grain: st.column_config.TextColumn(grain.replace("_"," "), help="SKU grain used in this view."),
        "Category": st.column_config.TextColumn("Category", help="Product category."),
        "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d",
                            help="Stores with at least one sale for the SKU in full history."),
        "Weeks": st.column_config.NumberColumn("Weeks (full)", format="%d",
                            help="Anchored weekly buckets counted over full dataset."),
        "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f",
                            help="SKU_Sales ÷ Active_Stores ÷ Weeks."),
        "SKU_Sales": st.column_config.NumberColumn("SKU Sales", format="$%.0f"),
        "Category_Sales": st.column_config.NumberColumn("Category Sales", format="$%.0f"),
        "% of Category Sales": st.column_config.ProgressColumn("% of Category Sales", min_value=0, max_value=100, format="%.2f%%"),
        "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d"),
        "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f")
    }
    if grain=="Item" and "Brand_Name" in view.columns:
        column_config["Brand_Name"] = st.column_config.TextColumn("Brand")
    if grain=="Brand_Name" and "Item" in view.columns:
        column_config["Item"] = st.column_config.TextColumn("Top Item")

    view = view.loc[:, ~view.columns.duplicated()]
    st.dataframe(
        view.sort_values(["Category","Velocity_$/SKU/Store/Week"], ascending=[True,False]),
        hide_index=True, use_container_width=True, column_config=column_config
    )

    st.divider()
    st.subheader("B) SKU Rationalization Tool")
    st.markdown(
        """
        <div style="margin-top:-6px;margin-bottom:8px;">
        <em>How to use:</em> Set thresholds. Items flagged if they meet <b>any</b> of: Low Velocity, Low Share, Similar Price Tier Duplicate.<br>
        <em>Interpret:</em> Sort by velocity/share; check brand/category clusters for redundancy.
        </div>
        """, unsafe_allow_html=True
    )
    perc = st.slider("Low Velocity threshold (percentile)", 5, 50, 25, step=5, key="aso_lowvel")
    share_min = st.number_input("Minimum Category Share to keep ( % )", 0.0, 100.0, 1.0, step=0.5, key="aso_share")
    price_band = st.slider("Price tier proximity (± % of median Unit Price)", 1, 30, 10, step=1, key="aso_price")
    show_only_flagged = st.checkbox("Show only flagged", value=True, key="aso_showflag")

    rat_df = view.copy()
    if rat_df.empty:
        st.info("No SKUs to evaluate."); return

    rat_df["Low_Velocity_Flag"] = False
    for cat, grp in rat_df.groupby("Category"):
        cutoff = np.percentile(grp["Velocity_$/SKU/Store/Week"], perc) if len(grp) else 0.0
        idx = rat_df["Category"] == cat
        rat_df.loc[idx, "Low_Velocity_Flag"] = rat_df.loc[idx, "Velocity_$/SKU/Store/Week"] <= cutoff

    rat_df["Low_Share_Flag"] = rat_df["% of Category Sales"] < share_min

    if "Median_Unit_Price" in rat_df.columns and "Brand_Name" in rat_df.columns:
        base2 = rat_df.copy()
        base2["Redundancy_Group"] = base2.get("Brand_Name","").astype(str) + " | " + base2["Category"].astype(str)
        def _tag(g):
            med = g["Median_Unit_Price"].median()
            lo, hi = med*(1-price_band/100.0), med*(1+price_band/100.0)
            g["Similar_Price_Tier"] = g["Median_Unit_Price"].between(lo, hi)
            return g
        base2 = base2.groupby("Redundancy_Group", group_keys=False).apply(_tag)
        grp_size = base2.groupby("Redundancy_Group")["SKU_Sales"].transform("size")
        base2["Redundant_Candidate"] = (grp_size>1) & base2["Similar_Price_Tier"] & base2["Low_Share_Flag"]
        rat_df = base2
    else:
        rat_df["Similar_Price_Tier"] = False
        rat_df["Redundant_Candidate"] = False

    rat_df["Rationalize?"] = rat_df["Low_Velocity_Flag"] | rat_df["Low_Share_Flag"] | rat_df["Redundant_Candidate"]

    flagged = rat_df[rat_df["Rationalize?"]]
    c1,c2,c3 = st.columns(3)
    c1.metric("SKUs evaluated", f"{len(rat_df):,}")
    c2.metric("Flagged SKUs", f"{int(rat_df['Rationalize?'].sum()):,}")
    c3.metric("% Flagged", f"{(100.0*rat_df['Rationalize?'].mean() if len(rat_df) else 0):.1f}%")

    if not flagged.empty:
        reason_counts = pd.DataFrame({
            "Reason": ["Low Velocity","Low Share","Similar Price-Tier Duplicate"],
            "Count": [int(flagged["Low_Velocity_Flag"].sum()),
                      int(flagged["Low_Share_Flag"].sum()),
                      int(flagged["Redundant_Candidate"].sum())]
        })
        fig_reason = px.bar(reason_counts, x="Reason", y="Count", title="Flag Reasons", text_auto=True)
        st.plotly_chart(fig_reason, use_container_width=True)

    view_df = flagged if show_only_flagged else rat_df
    keep_cols = [
        grain, "Brand_Name", "Item", "Category",
        "Median_Unit_Price", "Velocity_$/SKU/Store/Week", "% of Category Sales",
        "Velocity_Rank_in_Category", "Low_Velocity_Flag", "Low_Share_Flag",
        "Similar_Price_Tier", "Redundant_Candidate", "Rationalize?"
    ]
    view_df = view_df[[c for c in keep_cols if c in view_df.columns]].copy()
    view_df = view_df.loc[:, ~view_df.columns.duplicated()]

    st.dataframe(
        view_df.sort_values(["Rationalize?","Category","Velocity_$/SKU/Store/Week"],
                            ascending=[False,True,True]).reset_index(drop=True),
        hide_index=True, use_container_width=True,
        column_config={
            grain: st.column_config.TextColumn(grain),
            "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f"),
            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f"),
            "% of Category Sales": st.column_config.NumberColumn("% of Category Sales", format="%.2f"),
            "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d"),
            "Low_Velocity_Flag": st.column_config.CheckboxColumn("Low Velocity"),
            "Low_Share_Flag": st.column_config.CheckboxColumn("Low Share"),
            "Similar_Price_Tier": st.column_config.CheckboxColumn("Similar Price Tier"),
            "Redundant_Candidate": st.column_config.CheckboxColumn("Redundant Candidate"),
            "Rationalize?": st.column_config.CheckboxColumn("Flag"),
        }
    )

    st.divider()
    st.subheader("C) Assortment Opportunity Map")
    st.markdown(
        """
        <em>How to use:</em> Pick a geo grain and product grain. We compare local share vs overall to compute an <b>Index</b> (100 = average).  
        <b>&gt;100</b> over-index; <b>&lt;100</b> under-index (white space).
        """, unsafe_allow_html=True
    )
    geo_col = st.selectbox("Geo", [g for g in ["Store_State","Store_City","Store_ID"] if g in raw_df.columns], index=0, key="aso_geo")
    map_dim = st.selectbox("Analyze by", [d for d in ["Item","Brand_Name"] if d in raw_df.columns], index=0, key="aso_geo_dim")
    topn_geo = st.slider("Top-N per Geo", 3, 15, 5, key="aso_geo_topn")

    overall_share, geo_totals, share = MDL["geo_base"].get(geo_col, {}).get(map_dim, (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()))
    if share.empty:
        st.info("No geo share base available."); 
    else:
        # local product filters, if any
        for col, vals in filt.items():
            if vals and col in share.columns:
                share = share[share[col].isin(vals)]
        if share.empty:
            st.info("No rows after filters."); 
        else:
            top_by_geo = (share.sort_values("Index_vs_Overall", ascending=False)
                               .groupby(geo_col).head(int(topn_geo)).reset_index(drop=True))
            st.dataframe(
                top_by_geo[[c for c in [geo_col, map_dim, "Index_vs_Overall", "Geo_Share", "Overall_Share", "Geo_Sales"] if c in top_by_geo.columns]],
                hide_index=True, use_container_width=True,
                column_config={
                    "Index_vs_Overall": st.column_config.NumberColumn("Index vs Overall", format="%.1f"),
                    "Geo_Share": st.column_config.NumberColumn("Geo Share", format="%.2%"),
                    "Overall_Share": st.column_config.NumberColumn("Overall Share", format="%.2%"),
                    "Geo_Sales": st.column_config.NumberColumn("Geo Sales", format="$%.0f"),
                }
            )
            with st.expander("Heatmap (optional)", expanded=False):
                try:
                    piv = top_by_geo.pivot(index=geo_col, columns=map_dim, values="Index_vs_Overall").fillna(100.0)
                    fig_hm = px.imshow(piv, aspect="auto", title=f"Over/Under Index Heatmap by {geo_col}")
                    st.plotly_chart(fig_hm, use_container_width=True)
                except Exception:
                    st.info("Heatmap not available for current selection.")

    st.divider()
    st.subheader("D) New Item Tracker")
    new_window = st.select_slider("Define 'New' as first sold within (days)", options=[30,60,90,120,180], value=90, key="aso_new_win")
    first = MDL["first_sale"].copy()
    if first.empty:
        st.info("No 'Item' column for new item tracking."); return
    end_local = pd.to_datetime(date_range[1])
    cutoff = end_local - pd.Timedelta(days=int(new_window))
    first["Is_New"] = first["First_Sale_Date"] >= cutoff

    fraw = apply_local_filters_df(raw_df, date_range, filt)
    if fraw.empty:
        st.info("No data in selected window."); return
    merged = fraw.merge(first, on="Item", how="left")
    new_items = merged[merged["Is_New"] == True].copy()
    if new_items.empty:
        st.info("No new items in the selected window."); return

    # Quick velocity vs benchmark (category-level)
    perf = (merged.groupby(["Item","Category"], dropna=False)["Total_Sale"].sum().rename("Item_Sales").reset_index()
                 .merge(merged.groupby("Item")["Store_ID"].nunique().rename("Active_Stores").reset_index(), on="Item", how="left"))
    # approximate weeks from selected date range
    span_days = (pd.to_datetime(date_range[1]) - pd.to_datetime(date_range[0])).days
    weeks = max(int(np.ceil(span_days/7.0)), 1)
    perf["Weeks"] = weeks
    perf["Velocity_$/SKU/Store/Week"] = np.where(
        (perf["Active_Stores"]>0) & (perf["Weeks"]>0),
        perf["Item_Sales"]/(perf["Active_Stores"]*perf["Weeks"]), 0.0
    )
    # benchmark (non-new) by category
    not_new = merged.merge(first[first["Is_New"] == False][["Item"]].assign(Not_New=True), on="Item", how="inner")
    bench = (not_new.groupby(["Category","Item"])["Total_Sale"].sum().reset_index()
                    .merge(not_new.groupby("Item")["Store_ID"].nunique().rename("Active_Stores").reset_index(), on="Item", how="left"))
    bench["Weeks"] = weeks
    bench["Velocity_$/SKU/Store/Week"] = np.where(
        (bench["Active_Stores"]>0) & (bench["Weeks"]>0),
        bench["Total_Sale"]/(bench["Active_Stores"]*bench["Weeks"]), 0.0
    )
    bench_avg = bench.groupby("Category")["Velocity_$/SKU/Store/Week"].mean().rename("Benchmark_Velocity").reset_index()

    new_perf = (new_items[["Item","Category"]].drop_duplicates()
                .merge(perf, on=["Item","Category"], how="left")
                .merge(bench_avg, on="Category", how="left"))
    new_perf["Velocity_vs_Benchmark_%"] = np.where(
        new_perf["Benchmark_Velocity"]>0,
        100.0*new_perf["Velocity_$/SKU/Store/Week"]/new_perf["Benchmark_Velocity"], np.nan
    )
    st.dataframe(
        new_perf[["Item","Category","Active_Stores","Weeks","Velocity_$/SKU/Store/Week","Benchmark_Velocity","Velocity_vs_Benchmark_%"]],
        hide_index=True, use_container_width=True,
        column_config={
            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f"),
            "Benchmark_Velocity": st.column_config.NumberColumn("Benchmark Velocity", format="$%.2f"),
            "Velocity_vs_Benchmark_%": st.column_config.NumberColumn("Velocity vs Benchmark (%)", format="%.1f"),
            "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d"),
            "Weeks": st.column_config.NumberColumn("Weeks", format="%d"),
        }
    )

# ---------------------------------
# Main Tabs
# ---------------------------------
tabs = st.tabs([
    "📊 KPI Overview",
    "📈 KPI Trends",
    "🏆 Top-N Views",
    "🧺 Basket Affinity",
    "💲 Price Ladder",
    "🗺️ Store Map",
    "📦 Assortment & Space Optimization",
])

with tabs[0]:
    display_kpi_overview_tab()
with tabs[1]:
    display_kpi_trends_tab()
with tabs[2]:
    display_top_n_tab()
with tabs[3]:
    display_basket_affinity_tab()
with tabs[4]:
    display_price_ladder_tab()
with tabs[5]:
    display_store_map_tab()
with tabs[6]:
    display_assortment_space_optimization_tab()

# Footer: quick data/filters feedback
with st.expander("🧯 Data Snapshot", expanded=False):
    st.caption(f"Rows: **{len(raw_df):,}** | Baskets: **{raw_df['Transaction_ID'].nunique():,}** | "
               f"Date Range: **{raw_df['Date'].min().date()} → {raw_df['Date'].max().date()}**")
