# C-Store KPI Dashboard — Department View + Per-Tab Local Filters
# - Single compact global control (Department) in a sidebar expander
# - All other filters are local to each tab, do not affect other tabs
# - No CSV upload (reads fixed DATA_PATH)
# - Forecasts are conservative (ETS vs seasonal-naive) and honor local tab filters

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from itertools import combinations
from collections import Counter
from typing import Dict, Tuple, List

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="C-Store KPI Dashboard", layout="wide", initial_sidebar_state="expanded")
pio.templates.default = "plotly_white"

# -----------------------------
# Constants & Department Map
# -----------------------------
DATA_PATH = "cstorereal.csv"

DEPT_REPORTS = {
    "All Departments": ["kpi", "trends", "topn", "affinity", "price", "map", "assortment", "forecasts"],
    "Strategy & Finance": ["kpi", "forecasts"],
    "Merchandising & Category": ["topn", "price", "assortment", "affinity"],
    "Marketing & CX": ["trends", "topn", "affinity"],
    "Store Operations & Supply Chain": ["kpi", "trends", "map"],
    "Data & eCommerce": ["kpi", "trends", "topn", "affinity", "price", "map", "assortment", "forecasts"],
}

RULE_MAP = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}

# -----------------------------
# Helpers
# -----------------------------
def to_num_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

@st.cache_data(show_spinner=False)
def load_and_normalize(path_or_buffer) -> pd.DataFrame:
    if not os.path.exists(path_or_buffer):
        st.error(f"CSV not found at: {os.path.abspath(path_or_buffer)}")
        st.stop()

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
        st.error(f"Missing required columns: {missing}.")
        st.stop()

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

def multiselect_all_inline(label: str, options: list, key: str, default_all=False) -> list:
    """Inline multiselect with an 'All' fake option behavior; defaults to empty unless default_all=True."""
    all_label = "All"
    opts = [all_label] + list(options)
    default = [all_label] if default_all else []
    chosen = st.multiselect(label, opts, default=default, key=key)
    return list(options) if all_label in chosen or (default_all and not chosen) else [x for x in chosen if x != all_label]

def _weeks_in_range(df: pd.DataFrame, start_date, end_date) -> int:
    if df.empty:
        return 1
    try:
        n = df.set_index("Date").resample("W-SUN")["Total_Sale"].sum().shape[0]
        return max(int(n), 1)
    except Exception:
        span_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        return max(int(np.ceil(span_days / 7.0)), 1)

def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    total_sales = df["Total_Sale"].sum() if "Total_Sale" in df.columns else 0.0
    total_qty   = df["Quantity"].sum()    if "Quantity"   in df.columns else 0.0
    tx          = df["Transaction_ID"].nunique() if "Transaction_ID" in df.columns else 0
    spend_per_basket = (total_sales / tx) if tx else 0.0
    asp = (total_sales / total_qty) if total_qty else 0.0
    return dict(total_sales=total_sales, total_qty=total_qty, tx=tx, spend_per_basket=spend_per_basket, asp=asp)

def calculate_trends(df: pd.DataFrame, rule: str, group_dim: str or None) -> pd.DataFrame:
    group_cols = [pd.Grouper(key="Date", freq=rule)]
    if group_dim and group_dim in df.columns:
        group_cols.append(group_dim)
    trend_df = (
        df.groupby(group_cols, dropna=False)
          .agg(
              Total_Sale=("Total_Sale", "sum"),
              Quantity=("Quantity", "sum"),
              Transactions=("Transaction_ID", "nunique")
          )
          .reset_index()
          .sort_values("Date")
    )
    trend_df["Spend per Basket"] = np.where(trend_df["Transactions"] > 0, trend_df["Total_Sale"] / trend_df["Transactions"], 0.0)
    return trend_df

def calculate_top_n(df: pd.DataFrame, dim: str, n: int) -> Tuple[pd.DataFrame, str]:
    top_df = (df.groupby(dim)["Total_Sale"].sum().sort_values(ascending=False).head(n).reset_index())
    return top_df, "Total_Sale"

def calculate_affinity_rules(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    scope = df.dropna(subset=["Transaction_ID", key_col]).copy()
    scope["Transaction_ID"] = scope["Transaction_ID"].astype(str)
    scope[key_col] = scope[key_col].astype(str)
    tx_count_total = scope["Transaction_ID"].nunique()
    if tx_count_total == 0:
        return pd.DataFrame()
    basket_sales = scope.groupby("Transaction_ID")["Total_Sale"].sum()
    tx_keys = scope.groupby("Transaction_ID")[key_col].apply(lambda s: tuple(sorted(set(s))))

    item_counts = Counter()
    pair_counts = Counter()
    for keys in tx_keys:
        for k in keys:
            item_counts[k] += 1
        for a, b in combinations(keys, 2):
            pair_counts[tuple(sorted((a, b)))] += 1
    if not pair_counts:
        return pd.DataFrame()

    item_txids_map = scope.groupby(key_col)["Transaction_ID"].apply(set)

    def qty_in_txids(item: str, tx_ids: set) -> float:
        if "Quantity" not in scope.columns or not tx_ids:
            return 0.0
        view = scope[(scope["Transaction_ID"].isin(tx_ids)) & (scope[key_col] == item)]
        return float(view["Quantity"].sum())

    rows = []
    for (a, b), n_ab in pair_counts.items():
        count_ab = n_ab
        count_a = item_counts[a]
        count_b = item_counts[b]
        support_ab = count_ab / tx_count_total
        lift = (support_ab / ((count_a / tx_count_total) * (count_b / tx_count_total))) if (count_a and count_b) else 0.0
        tx_a = item_txids_map.get(a, set())
        tx_b = item_txids_map.get(b, set())
        co_tx = tx_a & tx_b
        total_co_spend = float(basket_sales.loc[list(co_tx)].sum()) if co_tx else 0.0
        avg_co_spend = (total_co_spend / count_ab) if count_ab else 0.0
        qty_a_in_co = qty_in_txids(a, co_tx)
        qty_b_in_co = qty_in_txids(b, co_tx)

        rows.append({
            "Antecedent": a, "Consequent": b, "Total Co-Baskets": count_ab,
            "Support (A,B)": support_ab, "Confidence (A->B)": (count_ab / count_a) if count_a else 0.0,
            "Lift (A,B)": lift, "Total_Antecedent_Qty_in_CoBasket": qty_a_in_co,
            "Avg_Antecedent_Qty_in_CoBasket": (qty_a_in_co / count_ab) if count_ab else 0.0,
            "Total_CoBasket_Sales_Value": total_co_spend, "Avg_CoBasket_Spend": avg_co_spend,
        })
        rows.append({
            "Antecedent": b, "Consequent": a, "Total Co-Baskets": count_ab,
            "Support (A,B)": support_ab, "Confidence (A->B)": (count_ab / count_b) if count_b else 0.0,
            "Lift (A,B)": lift, "Total_Antecedent_Qty_in_CoBasket": qty_b_in_co,
            "Avg_Antecedent_Qty_in_CoBasket": (qty_b_in_co / count_ab) if count_ab else 0.0,
            "Total_CoBasket_Sales_Value": total_co_spend, "Avg_CoBasket_Spend": avg_co_spend,
        })
    return pd.DataFrame(rows).sort_values(["Lift (A,B)", "Confidence (A->B)"], ascending=[False, False]).reset_index(drop=True)

# ---------- Forecast helpers ----------
def _forecast_freq_meta(rule: str):
    if rule == "D":
        return 30, 7, "D", "next 30 days"
    if rule.startswith("W-"):
        return 4, 52, rule, "next 4 weeks"
    if rule == "MS":
        return 1, 12, "MS", "next month"
    return 30, 7, "D", "next 30 days"

def _series_from_raw(df: pd.DataFrame, rule: str, metric: str) -> pd.Series:
    s = (
        df.set_index("Date")[metric]
          .sort_index()
          .resample(rule)
          .sum()
          .astype(float)
          .fillna(0.0)
    )
    s.index.name = "Date"
    return s

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
        denom = (np.abs(a) + np.abs(f))
        denom[denom == 0] = 1.0
        return float(np.mean(2.0 * np.abs(a - f) / denom))

    a = hold.to_numpy()
    naive_err = smape(a, naive_fc)
    ets_err = smape(a, ets_fc) if ets_fc is not None else np.inf
    return ("ets", ets_model, resid_std) if ets_err < naive_err else ("naive", None, None)

def _forecast_series_conservative(df: pd.DataFrame, rule: str, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    steps, seasonal_periods, freq, _ = _forecast_freq_meta(rule)
    ts = _series_from_raw(df, rule, metric).clip(lower=0)
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

# -----------------------------
# Local filter UI (per tab)
# -----------------------------
def render_local_filters(df: pd.DataFrame, prefix: str):
    """
    Renders tab-scoped filters and returns (filtered_df, rule, start_date, end_date).
    Keys are namespaced with `prefix` so each tab is independent.
    """
    with st.expander("Filters", expanded=False):
        stores = sorted(df["Store_ID"].unique().tolist())
        cats   = sorted(df["Category"].unique().tolist())
        brands = sorted(df["Brand_Name"].unique().tolist()) if "Brand_Name" in df.columns else []
        prods  = sorted(df["Item"].unique().tolist())
        pays   = sorted(df["Payment_Method"].unique().tolist()) if "Payment_Method" in df.columns else []

        min_date = df["Date"].min().date()
        max_date = df["Date"].max().date()

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            sel_stores = multiselect_all_inline("Store(s)", stores, key=f"{prefix}_stores")
            sel_cats   = multiselect_all_inline("Category", cats, key=f"{prefix}_cats")
        with c2:
            sel_brands = multiselect_all_inline("Brand", brands, key=f"{prefix}_brands") if brands else []
            sel_prods  = multiselect_all_inline("Product", prods, key=f"{prefix}_prods")
        with c3:
            sel_pays   = multiselect_all_inline("Payment Method", pays, key=f"{prefix}_pays") if pays else []

        c4, c5, c6 = st.columns([1,1,1])
        with c4:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date, key=f"{prefix}_start")
        with c5:
            end_date   = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date, key=f"{prefix}_end")
        with c6:
            freq_label = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key=f"{prefix}_freq")
        rule = RULE_MAP[freq_label]

    # Apply only to this tab
    mask = (df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))
    out = df.loc[mask].copy()
    if sel_stores:
        out = out[out["Store_ID"].isin(sel_stores)]
    if sel_cats:
        out = out[out["Category"].isin(sel_cats)]
    if sel_brands and "Brand_Name" in out.columns:
        out = out[out["Brand_Name"].isin(sel_brands)]
    if sel_prods:
        out = out[out["Item"].isin(sel_prods)]
    if sel_pays and "Payment_Method" in out.columns:
        out = out[out["Payment_Method"].isin(sel_pays)]
    return out, rule, start_date, end_date

# -----------------------------
# Displays (tab bodies)
# -----------------------------
def tab_kpi(df: pd.DataFrame):
    st.header("📊 KPI Overview")
    df_local, _, start_d, end_d = render_local_filters(df, prefix="kpi")
    if df_local.empty:
        st.info("No data for selected filters.")
        return
    kpis = calculate_kpis(df_local)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", f"${kpis['total_sales']:,.0f}")
    c2.metric("Transactions", f"{kpis['tx']:,}")
    c3.metric("Units", f"{int(kpis['total_qty']):,}")
    c4.metric("Spend/Basket", f"${kpis['spend_per_basket']:,.2f}")
    c5.metric("ASP", f"${kpis['asp']:,.2f}")
    with st.expander("Data Summary", expanded=False):
        st.caption(f"Date Range: **{pd.to_datetime(start_d).date()} → {pd.to_datetime(end_d).date()}** | Rows: **{len(df_local):,}** | Baskets: **{df_local['Transaction_ID'].nunique():,}**")

def tab_trends(df: pd.DataFrame):
    st.header("📈 KPI Trends")
    df_local, rule, *_ = render_local_filters(df, prefix="trends")
    if df_local.empty:
        st.info("No data for selected filters.")
        return
    metric = st.selectbox("Select Metric", ["Total_Sale", "Quantity", "Spend per Basket", "Transactions"], index=0, key="trend_metric_local")
    group_dim = None
    # choose a split if user wants (optional)
    split_dim = st.selectbox("Split by (optional)", ["(None)", "Store_ID", "Category", "Brand_Name", "Item"], index=0, key="trend_split")
    if split_dim != "(None)" and split_dim in df_local.columns:
        group_dim = split_dim

    trend_df = calculate_trends(df_local, rule, group_dim)
    if trend_df.empty or metric not in trend_df.columns or trend_df[metric].dropna().empty:
        st.info("No trend data for current filters and metric.")
        return

    fig = px.line(trend_df, x="Date", y=metric, color=group_dim, title=f"{metric} Over Time" + (f" by {group_dim}" if group_dim else ""))
    fig.update_layout(hovermode="x unified")
    show_fc = st.checkbox("Show forecast for aggregates", value=(group_dim is None and metric in ["Total_Sale", "Quantity"]), key="trend_fc_show")
    if show_fc and group_dim is None and metric in ["Total_Sale", "Quantity"]:
        fc_df = _forecast_series_conservative(df_local, rule=rule, metric=metric, alpha=0.05)
        fig.add_scatter(x=fc_df["Date"], y=fc_df["yhat"], mode="lines", name=f"{metric} forecast", line=dict(dash="dash"))
        fig.add_traces([
            dict(
                x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
                y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
                fill="toself", fillcolor="rgba(99,110,250,0.15)", line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip", name="95% interval"
            )
        ])
    st.plotly_chart(fig, use_container_width=True)

def tab_topn(df: pd.DataFrame):
    st.header("🏆 Top-N Views")
    df_local, _, *_ = render_local_filters(df, prefix="topn")
    if df_local.empty:
        st.info("No data for selected filters.")
        return
    dims_available = [d for d in ["Category", "Brand_Name", "Store_ID", "Item"] if d in df_local.columns]
    if not dims_available:
        st.info("No dimensions available.")
        return
    col1, col2 = st.columns([1, 4])
    with col1:
        dim = st.selectbox("Top-N by", dims_available, index=0, key="topn_dim_local")
    with col2:
        n = st.slider("N", 5, 30, 10, key="topn_n_local")
    top_df, y_col = calculate_top_n(df_local, dim, n)
    fig_bar = px.bar(top_df, x=dim, y=y_col, title=f"Top {n} {dim} by {y_col}", text_auto=".2s")
    st.plotly_chart(fig_bar, use_container_width=True)

def tab_affinity(df: pd.DataFrame):
    st.header("🧺 Targeted Basket Affinity Report")
    st.caption("Identify the strongest co-purchase relationships at Item/Brand/Category level.")
    df_local, _, *_ = render_local_filters(df, prefix="aff")
    if df_local.empty:
        st.info("No data for selected filters.")
        return
    valid_grans = [g for g in ["Item", "Brand_Name", "Category"] if g in df_local.columns]
    if not valid_grans:
        st.info("No valid columns for affinity.")
        return
    col_gran, col_target = st.columns([1, 3])
    with col_gran:
        default_index = valid_grans.index("Item") if "Item" in valid_grans else 0
        granularity = st.radio("Granularity", valid_grans, index=default_index, horizontal=True, key="aff_gran_local")
        key_col = granularity
    targets = sorted(df_local[key_col].astype(str).unique().tolist())
    with col_target:
        target_product = st.selectbox("Select Target", targets, index=0 if targets else None, key="aff_target_local")
    if not target_product:
        st.warning("Please select a Target.")
        return
    rules_df = calculate_affinity_rules(df_local, key_col)
    if rules_df.empty:
        st.info("No co-basket pairs found. Try widening the date range or filters.")
        return
    tdf = rules_df[rules_df["Antecedent"] == str(target_product)].copy()
    if tdf.empty:
        st.info(f"No outbound associations for **{target_product}**.")
        return
    tdf["Associated Item"] = tdf["Consequent"]
    display_df = tdf[[
        "Associated Item", "Confidence (A->B)", "Lift (A,B)", "Total Co-Baskets",
        "Total_Antecedent_Qty_in_CoBasket", "Avg_Antecedent_Qty_in_CoBasket",
        "Total_CoBasket_Sales_Value", "Avg_CoBasket_Spend",
    ]].copy()
    display_df["Confidence (%)"] = (display_df["Confidence (A->B)"] * 100.0).round(2)
    display_df = display_df.drop(columns=["Confidence (A->B)"]).rename(columns={
        "Lift (A,B)": "Lift",
        "Total_Antecedent_Qty_in_CoBasket": f"Total Qty of {target_product}",
        "Avg_Antecedent_Qty_in_CoBasket":   f"Avg Qty of {target_product}",
        "Total_CoBasket_Sales_Value":       "Total Co-Basket Sales",
        "Avg_CoBasket_Spend":               "Avg Co-Basket Spend",
    })
    display_df = display_df.sort_values(["Lift", "Confidence (%)"], ascending=False).reset_index(drop=True)
    display_df.insert(0, "Rank", range(1, 1 + len(display_df)))
    st.dataframe(
        display_df, hide_index=True, use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Associated Item": st.column_config.TextColumn("Associated Item", width="large"),
            "Confidence (%)": st.column_config.ProgressColumn(
                f"Confidence ({target_product} → Associated)", min_value=0, max_value=100, format="%.1f%%",
                help=f"Share of {target_product} baskets that also contained the Associated Item."
            ),
            "Lift": st.column_config.NumberColumn("Lift", format="%.2f", help=">1 indicates positive association."),
            "Total Co-Basket Sales": st.column_config.NumberColumn("Total Co-Basket Sales", format="$%.2f"),
            "Avg Co-Basket Spend": st.column_config.NumberColumn("Avg Co-Basket Spend", format="$%.2f"),
            f"Total Qty of {target_product}": st.column_config.NumberColumn(f"Total Qty of {target_product}", format="%d"),
            f"Avg Qty of {target_product}":   st.column_config.NumberColumn(f"Avg Qty of {target_product}", format="%.2f"),
            "Total Co-Baskets": st.column_config.NumberColumn("Total Co-Baskets", format="%d"),
        }
    )

def tab_price_ladder(df: pd.DataFrame):
    st.header("💲 Price Ladder")
    df_local, _, *_ = render_local_filters(df, prefix="price")
    if df_local.empty or "Unit_Price" not in df_local.columns:
        st.info("No price data available.")
        return
    levels = [l for l in ["Item", "Brand_Name", "Category"] if l in df_local.columns]
    if not levels:
        st.info("Need Item/Brand_Name/Category columns.")
        return
    col_lvl, col_sort = st.columns([1,1])
    with col_lvl:
        ladder_level = st.selectbox("Price Level", levels, index=0, key="pl_level_local")
    with col_sort:
        sort_metric = st.selectbox("Sort by", ["Median Price", "Average Price", "Count"], index=0, key="pl_sort_local")
    agg = (
        df_local.groupby(ladder_level)
          .agg(Avg_Price=("Unit_Price","mean"), Median_Price=("Unit_Price","median"), Count=("Unit_Price","size"))
          .reset_index()
    )
    sort_col = {"Median Price":"Median_Price","Average Price":"Avg_Price","Count":"Count"}[sort_metric]
    agg = agg.sort_values(sort_col, ascending=False)
    fig = px.bar(agg, x=ladder_level, y="Median_Price",
                 hover_data={"Avg_Price":":.2f","Median_Price":":.2f","Count":":,"},
                 title=f"Price Ladder (Median) by {ladder_level}", text_auto=".2f")
    fig.update_layout(xaxis_title=ladder_level, yaxis_title="Median Price")
    st.plotly_chart(fig, use_container_width=True)
    if ladder_level == "Item":
        st.caption("Distribution of Unit_Price by Item (jittered)")
        tmp = df_local.copy()
        tmp["jitter"] = np.random.uniform(-0.2, 0.2, size=len(tmp))
        fig_sc = px.strip(tmp, x="Item", y="Unit_Price", title="Unit Price Distribution by Item")
        st.plotly_chart(fig_sc, use_container_width=True)

def tab_store_map(df: pd.DataFrame):
    st.header("🗺️ Store Map (Hover for KPIs)")
    df_local, _, *_ = render_local_filters(df, prefix="map")
    needed_cols = {"Store_ID", "Store_Latitude", "Store_Longitude"}
    if not needed_cols.issubset(df_local.columns):
        st.info("Expecting Store_ID, Store_Latitude, Store_Longitude.")
        return
    if df_local.empty:
        st.info("No data for selected filters.")
        return
    with st.expander("ℹ️ Metric Definitions", expanded=False):
        st.markdown(
            "- **Total Sales** = Sum of Total_Sale\n"
            "- **Transactions** = Unique Transaction_ID count\n"
            "- **Units** = Sum of Quantity\n"
            "- **Spend/Basket** = Total Sales ÷ Transactions\n"
            "- **ASP** = Total Sales ÷ Units"
        )
    df_map = df_local.dropna(subset=["Store_Latitude", "Store_Longitude"]).copy()
    if df_map.empty:
        st.info("No stores with coordinates in current selection.")
        return
    kpi_agg = (
        df_map.groupby("Store_ID", as_index=False)
              .agg(Total_Sale=("Total_Sale", "sum"),
                   Quantity=("Quantity", "sum"),
                   Transactions=("Transaction_ID", "nunique"))
    )
    kpi_agg["Spend_per_Basket"] = np.where(kpi_agg["Transactions"] > 0, kpi_agg["Total_Sale"]/kpi_agg["Transactions"], 0.0)
    kpi_agg["ASP"] = np.where(kpi_agg["Quantity"] > 0, kpi_agg["Total_Sale"]/kpi_agg["Quantity"], 0.0)
    loc_cols = ["Store_ID", "Store_City", "Store_State", "Store_Latitude", "Store_Longitude"]
    locs = df_map[[c for c in loc_cols if c in df_map.columns]].drop_duplicates(subset=["Store_ID"])
    store_kpis = kpi_agg.merge(locs, on="Store_ID", how="left")
    if {"Store_City","Store_State"}.issubset(store_kpis.columns):
        store_kpis["Store_Label"] = store_kpis["Store_ID"].astype(str) + " — " + store_kpis["Store_City"].astype(str) + ", " + store_kpis["Store_State"].astype(str)
    else:
        store_kpis["Store_Label"] = store_kpis["Store_ID"].astype(str)
    c1, c2 = st.columns([1,1])
    with c1:
        size_metric = st.selectbox("Bubble Size", ["Total_Sale", "Transactions", "Quantity"], index=0, key="map_size_local")
    with c2:
        color_metric = st.selectbox("Bubble Color", ["Total_Sale", "Spend_per_Basket", "ASP", "Transactions", "Quantity"], index=0, key="map_color_local")
    fig = px.scatter_mapbox(
        store_kpis, lat="Store_Latitude", lon="Store_Longitude",
        size=size_metric, color=color_metric, size_max=28, zoom=3, center={"lat":39.5,"lon":-98.35},
        hover_name="Store_Label",
        custom_data=np.stack([
            store_kpis["Total_Sale"].values,
            store_kpis["Transactions"].values,
            store_kpis["Quantity"].values,
            store_kpis["Spend_per_Basket"].values,
            store_kpis["ASP"].values,
        ], axis=-1),
        mapbox_style="open-street-map"
    )
    fig.update_traces(
        hovertemplate="<b>%{hovertext}</b><br>" +
                      "Total Sales: $%{customdata[0]:,.0f}<br>" +
                      "Transactions: %{customdata[1]:,}<br>" +
                      "Units: %{customdata[2]:,}<br>" +
                      "Spend/Basket: $%{customdata[3]:,.2f}<br>" +
                      "ASP: $%{customdata[4]:,.2f}<br><extra></extra>"
    )
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=600)
    st.plotly_chart(fig, use_container_width=True)

def tab_assortment(df: pd.DataFrame):
    st.header("📦 Assortment & Space Optimization")
    df_local, _, start_date, end_date = render_local_filters(df, prefix="aso")
    if df_local.empty:
        st.info("No data for selected filters.")
        return

    # --- SKU Productivity ---
    st.subheader("A) SKU Productivity Report")
    prod_dim_options = [c for c in ["Item", "Brand_Name"] if c in df_local.columns and df_local[c].notna().any()]
    if not prod_dim_options:
        st.info("Need Item or Brand_Name.")
        return
    colp = st.columns([1,1,2])
    with colp[0]:
        prod_dim = st.selectbox("Granularity", options=prod_dim_options, index=0, key="aso_prod_dim_local")
    price_col = "Unit_Price" if "Unit_Price" in df_local.columns else None
    with colp[1]:
        show_price = st.checkbox("Show median Unit Price", value=bool(price_col), disabled=not bool(price_col), key="aso_show_price_local")

    n_weeks = _weeks_in_range(df_local, start_date, end_date)

    sku_store = df_local.groupby([prod_dim, "Store_ID"], dropna=False)["Total_Sale"].sum().reset_index()
    active_stores = sku_store.groupby(prod_dim)["Store_ID"].nunique().rename("Active_Stores")

    if "Category" not in df_local.columns:
        st.info("Category column missing.")
        return

    sku_sales = df_local.groupby([prod_dim, "Category"], dropna=False)["Total_Sale"].sum().rename("SKU_Sales")
    cat_sales = df_local.groupby("Category", dropna=False)["Total_Sale"].sum().rename("Category_Sales")

    prod_df = sku_sales.reset_index().merge(active_stores, on=prod_dim, how="left")
    prod_df["Active_Stores"] = prod_df["Active_Stores"].fillna(0).astype(int)
    prod_df["Weeks"] = n_weeks
    prod_df["Velocity_$/SKU/Store/Week"] = np.where(
        (prod_df["Active_Stores"] > 0) & (prod_df["Weeks"] > 0),
        prod_df["SKU_Sales"] / (prod_df["Active_Stores"] * prod_df["Weeks"]),
        0.0
    )
    prod_df = prod_df.merge(cat_sales.reset_index(), on="Category", how="left")
    prod_df["% of Category Sales"] = np.where(
        prod_df["Category_Sales"] > 0,
        100.0 * prod_df["SKU_Sales"] / prod_df["Category_Sales"],
        0.0
    )
    if not prod_df.empty:
        prod_df["Velocity_Rank_in_Category"] = (
            prod_df.groupby("Category")["Velocity_$/SKU/Store/Week"].rank(ascending=False, method="dense").astype(int)
        )

    if price_col and show_price:
        med_price = df_local.groupby(prod_dim)[price_col].median().rename("Median_Unit_Price")
        prod_df = prod_df.merge(med_price.reset_index(), on=prod_dim, how="left")

    # show table
    base_cols = [
        prod_dim, "Category", "Active_Stores", "Weeks", "Velocity_$/SKU/Store/Week",
        "SKU_Sales", "Category_Sales", "% of Category Sales", "Velocity_Rank_in_Category"
    ]
    if "Median_Unit_Price" in prod_df.columns and show_price:
        base_cols = [prod_dim, "Median_Unit_Price"] + [c for c in base_cols if c not in [prod_dim, "Median_Unit_Price"]]
    base_cols = [c for c in base_cols if c in prod_df.columns]
    prod_df_view = prod_df[base_cols].copy()
    prod_df_view = prod_df_view.loc[:, ~prod_df_view.columns.duplicated()]
    prod_df_view = prod_df_view.sort_values(["Category", "Velocity_$/SKU/Store/Week"], ascending=[True, False]).reset_index(drop=True)

    st.dataframe(
        prod_df_view, hide_index=True, use_container_width=True,
        column_config={
            prod_dim: st.column_config.TextColumn(("Item" if prod_dim=="Item" else "Brand"),
                                                  help="SKU grain used in this view."),
            "Category": st.column_config.TextColumn("Category", help="Product category."),
            "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d", help="Distinct stores with ≥1 sale."),
            "Weeks": st.column_config.NumberColumn("Weeks", format="%d", help="Anchored weekly buckets in selected range."),
            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f", help="Sales ÷ Active Stores ÷ Weeks."),
            "SKU_Sales": st.column_config.NumberColumn("SKU Sales", format="$%.0f"),
            "Category_Sales": st.column_config.NumberColumn("Category Sales", format="$%.0f"),
            "% of Category Sales": st.column_config.ProgressColumn("% of Category Sales", min_value=0, max_value=100, format="%.2f%%"),
            "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d"),
            "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f") if "Median_Unit_Price" in prod_df_view.columns else None,
        }
    )

    st.divider()

    # --- SKU Rationalization ---
    st.subheader("B) SKU Rationalization Tool")
    st.markdown(
        "<em>How to use:</em> Tune low-velocity percentile, min category share, and price-tier proximity. "
        "<em>Interpretation:</em> A SKU is flagged if any of these conditions are met.",
        unsafe_allow_html=True
    )
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    with c1:
        perc = st.slider("Low Velocity pct", 5, 50, 25, step=5, key="aso_lowvel_local")
    with c2:
        share_min = st.number_input("Min Category Share (%)", min_value=0.0, max_value=100.0, value=1.0, step=0.5, key="aso_share_local")
    with c3:
        price_var = st.slider("Price proximity (±%)", 1, 30, 10, step=1, key="aso_priceband_local")
    with c4:
        show_only_flagged = st.checkbox("Show only flagged", value=True, key="aso_onlyflagged_local")

    rat_df = prod_df.copy()
    rat_df["Low_Velocity_Flag"] = False
    for cat, grp in rat_df.groupby("Category"):
        cutoff = np.percentile(grp["Velocity_$/SKU/Store/Week"], perc) if len(grp) else 0.0
        idx = rat_df["Category"] == cat
        rat_df.loc[idx, "Low_Velocity_Flag"] = rat_df.loc[idx, "Velocity_$/SKU/Store/Week"] <= cutoff
    rat_df["Low_Share_Flag"] = rat_df["% of Category Sales"] < share_min

    price_col = "Unit_Price" if "Unit_Price" in df_local.columns else None
    if price_col and "Median_Unit_Price" in rat_df.columns and "Brand_Name" in df_local.columns:
        base = rat_df.copy()
        base["Redundancy_Group"] = base.get("Brand_Name", "").astype(str) + " | " + base["Category"].astype(str)
        def _redundant_group(g):
            med = g["Median_Unit_Price"].median()
            band_lo = med * (1 - price_var/100.0)
            band_hi = med * (1 + price_var/100.0)
            g["Similar_Price_Tier"] = g["Median_Unit_Price"].between(band_lo, band_hi)
            return g
        base = base.groupby("Redundancy_Group", group_keys=False).apply(_redundant_group)
        grp_size = base.groupby("Redundancy_Group")["SKU_Sales"].transform("size")
        base["Redundant_Candidate"] = (grp_size > 1) & base["Similar_Price_Tier"] & base["Low_Share_Flag"]
        rat_df = base
    else:
        rat_df["Similar_Price_Tier"] = False
        rat_df["Redundant_Candidate"] = False
    rat_df["Rationalize?"] = rat_df["Low_Velocity_Flag"] | rat_df["Low_Share_Flag"] | rat_df["Redundant_Candidate"]

    flagged = rat_df[rat_df["Rationalize?"]].copy()
    view_df = flagged if show_only_flagged else rat_df
    keep_cols = [c for c in [
        prod_dim, "Brand_Name", "Item", "Category", "Median_Unit_Price",
        "Velocity_$/SKU/Store/Week", "% of Category Sales", "Velocity_Rank_in_Category",
        "Low_Velocity_Flag", "Low_Share_Flag", "Redundant_Candidate", "Rationalize?"
    ] if c in view_df.columns]
    view_df = view_df[keep_cols].copy()
    view_df = view_df.loc[:, ~view_df.columns.duplicated()]
    view_df = view_df.sort_values(["Rationalize?", "Category", "Velocity_$/SKU/Store/Week"], ascending=[False, True, True]).reset_index(drop=True)

    st.dataframe(
        view_df, hide_index=True, use_container_width=True,
        column_config={
            prod_dim: st.column_config.TextColumn(("Item" if prod_dim=="Item" else "Brand")),
            "Brand_Name": st.column_config.TextColumn("Brand"),
            "Item": st.column_config.TextColumn("Item"),
            "Category": st.column_config.TextColumn("Category"),
            "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f") if "Median_Unit_Price" in view_df.columns else None,
            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f"),
            "% of Category Sales": st.column_config.NumberColumn("% of Category Sales", format="%.2f"),
            "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d"),
            "Low_Velocity_Flag": st.column_config.CheckboxColumn("Low Velocity"),
            "Low_Share_Flag": st.column_config.CheckboxColumn("Low Share"),
            "Redundant_Candidate": st.column_config.CheckboxColumn("Similar Price-Tier Duplicate"),
            "Rationalize?": st.column_config.CheckboxColumn("Flag"),
        }
    )

    st.divider()

    # --- Assortment Opportunity Map ---
    st.subheader("C) Assortment Opportunity Map")
    geo_cols_all = [c for c in ["Store_State", "Store_City", "Store_ID"] if c in df_local.columns]
    if not geo_cols_all:
        st.info("No geography columns found.")
        return
    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        geo_col = st.selectbox("Geo Dimension", options=geo_cols_all, index=0, key="aso_geo_local")
    with c2:
        map_dim = st.selectbox("Analyze by", options=[c for c in ["Item", "Brand_Name"] if c in df_local.columns], index=0, key="aso_geo_dim_local")
    with c3:
        topn_geo = st.slider("Top-N per Geo", 3, 15, 5, key="aso_geo_topn_local")

    overall_sales = df_local.groupby(map_dim)["Total_Sale"].sum().rename("Overall_Sales")
    overall_total = overall_sales.sum()
    overall_share = (overall_sales / overall_total).rename("Overall_Share").reset_index()

    geo_sales = df_local.groupby([geo_col, map_dim])["Total_Sale"].sum().rename("Geo_Sales").reset_index()
    geo_totals = df_local.groupby(geo_col)["Total_Sale"].sum().rename("Geo_Total").reset_index()
    geo_share = geo_sales.merge(geo_totals, on=geo_col, how="left")
    geo_share["Geo_Share"] = np.where(geo_share["Geo_Total"] > 0, geo_share["Geo_Sales"] / geo_share["Geo_Total"], 0.0)
    geo_share = geo_share.merge(overall_share, on=map_dim, how="left")
    geo_share["Index_vs_Overall"] = np.where(geo_share["Overall_Share"] > 0, 100.0 * geo_share["Geo_Share"] / geo_share["Overall_Share"], 0.0)

    top_by_geo = (
        geo_share.sort_values(["Index_vs_Overall"], ascending=False)
        .groupby(geo_col).head(topn_geo).reset_index(drop=True)
    )
    st.caption("Index > 100 = over-indexing (local favorite); Index < 100 = under-indexing (white space).")
    st.dataframe(
        top_by_geo[[geo_col, map_dim, "Index_vs_Overall", "Geo_Share", "Overall_Share", "Geo_Sales"]],
        hide_index=True, use_container_width=True,
        column_config={
            geo_col: st.column_config.TextColumn(geo_col.replace("_"," ")),
            map_dim: st.column_config.TextColumn(map_dim.replace("_"," ")),
            "Index_vs_Overall": st.column_config.NumberColumn("Index vs Overall", format="%.1f"),
            "Geo_Share": st.column_config.NumberColumn("Geo Share", format="%.2%"),
            "Overall_Share": st.column_config.NumberColumn("Overall Share", format="%.2%"),
            "Geo_Sales": st.column_config.NumberColumn("Geo Sales", format="$%.0f"),
        }
    )
    with st.expander("Heatmap (optional)", expanded=False):
        try:
            pivot = top_by_geo.pivot(index=geo_col, columns=map_dim, values="Index_vs_Overall").fillna(100.0)
            fig_hm = px.imshow(pivot, aspect="auto", title=f"Over/Under Index Heatmap by {geo_col} vs Overall")
            st.plotly_chart(fig_hm, use_container_width=True)
        except Exception:
            st.info("Heatmap not available for current selection.")

def tab_forecasts(df: pd.DataFrame):
    st.header("📉 Forecasts (Aggregate)")
    df_local, rule, *_ = render_local_filters(df, prefix="fc")
    if df_local.empty:
        st.info("No data for selected filters.")
        return
    for metric in ["Total_Sale", "Quantity"]:
        with st.container(border=True):
            st.subheader(f"{metric.replace('_',' ').title()} — History + Forecast")
            hist = (
                df_local.set_index("Date")[metric]
                        .sort_index()
                        .resample(rule).sum()
                        .reset_index().rename(columns={metric: "value"})
            )
            fc_df = _forecast_series_conservative(df_local, rule=rule, metric=metric, alpha=0.05)
            fig = px.line(hist, x="Date", y="value", title=f"{metric.replace('_',' ').title()} Forecast")
            fig.add_scatter(x=fc_df["Date"], y=fc_df["yhat"], mode="lines", name="forecast", line=dict(dash="dash"))
            fig.add_traces([
                dict(
                    x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
                    y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
                    fill="toself", fillcolor="rgba(99,110,250,0.15)",
                    line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip", name="95% interval"
                )
            ])
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                f"Projected {metric.replace('_',' ').title()}: **{fc_df['yhat'].sum():,.0f}** "
                f"(95% CI: {fc_df['yhat_lower'].sum():,.0f} – {fc_df['yhat_upper'].sum():,.0f})"
            )

# -----------------------------
# Load data (no upload)
# -----------------------------
raw_df = load_and_normalize(DATA_PATH)

# -----------------------------
# Compact Global Sidebar (Department only)
# -----------------------------
with st.sidebar.expander("Department View", expanded=True):
    dept_choice = st.selectbox(
        "Department",
        list(DEPT_REPORTS.keys()),
        index=0,
        help="Shows only the tabs that matter for this department."
    )
st.sidebar.caption("All other filters live inside each tab.")

# -----------------------------
# Tabs (driven by department)
# -----------------------------
section_order = ["kpi", "trends", "forecasts", "topn", "affinity", "price", "map", "assortment"]
labels = {
    "kpi": "📊 KPI Overview",
    "trends": "📈 KPI Trends",
    "forecasts": "📉 Forecasts",
    "topn": "🏆 Top-N Views",
    "affinity": "🧺 Basket Affinity",
    "price": "💲 Price Ladder",
    "map": "🗺️ Store Map",
    "assortment": "📦 Assortment & Space Optimization",
}
funcs = {
    "kpi": lambda: tab_kpi(raw_df),
    "trends": lambda: tab_trends(raw_df),
    "forecasts": lambda: tab_forecasts(raw_df),
    "topn": lambda: tab_topn(raw_df),
    "affinity": lambda: tab_affinity(raw_df),
    "price": lambda: tab_price_ladder(raw_df),
    "map": lambda: tab_store_map(raw_df),
    "assortment": lambda: tab_assortment(raw_df),
}

selected_sections = [s for s in section_order if s in DEPT_REPORTS.get(dept_choice, [])]
tab_labels = [labels[s] for s in selected_sections]
if not tab_labels:
    tab_labels = [labels["kpi"]]
    selected_sections = ["kpi"]

tabs = st.tabs(tab_labels)
for tab, section_key in zip(tabs, selected_sections):
    with tab:
        funcs[section_key]()
