# C-Store KPI Dashboard (Dept switchboard + Per-Tab Local Filters)
# - Single compact global sidebar: Department
# - Local filters INSIDE each tab; act only on that tab
# - Options (Stores, Categories, Brands, Items, Payment) come from raw_df (full list)
# - KPI Overview, Trends (w/ conservative forecasts), Top-N, Basket Affinity
# - Price Ladder, Store Map, Assortment & Space Optimization
# - No CSV upload (fixed path)

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from itertools import combinations
from collections import Counter
from typing import Dict, Tuple, List, Optional
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="C-Store KPI Dashboard", layout="wide", initial_sidebar_state="expanded")
pio.templates.default = "plotly_white"

# ---------------------------------
# Tiny CSS to make the single sidebar filter compact
# ---------------------------------
st.markdown(
    """
    <style>
      section[data-testid="stSidebar"] { width: 260px; min-width: 260px; }
      [data-testid="stSidebar"] .block-container { padding-top: .5rem; padding-bottom: .5rem; }
      [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { margin: .25rem 0 .25rem 0; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------
# Helpers
# ---------------------------------
RULE_MAP = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}

def to_num_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

@st.cache_data(show_spinner=False)
def load_and_normalize(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)

    # Canonicalize columns
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
            needs = (df["Total_Sale"] == 0) | (df["Total_Sale"].isna())
            df.loc[needs, "Total_Sale"] = (df.loc[needs, "Quantity"] * df.loc[needs, "Unit_Price"]).round(2)

    return df

def multiselect_all_ui(label: str, options: list, default_all: bool, key: str):
    """Multiselect with explicit 'All' behavior."""
    all_label = "All"
    opts = [all_label] + list(options)
    default_vals = [all_label] if default_all else []
    raw = st.multiselect(label, opts, default=default_vals, key=key)
    if all_label in raw or len(raw) == 0:
        return list(options)  # All
    return [v for v in raw if v != all_label]

def local_filters_block(raw_df: pd.DataFrame, key_prefix: str):
    """
    Per-tab local filters; options come from raw_df so the full lists are always available
    (e.g., all 10 stores). Returns (df_filtered, rule, start_date, end_date).
    """
    # Full option lists from the entire dataset
    stores = sorted(raw_df["Store_ID"].unique().tolist()) if "Store_ID" in raw_df.columns else []
    cats   = sorted(raw_df["Category"].unique().tolist()) if "Category" in raw_df.columns else []
    brands = sorted(raw_df["Brand_Name"].unique().tolist()) if "Brand_Name" in raw_df.columns else []
    prods  = sorted(raw_df["Item"].unique().tolist()) if "Item" in raw_df.columns else []
    pays   = sorted(raw_df["Payment_Method"].unique().tolist()) if "Payment_Method" in raw_df.columns else []

    # Dates
    min_date = raw_df["Date"].min().date()
    max_date = raw_df["Date"].max().date()

    with st.expander("Filters", expanded=False):
        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        with c1:
            sel_stores = multiselect_all_ui("Store(s)", stores, default_all=True, key=f"{key_prefix}_stores")
            sel_cats   = multiselect_all_ui("Category", cats, default_all=True, key=f"{key_prefix}_cats")
            sel_brands = multiselect_all_ui("Brand", brands, default_all=True, key=f"{key_prefix}_brands") if brands else []
        with c2:
            sel_prods  = multiselect_all_ui("Product", prods, default_all=True, key=f"{key_prefix}_prods")
            sel_pays   = multiselect_all_ui("Payment Method", pays, default_all=True, key=f"{key_prefix}_pays") if pays else []
        with c3:
            start_date = st.date_input("Start", value=min_date, min_value=min_date, max_value=max_date, key=f"{key_prefix}_start")
            end_date   = st.date_input("End",   value=max_date, min_value=min_date, max_value=max_date, key=f"{key_prefix}_end")
            freq = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=1, key=f"{key_prefix}_freq")

    # Apply only to this tab
    df = raw_df[
        (raw_df["Date"] >= pd.to_datetime(start_date)) &
        (raw_df["Date"] <= pd.to_datetime(end_date))
    ].copy()

    if sel_stores and "Store_ID" in df.columns:
        df = df[df["Store_ID"].isin(sel_stores)]
    if sel_cats and "Category" in df.columns:
        df = df[df["Category"].isin(sel_cats)]
    if sel_brands and "Brand_Name" in df.columns:
        df = df[df["Brand_Name"].isin(sel_brands)]
    if sel_prods and "Item" in df.columns:
        df = df[df["Item"].isin(sel_prods)]
    if sel_pays and "Payment_Method" in df.columns:
        df = df[df["Payment_Method"].isin(sel_pays)]

    rule = RULE_MAP[freq]
    return df, rule, start_date, end_date

# ---------------------------------
# Metrics & Computations (cached)
# ---------------------------------
@st.cache_data(show_spinner=False)
def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    total_sales = df["Total_Sale"].sum() if "Total_Sale" in df.columns else 0.0
    total_qty   = df["Quantity"].sum() if "Quantity" in df.columns else 0.0
    tx          = df["Transaction_ID"].nunique() if "Transaction_ID" in df.columns else 0
    spend_per_basket = (total_sales / tx) if tx else 0.0
    asp = (total_sales / total_qty) if total_qty else 0.0
    return dict(total_sales=total_sales, total_qty=total_qty, tx=tx, spend_per_basket=spend_per_basket, asp=asp)

@st.cache_data(show_spinner=False)
def calculate_trends(df: pd.DataFrame, rule: str, group_dim: Optional[str]) -> pd.DataFrame:
    group_cols = [pd.Grouper(key="Date", freq=rule)]
    if group_dim and group_dim in df.columns:
        group_cols.append(group_dim)
    trend_df = (
        df.groupby(group_cols, dropna=False)
          .agg(Total_Sale=("Total_Sale", "sum"),
               Quantity=("Quantity", "sum"),
               Transactions=("Transaction_ID", "nunique"))
          .reset_index()
          .sort_values("Date")
    )
    trend_df["Spend per Basket"] = np.where(trend_df["Transactions"] > 0,
                                            trend_df["Total_Sale"] / trend_df["Transactions"], 0.0)
    return trend_df

@st.cache_data(show_spinner=False)
def calculate_top_n(df: pd.DataFrame, dim: str, n: int) -> Tuple[pd.DataFrame, str]:
    top_df = df.groupby(dim)["Total_Sale"].sum().sort_values(ascending=False).head(n).reset_index()
    return top_df, "Total_Sale"

@st.cache_data(show_spinner=False)
def calculate_affinity_rules(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    scope = df.dropna(subset=["Transaction_ID", key_col]).copy()
    if scope.empty:
        return pd.DataFrame()
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

        rows.append({"Antecedent": a, "Consequent": b,
                     "Total Co-Baskets": count_ab,
                     "Support (A,B)": support_ab,
                     "Confidence (A->B)": (count_ab / count_a) if count_a else 0.0,
                     "Lift (A,B)": lift,
                     "Total_Antecedent_Qty_in_CoBasket": qty_a_in_co,
                     "Avg_Antecedent_Qty_in_CoBasket": (qty_a_in_co / count_ab) if count_ab else 0.0,
                     "Total_CoBasket_Sales_Value": total_co_spend,
                     "Avg_CoBasket_Spend": avg_co_spend})
        rows.append({"Antecedent": b, "Consequent": a,
                     "Total Co-Baskets": count_ab,
                     "Support (A,B)": support_ab,
                     "Confidence (A->B)": (count_ab / count_b) if count_b else 0.0,
                     "Lift (A,B)": lift,
                     "Total_Antecedent_Qty_in_CoBasket": qty_b_in_co,
                     "Avg_Antecedent_Qty_in_CoBasket": (qty_b_in_co / count_ab) if count_ab else 0.0,
                     "Total_CoBasket_Sales_Value": total_co_spend,
                     "Avg_CoBasket_Spend": avg_co_spend})

    out = pd.DataFrame(rows).sort_values(["Lift (A,B)", "Confidence (A->B)"], ascending=[False, False]).reset_index(drop=True)
    return out

# ---------- Forecast helpers ----------
def _forecast_freq_meta(rule: str):
    if rule == "D": return 30, 7, "D", "next 30 days"
    if rule.startswith("W-"): return 4, 52, rule, "next 4 weeks"
    if rule == "MS": return 1, 12, "MS", "next month"
    return 30, 7, "D", "next 30 days"

def _series_from_raw(df: pd.DataFrame, rule: str, metric: str) -> pd.Series:
    s = (df.set_index("Date")[metric].sort_index().resample(rule).sum().astype(float).fillna(0.0))
    s.index.name = "Date"
    return s

def _fit_ets_safe(ts: pd.Series, rule: str, seasonal_periods: int):
    use_fixed_smoothing = rule.startswith("W-")
    use_season = seasonal_periods and (len(ts) >= 2 * seasonal_periods)
    try:
        trend_type = 'add' if use_fixed_smoothing else None
        fit_params = {'smoothing_level': 0.1, 'smoothing_trend': 0.01, 'optimized': False} if use_fixed_smoothing else {'optimized': True}
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

def forecast_series_conservative(df: pd.DataFrame, rule: str, metric: str, alpha: float = 0.05) -> pd.DataFrame:
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
    lookback = min(len(ts), 28 if rule == "D" else (12 if rule == "MS" else 12))
    recent_mean = float(ts.iloc[-lookback:].mean()) if lookback > 0 else 0.0
    cap_lo = recent_mean * steps * 0.7
    cap_hi = recent_mean * steps * 1.3
    horizon_sum = float(mean_fc.sum())
    if cap_hi > 0 and horizon_sum > cap_hi:
        scale = cap_hi / horizon_sum
        mean_fc *= scale; lower *= scale; upper *= scale
    elif horizon_sum < cap_lo and horizon_sum > 0:
        scale = cap_lo / horizon_sum
        mean_fc *= scale; lower *= scale; upper *= scale
    idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.DataFrame({"Date": idx, "yhat": mean_fc, "yhat_lower": lower, "yhat_upper": upper})

# ---------------------------------
# Displays (each begins with its own local filter block)
# ---------------------------------
def tab_kpi_overview(raw_df: pd.DataFrame):
    st.header("📊 KPI Overview")
    df, _, _, _ = local_filters_block(raw_df, key_prefix="kpi")
    if df.empty:
        st.info("No data for selected filters.")
        return
    kpis = calculate_kpis(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Sales", f"${kpis['total_sales']:,.0f}")
    c2.metric("Transactions", f"{kpis['tx']:,}")
    c3.metric("Units", f"{int(kpis['total_qty']):,}")
    c4.metric("Spend/Basket", f"${kpis['spend_per_basket']:,.2f}")
    c5.metric("ASP", f"${kpis['asp']:,.2f}")

def tab_kpi_trends(raw_df: pd.DataFrame):
    st.header("📈 KPI Trends")
    df, rule, _, _ = local_filters_block(raw_df, key_prefix="trend")
    if df.empty:
        st.info("No data for selected filters.")
        return

    metric = st.selectbox("Metric", ["Total_Sale", "Quantity", "Spend per Basket", "Transactions"], index=0, key="trend_metric_sel")
    # auto-pick a split if >1 chosen in any dimension
    group_dim = None
    # Use a small chooser to explicitly select split (None / Store / Category / Brand / Item)
    dims_available = [d for d in ["Store_ID","Category","Brand_Name","Item"] if d in df.columns]
    group_dim = st.selectbox("Split by (optional)", ["None"] + dims_available, index=0, key="trend_split")
    group_dim = None if group_dim == "None" else group_dim

    trend_df = calculate_trends(df, rule, group_dim)
    if trend_df.empty or metric not in trend_df.columns:
        st.info("No trend data for current filters and metric.")
        return

    fig = px.line(trend_df, x="Date", y=metric, color=group_dim, title=f"{metric} over time" + (f" by {group_dim}" if group_dim else ""))
    fig.update_layout(hovermode="x unified")

    aggregate_only = (group_dim is None)
    can_forecast = aggregate_only and (metric in ["Total_Sale", "Quantity"])
    _, _, _, horizon_label = _forecast_freq_meta(rule)

    show_fc = st.checkbox(
        f"Show {metric} forecast ({horizon_label})",
        value=True if can_forecast else False,
        disabled=not can_forecast,
        key="trend_show_fc_cb"
    )

    if can_forecast and show_fc:
        fc_df = forecast_series_conservative(df, rule=rule, metric=metric, alpha=0.05)
        fig.add_scatter(x=fc_df["Date"], y=fc_df["yhat"], mode="lines", name=f"{metric} forecast", line=dict(dash="dash"))
        fig.add_traces([dict(
            x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
            y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% interval"
        )])
        st.caption(
            f"**Forecast summary ({horizon_label})** — Projected {metric.replace('_',' ').title()}: "
            f"**{fc_df['yhat'].sum():,.0f}** "
            f"(95% CI: {fc_df['yhat_lower'].sum():,.0f} – {fc_df['yhat_upper'].sum():,.0f})"
        )
        st.download_button(
            "📥 Download forecast (CSV)",
            data=fc_df.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{metric.lower()}_{rule}.csv",
            mime="text/csv",
            key="dl_fc_trend"
        )

    st.plotly_chart(fig, use_container_width=True)

def tab_top_n(raw_df: pd.DataFrame):
    st.header("🏆 Top-N Views")
    df, _, _, _ = local_filters_block(raw_df, key_prefix="topn")
    if df.empty:
        st.info("No data for selected filters.")
        return
    dims_available = [d for d in ["Category", "Brand_Name", "Store_ID", "Item"] if d in df.columns]
    dim = st.selectbox("Top-N by", dims_available, index=0, key="topn_dim_sel")
    n = st.slider("N", 5, 30, 10, key="topn_n_sel")
    top_df, y_col = calculate_top_n(df, dim, n)
    fig_bar = px.bar(top_df, x=dim, y=y_col, title=f"Top {n} {dim} by {y_col}", text_auto=".2s")
    st.plotly_chart(fig_bar, use_container_width=True)

def tab_basket_affinity(raw_df: pd.DataFrame):
    st.header("🧺 Targeted Basket Affinity Report")
    st.caption("Identify items most frequently purchased alongside a selected Target at Item / Brand / Category level.")
    df, _, _, _ = local_filters_block(raw_df, key_prefix="aff")
    if df.empty:
        st.info("No data for selected filters.")
        return

    valid_grans = [g for g in ["Item", "Brand_Name", "Category"] if g in df.columns]
    default_index = valid_grans.index("Item") if "Item" in valid_grans else 0
    granularity = st.radio("Granularity", valid_grans, index=default_index, horizontal=True, key="aff_granularity_sel")
    key_col = granularity

    unique_targets = sorted(df[key_col].astype(str).unique().tolist())
    target_product = st.selectbox("Select Target", unique_targets, index=0 if unique_targets else None, key="aff_target_sel")
    if not target_product:
        st.warning("Please select a Target to generate the affinity report.")
        return

    all_rules_df = calculate_affinity_rules(df, key_col)
    if all_rules_df.empty:
        st.info("No co-basket pairs found. Try widening the date range or filters.")
        return

    target_rules_df = all_rules_df[all_rules_df["Antecedent"] == str(target_product)].copy()
    if target_rules_df.empty:
        st.info(f"No outbound associations found for **{target_product}**.")
        return

    target_rules_df["Associated Item"] = target_rules_df["Consequent"]
    display_df = target_rules_df[[
        "Associated Item",
        "Confidence (A->B)",
        "Lift (A,B)",
        "Total Co-Baskets",
        "Total_Antecedent_Qty_in_CoBasket",
        "Avg_Antecedent_Qty_in_CoBasket",
        "Total_CoBasket_Sales_Value",
        "Avg_CoBasket_Spend",
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

    st.subheader(f"Items Associated with: {target_product}")
    st.markdown(
        f"**How to read:** Higher **Lift** and **Confidence** indicate stronger historical co-purchase with **{target_product}**. "
        f"Sales metrics reflect the **entire basket** value where both items appear."
    )
    st.dataframe(
        display_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Rank": st.column_config.NumberColumn("Rank", format="%d"),
            "Associated Item": st.column_config.TextColumn("Associated Item", width="large"),
            "Confidence (%)": st.column_config.ProgressColumn(
                f"Confidence ({target_product} → Associated)", min_value=0, max_value=100, format="%.1f%%",
                help=f"Share of {target_product} baskets that also contained the Associated Item."
            ),
            "Lift": st.column_config.NumberColumn(
                f"Lift ({target_product}, Associated)", format="%.2f",
                help="How much more often the pair appears together vs if independent (>1 = positive association)."
            ),
            "Total Co-Basket Sales": st.column_config.NumberColumn("Total Co-Basket Sales", format="$%.2f"),
            "Avg Co-Basket Spend": st.column_config.NumberColumn("Avg Co-Basket Spend", format="$%.2f"),
            f"Total Qty of {target_product}": st.column_config.NumberColumn(f"Total Qty of {target_product}", format="%d"),
            f"Avg Qty of {target_product}": st.column_config.NumberColumn(f"Avg Qty of {target_product}", format="%.2f"),
            "Total Co-Baskets": st.column_config.NumberColumn("Total Co-Baskets", format="%d"),
        },
        column_order=[
            "Rank", "Associated Item", "Lift", "Confidence (%)",
            "Total Co-Basket Sales", "Avg Co-Basket Spend",
            f"Total Qty of {target_product}", f"Avg Qty of {target_product}",
            "Total Co-Baskets",
        ],
    )

def tab_price_ladder(raw_df: pd.DataFrame):
    st.header("💲 Price Ladder")
    df, _, _, _ = local_filters_block(raw_df, key_prefix="pl")
    if df.empty or "Unit_Price" not in df.columns:
        st.info("No price data available.")
        return
    levels = [l for l in ["Item", "Brand_Name", "Category"] if l in df.columns]
    ladder_level = st.selectbox("Price Level", levels, index=0, key="pl_level_sel")
    sort_metric = st.selectbox("Sort by", ["Median Price", "Average Price", "Count"], index=0, key="pl_sort_sel")
    agg = (df.groupby(ladder_level).agg(Avg_Price=("Unit_Price","mean"),
                                        Median_Price=("Unit_Price","median"),
                                        Count=("Unit_Price","size")).reset_index())
    sort_col = {"Median Price":"Median_Price","Average Price":"Avg_Price","Count":"Count"}[sort_metric]
    agg = agg.sort_values(sort_col, ascending=False)
    fig = px.bar(agg, x=ladder_level, y="Median_Price",
                 hover_data={"Avg_Price":":.2f","Median_Price":":.2f","Count":":,"},
                 title=f"Price Ladder (Median) by {ladder_level}", text_auto=".2f")
    fig.update_layout(xaxis_title=ladder_level, yaxis_title="Median Price")
    st.plotly_chart(fig, use_container_width=True)
    if ladder_level == "Item":
        st.caption("Distribution of Unit_Price by Item (jittered)")
        tmp = df.copy()
        tmp["jitter"] = np.random.uniform(-0.2, 0.2, size=len(tmp))
        fig_sc = px.strip(tmp, x="Item", y="Unit_Price", title="Unit Price Distribution by Item")
        st.plotly_chart(fig_sc, use_container_width=True)

def tab_store_map(raw_df: pd.DataFrame):
    st.header("🗺️ Store Map (Hover for KPIs)")
    df, _, _, _ = local_filters_block(raw_df, key_prefix="map")
    needed_cols = {"Store_ID", "Store_Latitude", "Store_Longitude"}
    if not needed_cols.issubset(df.columns):
        st.info("Store location columns not found. Expecting Store_ID, Store_Latitude, Store_Longitude.")
        return
    if df.empty:
        st.info("No data for selected filters.")
        return

    with st.expander("ℹ️ Metric Definitions", expanded=False):
        st.markdown("""
- **Total Sales**: Sum of `Total_Sale` within current filters.
- **Transactions**: Count of unique `Transaction_ID`.
- **Units**: Sum of `Quantity`.
- **Spend/Basket**: `Total Sales ÷ Transactions`.
- **ASP**: `Total Sales ÷ Units`.
        """)

    df_map = df.dropna(subset=["Store_Latitude", "Store_Longitude"]).copy()
    if df_map.empty:
        st.info("No stores with coordinates in the current filter selection.")
        return

    kpi_agg = (df_map.groupby("Store_ID", as_index=False)
                    .agg(Total_Sale=("Total_Sale","sum"),
                         Quantity=("Quantity","sum"),
                         Transactions=("Transaction_ID","nunique")))
    kpi_agg["Spend_per_Basket"] = np.where(kpi_agg["Transactions"]>0,
                                           kpi_agg["Total_Sale"]/kpi_agg["Transactions"],0.0)
    kpi_agg["ASP"] = np.where(kpi_agg["Quantity"]>0,
                              kpi_agg["Total_Sale"]/kpi_agg["Quantity"],0.0)

    loc_cols = ["Store_ID","Store_City","Store_State","Store_Latitude","Store_Longitude"]
    locs = df_map[[c for c in loc_cols if c in df_map.columns]].drop_duplicates(subset=["Store_ID"])
    store_kpis = kpi_agg.merge(locs, on="Store_ID", how="left")

    if {"Store_City", "Store_State"}.issubset(store_kpis.columns):
        store_kpis["Store_Label"] = (store_kpis["Store_ID"].astype(str) + " — " +
                                     store_kpis["Store_City"].astype(str) + ", " +
                                     store_kpis["Store_State"].astype(str))
    else:
        store_kpis["Store_Label"] = store_kpis["Store_ID"].astype(str)

    col1, col2 = st.columns([1, 1])
    size_metric = col1.selectbox("Bubble Size", ["Total_Sale","Transactions","Quantity"], index=0, key="map_size_sel")
    color_metric = col2.selectbox("Bubble Color", ["Total_Sale","Spend_per_Basket","ASP","Transactions","Quantity"], index=0, key="map_color_sel")

    fig = px.scatter_mapbox(
        store_kpis, lat="Store_Latitude", lon="Store_Longitude",
        size=size_metric, color=color_metric, size_max=28,
        zoom=3, center={"lat":39.5, "lon":-98.35},
        hover_name="Store_Label",
        custom_data=np.stack([
            store_kpis["Total_Sale"].values,
            store_kpis["Transactions"].values,
            store_kpis["Quantity"].values,
            store_kpis["Spend_per_Basket"].values,
            store_kpis["ASP"].values,
        ], axis=-1),
        mapbox_style="open-street-map", title=None
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Total Sales: $%{customdata[0]:,.0f}<br>"
            "Transactions: %{customdata[1]:,}<br>"
            "Units: %{customdata[2]:,}<br>"
            "Spend/Basket: $%{customdata[3]:,.2f}<br>"
            "ASP: $%{customdata[4]:,.2f}<br>"
            "<extra></extra>"
        )
    )
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)
    st.plotly_chart(fig, use_container_width=True, key=f"store_map_{size_metric}_{color_metric}")

def tab_assortment_space(raw_df: pd.DataFrame):
    st.header("📦 Assortment & Space Optimization")
    df, _, start_date, end_date = local_filters_block(raw_df, key_prefix="aso")
    if df.empty:
        st.info("No data for selected filters.")
        return

    def _weeks_in_range(df_in: pd.DataFrame) -> int:
        try:
            n = df_in.set_index("Date").resample("W-SUN")["Total_Sale"].sum().shape[0]
            return max(int(n), 1)
        except Exception:
            span_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
            return max(int(np.ceil(span_days/7.0)), 1)

    n_weeks = _weeks_in_range(df)
    price_col = "Unit_Price" if "Unit_Price" in df.columns else None

    # --- A) SKU Productivity ---
    st.subheader("A) SKU Productivity Report")
    prod_dim_options = [c for c in ["Item","Brand_Name"] if c in df.columns and df[c].notna().any()]
    if not prod_dim_options:
        st.info("No valid SKU fields (Item/Brand_Name).")
        return
    with st.expander("Controls — Productivity", expanded=True):
        prod_dim = st.selectbox("Granularity", options=prod_dim_options, index=0, key="aso_prod_dim_sel")
        can_price = bool(price_col)
        show_price = st.checkbox("Show median Unit Price column", value=True if can_price else False, disabled=not can_price, key="aso_prod_price_cb")

    sku_store = df.groupby([prod_dim,"Store_ID"], dropna=False)["Total_Sale"].sum().reset_index()
    active_stores = sku_store.groupby(prod_dim)["Store_ID"].nunique().rename("Active_Stores")
    if "Category" not in df.columns:
        st.info("Category column missing; cannot compute % of category sales.")
        return
    sku_sales = df.groupby([prod_dim,"Category"], dropna=False)["Total_Sale"].sum().rename("SKU_Sales")
    cat_sales = df.groupby("Category", dropna=False)["Total_Sale"].sum().rename("Category_Sales")

    prod_df = sku_sales.reset_index().merge(active_stores, on=prod_dim, how="left")
    prod_df["Active_Stores"] = prod_df["Active_Stores"].fillna(0).astype(int)
    prod_df["Weeks"] = n_weeks
    prod_df["Velocity_$/SKU/Store/Week"] = np.where(
        (prod_df["Active_Stores"]>0) & (prod_df["Weeks"]>0),
        prod_df["SKU_Sales"]/(prod_df["Active_Stores"]*prod_df["Weeks"]),
        0.0
    )
    prod_df = prod_df.merge(cat_sales.reset_index(), on="Category", how="left")
    prod_df["% of Category Sales"] = np.where(
        prod_df["Category_Sales"]>0,
        100.0*prod_df["SKU_Sales"]/prod_df["Category_Sales"],
        0.0
    )
    if not prod_df.empty:
        prod_df["Velocity_Rank_in_Category"] = (
            prod_df.groupby("Category")["Velocity_$/SKU/Store/Week"].rank(ascending=False, method="dense").astype(int)
        )

    if price_col and show_price:
        med_price = df.groupby(prod_dim)[price_col].median().rename("Median_Unit_Price")
        prod_df = prod_df.merge(med_price.reset_index(), on=prod_dim, how="left")

    meta_cols = []
    if prod_dim != "Brand_Name" and "Brand_Name" in df.columns: meta_cols.append("Brand_Name")
    if prod_dim != "Item" and "Item" in df.columns: meta_cols.append("Item")
    for c in meta_cols:
        try:
            top_map = (df.groupby([prod_dim,c])["Total_Sale"].sum().reset_index()
                         .sort_values([prod_dim,"Total_Sale"], ascending=[True,False])
                         .drop_duplicates(subset=[prod_dim])[[prod_dim,c]])
            prod_df = prod_df.merge(top_map, on=prod_dim, how="left")
        except Exception:
            pass

    show_cols = [prod_dim, "Category", "Active_Stores", "Weeks",
                 "Velocity_$/SKU/Store/Week", "SKU_Sales", "Category_Sales",
                 "% of Category Sales", "Velocity_Rank_in_Category"]
    if "Median_Unit_Price" in prod_df.columns and show_price:
        show_cols.insert(1, "Median_Unit_Price")
    for c in ["Brand_Name","Item"]:
        if c in prod_df.columns and c not in show_cols:
            show_cols.append(c)

    prod_df_view = prod_df[[c for c in show_cols if c in prod_df.columns]].copy()
    prod_df_view = prod_df_view.loc[:, ~prod_df_view.columns.duplicated()]
    prod_df_view = prod_df_view.sort_values(["Category","Velocity_$/SKU/Store/Week"], ascending=[True,False]).reset_index(drop=True)

    # Column helps
    id_label = "Item" if prod_dim == "Item" else "Brand"
    st.dataframe(
        prod_df_view,
        hide_index=True,
        use_container_width=True,
        column_config={
            prod_dim: st.column_config.TextColumn(id_label, help="SKU grain used in this table."),
            "Category": st.column_config.TextColumn("Category", help="Product category of the SKU."),
            "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d", help="Stores where the SKU sold at least once in the period."),
            "Weeks": st.column_config.NumberColumn("Weeks", format="%d", help="Anchored weekly buckets in the selected range."),
            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f",
                                                                        help="SKU_Sales ÷ Active_Stores ÷ Weeks."),
            "SKU_Sales": st.column_config.NumberColumn("SKU Sales", format="$%.0f"),
            "Category_Sales": st.column_config.NumberColumn("Category Sales", format="$%.0f"),
            "% of Category Sales": st.column_config.ProgressColumn("% of Category Sales", min_value=0, max_value=100, format="%.2f%%"),
            "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d", help="Dense rank by Velocity within Category."),
            "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f")
        }
    )

    st.download_button(
        "📥 Download SKU Productivity (CSV)",
        data=prod_df_view.to_csv(index=False).encode("utf-8"),
        file_name="sku_productivity.csv",
        mime="text/csv",
        key="dl_aso_prod"
    )

    st.divider()

    # --- B) SKU Rationalization ---
    st.subheader("B) SKU Rationalization Tool")
    st.markdown(
        """
        <div style="margin-top:-6px;margin-bottom:8px;">
        <em>How to use:</em> Set thresholds for <b>low velocity</b> (percentile), minimum <b>category share</b>, and optional <b>similar price-tier</b> redundancy within brand+category.  
        <br><em>Interpretation:</em> A SKU is flagged if any of these are true. Sort the table to prioritize action.
        </div>
        """,
        unsafe_allow_html=True
    )
    with st.expander("Controls — Rationalization", expanded=True):
        perc = st.slider("Low Velocity threshold (percentile)", 5, 50, 25, step=5, key="aso_lowvel_pct_sel")
        share_min = st.number_input("Minimum Category Share to avoid flag (%, e.g., 1.0)", min_value=0.0, max_value=100.0, value=1.0, step=0.5, key="aso_share_min_num")
        price_var = st.slider("Price tier proximity for redundancy (± % of median Unit Price)", 1, 30, 10, step=1, key="aso_price_band_sel")
        show_only_flagged = st.checkbox("Show only flagged SKUs", value=True, key="aso_rat_flagged_only_cb")

    rat_df = prod_df.copy()
    rat_df["Low_Velocity_Flag"] = False
    for cat, grp in rat_df.groupby("Category"):
        cutoff = np.percentile(grp["Velocity_$/SKU/Store/Week"], perc) if len(grp) else 0.0
        idx = rat_df["Category"] == cat
        rat_df.loc[idx, "Low_Velocity_Flag"] = rat_df.loc[idx, "Velocity_$/SKU/Store/Week"] <= cutoff
    rat_df["Low_Share_Flag"] = rat_df["% of Category Sales"] < share_min

    if price_col and "Median_Unit_Price" in rat_df.columns and "Brand_Name" in rat_df.columns:
        base = rat_df.copy()
        base["Redundancy_Group"] = base["Brand_Name"].astype(str) + " | " + base["Category"].astype(str)
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
        rat_df["Redundancy_Group"] = (rat_df.get("Brand_Name", "").astype(str) + " | " + rat_df["Category"].astype(str))
        rat_df["Similar_Price_Tier"] = False
        rat_df["Redundant_Candidate"] = False

    rat_df["Rationalize?"] = rat_df["Low_Velocity_Flag"] | rat_df["Low_Share_Flag"] | rat_df["Redundant_Candidate"]

    total_skus = int(rat_df.shape[0])
    flagged_count = int(rat_df["Rationalize?"].sum())
    pct_flagged = (flagged_count / total_skus * 100.0) if total_skus else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("SKUs evaluated", f"{total_skus:,}")
    c2.metric("Flagged SKUs", f"{flagged_count:,}")
    c3.metric("% Flagged", f"{pct_flagged:.1f}%")

    flagged = rat_df[rat_df["Rationalize?"]].copy()
    if not flagged.empty:
        reason_counts = pd.DataFrame({
            "Reason": ["Low Velocity", "Low Share", "Similar Price-Tier Duplicate"],
            "Count": [
                int(flagged["Low_Velocity_Flag"].sum()),
                int(flagged["Low_Share_Flag"].sum()),
                int(flagged["Redundant_Candidate"].sum())
            ]
        })
        fig_reason = px.bar(reason_counts, x="Reason", y="Count", title="Flag Reasons (count)", text_auto=True)
        st.plotly_chart(fig_reason, use_container_width=True)
    else:
        st.info("No SKUs are currently flagged with the selected thresholds.")

    view_df = flagged if show_only_flagged else rat_df
    # safe columns (avoid duplicates)
    rat_cols = [
        prod_dim, "Brand_Name", "Item", "Category",
        "Median_Unit_Price", "Velocity_$/SKU/Store/Week",
        "% of Category Sales", "Velocity_Rank_in_Category",
        "Low_Velocity_Flag", "Low_Share_Flag", "Redundant_Candidate", "Rationalize?"
    ]
    rat_cols = [c for c in rat_cols if c in view_df.columns]
    view_df = view_df[rat_cols].copy()
    view_df = view_df.loc[:, ~view_df.columns.duplicated()]
    view_df = view_df.sort_values(["Rationalize?","Category","Velocity_$/SKU/Store/Week"], ascending=[False,True,True]).reset_index(drop=True)

    id_label = "Item" if prod_dim == "Item" else "Brand"
    st.dataframe(
        view_df,
        hide_index=True,
        use_container_width=True,
        column_config={
            prod_dim: st.column_config.TextColumn(id_label, help="SKU grain used in this view (Item or Brand)."),
            "Brand_Name": st.column_config.TextColumn("Brand", help="Brand associated with the SKU (when viewing by Item)."),
            "Item": st.column_config.TextColumn("Item", help="Top-selling item for the brand (when viewing by Brand)."),
            "Category": st.column_config.TextColumn("Category", help="Product category."),
            "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f"),
            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f"),
            "% of Category Sales": st.column_config.NumberColumn("% of Category Sales", format="%.2f"),
            "Velocity_Rank_in_Category": st.column_config.NumberColumn("Rank in Category", format="%d"),
            "Low_Velocity_Flag": st.column_config.CheckboxColumn("Low Velocity"),
            "Low_Share_Flag": st.column_config.CheckboxColumn("Low Share"),
            "Redundant_Candidate": st.column_config.CheckboxColumn("Similar Price-Tier Duplicate"),
            "Rationalize?": st.column_config.CheckboxColumn("Flag"),
        }
    )
    st.download_button(
        "📥 Download Rationalization Output (CSV)",
        data=view_df.to_csv(index=False).encode("utf-8"),
        file_name="sku_rationalization.csv",
        mime="text/csv",
        key="dl_aso_rat"
    )

    st.divider()

    # --- C) Opportunity Map ---
    st.subheader("C) Assortment Opportunity Map")
    st.markdown(
        """
        <div style="margin-top:-6px;margin-bottom:8px;">
        <em>How to use:</em> Choose geography & product grain. We compare each geo’s share to overall to compute an <b>Index</b>.  
        <br><em>Interpretation:</em> Index &gt; 100 = over-indexing; Index &lt; 100 = under-indexing.
        </div>
        """,
        unsafe_allow_html=True
    )
    geo_cols_all = [c for c in ["Store_State","Store_City","Store_ID"] if c in df.columns]
    if not geo_cols_all:
        st.info("No geography columns found (need Store_State/Store_City/Store_ID).")
        return

    with st.expander("Controls — Opportunity Map", expanded=True):
        geo_col = st.selectbox("Geo Dimension", options=geo_cols_all, index=0, key="aso_geo_sel")
        map_dim = st.selectbox("Analyze by", options=[c for c in ["Item","Brand_Name"] if c in df.columns], index=0, key="aso_geo_dim_sel")
        topn_geo = st.slider("Top-N by Geo (highest over-index)", 3, 15, 5, key="aso_geo_topn_sel")

    overall_sales = df.groupby(map_dim)["Total_Sale"].sum().rename("Overall_Sales")
    overall_total = overall_sales.sum()
    overall_share = (overall_sales / overall_total).rename("Overall_Share").reset_index()

    geo_sales = df.groupby([geo_col, map_dim])["Total_Sale"].sum().rename("Geo_Sales").reset_index()
    geo_totals = df.groupby(geo_col)["Total_Sale"].sum().rename("Geo_Total").reset_index()
    geo_share = geo_sales.merge(geo_totals, on=geo_col, how="left")
    geo_share["Geo_Share"] = np.where(geo_share["Geo_Total"]>0, geo_share["Geo_Sales"]/geo_share["Geo_Total"], 0.0)
    geo_share = geo_share.merge(overall_share, on=map_dim, how="left")
    geo_share["Index_vs_Overall"] = np.where(geo_share["Overall_Share"]>0, 100.0*geo_share["Geo_Share"]/geo_share["Overall_Share"], 0.0)

    top_by_geo = (geo_share.sort_values(["Index_vs_Overall"], ascending=False)
                         .groupby(geo_col).head(topn_geo).reset_index(drop=True))

    st.dataframe(
        top_by_geo[[geo_col, map_dim, "Index_vs_Overall", "Geo_Share", "Overall_Share", "Geo_Sales"]],
        hide_index=True, use_container_width=True,
        column_config={
            geo_col: st.column_config.TextColumn(geo_col.replace("_"," "), help="Selected geography level."),
            map_dim: st.column_config.TextColumn(map_dim.replace("_"," "), help="Selected product grain."),
            "Index_vs_Overall": st.column_config.NumberColumn("Index vs Overall", format="%.1f",
                                                              help="100 × Geo_Share ÷ Overall_Share."),
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

    st.download_button(
        "📥 Download Opportunity Map (CSV)",
        data=geo_share.to_csv(index=False).encode("utf-8"),
        file_name="assortment_opportunity_map.csv",
        mime="text/csv",
        key="dl_aso_opp"
    )

    st.divider()

    # --- D) New Item Tracker ---
    st.subheader("D) New Item Tracker")
    if "Item" not in df.columns:
        st.info("Item column not found for New Item Tracker.")
        return
    with st.expander("Controls — New Items", expanded=True):
        new_window = st.select_slider("Define 'New' as items first sold within the last:", options=[30,60,90,120,180], value=90, key="aso_new_window_sel")
        tier_bins = st.selectbox("Price Tiering (within Category)", options=[3,4,5], index=0, key="aso_tiers_sel")

    # First sale computed on filtered view to respect your local filters
    all_first = df.groupby("Item")["Date"].min().rename("First_Sale_Date").reset_index()
    new_cutoff = pd.to_datetime(end_date) - pd.Timedelta(days=int(new_window))
    all_first["Is_New"] = all_first["First_Sale_Date"] >= new_cutoff

    f = df.merge(all_first, on="Item", how="left")
    new_items = f[f["Is_New"] == True].copy()
    if new_items.empty:
        st.info(f"No items with first sale within the last {new_window} days for current filters.")
        return

    price_col_here = "Unit_Price" if "Unit_Price" in df.columns else None
    price_meta = None
    if price_col_here:
        med_price_item = f.groupby("Item")[price_col_here].median().rename("Median_Unit_Price")
        cats_item = f.groupby("Item")["Category"].agg(lambda x: x.mode().iloc[0] if len(x) else None).rename("Category_Assigned")
        price_meta = pd.concat([med_price_item, cats_item], axis=1).reset_index()
        def _apply_tiers(g):
            g = g.sort_values("Median_Unit_Price")
            g["Price_Tier"] = pd.qcut(g["Median_Unit_Price"].rank(method="first"), q=tier_bins, labels=[f"T{i+1}" for i in range(tier_bins)])
            return g
        tier_ref = price_meta.groupby("Category_Assigned", group_keys=True).apply(_apply_tiers).reset_index(drop=True)
        new_items = new_items.merge(price_meta, on="Item", how="left")
        new_items = new_items.merge(tier_ref[["Item","Price_Tier"]], on="Item", how="left")
    else:
        new_items["Median_Unit_Price"] = np.nan
        new_items["Category_Assigned"] = new_items["Category"]
        new_items["Price_Tier"] = "NA"

    perf_base = (f.groupby(["Item","Category"], dropna=False)["Total_Sale"].sum().rename("Item_Sales").reset_index()
                   .merge(f.groupby(["Item"])["Store_ID"].nunique().rename("Active_Stores").reset_index(),
                          on="Item", how="left"))
    perf_base["Weeks"] = n_weeks
    perf_base["Velocity_$/SKU/Store/Week"] = np.where(
        (perf_base["Active_Stores"]>0) & (perf_base["Weeks"]>0),
        perf_base["Item_Sales"]/(perf_base["Active_Stores"]*perf_base["Weeks"]),
        0.0
    )
    new_perf = new_items[["Item","Category","Price_Tier"]].drop_duplicates().merge(perf_base, on=["Item","Category"], how="left")

    # Benchmark: non-new in same category (+tier if available)
    bench = f.merge(all_first[all_first["Is_New"]==False][["Item"]].assign(Not_New=True), on="Item", how="inner")
    if price_meta is not None:
        bench = bench.merge(price_meta, on="Item", how="left")
    else:
        bench["Category_Assigned"] = bench["Category"]; bench["Price_Tier"] = "NA"

    bench_perf = (bench.groupby(["Item","Category_Assigned","Price_Tier"], dropna=False)["Total_Sale"].sum().rename("Item_Sales").reset_index()
                    .merge(bench.groupby(["Item"])["Store_ID"].nunique().rename("Active_Stores").reset_index(),
                           on="Item", how="left"))
    bench_perf["Weeks"] = n_weeks
    bench_perf["Velocity_$/SKU/Store/Week"] = np.where(
        (bench_perf["Active_Stores"]>0) & (bench_perf["Weeks"]>0),
        bench_perf["Item_Sales"]/(bench_perf["Active_Stores"]*bench_perf["Weeks"]),
        0.0
    )

    if "Price_Tier" in bench_perf.columns and bench_perf["Price_Tier"].notna().any():
        bench_avg = bench_perf.groupby(["Category_Assigned","Price_Tier"])["Velocity_$/SKU/Store/Week"].mean().rename("Benchmark_Velocity").reset_index()
        new_perf = new_perf.merge(bench_avg, left_on=["Category","Price_Tier"], right_on=["Category_Assigned","Price_Tier"], how="left").drop(columns=["Category_Assigned"])
    else:
        bench_avg = bench_perf.groupby(["Category_Assigned"])["Velocity_$/SKU/Store/Week"].mean().rename("Benchmark_Velocity").reset_index()
        new_perf = new_perf.merge(bench_avg, left_on=["Category"], right_on=["Category_Assigned"], how="left").drop(columns=["Category_Assigned"])

    new_perf["Benchmark_Velocity"] = new_perf["Benchmark_Velocity"].fillna(0.0)
    new_perf["Velocity_vs_Benchmark_%"] = np.where(
        new_perf["Benchmark_Velocity"]>0,
        100.0*new_perf["Velocity_$/SKU/Store/Week"]/new_perf["Benchmark_Velocity"],
        np.nan
    )
    if price_meta is not None:
        new_perf = new_perf.merge(price_meta[["Item","Median_Unit_Price"]], on="Item", how="left")

    new_perf = new_perf.sort_values(["Category","Velocity_vs_Benchmark_%"], ascending=[True,False]).reset_index(drop=True)

    st.dataframe(
        new_perf[[c for c in ["Item","Category","Price_Tier","Median_Unit_Price","Active_Stores","Weeks",
                              "Velocity_$/SKU/Store/Week","Benchmark_Velocity","Velocity_vs_Benchmark_%"] if c in new_perf.columns]],
        hide_index=True, use_container_width=True,
        column_config={
            "Median_Unit_Price": st.column_config.NumberColumn("Median Unit Price", format="$%.2f"),
            "Velocity_$/SKU/Store/Week": st.column_config.NumberColumn("Velocity ($/SKU/store/week)", format="$%.2f"),
            "Benchmark_Velocity": st.column_config.NumberColumn("Benchmark Velocity", format="$%.2f"),
            "Velocity_vs_Benchmark_%": st.column_config.NumberColumn("Velocity vs Benchmark (%)", format="%.1f"),
            "Active_Stores": st.column_config.NumberColumn("Active Stores", format="%d"),
            "Weeks": st.column_config.NumberColumn("Weeks", format="%d"),
        }
    )
    try:
        chart_df = new_perf.dropna(subset=["Velocity_vs_Benchmark_%"]).head(20)
        fig_new = px.bar(chart_df, x="Item", y="Velocity_vs_Benchmark_%", color="Category",
                         title="New Item Velocity vs Benchmark (%) — Top 20", text_auto=".1f")
        fig_new.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_new, use_container_width=True)
    except Exception:
        pass
    st.download_button(
        "📥 Download New Item Tracker (CSV)",
        data=new_perf.to_csv(index=False).encode("utf-8"),
        file_name="new_item_tracker.csv",
        mime="text/csv",
        key="dl_aso_new"
    )

# ---------------------------------
# Data Load (no upload)
# ---------------------------------
DATA_PATH = "cstorereal.csv"
if not os.path.exists(DATA_PATH):
    st.error(f"CSV not found at: {os.path.abspath(DATA_PATH)}")
    st.stop()

try:
    raw_df = load_and_normalize(DATA_PATH)
except Exception as e:
    st.error(f"Error loading or cleaning data: {e}")
    st.stop()

# ---------------------------------
# Global Sidebar: Department only
# ---------------------------------
st.sidebar.header("Department")

TAB_ORDER = [
    "📊 KPI Overview",
    "📈 KPI Trends",
    "🏆 Top-N Views",
    "🧺 Basket Affinity",
    "💲 Price Ladder",
    "🗺️ Store Map",
    "📦 Assortment & Space Optimization",
]

DEPARTMENT_TO_TABS = {
    "Strategy & Finance": [
        "📊 KPI Overview",
        "📈 KPI Trends",
    ],
    "Sales & Category": [
        "📊 KPI Overview",
        "🏆 Top-N Views",
        "🧺 Basket Affinity",
        "💲 Price Ladder",
        "🗺️ Store Map",
    ],
    "Marketing & CRM": [
        "📊 KPI Overview",
        "📈 KPI Trends",
        "🧺 Basket Affinity",
        "🏆 Top-N Views",
    ],
    "Operations & Supply": [
        "🗺️ Store Map",
        "📊 KPI Overview",
    ],
    "Merchandising & Space": [
        "📦 Assortment & Space Optimization",
        "💲 Price Ladder",
        "🏆 Top-N Views",
    ],
    "Exec & Field": [
        "📊 KPI Overview",
        "🗺️ Store Map",
        "📈 KPI Trends",
    ],
    "All Departments": TAB_ORDER,
}

dept = st.sidebar.selectbox("Choose department", list(DEPARTMENT_TO_TABS.keys()), index=0, key="dept_select")

# ---------------------------------
# Main
# ---------------------------------
if not raw_df.empty:
    with st.expander("🧯 Data Validation (Full Dataset)", expanded=False):
        rows = len(raw_df)
        baskets = raw_df["Transaction_ID"].nunique()
        min_d = raw_df["Date"].min().date()
        max_d = raw_df["Date"].max().date()
        st.caption(f"Rows: **{rows:,}** | Baskets: **{baskets:,}** | Date Range: **{min_d} → {max_d}**")
        issues = []
        if "Quantity" in raw_df.columns and (raw_df["Quantity"] <= 0).any(): issues.append("Non-positive Quantity")
        if "Total_Sale" in raw_df.columns and (raw_df["Total_Sale"] < 0).any(): issues.append("Negative Total_Sale")
        if issues: st.warning(" | ".join(issues))
        else: st.success("No obvious data issues detected.")

    allowed_tabs = [t for t in TAB_ORDER if t in DEPARTMENT_TO_TABS.get(dept, [])]
    tabs = st.tabs(allowed_tabs)

    for label, container in zip(allowed_tabs, tabs):
        with container:
            if label == "📊 KPI Overview":
                tab_kpi_overview(raw_df)
            elif label == "📈 KPI Trends":
                tab_kpi_trends(raw_df)
            elif label == "🏆 Top-N Views":
                tab_top_n(raw_df)
            elif label == "🧺 Basket Affinity":
                tab_basket_affinity(raw_df)
            elif label == "💲 Price Ladder":
                tab_price_ladder(raw_df)
            elif label == "🗺️ Store Map":
                tab_store_map(raw_df)
            elif label == "📦 Assortment & Space Optimization":
                tab_assortment_space(raw_df)
else:
    st.info("No data available.")
