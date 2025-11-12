# C-Store KPI Dashboard (Full App with Price Ladder + Store Map + Conservative Forecasts)
# - Safe session_state defaults (no post-instantiation writes)
# - Unique widget keys
# - Robust "All" multiselect
# - KPI, Trends (with conservative 1-month forecast), Top-N, Basket Affinity
# - Price Ladder visualization
# - Store Map with hover KPIs (no Mapbox token needed)

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from itertools import combinations
from collections import Counter
from typing import Dict, Tuple, List

# Forecasting
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ---------------------------------
# App Config
# ---------------------------------
st.set_page_config(page_title="C-Store KPI Dashboard", layout="wide", initial_sidebar_state="expanded")
pio.templates.default = "plotly_white"

# ---------------------------------
# UI Helpers
# ---------------------------------
def multiselect_all(label: str, options: list, key: str) -> list:
    """
    Sidebar multiselect with an 'All' option.
    Starts empty by default until the user makes a selection.
    """
    all_label = "All"
    opts = [all_label] + list(options)

    # Default to empty on first load
    prev_selection = st.session_state.get(key, [])

    # Check if previous selection was 'All'
    use_all = (set(prev_selection) == set(options)) and (len(prev_selection) > 0)

    # Default widget selection
    default_raw = [all_label] if use_all else [v for v in prev_selection if v in options]

    chosen = st.sidebar.multiselect(label, opts, default=default_raw, key=f"{key}_raw")

    if all_label in chosen:
        st.session_state[key] = list(options)
        return list(options)
    else:
        final = [v for v in chosen if v != all_label]
        st.session_state[key] = final
        return final

def to_num_clean(series: pd.Series) -> pd.Series:
    """Coerce strings like '$1,234.50' -> 1234.50 (floats)."""
    return pd.to_numeric(series.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")

def ensure_session_defaults(df: pd.DataFrame):
    """
    Initialize session_state defaults BEFORE widgets render.
    Never overwrite once widget keys exist.
    """
    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    st.session_state.setdefault("global_start_date", min_date)
    st.session_state.setdefault("global_end_date",   max_date)
    st.session_state.setdefault("global_freq",       "Weekly")
    return min_date, max_date

# ---------------------------------
# Data Load & Normalize
# ---------------------------------
@st.cache_data(show_spinner=False)
def load_and_normalize(path_or_buffer) -> pd.DataFrame:
    """Load, clean data, coerce types, and compute Total_Sale if needed."""
    try:
        df = pd.read_csv(path_or_buffer)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        return pd.DataFrame()

    # Map common source headers -> canonical names
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

    # Coerce and clean essential columns
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Transaction_ID"])

    for c in ["Transaction_ID", "Store_ID", "Item", "Brand_Name", "Category", "Payment_Method"]:
        if c in df.columns:
            df[c] = df[c].astype(str)

    # Numeric columns
    if "Unit_Price" in df.columns:
        df["Unit_Price"] = to_num_clean(df["Unit_Price"]).fillna(0.0)
    if "Quantity" in df.columns:
        df["Quantity"] = to_num_clean(df["Quantity"]).fillna(0.0)
    if "Total_Sale" in df.columns:
        df["Total_Sale"] = to_num_clean(df["Total_Sale"]).fillna(0.0)

    # Impute Total_Sale if missing/zero and we have Quantity & Unit_Price
    if {"Quantity", "Unit_Price"} <= set(df.columns):
        if "Total_Sale" not in df.columns:
            df["Total_Sale"] = (df["Quantity"] * df["Unit_Price"]).round(2)
        else:
            needs_impute = (df["Total_Sale"] == 0) | (df["Total_Sale"].isna())
            df.loc[needs_impute, "Total_Sale"] = (df.loc[needs_impute, "Quantity"] * df.loc[needs_impute, "Unit_Price"]).round(2)

    return df

# Replace with your path if needed
DATA_PATH = "cstorereal.csv"
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])

try:
    if uploaded is not None:
        raw_df = load_and_normalize(uploaded)
        st.sidebar.success("✅ CSV uploaded.")
    else:
        if not os.path.exists(DATA_PATH):
            st.warning(f"CSV not found at: {os.path.abspath(DATA_PATH)}. Upload a CSV on the sidebar.")
            st.stop()
        raw_df = load_and_normalize(DATA_PATH)
except Exception as e:
    st.error(f"Error loading or cleaning data: {e}")
    st.stop()

# ---------------------------------
# Global Filters
# ---------------------------------
min_date, max_date = ensure_session_defaults(raw_df)
st.sidebar.header("Global Filters")

stores = sorted(raw_df["Store_ID"].unique().tolist())
cats   = sorted(raw_df["Category"].unique().tolist())
brands = sorted(raw_df["Brand_Name"].unique().tolist()) if "Brand_Name" in raw_df.columns else []
prods  = sorted(raw_df["Item"].unique().tolist())
pays   = sorted(raw_df["Payment_Method"].unique().tolist()) if "Payment_Method" in raw_df.columns else []

selected_stores     = multiselect_all("Store(s)", stores, key="global_stores")
selected_categories = multiselect_all("Category", cats, key="global_categories")
selected_brands     = multiselect_all("Brand", brands, key="global_brands") if brands else []
selected_products   = multiselect_all("Product", prods, key="global_products")
selected_payments   = multiselect_all("Payment Method", pays, key="global_payments") if pays else []

start_date = st.sidebar.date_input("Start Date", value=st.session_state["global_start_date"], key="global_start_date_widget")
end_date   = st.sidebar.date_input("End Date",   value=st.session_state["global_end_date"],   key="global_end_date_widget")

# Consistent, anchored time rules (very important for correct forecasting)
RULE_MAP = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "MS"}
freq = st.sidebar.radio(
    "Time Granularity",
    ["Daily", "Weekly", "Monthly"],
    index=["Daily","Weekly","Monthly"].index(st.session_state["global_freq"]),
    horizontal=True,
    key="global_freq_widget"
)
rule = RULE_MAP[freq]

# Sync back canonical state
st.session_state["global_start_date"] = start_date
st.session_state["global_end_date"]   = end_date
st.session_state["global_freq"]       = freq

# Apply Filters
mask = (raw_df["Date"] >= pd.to_datetime(start_date)) & (raw_df["Date"] <= pd.to_datetime(end_date))
filtered_df = raw_df.loc[mask].copy()

if selected_stores:
    filtered_df = filtered_df[filtered_df["Store_ID"].isin(selected_stores)]
if selected_categories:
    filtered_df = filtered_df[filtered_df["Category"].isin(selected_categories)]
if selected_brands and "Brand_Name" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Brand_Name"].isin(selected_brands)]
if selected_products:
    filtered_df = filtered_df[filtered_df["Item"].isin(selected_products)]
if selected_payments and "Payment_Method" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Payment_Method"].isin(selected_payments)]

# ---------------------------------
# Calculations
# ---------------------------------
@st.cache_data(show_spinner="Calculating KPIs...")
def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    total_sales = df["Total_Sale"].sum() if "Total_Sale" in df.columns else 0.0
    total_qty   = df["Quantity"].sum()    if "Quantity"   in df.columns else 0.0
    tx          = df["Transaction_ID"].nunique() if "Transaction_ID" in df.columns else 0
    spend_per_basket = (total_sales / tx) if tx else 0.0
    asp = (total_sales / total_qty) if total_qty else 0.0
    return dict(total_sales=total_sales, total_qty=total_qty, tx=tx, spend_per_basket=spend_per_basket, asp=asp)

@st.cache_data(show_spinner="Calculating Trend Data...")
def calculate_trends(df: pd.DataFrame, rule: str, group_dim: str or None) -> pd.DataFrame:
    """
    Aggregates by the same anchored frequency as the forecast (D, W-SUN, MS)
    so chart & model see identical buckets.
    """
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

@st.cache_data(show_spinner="Calculating Top-N Data...")
def calculate_top_n(df: pd.DataFrame, dim: str, n: int) -> Tuple[pd.DataFrame, str]:
    top_df = (df.groupby(dim)["Total_Sale"].sum().sort_values(ascending=False).head(n).reset_index())
    return top_df, "Total_Sale"

@st.cache_data(show_spinner="Calculating Basket Associations...")
def calculate_affinity_rules(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
    """
    Symmetric pair counting over transactions for (Item/Brand/Category).
    Returns both directions (A->B and B->A) with:
      - Support, Confidence, Lift
      - Total Co-Baskets
      - Co-basket Sales (total & avg)
      - Antecedent quantity in co-baskets (total & avg)
    """
    scope = df.dropna(subset=["Transaction_ID", key_col]).copy()
    scope["Transaction_ID"] = scope["Transaction_ID"].astype(str)
    scope[key_col] = scope[key_col].astype(str)

    tx_count_total = scope["Transaction_ID"].nunique()
    if tx_count_total == 0:
        return pd.DataFrame()

    # Basket sales
    basket_sales = scope.groupby("Transaction_ID")["Total_Sale"].sum()

    # Group each transaction to a unique set of keys
    tx_keys = scope.groupby("Transaction_ID")[key_col].apply(lambda s: tuple(sorted(set(s))))

    # Item & pair counts
    item_counts = Counter()
    pair_counts = Counter()
    for keys in tx_keys:
        for k in keys:
            item_counts[k] += 1
        for a, b in combinations(keys, 2):
            pair_counts[tuple(sorted((a, b)))] += 1

    if not pair_counts:
        return pd.DataFrame()

    # Map item -> set(tx_ids) for fast co-occurrence pulls
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

        # Co-basket spend metrics
        total_co_spend = float(basket_sales.loc[list(co_tx)].sum()) if co_tx else 0.0
        avg_co_spend = (total_co_spend / count_ab) if count_ab else 0.0

        # Quantities for antecedent in co-baskets
        qty_a_in_co = qty_in_txids(a, co_tx)
        qty_b_in_co = qty_in_txids(b, co_tx)

        # A -> B
        rows.append({
            "Antecedent": a,
            "Consequent": b,
            "Total Co-Baskets": count_ab,
            "Support (A,B)": support_ab,
            "Confidence (A->B)": (count_ab / count_a) if count_a else 0.0,
            "Lift (A,B)": lift,
            "Total_Antecedent_Qty_in_CoBasket": qty_a_in_co,
            "Avg_Antecedent_Qty_in_CoBasket": (qty_a_in_co / count_ab) if count_ab else 0.0,
            "Total_CoBasket_Sales_Value": total_co_spend,
            "Avg_CoBasket_Spend": avg_co_spend,
        })
        # B -> A
        rows.append({
            "Antecedent": b,
            "Consequent": a,
            "Total Co-Baskets": count_ab,
            "Support (A,B)": support_ab,
            "Confidence (A->B)": (count_ab / count_b) if count_b else 0.0,
            "Lift (A,B)": lift,
            "Total_Antecedent_Qty_in_CoBasket": qty_b_in_co,
            "Avg_Antecedent_Qty_in_CoBasket": (qty_b_in_co / count_ab) if count_ab else 0.0,
            "Total_CoBasket_Sales_Value": total_co_spend,
            "Avg_CoBasket_Spend": avg_co_spend,
        })

    out = pd.DataFrame(rows)
    # Sort by Lift then Confidence
    return out.sort_values(["Lift (A,B)", "Confidence (A->B)"], ascending=[False, False]).reset_index(drop=True)



# ---------------------------------
# Forecast helpers (Conservative)
# ---------------------------------
def _forecast_freq_meta(rule: str):
    """Return (horizon_steps, seasonal_periods, pandas_freq, horizon_label)."""
    if rule == "D":
        return 30, 7, "D", "next 30 days"
    if rule.startswith("W-"):
        # Weekly data often suffers from short-term noise that impacts the long-term forecast.
        return 4, 52, rule, "next 4 weeks"
    if rule == "MS":
        return 1, 12, "MS", "next month"
    return 30, 7, "D", "next 30 days"

def _series_from_raw(df: pd.DataFrame, rule: str, metric: str) -> pd.Series:
    """
    Build the time series directly from filtered raw data using resample(sum),
    ensuring buckets match chart granularity and anchoring exactly.
    """
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
    """
    Conservative ETS fitting:
    - For weekly ('W-') data, uses fixed, low smoothing parameters and an
      additive trend to prevent the model from overreacting to a sharp recent drop.
    - For other data (D, MS), uses level-only trend and optimizes parameters
      to maintain a conservative forecast.
    Returns (model, residual_std) or (None, None).
    """
    use_fixed_smoothing = rule.startswith("W-")
    use_season = seasonal_periods and (len(ts) >= 2 * seasonal_periods)
    
    try:
        # Determine trend and fit parameters based on the time series frequency
        if use_fixed_smoothing:
            # Option 1 for 'W-': Fixed, low smoothing parameters to average over entire history
            # and ignore the recent dip, forcing a conservative additive trend.
            trend_type = 'add'
            fit_params = {
                'smoothing_level': 0.1,  # Low alpha: gives more weight to older data
                'smoothing_trend': 0.01, # Very low beta: keeps the trend from being affected by the recent slope
                'optimized': False       # Forces the use of the manual parameters above
            }
        else:
            # Default conservative for D/MS: level-only, let model optimize for best fit
            trend_type = None
            fit_params = {'optimized': True}

        model = ExponentialSmoothing(
            ts,
            trend=trend_type,
            damped_trend=False,
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
    """Calculate a simple Seasonal Naive forecast."""
    if seasonal_periods and len(ts) >= seasonal_periods:
        rep = np.tile(ts.iloc[-seasonal_periods:].to_numpy(), int(np.ceil(steps/seasonal_periods)))[:steps]
        return rep.astype(float)
    return np.full(steps, float(ts.iloc[-1]) if len(ts) else 0.0)

def _choose_model(ts: pd.Series, rule: str, seasonal_periods: int):
    """
    Tiny holdout (≈12% of history, min 6) to compare ETS vs seasonal-naive.
    Pick lower sMAPE to avoid overfit.
    """
    n = len(ts)
    if n < max(12, seasonal_periods + 4):
        return "naive", None, None  # too short → naive

    h = max(6, int(round(n * 0.12)))
    train, hold = ts.iloc[:-h], ts.iloc[-h:]

    # Pass the 'rule' to the fitting function
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

    if ets_err < naive_err:
        return "ets", ets_model, resid_std
    else:
        return "naive", None, None

def _forecast_series_conservative(df: pd.DataFrame, rule: str, metric: str, alpha: float = 0.05) -> pd.DataFrame:
    """
    End-to-end conservative forecast on the *aggregate* series built from df:
      - resample(sum) with anchored freq (D, W-SUN, MS)
      - choose ETS vs seasonal-naive via holdout sMAPE
      - non-negative bounds; horizon total capped to recent average ±30%
    """
    steps, seasonal_periods, freq, _ = _forecast_freq_meta(rule)
    ts = _series_from_raw(df, rule, metric).clip(lower=0)

    # Pass the 'rule' to the model choice function
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

    # Guardrail: cap horizon sum around recent average (prevents runaway)
    lookback = min(len(ts), 28 if rule == "D" else (12 if rule == "MS" else 12))
    recent_mean = float(ts.iloc[-lookback:].mean()) if lookback > 0 else 0.0
    cap_lo = recent_mean * steps * 0.7
    cap_hi = recent_mean * steps * 1.3
    horizon_sum = float(mean_fc.sum())
    
    # Apply capping logic
    if cap_hi > 0 and horizon_sum > cap_hi:
        scale = cap_hi / horizon_sum
        mean_fc *= scale; lower *= scale; upper *= scale
    elif horizon_sum < cap_lo and horizon_sum > 0:
        scale = cap_lo / horizon_sum
        mean_fc *= scale; lower *= scale; upper *= scale

    idx = pd.date_range(ts.index[-1], periods=steps + 1, freq=freq)[1:]
    return pd.DataFrame({"Date": idx, "yhat": mean_fc, "yhat_lower": lower, "yhat_upper": upper})

# ---------------------------------
# Displays
# ---------------------------------
def display_kpi_overview(df: pd.DataFrame):
    st.header("📊 KPI Overview")
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


def display_kpi_trends(df: pd.DataFrame, rule: str, selected_stores, selected_categories, selected_brands, selected_products):
    st.header("📈 KPI Trends")
    if df.empty:
        st.info("No data for selected filters.")
        return

    metric = st.selectbox(
        "Select Metric",
        ["Total_Sale", "Quantity", "Spend per Basket", "Transactions"],
        index=0,
        key="trend_metric"
    )

    # Pick a group dimension if any filter has multiple selections
    group_dim = None
    for dim, vals in {
        "Store_ID": selected_stores,
        "Category": selected_categories,
        "Brand_Name": selected_brands,
        "Item": selected_products
    }.items():
        if isinstance(vals, list) and len(vals) > 1:
            group_dim = dim
            break

    trend_df = calculate_trends(df, rule, group_dim)
    if trend_df.empty or metric not in trend_df.columns or trend_df[metric].dropna().empty:
        st.info("No trend data for current filters and metric.")
        return

    # Base line chart (historical)
    fig = px.line(
        trend_df,
        x="Date",
        y=metric,
        color=group_dim,
        title=f"{metric} Over Time" + (f" by {group_dim}" if group_dim else "")
    )
    fig.update_layout(hovermode="x unified")

    # --- Forecasts are AGGREGATE ONLY (no grouping) and only for additive metrics ---
    aggregate_only = (group_dim is None)
    can_forecast = aggregate_only and (metric in ["Total_Sale", "Quantity"])
    _, _, _, horizon_label = _forecast_freq_meta(rule)

    show_fc = st.checkbox(
        f"Show {metric} forecast ({horizon_label})",
        value=True if can_forecast else False,
        disabled=not can_forecast,
        key="trend_show_fc"
    )

    if (not aggregate_only) and metric in ["Total_Sale", "Quantity"]:
        st.caption("🔎 Forecasts are shown only on the **aggregate** view (when the chart isn’t split by Store/Category/Brand/Item).")

    if can_forecast and show_fc:
        # Forecast from RAW filtered df (aggregate); resampling happens in helper
        fc_df = _forecast_series_conservative(df, rule=rule, metric=metric, alpha=0.05)

        # Forecast line
        fig.add_scatter(
            x=fc_df["Date"], y=fc_df["yhat"],
            mode="lines",
            name=f"{metric} forecast",
            line=dict(dash="dash"),
        )
        # Confidence band
        fig.add_traces([
            dict(
                x=pd.concat([fc_df["Date"], fc_df["Date"][::-1]]),
                y=pd.concat([fc_df["yhat_upper"], fc_df["yhat_lower"][::-1]]),
                fill="toself",
                fillcolor="rgba(99,110,250,0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="95% interval"
            )
        ])

        # Summary line
        st.caption(
            f"**Forecast summary ({horizon_label})** — Projected {metric.replace('_',' ').title()}: "
            f"**{fc_df['yhat'].sum():,.0f}** "
            f"(95% CI: {fc_df['yhat_lower'].sum():,.0f} – {fc_df['yhat_upper'].sum():,.0f})"
        )

        # Optional download
        st.download_button(
            "📥 Download forecast (CSV)",
            data=fc_df.to_csv(index=False).encode("utf-8"),
            file_name=f"forecast_{metric.lower()}_{rule}.csv",
            mime="text/csv",
            key=f"dl_fc_{metric}_{rule}"
        )

    st.plotly_chart(fig, use_container_width=True)


def display_top_n_views(df: pd.DataFrame):
    st.header("🏆 Top-N Views")
    if df.empty:
        st.info("No data for selected filters.")
        return

    dims_available = [d for d in ["Category", "Brand_Name", "Store_ID", "Item"] if d in df.columns]
    if not dims_available:
        st.info("No dimensions available (Need Category/Brand_Name/Store_ID/Item).")
        return

    col1, col2 = st.columns([1, 4])
    with col1:
        dim = st.selectbox("Top-N by", dims_available, index=0, key="topn_dim")
    with col2:
        n = st.slider("N", 5, 30, 10, key="topn_n")

    top_df, y_col = calculate_top_n(df, dim, n)
    fig_bar = px.bar(top_df, x=dim, y=y_col, title=f"Top {n} {dim} by {y_col}", text_auto=".2s")
    st.plotly_chart(fig_bar, use_container_width=True)


def display_basket_affinity(df: pd.DataFrame):
    st.header("🧺 Targeted Basket Affinity Report")
    st.caption("Identify items most frequently purchased alongside a selected Target at Item / Brand / Category level.")

    if df.empty:
        st.info("No data for selected filters.")
        return

    valid_grans = [g for g in ["Item", "Brand_Name", "Category"] if g in df.columns]
    if not valid_grans:
        st.info("No valid columns for affinity (need Item/Brand_Name/Category).")
        return

    col_gran, col_target = st.columns([1, 3])
    with col_gran:
        default_index = valid_grans.index("Item") if "Item" in valid_grans else 0
        granularity = st.radio("Granularity", valid_grans, index=default_index, horizontal=True, key="aff_granularity")
        key_col = granularity

    unique_targets = sorted(df[key_col].astype(str).unique().tolist())

    with col_target:
        target_product = st.selectbox("Select Target", unique_targets, index=0 if unique_targets else None, key="aff_target")

    if not target_product:
        st.warning("Please select a Target to generate the affinity report.")
        return

    all_rules_df = calculate_affinity_rules(df, key_col)
    if all_rules_df.empty:
        st.info("No co-basket pairs found. Try widening the date range or filters.")
        return

    # Keep only rules where selected target is the antecedent
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

    # Confidence as %
    display_df["Confidence (%)"] = (display_df["Confidence (A->B)"] * 100.0).round(2)
    display_df = display_df.drop(columns=["Confidence (A->B)"])

    # Friendly column names
    display_df = display_df.rename(columns={
        "Lift (A,B)": "Lift",
        "Total_Antecedent_Qty_in_CoBasket": f"Total Qty of {target_product}",
        "Avg_Antecedent_Qty_in_CoBasket":   f"Avg Qty of {target_product}",
        "Total_CoBasket_Sales_Value":       "Total Co-Basket Sales",
        "Avg_CoBasket_Spend":               "Avg Co-Basket Spend",
    })

    # Rank & sort
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
                f"Confidence ({target_product} → Associated)",
                min_value=0, max_value=100, format="%.1f%%",
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

    st.download_button(
        f"📥 Download Associations for {target_product} (CSV)",
        data=display_df.to_csv(index=False).encode("utf-8"),
        file_name=f"association_rules_{target_product}_{key_col}.csv",
        mime="text/csv"
    )


def display_price_ladder(df: pd.DataFrame):
    st.header("💲 Price Ladder")
    if df.empty or "Unit_Price" not in df.columns:
        st.info("No price data available.")
        return

    # Choose level
    levels = [l for l in ["Item", "Brand_Name", "Category"] if l in df.columns]
    if not levels:
        st.info("Need Item/Brand_Name/Category columns.")
        return

    col_lvl, col_sort = st.columns([1,1])
    with col_lvl:
        ladder_level = st.selectbox("Price Level", levels, index=0, key="pl_level")
    with col_sort:
        sort_metric = st.selectbox("Sort by", ["Median Price", "Average Price", "Count"], index=0, key="pl_sort")

    agg = (
        df.groupby(ladder_level)
          .agg(Avg_Price=("Unit_Price","mean"),
               Median_Price=("Unit_Price","median"),
               Count=("Unit_Price","size"))
          .reset_index()
    )

    sort_col = {"Median Price":"Median_Price","Average Price":"Avg_Price","Count":"Count"}[sort_metric]
    agg = agg.sort_values(sort_col, ascending=False)

    # Bar chart
    fig = px.bar(
        agg,
        x=ladder_level,
        y="Median_Price",
        hover_data={"Avg_Price":":.2f", "Median_Price":":.2f", "Count":":,"},
        title=f"Price Ladder (Median) by {ladder_level}",
        text_auto=".2f"
    )
    fig.update_layout(xaxis_title=ladder_level, yaxis_title="Median Price")
    st.plotly_chart(fig, use_container_width=True)

    # Optional: Scatter of all unit prices vs item
    if ladder_level == "Item":
        st.caption("Distribution of Unit_Price by Item (jittered)")
        tmp = df.copy()
        tmp["jitter"] = np.random.uniform(-0.2, 0.2, size=len(tmp))
        fig_sc = px.strip(tmp, x="Item", y="Unit_Price", title="Unit Price Distribution by Item")
        st.plotly_chart(fig_sc, use_container_width=True)


def display_store_map(df: pd.DataFrame):
    st.header("🗺️ Store Map (Hover for KPIs)")
    needed_cols = {"Store_ID", "Store_Latitude", "Store_Longitude"}
    if not needed_cols.issubset(df.columns):
        st.info("Store location columns not found. Expecting Store_ID, Store_Latitude, Store_Longitude.")
        return

    if df.empty:
        st.info("No data for selected filters.")
        return

    # --- Metric definitions note ---
    with st.expander("ℹ️ Metric Definitions", expanded=False):
        st.markdown("""
- **Total Sales** — Sum of `Total_Sale` for the store within the current filters (date, category, etc.).
- **Transactions** — Count of unique `Transaction_ID` for the store within the current filters.
- **Units** — Sum of `Quantity` for the store within the current filters.
- **Spend/Basket** — `Total Sales ÷ Transactions`. Average spend per transaction.
- **ASP** — `Total Sales ÷ Units`. Average selling price per unit.
- **Bubble Size** — Control sets which metric drives marker size (e.g., Total Sales).
- **Bubble Color** — Control sets which metric drives marker color (e.g., ASP).
        """)

    # Keep only rows with valid coordinates
    df_map = df.dropna(subset=["Store_Latitude", "Store_Longitude"]).copy()
    if df_map.empty:
        st.info("No stores with coordinates in the current filter selection.")
        return

    # Build per-store KPI table from the *filtered* data
    kpi_agg = (
        df_map.groupby("Store_ID", as_index=False)
              .agg(
                  Total_Sale=("Total_Sale", "sum"),
                  Quantity=("Quantity", "sum"),
                  Transactions=("Transaction_ID", "nunique")
              )
    )
    kpi_agg["Spend_per_Basket"] = np.where(
        kpi_agg["Transactions"] > 0,
        kpi_agg["Total_Sale"] / kpi_agg["Transactions"],
        0.0
    )
    kpi_agg["ASP"] = np.where(
        kpi_agg["Quantity"] > 0,
        kpi_agg["Total_Sale"] / kpi_agg["Quantity"],
        0.0
    )

    # Unique store locations & labels from the *filtered* data
    loc_cols = ["Store_ID", "Store_City", "Store_State", "Store_Latitude", "Store_Longitude"]
    locs = df_map[[c for c in loc_cols if c in df_map.columns]].drop_duplicates(subset=["Store_ID"])
    store_kpis = kpi_agg.merge(locs, on="Store_ID", how="left")

    # Friendly label
    if {"Store_City", "Store_State"}.issubset(store_kpis.columns):
        store_kpis["Store_Label"] = (
            store_kpis["Store_ID"].astype(str) + " — " +
            store_kpis["Store_City"].astype(str) + ", " +
            store_kpis["Store_State"].astype(str)
        )
    else:
        store_kpis["Store_Label"] = store_kpis["Store_ID"].astype(str)

    # Controls
    col1, col2 = st.columns([1, 1])
    with col1:
        size_metric = st.selectbox(
            "Bubble Size",
            ["Total_Sale", "Transactions", "Quantity"],
            index=0,
            key="map_size_metric"
        )
    with col2:
        color_metric = st.selectbox(
            "Bubble Color",
            ["Total_Sale", "Spend_per_Basket", "ASP", "Transactions", "Quantity"],
            index=0,
            key="map_color_metric"
        )

    # Plot
    fig = px.scatter_mapbox(
        store_kpis,
        lat="Store_Latitude",
        lon="Store_Longitude",
        size=size_metric,
        color=color_metric,
        size_max=28,
        zoom=3,
        center={"lat": 39.5, "lon": -98.35},  # US centroid-ish
        hover_name="Store_Label",
        custom_data=np.stack([
            store_kpis["Total_Sale"].values,
            store_kpis["Transactions"].values,
            store_kpis["Quantity"].values,
            store_kpis["Spend_per_Basket"].values,
            store_kpis["ASP"].values,
        ], axis=-1),
        mapbox_style="open-street-map",
        title=None,
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
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=600)

    # Dynamic key ensures Streamlit re-renders when dropdowns change
    st.plotly_chart(fig, use_container_width=True, key=f"store_map_{size_metric}_{color_metric}")

# ---------------------------------
# Main
# ---------------------------------
if not filtered_df.empty:
    with st.expander("🧯 Data Validation & Filter Feedback", expanded=False):
        rows = len(filtered_df)
        baskets = filtered_df["Transaction_ID"].nunique()
        st.caption(
            f"Rows: **{rows:,}** | Baskets: **{baskets:,}** | "
            f"Date Range: **{pd.to_datetime(start_date).date()} → {pd.to_datetime(end_date).date()}**"
        )
        issues = []
        if "Quantity" in filtered_df.columns and (filtered_df["Quantity"] <= 0).any():
            issues.append("Non-positive Quantity")
        if "Total_Sale" in filtered_df.columns and (filtered_df["Total_Sale"] < 0).any():
            issues.append("Negative Total_Sale")
        if issues:
            st.warning(" | ".join(issues))
        else:
            st.success("No obvious data issues detected.")

    # Tabs (Price Ladder + Store Map included)
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 KPI Overview",
        "📈 KPI Trends",
        "🏆 Top-N Views",
        "🧺 Basket Affinity",
        "💲 Price Ladder",
        "🗺️ Store Map",
    ])
    with tab1:
        display_kpi_overview(filtered_df)
    with tab2:
        display_kpi_trends(filtered_df, rule, selected_stores, selected_categories, selected_brands, selected_products)
    with tab3:
        display_top_n_views(filtered_df)
    with tab4:
        display_basket_affinity(filtered_df)
    with tab5:
        display_price_ladder(filtered_df)
    with tab6:
        display_store_map(filtered_df)
else:
    st.info("No data available after applying filters.")
