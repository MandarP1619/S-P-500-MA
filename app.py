import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests

from io import StringIO
from datetime import date

import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="SMA Tactical Strategy Optimizer",
    layout="wide"
)

st.title("SMA Tactical Strategy Optimizer")

st.write(
    "This app evaluates moving-average tactical strategies against buy-and-hold "
    "and simultaneously optimizes the SMA window and buffer percentage using a "
    "weighted composite score."
)


# ============================================================
# SIDEBAR INPUTS
# ============================================================
st.sidebar.header("Base Strategy Configuration")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY"
).strip().upper()

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(1995, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today()
)

base_sma_window = st.sidebar.number_input(
    "User-Selected Base SMA",
    min_value=20,
    max_value=700,
    value=315,
    step=5
)

base_buffer_pct = st.sidebar.number_input(
    "User-Selected Base Buffer %",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.05
)

starting_balance = st.sidebar.number_input(
    "Starting Balance",
    min_value=1000,
    value=10000,
    step=1000
)

st.sidebar.header("Simultaneous Optimization Range")

minimum_sma = st.sidebar.number_input(
    "Minimum SMA Window",
    min_value=20,
    max_value=700,
    value=50,
    step=5
)

maximum_sma = st.sidebar.number_input(
    "Maximum SMA Window",
    min_value=20,
    max_value=700,
    value=400,
    step=5
)

sma_step = st.sidebar.number_input(
    "SMA Step",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

minimum_buffer = st.sidebar.number_input(
    "Minimum Buffer %",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.05
)

maximum_buffer = st.sidebar.number_input(
    "Maximum Buffer %",
    min_value=0.0,
    max_value=20.0,
    value=2.0,
    step=0.05
)

buffer_step = st.sidebar.number_input(
    "Buffer Step %",
    min_value=0.05,
    max_value=5.0,
    value=0.05,
    step=0.05
)

st.sidebar.header("Composite Score Weights")

weight_drawdown = st.sidebar.number_input(
    "Max Drawdown Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.35,
    step=0.05
)

weight_cagr = st.sidebar.number_input(
    "CAGR Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05
)

weight_sharpe = st.sidebar.number_input(
    "Sharpe Ratio Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.05
)

weight_calmar = st.sidebar.number_input(
    "Calmar Ratio Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.15,
    step=0.05
)

weight_trade_count = st.sidebar.number_input(
    "Trade Count Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.05
)

weight_time_in_market = st.sidebar.number_input(
    "Time in Market Weight",
    min_value=0.0,
    max_value=1.0,
    value=0.05,
    step=0.05
)

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()


# ============================================================
# VALIDATE INPUTS
# ============================================================
if start_date >= end_date:
    st.error("The start date must be earlier than the end date.")
    st.stop()

if minimum_sma > maximum_sma:
    st.error("Minimum SMA must be less than or equal to Maximum SMA.")
    st.stop()

if minimum_buffer > maximum_buffer:
    st.error("Minimum Buffer must be less than or equal to Maximum Buffer.")
    st.stop()

weight_total = (
    weight_drawdown
    + weight_cagr
    + weight_sharpe
    + weight_calmar
    + weight_trade_count
    + weight_time_in_market
)

if not np.isclose(weight_total, 1.0):
    st.error(
        f"Composite-score weights must add to 1.00. "
        f"Current total: {weight_total:.2f}"
    )
    st.stop()


# ============================================================
# CURRENT SETTINGS
# ============================================================
st.write("### Current Settings")

settings_col1, settings_col2, settings_col3 = st.columns(3)

with settings_col1:
    st.write(f"Ticker: **{ticker}**")
    st.write(f"Start Date: **{start_date}**")
    st.write(f"End Date: **{end_date}**")

with settings_col2:
    st.write(f"Base SMA: **{base_sma_window} days**")
    st.write(f"Base Buffer: **{base_buffer_pct:.2f}%**")
    st.write(f"Starting Balance: **${starting_balance:,.0f}**")

with settings_col3:
    st.write(
        f"SMA Search: **{minimum_sma} to {maximum_sma}, "
        f"step {sma_step}**"
    )
    st.write(
        f"Buffer Search: **{minimum_buffer:.2f}% to "
        f"{maximum_buffer:.2f}%, step {buffer_step:.2f}%**"
    )


# ============================================================
# DATA LOADING
# ============================================================
def download_fred_tbill_data(start_date_value):
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            fred_url,
            headers=headers,
            timeout=30
        )

        response.raise_for_status()

        tbill = pd.read_csv(StringIO(response.text))

        required_columns = {"observation_date", "TB3MS"}

        if not required_columns.issubset(tbill.columns):
            raise ValueError(
                f"Unexpected FRED columns: {list(tbill.columns)}"
            )

        return tbill, False

    except Exception:
        fallback = pd.DataFrame({
            "observation_date": [start_date_value],
            "TB3MS": [0.0]
        })

        return fallback, True


@st.cache_data(ttl=3600)
def load_data(ticker_value, start_date_value, end_date_value):
    asset = yf.download(
        ticker_value,
        start=start_date_value,
        end=end_date_value,
        auto_adjust=False,
        progress=False
    )

    if asset.empty:
        return None, False

    if isinstance(asset.columns, pd.MultiIndex):
        asset.columns = asset.columns.get_level_values(0)

    required_asset_columns = {"Close", "Adj Close"}

    if not required_asset_columns.issubset(asset.columns):
        return None, False

    asset = asset[["Close", "Adj Close"]].copy()

    asset = asset.rename(columns={
        "Close": "Asset_Close",
        "Adj Close": "Asset_Adj_Close"
    })

    asset = asset.reset_index()
    asset = asset.rename(columns={asset.columns[0]: "Date"})

    asset["Date"] = pd.to_datetime(
        asset["Date"],
        errors="coerce"
    ).dt.date

    tbill, used_fallback = download_fred_tbill_data(start_date_value)

    tbill = tbill.rename(columns={
        "observation_date": "Date",
        "TB3MS": "TBill_3M_Yield"
    })

    tbill = tbill[[
        "Date",
        "TBill_3M_Yield"
    ]].copy()

    tbill["Date"] = pd.to_datetime(
        tbill["Date"],
        errors="coerce"
    ).dt.date

    tbill["TBill_3M_Yield"] = pd.to_numeric(
        tbill["TBill_3M_Yield"],
        errors="coerce"
    )

    tbill = tbill.dropna(
        subset=["Date", "TBill_3M_Yield"]
    )

    full_dates = pd.DataFrame({
        "Date": pd.date_range(
            start=start_date_value,
            end=end_date_value
        ).date
    })

    tbill_daily = pd.merge(
        full_dates,
        tbill,
        on="Date",
        how="left"
    )

    tbill_daily["TBill_3M_Yield"] = (
        tbill_daily["TBill_3M_Yield"]
        .ffill()
        .fillna(0.0)
    )

    combined = pd.merge(
        asset,
        tbill_daily,
        on="Date",
        how="left"
    )

    combined["Date"] = pd.to_datetime(combined["Date"])

    combined["TBill_3M_Yield"] = (
        combined["TBill_3M_Yield"]
        .ffill()
        .fillna(0.0)
    )

    combined = combined.dropna(
        subset=["Asset_Close", "Asset_Adj_Close"]
    ).reset_index(drop=True)

    return combined, used_fallback


# ============================================================
# STRATEGY ENGINE
# ============================================================
def run_strategy(
    source_df,
    sma_window,
    starting_balance,
    buffer_pct=0.0
):
    strategy_df = source_df.copy().reset_index(drop=True)

    buffer_decimal = buffer_pct / 100.0

    strategy_df["Asset_Return"] = (
        strategy_df["Asset_Adj_Close"]
        .pct_change()
        .fillna(0.0)
    )

    strategy_df["Cash_Return"] = (
        (
            1 + strategy_df["TBill_3M_Yield"] / 100.0
        ) ** (1 / 252)
    ) - 1

    strategy_df["SMA"] = (
        strategy_df["Asset_Close"]
        .rolling(window=int(sma_window))
        .mean()
    )

    strategy_df["Upper_Buffer"] = (
        strategy_df["SMA"] * (1 + buffer_decimal)
    )

    strategy_df["Lower_Buffer"] = (
        strategy_df["SMA"] * (1 - buffer_decimal)
    )

    strategy_df["Above_Buffer"] = (
        strategy_df["Asset_Close"]
        > strategy_df["Upper_Buffer"]
    )

    strategy_df["Below_Buffer"] = (
        strategy_df["Asset_Close"]
        < strategy_df["Lower_Buffer"]
    )

    strategy_df["Confirmed_Above"] = (
        strategy_df["Above_Buffer"]
        & strategy_df["Above_Buffer"].shift(1).fillna(False)
    )

    strategy_df["Confirmed_Below"] = (
        strategy_df["Below_Buffer"]
        & strategy_df["Below_Buffer"].shift(1).fillna(False)
    )

    positions = []
    current_position = 1

    for row_index in range(len(strategy_df)):
        if strategy_df.loc[row_index, "Confirmed_Below"]:
            current_position = 0

        elif strategy_df.loc[row_index, "Confirmed_Above"]:
            current_position = 1

        positions.append(current_position)

    strategy_df["Position"] = positions

    prior_position = strategy_df["Position"].shift(1).fillna(1)

    strategy_df["Strategy_Return"] = np.where(
        prior_position == 1,
        strategy_df["Asset_Return"],
        strategy_df["Cash_Return"]
    )

    strategy_df["Strategy_Return"] = (
        strategy_df["Strategy_Return"].fillna(0.0)
    )

    strategy_df["Strategy_Value"] = (
        starting_balance
        * (1 + strategy_df["Strategy_Return"]).cumprod()
    )

    strategy_df["Buy_Hold_Value"] = (
        starting_balance
        * (1 + strategy_df["Asset_Return"]).cumprod()
    )

    strategy_df["Cash_Value"] = (
        starting_balance
        * (1 + strategy_df["Cash_Return"]).cumprod()
    )

    return strategy_df


# ============================================================
# PERFORMANCE METRICS
# ============================================================
def calculate_ending_value(values):
    clean_values = values.dropna()

    if clean_values.empty:
        return np.nan

    return clean_values.iloc[-1]


def calculate_cagr(values, dates):
    valid_mask = values.notna() & dates.notna()

    clean_values = values.loc[valid_mask]
    clean_dates = dates.loc[valid_mask]

    if len(clean_values) < 2:
        return np.nan

    years = (
        clean_dates.iloc[-1] - clean_dates.iloc[0]
    ).days / 365.25

    if years <= 0 or clean_values.iloc[0] <= 0:
        return np.nan

    return (
        clean_values.iloc[-1] / clean_values.iloc[0]
    ) ** (1 / years) - 1


def calculate_volatility(returns):
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return np.nan

    return clean_returns.std() * np.sqrt(252)


def calculate_max_drawdown(values):
    clean_values = values.dropna()

    if clean_values.empty:
        return np.nan

    running_maximum = clean_values.cummax()
    drawdown = clean_values / running_maximum - 1

    return drawdown.min()


def calculate_sharpe_ratio(returns):
    clean_returns = returns.dropna()

    if clean_returns.empty:
        return np.nan

    standard_deviation = clean_returns.std()

    if standard_deviation == 0 or pd.isna(standard_deviation):
        return np.nan

    return (
        clean_returns.mean() * 252
    ) / (
        standard_deviation * np.sqrt(252)
    )


def calculate_calmar_ratio(cagr, max_drawdown):
    if (
        pd.isna(cagr)
        or pd.isna(max_drawdown)
        or max_drawdown == 0
    ):
        return np.nan

    return cagr / abs(max_drawdown)


def calculate_strategy_metrics(strategy_df):
    cagr = calculate_cagr(
        strategy_df["Strategy_Value"],
        strategy_df["Date"]
    )

    max_drawdown = calculate_max_drawdown(
        strategy_df["Strategy_Value"]
    )

    return {
        "Ending Portfolio Value": calculate_ending_value(
            strategy_df["Strategy_Value"]
        ),
        "CAGR": cagr,
        "Volatility": calculate_volatility(
            strategy_df["Strategy_Return"]
        ),
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": calculate_sharpe_ratio(
            strategy_df["Strategy_Return"]
        ),
        "Calmar Ratio": calculate_calmar_ratio(
            cagr,
            max_drawdown
        ),
        "Number of Trades": (
            strategy_df["Position"]
            .diff()
            .abs()
            .fillna(0)
            .sum()
        ),
        "Time in Market": strategy_df["Position"].mean()
    }


# ============================================================
# COMPOSITE SCORE
# ============================================================
def normalize_higher_better(series):
    numeric_series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = numeric_series.min()
    maximum = numeric_series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or np.isclose(maximum, minimum)
    ):
        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        numeric_series - minimum
    ) / (
        maximum - minimum
    )


def normalize_lower_better(series):
    numeric_series = pd.to_numeric(
        series,
        errors="coerce"
    )

    minimum = numeric_series.min()
    maximum = numeric_series.max()

    if (
        pd.isna(minimum)
        or pd.isna(maximum)
        or np.isclose(maximum, minimum)
    ):
        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        maximum - numeric_series
    ) / (
        maximum - minimum
    )


def add_composite_score(results_df):
    scored_df = results_df.copy()

    scored_df["Abs Max Drawdown"] = (
        scored_df["Max Drawdown"].abs()
    )

    scored_df["Score_Max_Drawdown"] = normalize_lower_better(
        scored_df["Abs Max Drawdown"]
    )

    scored_df["Score_CAGR"] = normalize_higher_better(
        scored_df["CAGR"]
    )

    scored_df["Score_Sharpe"] = normalize_higher_better(
        scored_df["Sharpe Ratio"]
    )

    scored_df["Score_Calmar"] = normalize_higher_better(
        scored_df["Calmar Ratio"]
    )

    scored_df["Score_Trade_Count"] = normalize_lower_better(
        scored_df["Number of Trades"]
    )

    scored_df["Score_Time_in_Market"] = normalize_lower_better(
        scored_df["Time in Market"]
    )

    scored_df["Composite Score"] = (
        weight_drawdown
        * scored_df["Score_Max_Drawdown"]
        + weight_cagr
        * scored_df["Score_CAGR"]
        + weight_sharpe
        * scored_df["Score_Sharpe"]
        + weight_calmar
        * scored_df["Score_Calmar"]
        + weight_trade_count
        * scored_df["Score_Trade_Count"]
        + weight_time_in_market
        * scored_df["Score_Time_in_Market"]
    )

    return scored_df


# ============================================================
# FORMATTING FUNCTIONS
# ============================================================
def format_metrics_table(metrics_df):
    display_df = metrics_df.astype(object).copy()

    if "Ending Portfolio Value" in display_df.index:
        display_df.loc[
            "Ending Portfolio Value", :
        ] = metrics_df.loc[
            "Ending Portfolio Value", :
        ].apply(
            lambda value: f"${value:,.0f}"
        )

    for row_name in [
        "CAGR",
        "Volatility",
        "Max Drawdown",
        "Time in Market"
    ]:
        if row_name in display_df.index:
            display_df.loc[
                row_name, :
            ] = metrics_df.loc[
                row_name, :
            ].apply(
                lambda value: f"{value:.2%}"
            )

    if "Sharpe Ratio" in display_df.index:
        display_df.loc[
            "Sharpe Ratio", :
        ] = metrics_df.loc[
            "Sharpe Ratio", :
        ].apply(
            lambda value: f"{value:.4f}"
        )

    if "Calmar Ratio" in display_df.index:
        display_df.loc[
            "Calmar Ratio", :
        ] = metrics_df.loc[
            "Calmar Ratio", :
        ].apply(
            lambda value: f"{value:.4f}"
        )

    if "Number of Trades" in display_df.index:
        display_df.loc[
            "Number of Trades", :
        ] = metrics_df.loc[
            "Number of Trades", :
        ].apply(
            lambda value: f"{value:.0f}"
        )

    return display_df


def format_optimization_table(results_df):
    display_df = results_df.copy()

    if "Ending Portfolio Value" in display_df.columns:
        display_df["Ending Portfolio Value"] = (
            display_df["Ending Portfolio Value"]
            .apply(lambda value: f"${value:,.0f}")
        )

    percentage_columns = [
        "Buffer %",
        "CAGR",
        "Volatility",
        "Max Drawdown",
        "Time in Market",
        "Abs Max Drawdown"
    ]

    for column in percentage_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda value: f"{value:.2%}"
            )

    decimal_columns = [
        "Sharpe Ratio",
        "Calmar Ratio",
        "Score_Max_Drawdown",
        "Score_CAGR",
        "Score_Sharpe",
        "Score_Calmar",
        "Score_Trade_Count",
        "Score_Time_in_Market",
        "Composite Score"
    ]

    for column in decimal_columns:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda value: f"{value:.4f}"
            )

    if "Number of Trades" in display_df.columns:
        display_df["Number of Trades"] = (
            display_df["Number of Trades"]
            .apply(lambda value: f"{value:.0f}")
        )

    if "SMA Window" in display_df.columns:
        display_df["SMA Window"] = (
            display_df["SMA Window"]
            .apply(lambda value: f"{int(value)}")
        )

    return display_df


# ============================================================
# CHART FUNCTIONS
# ============================================================
def create_portfolio_chart(
    strategy_df,
    ticker_value,
    sma_value,
    buffer_value
):
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=(
            (
                f"{sma_value}-Day SMA Strategy with "
                f"{buffer_value:.2f}% Buffer vs Buy & Hold"
            ),
            "3-Month T-Bill Yield"
        )
    )

    figure.add_trace(
        go.Scatter(
            x=strategy_df["Date"],
            y=strategy_df["Buy_Hold_Value"],
            mode="lines",
            name=f"Buy & Hold {ticker_value}",
            hovertemplate=(
                "Date: %{x}<br>"
                "Portfolio Value: $%{y:,.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=strategy_df["Date"],
            y=strategy_df["Strategy_Value"],
            mode="lines",
            name=(
                f"{sma_value}-Day SMA, "
                f"{buffer_value:.2f}% Buffer"
            ),
            hovertemplate=(
                "Date: %{x}<br>"
                "Portfolio Value: $%{y:,.2f}"
                "<extra></extra>"
            )
        ),
        row=1,
        col=1
    )

    figure.add_trace(
        go.Scatter(
            x=strategy_df["Date"],
            y=strategy_df["TBill_3M_Yield"],
            mode="lines",
            name="3M T-Bill Yield",
            hovertemplate=(
                "Date: %{x}<br>"
                "Yield: %{y:.2f}%"
                "<extra></extra>"
            )
        ),
        row=2,
        col=1
    )

    figure.update_layout(
        height=850,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.03,
            xanchor="left",
            x=0
        )
    )

    figure.update_yaxes(
        title_text="Portfolio Value ($)",
        row=1,
        col=1
    )

    figure.update_yaxes(
        title_text="Yield (%)",
        row=2,
        col=1
    )

    figure.update_xaxes(
        title_text="Date",
        row=2,
        col=1
    )

    return figure


def create_heatmap(grid_results_df):
    heatmap_data = grid_results_df.pivot(
        index="SMA Window",
        columns="Buffer %",
        values="Composite Score"
    )

    figure = go.Figure(
        data=go.Heatmap(
            x=heatmap_data.columns,
            y=heatmap_data.index,
            z=heatmap_data.values,
            colorscale="RdYlGn",
            colorbar=dict(
                title="Composite Score"
            ),
            hovertemplate=(
                "SMA Window: %{y}<br>"
                "Buffer: %{x:.2%}<br>"
                "Composite Score: %{z:.4f}"
                "<extra></extra>"
            )
        )
    )

    figure.update_layout(
        height=750,
        title="Composite Score by SMA Window and Buffer",
        xaxis_title="Buffer Percentage",
        yaxis_title="SMA Window"
    )

    figure.update_xaxes(
        tickformat=".2%"
    )

    return figure


def create_top_combinations_chart(top_results_df):
    chart_df = top_results_df.copy()

    chart_df["Combination"] = chart_df.apply(
        lambda row: (
            f"{int(row['SMA Window'])}-Day SMA | "
            f"{row['Buffer %']:.2%}"
        ),
        axis=1
    )

    figure = go.Figure(
        data=go.Bar(
            x=chart_df["Composite Score"],
            y=chart_df["Combination"],
            orientation="h",
            hovertemplate=(
                "%{y}<br>"
                "Composite Score: %{x:.4f}"
                "<extra></extra>"
            )
        )
    )

    figure.update_layout(
        height=650,
        title="Top Simultaneous Parameter Combinations",
        xaxis_title="Composite Score",
        yaxis_title="",
        yaxis=dict(
            autorange="reversed"
        )
    )

    return figure


# ============================================================
# LOAD DATA
# ============================================================
start_date_string = start_date.strftime("%Y-%m-%d")
end_date_string = end_date.strftime("%Y-%m-%d")

raw_df, used_tbill_fallback = load_data(
    ticker,
    start_date_string,
    end_date_string
)

if raw_df is None or raw_df.empty:
    st.error(
        "No asset data was returned. Check the ticker symbol "
        "and selected date range."
    )
    st.stop()

if used_tbill_fallback:
    st.warning(
        "FRED T-bill data could not be downloaded. "
        "The app is temporarily using a 0% cash yield. "
        "Refresh the app later to retry the FRED download."
    )


# ============================================================
# BASE STRATEGY
# ============================================================
base_df = run_strategy(
    raw_df,
    base_sma_window,
    starting_balance,
    buffer_pct=base_buffer_pct
)

base_metrics = calculate_strategy_metrics(base_df)

buy_hold_cagr = calculate_cagr(
    base_df["Buy_Hold_Value"],
    base_df["Date"]
)

buy_hold_max_drawdown = calculate_max_drawdown(
    base_df["Buy_Hold_Value"]
)

cash_cagr = calculate_cagr(
    base_df["Cash_Value"],
    base_df["Date"]
)

cash_max_drawdown = calculate_max_drawdown(
    base_df["Cash_Value"]
)

base_summary = pd.DataFrame({
    (
        f"Base Strategy: {base_sma_window}-Day SMA, "
        f"{base_buffer_pct:.2f}% Buffer"
    ): [
        base_metrics["Ending Portfolio Value"],
        base_metrics["CAGR"],
        base_metrics["Volatility"],
        base_metrics["Max Drawdown"],
        base_metrics["Sharpe Ratio"],
        base_metrics["Calmar Ratio"],
        base_metrics["Number of Trades"],
        base_metrics["Time in Market"]
    ],
    f"Buy & Hold {ticker}": [
        calculate_ending_value(base_df["Buy_Hold_Value"]),
        buy_hold_cagr,
        calculate_volatility(base_df["Asset_Return"]),
        buy_hold_max_drawdown,
        calculate_sharpe_ratio(base_df["Asset_Return"]),
        calculate_calmar_ratio(
            buy_hold_cagr,
            buy_hold_max_drawdown
        ),
        0,
        1.0
    ],
    "Cash / 3M T-Bill": [
        calculate_ending_value(base_df["Cash_Value"]),
        cash_cagr,
        calculate_volatility(base_df["Cash_Return"]),
        cash_max_drawdown,
        calculate_sharpe_ratio(base_df["Cash_Return"]),
        calculate_calmar_ratio(
            cash_cagr,
            cash_max_drawdown
        ),
        0,
        0.0
    ]
}, index=[
    "Ending Portfolio Value",
    "CAGR",
    "Volatility",
    "Max Drawdown",
    "Sharpe Ratio",
    "Calmar Ratio",
    "Number of Trades",
    "Time in Market"
])

st.write("### Base Strategy Performance Summary")

st.dataframe(
    format_metrics_table(base_summary),
    use_container_width=True
)

st.plotly_chart(
    create_portfolio_chart(
        base_df,
        ticker,
        base_sma_window,
        base_buffer_pct
    ),
    use_container_width=True
)


# ============================================================
# SIMULTANEOUS SMA + BUFFER OPTIMIZATION
# ============================================================
st.write("### Simultaneous SMA and Buffer Optimization")

sma_values = list(
    range(
        int(minimum_sma),
        int(maximum_sma) + 1,
        int(sma_step)
    )
)

buffer_values = np.arange(
    minimum_buffer,
    maximum_buffer + buffer_step / 2,
    buffer_step
)

number_of_combinations = (
    len(sma_values) * len(buffer_values)
)

st.write(
    f"The optimizer will test **{number_of_combinations:,}** "
    "SMA-buffer combinations."
)

grid_results = []

progress_bar = st.progress(0)
progress_text = st.empty()

combination_counter = 0

for test_sma in sma_values:
    for test_buffer in buffer_values:
        strategy_test_df = run_strategy(
            raw_df,
            test_sma,
            starting_balance,
            buffer_pct=float(test_buffer)
        )

        result_row = calculate_strategy_metrics(
            strategy_test_df
        )

        result_row.update({
            "SMA Window": int(test_sma),
            "Buffer %": float(test_buffer) / 100.0
        })

        grid_results.append(result_row)

        combination_counter += 1

        progress_bar.progress(
            combination_counter / number_of_combinations
        )

        progress_text.write(
            f"Testing combination "
            f"{combination_counter:,} of "
            f"{number_of_combinations:,}"
        )

progress_bar.empty()
progress_text.empty()

grid_results_df = pd.DataFrame(grid_results)

grid_results_df = grid_results_df.replace(
    [np.inf, -np.inf],
    np.nan
)

grid_results_df = grid_results_df.dropna(
    subset=[
        "CAGR",
        "Max Drawdown",
        "Sharpe Ratio",
        "Calmar Ratio"
    ]
).reset_index(drop=True)

if grid_results_df.empty:
    st.error(
        "No valid optimization results were produced."
    )
    st.stop()

grid_results_df = add_composite_score(
    grid_results_df
)

grid_results_df = grid_results_df.sort_values(
    "Composite Score",
    ascending=False
).reset_index(drop=True)

grid_results_df.insert(
    0,
    "Rank",
    np.arange(1, len(grid_results_df) + 1)
)

best_result = grid_results_df.iloc[0]

best_sma = int(best_result["SMA Window"])
best_buffer_decimal = float(best_result["Buffer %"])
best_buffer_pct = best_buffer_decimal * 100.0
best_composite_score = float(
    best_result["Composite Score"]
)

st.success(
    f"Best simultaneous combination: "
    f"{best_sma}-day SMA with a "
    f"{best_buffer_pct:.2f}% buffer. "
    f"Composite score: {best_composite_score:.4f}"
)


# ============================================================
# OPTIMIZATION HEATMAP
# ============================================================
st.write("### Parameter Optimization Heatmap")

st.plotly_chart(
    create_heatmap(grid_results_df),
    use_container_width=True
)


# ============================================================
# TOP COMBINATIONS
# ============================================================
top_count = st.slider(
    "Number of top parameter combinations to display",
    min_value=5,
    max_value=100,
    value=20,
    step=5
)

top_combinations = grid_results_df.head(
    top_count
).copy()

st.plotly_chart(
    create_top_combinations_chart(top_combinations),
    use_container_width=True
)

st.write("### Top Parameter Combinations")

st.dataframe(
    format_optimization_table(top_combinations),
    use_container_width=True,
    hide_index=True
)


# ============================================================
# FINAL OPTIMIZED STRATEGY
# ============================================================
optimized_df = run_strategy(
    raw_df,
    best_sma,
    starting_balance,
    buffer_pct=best_buffer_pct
)

optimized_metrics = calculate_strategy_metrics(
    optimized_df
)

final_comparison = pd.DataFrame({
    (
        f"Base: {base_sma_window}-Day SMA, "
        f"{base_buffer_pct:.2f}% Buffer"
    ): [
        base_metrics["Ending Portfolio Value"],
        base_metrics["CAGR"],
        base_metrics["Volatility"],
        base_metrics["Max Drawdown"],
        base_metrics["Sharpe Ratio"],
        base_metrics["Calmar Ratio"],
        base_metrics["Number of Trades"],
        base_metrics["Time in Market"]
    ],
    (
        f"Optimized: {best_sma}-Day SMA, "
        f"{best_buffer_pct:.2f}% Buffer"
    ): [
        optimized_metrics["Ending Portfolio Value"],
        optimized_metrics["CAGR"],
        optimized_metrics["Volatility"],
        optimized_metrics["Max Drawdown"],
        optimized_metrics["Sharpe Ratio"],
        optimized_metrics["Calmar Ratio"],
        optimized_metrics["Number of Trades"],
        optimized_metrics["Time in Market"]
    ],
    f"Buy & Hold {ticker}": [
        calculate_ending_value(
            optimized_df["Buy_Hold_Value"]
        ),
        calculate_cagr(
            optimized_df["Buy_Hold_Value"],
            optimized_df["Date"]
        ),
        calculate_volatility(
            optimized_df["Asset_Return"]
        ),
        calculate_max_drawdown(
            optimized_df["Buy_Hold_Value"]
        ),
        calculate_sharpe_ratio(
            optimized_df["Asset_Return"]
        ),
        calculate_calmar_ratio(
            calculate_cagr(
                optimized_df["Buy_Hold_Value"],
                optimized_df["Date"]
            ),
            calculate_max_drawdown(
                optimized_df["Buy_Hold_Value"]
            )
        ),
        0,
        1.0
    ]
}, index=[
    "Ending Portfolio Value",
    "CAGR",
    "Volatility",
    "Max Drawdown",
    "Sharpe Ratio",
    "Calmar Ratio",
    "Number of Trades",
    "Time in Market"
])

st.write("### Final Optimized Strategy Performance")

st.dataframe(
    format_metrics_table(final_comparison),
    use_container_width=True
)

st.plotly_chart(
    create_portfolio_chart(
        optimized_df,
        ticker,
        best_sma,
        best_buffer_pct
    ),
    use_container_width=True
)


# ============================================================
# COMPOSITE SCORE BREAKDOWN
# ============================================================
st.write("### Winning Composite Score Breakdown")

score_breakdown = pd.DataFrame({
    "Metric": [
        "Max Drawdown",
        "CAGR",
        "Sharpe Ratio",
        "Calmar Ratio",
        "Number of Trades",
        "Time in Market"
    ],
    "Weight": [
        weight_drawdown,
        weight_cagr,
        weight_sharpe,
        weight_calmar,
        weight_trade_count,
        weight_time_in_market
    ],
    "Normalized Score": [
        best_result["Score_Max_Drawdown"],
        best_result["Score_CAGR"],
        best_result["Score_Sharpe"],
        best_result["Score_Calmar"],
        best_result["Score_Trade_Count"],
        best_result["Score_Time_in_Market"]
    ]
})

score_breakdown["Weighted Contribution"] = (
    score_breakdown["Weight"]
    * score_breakdown["Normalized Score"]
)

score_breakdown_display = score_breakdown.copy()

score_breakdown_display["Weight"] = (
    score_breakdown_display["Weight"]
    .apply(lambda value: f"{value:.0%}")
)

score_breakdown_display["Normalized Score"] = (
    score_breakdown_display["Normalized Score"]
    .apply(lambda value: f"{value:.4f}")
)

score_breakdown_display["Weighted Contribution"] = (
    score_breakdown_display["Weighted Contribution"]
    .apply(lambda value: f"{value:.4f}")
)

st.dataframe(
    score_breakdown_display,
    use_container_width=True,
    hide_index=True
)

st.write(
    f"Composite Score = "
    f"**{score_breakdown['Weighted Contribution'].sum():.4f}**"
)


# ============================================================
# OPTIMIZED TRADE LEDGER
# ============================================================
st.write("### Optimized Strategy Trade Ledger")

optimized_df["Position_Change"] = (
    optimized_df["Position"].diff()
)

trade_ledger = optimized_df[
    optimized_df["Position_Change"].fillna(0) != 0
].copy()

trade_ledger["Trade_Action"] = np.where(
    trade_ledger["Position"] == 1,
    f"Sell T-Bill / Buy {ticker}",
    f"Sell {ticker} / Buy T-Bill"
)

trade_ledger["Reason"] = np.where(
    trade_ledger["Position"] == 1,
    (
        f"2 consecutive closes above "
        f"{best_buffer_pct:.2f}% upper buffer"
    ),
    (
        f"2 consecutive closes below "
        f"{best_buffer_pct:.2f}% lower buffer"
    )
)

trade_ledger["Strategy_Advantage"] = (
    trade_ledger["Strategy_Value"]
    - trade_ledger["Buy_Hold_Value"]
)

trade_ledger = trade_ledger[[
    "Date",
    "Trade_Action",
    "Reason",
    "Asset_Close",
    "SMA",
    "Upper_Buffer",
    "Lower_Buffer",
    "TBill_3M_Yield",
    "Strategy_Value",
    "Buy_Hold_Value",
    "Strategy_Advantage"
]].copy()

trade_ledger = trade_ledger.rename(columns={
    "Asset_Close": f"{ticker}_Close",
    "SMA": f"{best_sma}_Day_SMA",
    "Upper_Buffer": "Upper_Buffer_Level",
    "Lower_Buffer": "Lower_Buffer_Level",
    "TBill_3M_Yield": "3M_TBill_Yield"
})

show_all_trades = st.checkbox(
    "Show all optimized trade ledger entries",
    value=False
)

if show_all_trades:
    displayed_trade_ledger = trade_ledger
else:
    displayed_trade_ledger = trade_ledger.tail(20)

st.dataframe(
    displayed_trade_ledger,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOADS
# ============================================================
st.write("### Download Results")

grid_csv = grid_results_df.to_csv(
    index=False
).encode("utf-8")

top_csv = top_combinations.to_csv(
    index=False
).encode("utf-8")

trade_csv = trade_ledger.to_csv(
    index=False
).encode("utf-8")

download_col1, download_col2, download_col3 = st.columns(3)

with download_col1:
    st.download_button(
        label="Download Full Grid Results",
        data=grid_csv,
        file_name=(
            f"{ticker}_simultaneous_optimization.csv"
        ),
        mime="text/csv"
    )

with download_col2:
    st.download_button(
        label="Download Top Combinations",
        data=top_csv,
        file_name=(
            f"{ticker}_top_parameter_combinations.csv"
        ),
        mime="text/csv"
    )

with download_col3:
    st.download_button(
        label="Download Optimized Trade Ledger",
        data=trade_csv,
        file_name=(
            f"{ticker}_optimized_trade_ledger.csv"
        ),
        mime="text/csv"
    )

st.success(
    "Data loaded and simultaneous SMA-buffer optimization "
    "completed successfully."
)
