import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="SMA Tactical Strategy Optimizer",
    layout="wide"
)

st.title("SMA Tactical Strategy Optimizer")
st.write(
    "This app compares a moving-average tactical strategy against buy-and-hold, "
    "then optimizes both the SMA window and buffer percentage using a weighted composite score."
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Strategy Configuration")

ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

start_date = st.sidebar.date_input(
    "Start Date",
    value=date(1995, 1, 1)
)

end_date = st.sidebar.date_input(
    "End Date",
    value=date.today()
)

sma_window = st.sidebar.number_input(
    "User-Selected Base SMA for Buffer Optimization",
    min_value=20,
    max_value=700,
    value=315,
    step=5
)

starting_balance = st.sidebar.number_input(
    "Starting Balance",
    min_value=1000,
    value=10000,
    step=1000
)

st.sidebar.header("Buffer Optimization")

min_buffer = st.sidebar.number_input(
    "Minimum Buffer %",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.05
)

max_buffer = st.sidebar.number_input(
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

st.sidebar.header("Stage 1 SMA Optimization")

stage1_sma_min = st.sidebar.number_input(
    "Minimum SMA",
    min_value=20,
    max_value=700,
    value=50,
    step=5
)

stage1_sma_max = st.sidebar.number_input(
    "Maximum SMA",
    min_value=20,
    max_value=700,
    value=400,
    step=5
)

stage1_sma_step = st.sidebar.number_input(
    "SMA Step",
    min_value=1,
    max_value=50,
    value=5,
    step=1
)

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.write("### Current Settings")
st.write(f"Ticker: **{ticker}**")
st.write(f"Start Date: **{start_date}**")
st.write(f"End Date: **{end_date}**")
st.write(f"User-Selected Base SMA: **{sma_window} trading days**")
st.write(f"Starting Balance: **${starting_balance:,.0f}**")


# -----------------------------
# Data Loader
# -----------------------------
@st.cache_data(ttl=3600)
def load_data(ticker, start_date, end_date):
    asset = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False
    )

    if asset.empty:
        return None

    if isinstance(asset.columns, pd.MultiIndex):
        asset.columns = asset.columns.get_level_values(0)

    asset = asset[["Close", "Adj Close"]].copy()

    asset = asset.rename(columns={
        "Close": "Asset_Close",
        "Adj Close": "Asset_Adj_Close"
    })

    asset = asset.reset_index()
    asset = asset.rename(columns={asset.columns[0]: "Date"})
    asset["Date"] = pd.to_datetime(asset["Date"], errors="coerce").dt.date

    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS"
    tbill = pd.read_csv(fred_url)

    tbill = tbill.rename(columns={
        "observation_date": "Date",
        "TB3MS": "TBill_3M_Yield"
    })

    tbill = tbill[["Date", "TBill_3M_Yield"]].copy()
    tbill["Date"] = pd.to_datetime(tbill["Date"], errors="coerce").dt.date
    tbill["TBill_3M_Yield"] = pd.to_numeric(tbill["TBill_3M_Yield"], errors="coerce")
    tbill = tbill.dropna()

    full_dates = pd.DataFrame({
        "Date": pd.date_range(start=start_date, end=end_date).date
    })

    tbill_daily = pd.merge(
        full_dates,
        tbill,
        on="Date",
        how="left"
    )

    tbill_daily["TBill_3M_Yield"] = tbill_daily["TBill_3M_Yield"].ffill()

    df = pd.merge(
        asset,
        tbill_daily,
        on="Date",
        how="left"
    )

    df["Date"] = pd.to_datetime(df["Date"])
    df["TBill_3M_Yield"] = df["TBill_3M_Yield"].ffill()

    return df


# -----------------------------
# Strategy Function
# -----------------------------
def run_strategy(df, sma_window, starting_balance, buffer_pct=0.0):
    df = df.copy()

    buffer_decimal = buffer_pct / 100

    df["Asset_Return"] = df["Asset_Adj_Close"].pct_change().fillna(0)
    df["Cash_Return"] = ((1 + df["TBill_3M_Yield"] / 100) ** (1 / 252)) - 1

    df["SMA"] = df["Asset_Close"].rolling(window=sma_window).mean()

    df["Upper_Buffer"] = df["SMA"] * (1 + buffer_decimal)
    df["Lower_Buffer"] = df["SMA"] * (1 - buffer_decimal)

    df["Above_Buffer"] = df["Asset_Close"] > df["Upper_Buffer"]
    df["Below_Buffer"] = df["Asset_Close"] < df["Lower_Buffer"]

    df["Confirmed_Above"] = df["Above_Buffer"] & df["Above_Buffer"].shift(1)
    df["Confirmed_Below"] = df["Below_Buffer"] & df["Below_Buffer"].shift(1)

    position = []
    current_position = 1

    for i in range(len(df)):
        if df.loc[i, "Confirmed_Below"]:
            current_position = 0
        elif df.loc[i, "Confirmed_Above"]:
            current_position = 1

        position.append(current_position)

    df["Position"] = position

    df["Strategy_Return"] = np.where(
        df["Position"].shift(1) == 1,
        df["Asset_Return"],
        df["Cash_Return"]
    )

    df["Strategy_Return"] = df["Strategy_Return"].fillna(0)

    df["Strategy_Value"] = starting_balance * (1 + df["Strategy_Return"]).cumprod()
    df["Buy_Hold_Value"] = starting_balance * (1 + df["Asset_Return"]).cumprod()
    df["Cash_Value"] = starting_balance * (1 + df["Cash_Return"]).cumprod()

    return df


# -----------------------------
# Metric Functions
# -----------------------------
def calculate_ending_value(values):
    values = values.dropna()

    if values.empty:
        return np.nan

    return values.iloc[-1]


def calculate_cagr(values, dates):
    values = values.dropna()
    dates = dates.loc[values.index]

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25

    if years <= 0:
        return np.nan

    return (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1


def calculate_volatility(returns):
    return returns.std() * np.sqrt(252)


def calculate_max_drawdown(values):
    running_max = values.cummax()
    drawdown = values / running_max - 1
    return drawdown.min()


def calculate_sharpe_ratio(returns):
    if returns.std() == 0:
        return np.nan

    return (returns.mean() * 252) / (returns.std() * np.sqrt(252))


def calculate_calmar_ratio(cagr, max_drawdown):
    if max_drawdown == 0 or pd.isna(max_drawdown):
        return np.nan

    return cagr / abs(max_drawdown)


def calculate_strategy_metrics(strategy_df):
    ending_value = calculate_ending_value(strategy_df["Strategy_Value"])
    cagr = calculate_cagr(strategy_df["Strategy_Value"], strategy_df["Date"])
    volatility = calculate_volatility(strategy_df["Strategy_Return"])
    max_drawdown = calculate_max_drawdown(strategy_df["Strategy_Value"])
    sharpe = calculate_sharpe_ratio(strategy_df["Strategy_Return"])
    calmar = calculate_calmar_ratio(cagr, max_drawdown)
    trade_count = strategy_df["Position"].diff().abs().sum()
    time_in_market = strategy_df["Position"].mean()

    return {
        "Ending Portfolio Value": ending_value,
        "CAGR": cagr,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Number of Trades": trade_count,
        "Time in Market": time_in_market
    }

# -----------------------------
# Composite Score Functions
# -----------------------------
def normalize_higher_better(series):
    if series.max() == series.min():
        return pd.Series(1, index=series.index)

    return (series - series.min()) / (series.max() - series.min())


def normalize_lower_better(series):
    if series.max() == series.min():
        return pd.Series(1, index=series.index)

    return (series.max() - series) / (series.max() - series.min())


def add_composite_score(results_df):
    df = results_df.copy()

    df["Abs Max Drawdown"] = df["Max Drawdown"].abs()

    df["Score_Max_Drawdown"] = normalize_lower_better(df["Abs Max Drawdown"])
    df["Score_CAGR"] = normalize_higher_better(df["CAGR"])
    df["Score_Sharpe"] = normalize_higher_better(df["Sharpe Ratio"])
    df["Score_Calmar"] = normalize_higher_better(df["Calmar Ratio"])
    df["Score_Trade_Count"] = normalize_lower_better(df["Number of Trades"])
    df["Score_Time_in_Market"] = normalize_lower_better(df["Time in Market"])

    df["Composite Score"] = (
        0.35 * df["Score_Max_Drawdown"] +
        0.25 * df["Score_CAGR"] +
        0.15 * df["Score_Sharpe"] +
        0.15 * df["Score_Calmar"] +
        0.05 * df["Score_Trade_Count"] +
        0.05 * df["Score_Time_in_Market"]
    )

    return df


# -----------------------------
# Display Formatting
# -----------------------------
def format_metrics_table(metrics):
    display = metrics.astype(object).copy()

    if "Ending Portfolio Value" in display.index:
        display.loc["Ending Portfolio Value", :] = metrics.loc[
            "Ending Portfolio Value", :
        ].apply(lambda x: f"${x:,.0f}")

    for row in ["CAGR", "Volatility", "Max Drawdown"]:
        if row in display.index:
            display.loc[row, :] = metrics.loc[row, :].apply(lambda x: f"{x:.2%}")

    if "Time in Market" in display.index:
        display.loc["Time in Market", :] = metrics.loc["Time in Market", :].apply(
            lambda x: f"{x:.2%}"
        )

    if "Number of Trades" in display.index:
        display.loc["Number of Trades", :] = metrics.loc["Number of Trades", :].apply(
            lambda x: f"{x:.0f}"
        )

    if "Sharpe Ratio" in display.index:
        display.loc["Sharpe Ratio", :] = metrics.loc["Sharpe Ratio", :].apply(
            lambda x: f"{x:.2f}"
        )

    if "Calmar Ratio" in display.index:
        display.loc["Calmar Ratio", :] = metrics.loc["Calmar Ratio", :].apply(
            lambda x: f"{x:.2f}"
        )

    return display


def format_optimization_table(results_df):
    display = results_df.copy()

    currency_cols = [
        "Ending Portfolio Value"
    ]

    for col in currency_cols:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"${x:,.0f}")

    percentage_cols = [
        "Buffer %",
        "CAGR",
        "Volatility",
        "Max Drawdown",
        "Time in Market",
        "Abs Max Drawdown"
    ]

    for col in percentage_cols:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.2%}")

    decimal_cols = [
        "Sharpe Ratio",
        "Calmar Ratio",
        "Composite Score",
        "Score_Max_Drawdown",
        "Score_CAGR",
        "Score_Sharpe",
        "Score_Calmar",
        "Score_Trade_Count",
        "Score_Time_in_Market"
    ]

    for col in decimal_cols:
        if col in display.columns:
            display[col] = display[col].apply(lambda x: f"{x:.4f}")

    if "Number of Trades" in display.columns:
        display["Number of Trades"] = display["Number of Trades"].apply(
            lambda x: f"{x:.0f}"
        )

    return display


# -----------------------------
# Chart Functions
# -----------------------------
def create_portfolio_chart(strategy_df, ticker, sma_window, buffer_pct):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.75, 0.25],
        subplot_titles=(
            f"{sma_window}-Day SMA Strategy vs Buy & Hold {ticker}",
            "3-Month T-Bill Yield Over Time"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=strategy_df["Date"],
            y=strategy_df["Buy_Hold_Value"],
            mode="lines",
            name=f"Buy & Hold {ticker}",
            hovertemplate="Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=strategy_df["Date"],
            y=strategy_df["Strategy_Value"],
            mode="lines",
            name=f"{sma_window}-Day SMA Strategy | Buffer {buffer_pct:.2f}%",
            hovertemplate="Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=strategy_df["Date"],
            y=strategy_df["TBill_3M_Yield"],
            mode="lines",
            name="3M T-Bill Yield",
            hovertemplate="Date: %{x}<br>Yield: %{y:.2f}%<extra></extra>"
        ),
        row=2,
        col=1
    )

    fig.update_layout(
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

    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Yield (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


def create_score_line_chart(results_df, x_col, title, x_title):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=results_df[x_col],
            y=results_df["Composite Score"],
            mode="lines+markers",
            name="Composite Score",
            hovertemplate=(
                f"{x_title}: " + "%{x}<br>"
                "Composite Score: %{y:.4f}<extra></extra>"
            )
        )
    )

    fig.update_layout(
        height=500,
        title=title,
        xaxis_title=x_title,
        yaxis_title="Composite Score",
        hovermode="x unified"
    )

    if x_col == "Buffer %":
        fig.update_xaxes(tickformat=".2%")

    return fig


# -----------------------------
# Load Data
# -----------------------------
start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

raw_df = load_data(ticker, start_date_str, end_date_str)

if raw_df is None or raw_df.empty:
    st.error("No data found. Please check the ticker symbol or date range.")
    st.stop()


# -----------------------------
# Base Strategy
# -----------------------------
base_df = run_strategy(
    raw_df.copy(),
    sma_window,
    starting_balance,
    buffer_pct=0.0
)

base_metrics = calculate_strategy_metrics(base_df)

buy_hold_ending_value = calculate_ending_value(base_df["Buy_Hold_Value"])
buy_hold_cagr = calculate_cagr(base_df["Buy_Hold_Value"], base_df["Date"])
buy_hold_max_dd = calculate_max_drawdown(base_df["Buy_Hold_Value"])

cash_ending_value = calculate_ending_value(base_df["Cash_Value"])
cash_cagr = calculate_cagr(base_df["Cash_Value"], base_df["Date"])
cash_max_dd = calculate_max_drawdown(base_df["Cash_Value"])

metrics = pd.DataFrame({
    f"{sma_window}-Day SMA Strategy": [
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
        buy_hold_ending_value,
        buy_hold_cagr,
        calculate_volatility(base_df["Asset_Return"]),
        buy_hold_max_dd,
        calculate_sharpe_ratio(base_df["Asset_Return"]),
        calculate_calmar_ratio(buy_hold_cagr, buy_hold_max_dd),
        0,
        1.0
    ],
    "Cash / 3M T-Bill": [
        cash_ending_value,
        cash_cagr,
        calculate_volatility(base_df["Cash_Return"]),
        cash_max_dd,
        calculate_sharpe_ratio(base_df["Cash_Return"]),
        calculate_calmar_ratio(cash_cagr, cash_max_dd),
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
    format_metrics_table(metrics),
    use_container_width=True
)

st.write("### Base Strategy vs Buy & Hold")
st.plotly_chart(
    create_portfolio_chart(base_df, ticker, sma_window, 0.0),
    use_container_width=True
)

# -----------------------------
# Buffer Optimization for User-Selected Base SMA
# -----------------------------
st.write(f"### Buffer Optimization for User-Selected {sma_window}-Day SMA")

st.write(
    f"This section optimizes the buffer percentage while keeping the SMA fixed at "
    f"**{sma_window} days**."
)

buffer_values = np.arange(min_buffer, max_buffer + buffer_step, buffer_step)

buffer_results = []

for test_buffer in buffer_values:
    temp_df = run_strategy(
        raw_df.copy(),
        sma_window,
        starting_balance,
        buffer_pct=test_buffer
    )

    row = calculate_strategy_metrics(temp_df)

    row.update({
        "SMA Window": sma_window,
        "Buffer %": test_buffer / 100
    })

    buffer_results.append(row)

buffer_results_df = pd.DataFrame(buffer_results)
buffer_results_df = add_composite_score(buffer_results_df)

best_buffer_row = buffer_results_df.loc[
    buffer_results_df["Composite Score"].idxmax()
]

best_base_buffer = best_buffer_row["Buffer %"]

st.success(
    f"Best buffer for user-selected {sma_window}-day SMA: {best_base_buffer:.2%}"
)

st.plotly_chart(
    create_score_line_chart(
        buffer_results_df,
        "Buffer %",
        f"Composite Score by Buffer Percentage for {sma_window}-Day SMA",
        "Buffer %"
    ),
    use_container_width=True
)

display_buffer_results = buffer_results_df.sort_values(
    "Composite Score",
    ascending=False
).copy()

st.write("#### Buffer Optimization Results")
st.dataframe(
    format_optimization_table(display_buffer_results),
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Two-Stage Optimization Framework
# -----------------------------
st.write("### Two-Stage Optimization Framework")

st.write(
    """
    Stage 1 tests multiple SMA windows with no buffer to identify the strongest trend window.
    Stage 2 then optimizes the buffer percentage using the selected SMA window.
    """
)

# -----------------------------
# Stage 1: SMA Optimization
# -----------------------------
st.write("#### Stage 1: Find Best SMA Window")

if stage1_sma_min > stage1_sma_max:
    st.error("Minimum SMA must be less than or equal to Maximum SMA.")
    st.stop()

sma_values = list(range(stage1_sma_min, stage1_sma_max + 1, stage1_sma_step))

stage1_results = []

for test_sma in sma_values:
    temp_df = run_strategy(
        raw_df.copy(),
        test_sma,
        starting_balance,
        buffer_pct=0.0
    )

    row = calculate_strategy_metrics(temp_df)

    row.update({
        "SMA Window": test_sma,
        "Buffer %": 0.0
    })

    stage1_results.append(row)

stage1_results_df = pd.DataFrame(stage1_results)
stage1_results_df = add_composite_score(stage1_results_df)

best_stage1_row = stage1_results_df.loc[
    stage1_results_df["Composite Score"].idxmax()
]

best_stage1_sma = int(best_stage1_row["SMA Window"])

st.success(
    f"Best SMA window from Stage 1: {best_stage1_sma} days"
)

st.plotly_chart(
    create_score_line_chart(
        stage1_results_df,
        "SMA Window",
        "Stage 1: Composite Score by SMA Window",
        "SMA Window"
    ),
    use_container_width=True
)

display_stage1_results = stage1_results_df.sort_values(
    "Composite Score",
    ascending=False
).copy()

st.write("##### Stage 1 SMA Optimization Results")
st.dataframe(
    format_optimization_table(display_stage1_results),
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Stage 2: Buffer Optimization for Best SMA
# -----------------------------
st.write("#### Stage 2: Optimize Buffer for Best SMA Window")

st.write(
    f"Stage 2 tests buffer percentages using the selected SMA window: "
    f"**{best_stage1_sma} days**."
)

stage2_buffer_values = np.arange(min_buffer, max_buffer + buffer_step, buffer_step)

stage2_results = []

for test_buffer in stage2_buffer_values:
    temp_df = run_strategy(
        raw_df.copy(),
        best_stage1_sma,
        starting_balance,
        buffer_pct=test_buffer
    )

    row = calculate_strategy_metrics(temp_df)

    row.update({
        "SMA Window": best_stage1_sma,
        "Buffer %": test_buffer / 100
    })

    stage2_results.append(row)

stage2_results_df = pd.DataFrame(stage2_results)
stage2_results_df = add_composite_score(stage2_results_df)

best_stage2_row = stage2_results_df.loc[
    stage2_results_df["Composite Score"].idxmax()
]

best_stage2_sma = int(best_stage2_row["SMA Window"])
best_stage2_buffer = best_stage2_row["Buffer %"]

st.success(
    f"Final two-stage optimized strategy: "
    f"{best_stage2_sma}-day SMA with {best_stage2_buffer:.2%} buffer"
)

st.plotly_chart(
    create_score_line_chart(
        stage2_results_df,
        "Buffer %",
        f"Stage 2: Composite Score by Buffer for {best_stage1_sma}-Day SMA",
        "Buffer %"
    ),
    use_container_width=True
)

display_stage2_results = stage2_results_df.sort_values(
    "Composite Score",
    ascending=False
).copy()

st.write("##### Stage 2 Buffer Optimization Results")
st.dataframe(
    format_optimization_table(display_stage2_results),
    use_container_width=True,
    hide_index=True
)


# -----------------------------
# Final Optimized Strategy
# -----------------------------
st.write("### Final Optimized Strategy Performance")

optimized_df = run_strategy(
    raw_df.copy(),
    best_stage2_sma,
    starting_balance,
    buffer_pct=best_stage2_buffer * 100
)

optimized_metrics = calculate_strategy_metrics(optimized_df)

final_metrics = pd.DataFrame({
    "Base Strategy": [
        base_metrics["Ending Portfolio Value"],
        base_metrics["CAGR"],
        base_metrics["Volatility"],
        base_metrics["Max Drawdown"],
        base_metrics["Sharpe Ratio"],
        base_metrics["Calmar Ratio"],
        base_metrics["Number of Trades"],
        base_metrics["Time in Market"]
    ],
    "Optimized Strategy": [
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
        buy_hold_ending_value,
        buy_hold_cagr,
        calculate_volatility(base_df["Asset_Return"]),
        buy_hold_max_dd,
        calculate_sharpe_ratio(base_df["Asset_Return"]),
        calculate_calmar_ratio(buy_hold_cagr, buy_hold_max_dd),
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

st.dataframe(
    format_metrics_table(final_metrics),
    use_container_width=True
)

st.plotly_chart(
    create_portfolio_chart(
        optimized_df,
        ticker,
        best_stage2_sma,
        best_stage2_buffer * 100
    ),
    use_container_width=True
)


# -----------------------------
# Optimized Strategy Trade Ledger
# -----------------------------
st.write("### Optimized Strategy Trade Ledger")

optimized_df["Position_Change"] = optimized_df["Position"].diff()

trade_ledger = optimized_df[optimized_df["Position_Change"] != 0].copy()

trade_ledger["Trade_Action"] = np.where(
    trade_ledger["Position"] == 1,
    f"Sell T-Bill / Buy {ticker}",
    f"Sell {ticker} / Buy T-Bill"
)

trade_ledger["Reason"] = np.where(
    trade_ledger["Position"] == 1,
    "2 consecutive closes ABOVE upper buffer",
    "2 consecutive closes BELOW lower buffer"
)

trade_ledger["Strategy_Advantage"] = (
    trade_ledger["Strategy_Value"] - trade_ledger["Buy_Hold_Value"]
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
    "SMA": f"{best_stage2_sma}_Day_SMA",
    "Upper_Buffer": "Upper_Buffer_Level",
    "Lower_Buffer": "Lower_Buffer_Level",
    "TBill_3M_Yield": "3M_TBill_Yield"
})

show_all_trades = st.checkbox("Show all optimized trade ledger entries", value=False)

if show_all_trades:
    display_ledger = trade_ledger
else:
    display_ledger = trade_ledger.tail(20)

st.dataframe(
    display_ledger,
    use_container_width=True,
    hide_index=True
)

csv = trade_ledger.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Optimized Trade Ledger as CSV",
    data=csv,
    file_name=f"{ticker}_optimized_trade_ledger.csv",
    mime="text/csv"
)


# -----------------------------
# Download Optimization Results
# -----------------------------
st.write("### Download Optimization Results")

stage1_csv = stage1_results_df.to_csv(index=False).encode("utf-8")
stage2_csv = stage2_results_df.to_csv(index=False).encode("utf-8")
buffer_csv = buffer_results_df.to_csv(index=False).encode("utf-8")

col1, col2, col3 = st.columns(3)

with col1:
    st.download_button(
        label="Download Stage 1 SMA Results",
        data=stage1_csv,
        file_name=f"{ticker}_stage1_sma_optimization.csv",
        mime="text/csv"
    )

with col2:
    st.download_button(
        label="Download Stage 2 Buffer Results",
        data=stage2_csv,
        file_name=f"{ticker}_stage2_buffer_optimization.csv",
        mime="text/csv"
    )

with col3:
    st.download_button(
        label="Download Base Buffer Results",
        data=buffer_csv,
        file_name=f"{ticker}_base_buffer_optimization.csv",
        mime="text/csv"
    )

st.success("Data loaded and strategy optimization completed successfully.")
