import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="315 SMA Tactical Strategy",
    layout="wide"
)

st.title("315-Day SMA Tactical Strategy")
st.write("This app compares a moving-average tactical strategy against buy-and-hold.")

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
    "Moving Average Window",
    min_value=50,
    max_value=500,
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
    step=0.25
)

max_buffer = st.sidebar.number_input(
    "Maximum Buffer %",
    min_value=0.0,
    max_value=20.0,
    value=5.0,
    step=0.25
)

buffer_step = st.sidebar.number_input(
    "Buffer Step %",
    min_value=0.05,
    max_value=5.0,
    value=0.25,
    step=0.05
)

if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()
    st.rerun()

st.write("### Current Settings")
st.write(f"Ticker: **{ticker}**")
st.write(f"Start Date: **{start_date}**")
st.write(f"End Date: **{end_date}**")
st.write(f"SMA Window: **{sma_window} trading days**")
st.write(f"Starting Balance: **${starting_balance:,.0f}**")


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


def calculate_cagr(values, dates):
    values = values.dropna()
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


def format_metrics_table(metrics):
    display = metrics.astype(object).copy()

    for row in ["CAGR", "Volatility", "Max Drawdown"]:
        display.loc[row, :] = metrics.loc[row, :].apply(lambda x: f"{x:.2%}")

    display.loc["Sharpe Ratio", :] = metrics.loc["Sharpe Ratio", :].apply(
        lambda x: f"{x:.2f}"
    )

    if "Calmar Ratio" in display.index:
        display.loc["Calmar Ratio", :] = metrics.loc["Calmar Ratio", :].apply(
            lambda x: f"{x:.2f}"
        )

    return display


start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

raw_df = load_data(ticker, start_date_str, end_date_str)

if raw_df is None or raw_df.empty:
    st.error("No data found. Please check the ticker symbol or date range.")
    st.stop()

df = run_strategy(raw_df.copy(), sma_window, starting_balance, buffer_pct=0.0)

# -----------------------------
# Performance Summary
# -----------------------------
strategy_cagr = calculate_cagr(df["Strategy_Value"], df["Date"])
strategy_max_dd = calculate_max_drawdown(df["Strategy_Value"])

buy_hold_cagr = calculate_cagr(df["Buy_Hold_Value"], df["Date"])
buy_hold_max_dd = calculate_max_drawdown(df["Buy_Hold_Value"])

cash_cagr = calculate_cagr(df["Cash_Value"], df["Date"])
cash_max_dd = calculate_max_drawdown(df["Cash_Value"])

metrics = pd.DataFrame({
    f"{sma_window}-Day SMA Strategy": [
        strategy_cagr,
        calculate_volatility(df["Strategy_Return"]),
        strategy_max_dd,
        calculate_sharpe_ratio(df["Strategy_Return"]),
        calculate_calmar_ratio(strategy_cagr, strategy_max_dd)
    ],
    f"Buy & Hold {ticker}": [
        buy_hold_cagr,
        calculate_volatility(df["Asset_Return"]),
        buy_hold_max_dd,
        calculate_sharpe_ratio(df["Asset_Return"]),
        calculate_calmar_ratio(buy_hold_cagr, buy_hold_max_dd)
    ],
    "Cash / 3M T-Bill": [
        cash_cagr,
        calculate_volatility(df["Cash_Return"]),
        cash_max_dd,
        calculate_sharpe_ratio(df["Cash_Return"]),
        calculate_calmar_ratio(cash_cagr, cash_max_dd)
    ]
}, index=[
    "CAGR",
    "Volatility",
    "Max Drawdown",
    "Sharpe Ratio",
    "Calmar Ratio"
])

st.write("### Performance Summary")
st.dataframe(format_metrics_table(metrics), use_container_width=True)

# -----------------------------
# Strategy Chart
# -----------------------------
st.write("### Strategy vs Buy & Hold with T-Bill Yield")

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
        x=df["Date"],
        y=df["Buy_Hold_Value"],
        mode="lines",
        name=f"Buy & Hold {ticker}",
        hovertemplate="Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Strategy_Value"],
        mode="lines",
        name=f"{sma_window}-Day SMA Strategy",
        hovertemplate="Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>"
    ),
    row=1,
    col=1
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["TBill_3M_Yield"],
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

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# Buffer Optimization
# -----------------------------
st.write("### Buffer Optimization Using Composite Score")

buffer_values = np.arange(min_buffer, max_buffer + buffer_step, buffer_step)

buffer_results = []

for buffer in buffer_values:
    temp_df = run_strategy(
        raw_df.copy(),
        sma_window,
        starting_balance,
        buffer_pct=buffer
    )

    cagr = calculate_cagr(temp_df["Strategy_Value"], temp_df["Date"])
    volatility = calculate_volatility(temp_df["Strategy_Return"])
    max_drawdown = calculate_max_drawdown(temp_df["Strategy_Value"])
    sharpe = calculate_sharpe_ratio(temp_df["Strategy_Return"])
    calmar = calculate_calmar_ratio(cagr, max_drawdown)

    trade_count = temp_df["Position"].diff().abs().sum()
    time_in_market = temp_df["Position"].mean()

    buffer_results.append({
        "Buffer %": buffer / 100,
        "CAGR": cagr,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Number of Trades": trade_count,
        "Time in Market": time_in_market
    })

buffer_results_df = pd.DataFrame(buffer_results)
buffer_results_df = add_composite_score(buffer_results_df)

best_buffer_row = buffer_results_df.loc[
    buffer_results_df["Composite Score"].idxmax()
]

best_buffer = best_buffer_row["Buffer %"]

st.success(
    f"Best buffer based on composite score: {best_buffer:.2%}"
)

st.write("#### Composite Score Weights")
st.write(
    """
    - Max Drawdown: 35%
    - CAGR: 25%
    - Sharpe Ratio: 15%
    - Calmar Ratio: 15%
    - Number of Trades: 5%
    - Time in Market: 5%
    """
)

fig_buffer = go.Figure()

fig_buffer.add_trace(
    go.Scatter(
        x=buffer_results_df["Buffer %"],
        y=buffer_results_df["Composite Score"],
        mode="lines+markers",
        name="Composite Score",
        hovertemplate="Buffer: %{x:.2%}<br>Composite Score: %{y:.4f}<extra></extra>"
    )
)

fig_buffer.update_layout(
    height=500,
    title="Composite Score by Buffer Percentage",
    xaxis_title="Buffer %",
    yaxis_title="Composite Score",
    hovermode="x unified"
)

fig_buffer.update_xaxes(tickformat=".2%")

st.plotly_chart(fig_buffer, use_container_width=True)

display_buffer_results = buffer_results_df.sort_values(
    "Composite Score",
    ascending=False
).copy()

for col in ["Buffer %", "CAGR", "Volatility", "Max Drawdown", "Time in Market"]:
    display_buffer_results[col] = display_buffer_results[col].apply(lambda x: f"{x:.2%}")

for col in [
    "Sharpe Ratio",
    "Calmar Ratio",
    "Composite Score",
    "Score_Max_Drawdown",
    "Score_CAGR",
    "Score_Sharpe",
    "Score_Calmar",
    "Score_Trade_Count",
    "Score_Time_in_Market"
]:
    display_buffer_results[col] = display_buffer_results[col].apply(lambda x: f"{x:.4f}")

display_buffer_results["Number of Trades"] = display_buffer_results[
    "Number of Trades"
].apply(lambda x: f"{x:.0f}")

st.write("### Buffer Optimization Results Table")
st.dataframe(
    display_buffer_results,
    use_container_width=True,
    hide_index=True
)

# -----------------------------
# Trade Ledger
# -----------------------------
st.write("### Trade Ledger")

df["Position_Change"] = df["Position"].diff()

trade_ledger = df[df["Position_Change"] != 0].copy()

trade_ledger["Trade_Action"] = np.where(
    trade_ledger["Position"] == 1,
    f"Sell T-Bill / Buy {ticker}",
    f"Sell {ticker} / Buy T-Bill"
)

trade_ledger["Reason"] = np.where(
    trade_ledger["Position"] == 1,
    f"2 consecutive closes ABOVE {sma_window}-day SMA",
    f"2 consecutive closes BELOW {sma_window}-day SMA"
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
    "TBill_3M_Yield",
    "Strategy_Value",
    "Buy_Hold_Value",
    "Strategy_Advantage"
]].copy()

trade_ledger = trade_ledger.rename(columns={
    "Asset_Close": f"{ticker}_Close",
    "SMA": f"{sma_window}_Day_SMA",
    "TBill_3M_Yield": "3M_TBill_Yield"
})

show_all_trades = st.checkbox("Show all trade ledger entries", value=False)

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
    label="Download Full Trade Ledger as CSV",
    data=csv,
    file_name=f"{ticker}_trade_ledger.csv",
    mime="text/csv"
)

st.success("Data loaded and strategy calculated successfully.")
