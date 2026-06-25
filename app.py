import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
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


def run_strategy(df, sma_window, starting_balance):
    df = df.copy()

    df["Asset_Return"] = df["Asset_Adj_Close"].pct_change().fillna(0)

    df["Cash_Return"] = ((1 + df["TBill_3M_Yield"] / 100) ** (1 / 252)) - 1

    df["SMA"] = df["Asset_Close"].rolling(window=sma_window).mean()

    df["Above_SMA"] = df["Asset_Close"] > df["SMA"]
    df["Below_SMA"] = df["Asset_Close"] < df["SMA"]

    df["Confirmed_Above"] = df["Above_SMA"] & df["Above_SMA"].shift(1)
    df["Confirmed_Below"] = df["Below_SMA"] & df["Below_SMA"].shift(1)

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


start_date_str = start_date.strftime("%Y-%m-%d")
end_date_str = end_date.strftime("%Y-%m-%d")

df = load_data(ticker, start_date_str, end_date_str)

if df is None or df.empty:
    st.error("No data found. Please check the ticker symbol or date range.")
    st.stop()

df = run_strategy(df, sma_window, starting_balance)

# -----------------------------
# Performance Metrics
# -----------------------------
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


metrics = pd.DataFrame({
    "315 SMA Strategy": [
        calculate_cagr(df["Strategy_Value"], df["Date"]),
        calculate_volatility(df["Strategy_Return"]),
        calculate_max_drawdown(df["Strategy_Value"]),
        calculate_sharpe_ratio(df["Strategy_Return"])
    ],
    f"Buy & Hold {ticker}": [
        calculate_cagr(df["Buy_Hold_Value"], df["Date"]),
        calculate_volatility(df["Asset_Return"]),
        calculate_max_drawdown(df["Buy_Hold_Value"]),
        calculate_sharpe_ratio(df["Asset_Return"])
    ],
    "Cash / 3M T-Bill": [
        calculate_cagr(df["Cash_Value"], df["Date"]),
        calculate_volatility(df["Cash_Return"]),
        calculate_max_drawdown(df["Cash_Value"]),
        calculate_sharpe_ratio(df["Cash_Return"])
    ]
}, index=[
    "CAGR",
    "Volatility",
    "Max Drawdown",
    "Sharpe Ratio"
])

metrics_display = metrics.astype(object).copy()

for row in ["CAGR", "Volatility", "Max Drawdown"]:
    metrics_display.loc[row, :] = metrics.loc[row, :].apply(lambda x: f"{x:.2%}")

metrics_display.loc["Sharpe Ratio", :] = metrics.loc["Sharpe Ratio", :].apply(
    lambda x: f"{x:.2f}"
)

st.write("### Performance Summary")
st.dataframe(metrics_display)

# -----------------------------
# Interactive Portfolio + T-Bill Yield Chart
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
        line=dict(color="#2563EB"),
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
        line=dict(color="#F97316"),
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
        line=dict(color="#16A34A"),
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

# -----------------------------
# SMA Window Comparison
# -----------------------------
st.write("### SMA Strategy Comparison: 90 vs 150 vs 200 vs 315 Days")

comparison_windows = [90, 150, 200, 315]

comparison_df = df.copy()
comparison_results = {}

for window in comparison_windows:
    temp_df = run_strategy(comparison_df, window, starting_balance)
    comparison_results[f"{window}-Day SMA"] = temp_df["Strategy_Value"]

comparison_chart_df = pd.DataFrame(comparison_results)
comparison_chart_df["Date"] = df["Date"]

fig_sma_compare = go.Figure()

for window in comparison_windows:
    fig_sma_compare.add_trace(
        go.Scatter(
            x=comparison_chart_df["Date"],
            y=comparison_chart_df[f"{window}-Day SMA"],
            mode="lines",
            name=f"{window}-Day SMA Strategy",
            hovertemplate="Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>"
        )
    )

fig_sma_compare.update_layout(
    height=650,
    title="Portfolio Value Comparison Across SMA Windows",
    xaxis_title="Date",
    yaxis_title="Portfolio Value ($)",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.03,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig_sma_compare, use_container_width=True)

st.success("Data loaded and strategy calculated successfully.")
