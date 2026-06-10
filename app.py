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

st.write("### Current Settings")
st.write(f"Ticker: **{ticker}**")
st.write(f"Start Date: **{start_date}**")
st.write(f"End Date: **{end_date}**")
st.write(f"SMA Window: **{sma_window} trading days**")
st.write(f"Starting Balance: **${starting_balance:,.0f}**")


@st.cache_data
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


df = load_data(ticker, start_date, end_date)

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
# Interactive Price vs SMA Chart
# -----------------------------
st.write("### Price vs Moving Average")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["Asset_Close"],
    mode="lines",
    name=f"{ticker} Close",
    line=dict(color="#2563EB"),
    hovertemplate="Date: %{x}<br>Close: $%{y:,.2f}<extra></extra>"
))

fig.add_trace(go.Scatter(
    x=df["Date"],
    y=df["SMA"],
    mode="lines",
    name=f"{sma_window}-Day SMA",
    line=dict(color="#F97316"),
    hovertemplate="Date: %{x}<br>SMA: $%{y:,.2f}<extra></extra>"
))

fig.update_layout(
    title=f"{ticker} Price vs {sma_window}-Day Moving Average",
    xaxis_title="Date",
    yaxis_title="Price ($)",
    hovermode="x unified",
    height=650,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0
    )
)

st.plotly_chart(fig, use_container_width=True)

st.success("Data loaded and strategy calculated successfully.")

st.write("### Strategy Data Preview")
