import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import matplotlib.pyplot as plt
from datetime import date

st.set_page_config(
    page_title="315 SMA Tactical Strategy",
    layout="wide"
)

st.title("315-Day SMA Tactical Strategy")
st.write(
    "This app compares a moving-average tactical strategy against buy-and-hold."
)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Strategy Configuration")

ticker = st.sidebar.text_input(
    "Ticker Symbol",
    value="SPY"
).upper()

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

# -----------------------------
# Data Download Function
# -----------------------------
@st.cache_data
def load_data(ticker, start_date, end_date):
    # Download ticker price data from Yahoo Finance
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
    asset["Date"] = pd.to_datetime(asset["Date"])

    # Download 3-Month Treasury Bill Yield from FRED
    tbill = web.DataReader(
        "TB3MS",
        "fred",
        start=start_date,
        end=end_date
    )

    tbill = tbill.reset_index()
    tbill = tbill.rename(columns={
        tbill.columns[0]: "Date",
        "TB3MS": "TBill_3M_Yield"
    })
    tbill["Date"] = pd.to_datetime(tbill["Date"])

    # Merge ticker daily prices with latest available monthly T-Bill yield
    df = pd.merge_asof(
        asset.sort_values("Date"),
        tbill.sort_values("Date"),
        on="Date",
        direction="backward"
    )

    df["TBill_3M_Yield"] = df["TBill_3M_Yield"].ffill()

    return df

# -----------------------------
# Load Data
# -----------------------------
df = load_data(ticker, start_date, end_date)

if df is None or df.empty:
    st.error("No data found. Please check the ticker symbol or date range.")
    st.stop()

st.success("Data loaded successfully.")

st.write("### Data Preview")
st.dataframe(df.head())
