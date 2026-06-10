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
