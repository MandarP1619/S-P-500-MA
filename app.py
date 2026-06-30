import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="315 SMA Tactical Strategy", layout="wide")

st.title("315-Day SMA Tactical Strategy")
st.write("This app compares a moving-average tactical strategy against buy-and-hold.")

st.sidebar.header("Strategy Configuration")

ticker = st.sidebar.text_input("Ticker Symbol", value="SPY").upper()

start_date = st.sidebar.date_input("Start Date", value=date(1995, 1, 1))
end_date = st.sidebar.date_input("End Date", value=date.today())

sma_window = st.sidebar.number_input(
    "Moving Average Window",
    min_value=50,
    max_value=500,
    value=315,
    step=5
)

buffer_pct = st.sidebar.number_input(
    "SMA Buffer (%)",
    min_value=0.0,
    max_value=20.0,
    value=0.0,
    step=0.25
) / 100

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
st.write(f"SMA Buffer: **{buffer_pct:.2%}**")
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

    tbill_daily = pd.merge(full_dates, tbill, on="Date", how="left")
    tbill_daily["TBill_3M_Yield"] = tbill_daily["TBill_3M_Yield"].ffill()

    df = pd.merge(asset, tbill_daily, on="Date", how="left")
    df["Date"] = pd.to_datetime(df["Date"])
    df["TBill_3M_Yield"] = df["TBill_3M_Yield"].ffill()

    return df


def run_strategy(df, sma_window, starting_balance, buffer_pct=0):
    df = df.copy()

    df["Asset_Return"] = df["Asset_Adj_Close"].pct_change().fillna(0)
    df["Cash_Return"] = ((1 + df["TBill_3M_Yield"] / 100) ** (1 / 252)) - 1

    df["SMA"] = df["Asset_Close"].rolling(window=sma_window).mean()

    df["Upper_Band"] = df["SMA"] * (1 + buffer_pct)
    df["Lower_Band"] = df["SMA"] * (1 - buffer_pct)

    df["Above_SMA"] = df["Asset_Close"] > df["Upper_Band"]
    df["Below_SMA"] = df["Asset_Close"] < df["Lower_Band"]

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

df = run_strategy(raw_df.copy(), sma_window, starting_balance, buffer_pct)

strategy_cagr = calculate_cagr(df["Strategy_Value"], df["Date"])
strategy_max_dd = calculate_max_drawdown(df["Strategy_Value"])

buy_hold_cagr = calculate_cagr(df["Buy_Hold_Value"], df["Date"])
buy_hold_max_dd = calculate_max_drawdown(df["Buy_Hold_Value"])

cash_cagr = calculate_cagr(df["Cash_Value"], df["Date"])
cash_max_dd = calculate_max_drawdown(df["Cash_Value"])

metrics = pd.DataFrame({
    f"{sma_window}-Day SMA Strategy with {buffer_pct:.2%} Buffer": [
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
    f"2 consecutive closes ABOVE upper band",
    f"2 consecutive closes BELOW lower band"
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
    "Upper_Band",
    "Lower_Band",
    "TBill_3M_Yield",
    "Strategy_Value",
    "Buy_Hold_Value",
    "Strategy_Advantage"
]].copy()

trade_ledger = trade_ledger.rename(columns={
    "Asset_Close": f"{ticker}_Close",
    "SMA": f"{sma_window}_Day_SMA",
    "Upper_Band": "Upper_Buffer_Band",
    "Lower_Band": "Lower_Buffer_Band",
    "TBill_3M_Yield": "3M_TBill_Yield"
})

show_all_trades = st.checkbox("Show all trade ledger entries", value=False)

if show_all_trades:
    display_ledger = trade_ledger
else:
    display_ledger = trade_ledger.tail(20)

st.dataframe(display_ledger, use_container_width=True, hide_index=True)

csv = trade_ledger.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Full Trade Ledger as CSV",
    data=csv,
    file_name=f"{ticker}_trade_ledger.csv",
    mime="text/csv"
)

st.write("### SMA Strategy Comparison: 90 vs 150 vs 200 vs 315 Days")

comparison_windows = [90, 150, 200, 315]
comparison_metrics = {}

fig_sma_compare = go.Figure()

for window in comparison_windows:
    temp_df = run_strategy(raw_df.copy(), window, starting_balance, buffer_pct)

    temp_cagr = calculate_cagr(temp_df["Strategy_Value"], temp_df["Date"])
    temp_max_dd = calculate_max_drawdown(temp_df["Strategy_Value"])

    comparison_metrics[f"{window}-Day SMA"] = [
        temp_cagr,
        calculate_volatility(temp_df["Strategy_Return"]),
        temp_max_dd,
        calculate_sharpe_ratio(temp_df["Strategy_Return"]),
        calculate_calmar_ratio(temp_cagr, temp_max_dd)
    ]

    fig_sma_compare.add_trace(
        go.Scatter(
            x=temp_df["Date"],
            y=temp_df["Strategy_Value"],
            mode="lines",
            name=f"{window}-Day SMA Strategy",
            hovertemplate="Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>"
        )
    )

fig_sma_compare.update_layout(
    height=650,
    title=f"Portfolio Value Comparison Across SMA Windows with {buffer_pct:.2%} Buffer",
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

sma_comparison_metrics = pd.DataFrame(
    comparison_metrics,
    index=[
        "CAGR",
        "Volatility",
        "Max Drawdown",
        "Sharpe Ratio",
        "Calmar Ratio"
    ]
)

st.write("### SMA Window Performance Summary")
st.dataframe(format_metrics_table(sma_comparison_metrics), use_container_width=True)

st.write("### Parameter Heatmap: SMA Window Robustness")

sma_range = list(range(50, 401, 10))
heatmap_results = []

for window in sma_range:
    temp_df = run_strategy(raw_df.copy(), window, starting_balance, buffer_pct)

    cagr = calculate_cagr(temp_df["Strategy_Value"], temp_df["Date"])
    vol = calculate_volatility(temp_df["Strategy_Return"])
    max_dd = calculate_max_drawdown(temp_df["Strategy_Value"])
    sharpe = calculate_sharpe_ratio(temp_df["Strategy_Return"])
    calmar = calculate_calmar_ratio(cagr, max_dd)

    trade_count = temp_df["Position"].diff().abs().sum()
    time_in_market = temp_df["Position"].mean()

    heatmap_results.append({
        "SMA Window": window,
        "CAGR": cagr,
        "Volatility": vol,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Number of Trades": trade_count,
        "Time in Market": time_in_market
    })

heatmap_df = pd.DataFrame(heatmap_results)

selected_metric = st.selectbox(
    "Select metric for SMA window heatmap",
    [
        "Max Drawdown",
        "CAGR",
        "Volatility",
        "Sharpe Ratio",
        "Calmar Ratio",
        "Number of Trades",
        "Time in Market"
    ]
)

z_values = [heatmap_df[selected_metric].tolist()]

fig_heatmap = go.Figure(
    data=go.Heatmap(
        x=heatmap_df["SMA Window"].tolist(),
        y=[selected_metric],
        z=z_values,
        colorscale="RdYlGn",
        colorbar=dict(title=selected_metric),
        hovertemplate=(
            "SMA Window: %{x}<br>"
            f"{selected_metric}: " + "%{z:.4f}<extra></extra>"
        )
    )
)

fig_heatmap.update_layout(
    height=300,
    title=f"{selected_metric} by SMA Window",
    xaxis_title="SMA Window",
    yaxis_title=""
)

st.plotly_chart(fig_heatmap, use_container_width=True)

st.write("### SMA Parameter Results Table")

heatmap_table = heatmap_df.copy()

for col in ["CAGR", "Volatility", "Max Drawdown", "Time in Market"]:
    heatmap_table[col] = heatmap_table[col].apply(lambda x: f"{x:.2%}")

heatmap_table["Sharpe Ratio"] = heatmap_table["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
heatmap_table["Calmar Ratio"] = heatmap_table["Calmar Ratio"].apply(lambda x: f"{x:.2f}")
heatmap_table["Number of Trades"] = heatmap_table["Number of Trades"].apply(lambda x: f"{x:.0f}")

st.dataframe(heatmap_table, use_container_width=True, hide_index=True)

# -----------------------------
# SMA Buffer Optimization
# -----------------------------

st.write("### SMA Buffer Optimization")

st.write(
    "This section tests different SMA buffer percentages to determine which trigger zone produces the best risk-adjusted results."
)

buffer_range = np.arange(0, 0.105, 0.005)
buffer_results = []

for buffer in buffer_range:
    temp_df = run_strategy(raw_df.copy(), sma_window, starting_balance, buffer)

    cagr = calculate_cagr(temp_df["Strategy_Value"], temp_df["Date"])
    vol = calculate_volatility(temp_df["Strategy_Return"])
    max_dd = calculate_max_drawdown(temp_df["Strategy_Value"])
    sharpe = calculate_sharpe_ratio(temp_df["Strategy_Return"])
    calmar = calculate_calmar_ratio(cagr, max_dd)

    trade_count = temp_df["Position"].diff().abs().sum()
    time_in_market = temp_df["Position"].mean()
    final_value = temp_df["Strategy_Value"].iloc[-1]

    buffer_results.append({
        "Buffer %": buffer,
        "Final Value": final_value,
        "CAGR": cagr,
        "Volatility": vol,
        "Max Drawdown": max_dd,
        "Sharpe Ratio": sharpe,
        "Calmar Ratio": calmar,
        "Number of Trades": trade_count,
        "Time in Market": time_in_market
    })

buffer_df = pd.DataFrame(buffer_results)

best_by_calmar = buffer_df.sort_values(
    by=["Calmar Ratio", "Sharpe Ratio", "CAGR"],
    ascending=[False, False, False]
).head(1)

best_buffer_value = best_by_calmar["Buffer %"].iloc[0]
best_calmar_value = best_by_calmar["Calmar Ratio"].iloc[0]
best_cagr_value = best_by_calmar["CAGR"].iloc[0]
best_drawdown_value = best_by_calmar["Max Drawdown"].iloc[0]
best_trades_value = best_by_calmar["Number of Trades"].iloc[0]

st.write("#### Best Buffer Based on Calmar Ratio")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Best Buffer", f"{best_buffer_value:.2%}")
col2.metric("Calmar Ratio", f"{best_calmar_value:.2f}")
col3.metric("CAGR", f"{best_cagr_value:.2%}")
col4.metric("Max Drawdown", f"{best_drawdown_value:.2%}")
col5.metric("Trades", f"{best_trades_value:.0f}")

fig_buffer = go.Figure()

fig_buffer.add_trace(
    go.Scatter(
        x=buffer_df["Buffer %"],
        y=buffer_df["Calmar Ratio"],
        mode="lines+markers",
        name="Calmar Ratio",
        hovertemplate="Buffer: %{x:.2%}<br>Calmar Ratio: %{y:.2f}<extra></extra>"
    )
)

fig_buffer.add_trace(
    go.Scatter(
        x=buffer_df["Buffer %"],
        y=buffer_df["Sharpe Ratio"],
        mode="lines+markers",
        name="Sharpe Ratio",
        hovertemplate="Buffer: %{x:.2%}<br>Sharpe Ratio: %{y:.2f}<extra></extra>"
    )
)

fig_buffer.update_layout(
    height=550,
    title=f"SMA Buffer Optimization for {sma_window}-Day SMA",
    xaxis_title="Buffer Percentage",
    yaxis_title="Risk-Adjusted Metric",
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.03,
        xanchor="left",
        x=0
    )
)

fig_buffer.update_xaxes(tickformat=".0%")

st.plotly_chart(fig_buffer, use_container_width=True)

selected_buffer_metric = st.selectbox(
    "Select metric for buffer heatmap",
    [
        "Calmar Ratio",
        "Sharpe Ratio",
        "CAGR",
        "Max Drawdown",
        "Volatility",
        "Number of Trades",
        "Time in Market",
        "Final Value"
    ]
)

fig_buffer_heatmap = go.Figure(
    data=go.Heatmap(
        x=buffer_df["Buffer %"].tolist(),
        y=[selected_buffer_metric],
        z=[buffer_df[selected_buffer_metric].tolist()],
        colorscale="RdYlGn",
        colorbar=dict(title=selected_buffer_metric),
        hovertemplate=(
            "Buffer: %{x:.2%}<br>"
            f"{selected_buffer_metric}: " + "%{z:.4f}<extra></extra>"
        )
    )
)

fig_buffer_heatmap.update_layout(
    height=300,
    title=f"{selected_buffer_metric} by SMA Buffer",
    xaxis_title="Buffer Percentage",
    yaxis_title=""
)

fig_buffer_heatmap.update_xaxes(tickformat=".0%")

st.plotly_chart(fig_buffer_heatmap, use_container_width=True)

st.write("#### SMA Buffer Optimization Results Table")

buffer_table = buffer_df.copy()

buffer_table["Buffer %"] = buffer_table["Buffer %"].apply(lambda x: f"{x:.2%}")
buffer_table["Final Value"] = buffer_table["Final Value"].apply(lambda x: f"${x:,.2f}")

for col in ["CAGR", "Volatility", "Max Drawdown", "Time in Market"]:
    buffer_table[col] = buffer_table[col].apply(lambda x: f"{x:.2%}")

buffer_table["Sharpe Ratio"] = buffer_table["Sharpe Ratio"].apply(lambda x: f"{x:.2f}")
buffer_table["Calmar Ratio"] = buffer_table["Calmar Ratio"].apply(lambda x: f"{x:.2f}")
buffer_table["Number of Trades"] = buffer_table["Number of Trades"].apply(lambda x: f"{x:.0f}")

st.dataframe(buffer_table, use_container_width=True, hide_index=True)

buffer_csv = buffer_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Buffer Optimization Results as CSV",
    data=buffer_csv,
    file_name=f"{ticker}_sma_buffer_optimization.csv",
    mime="text/csv"
)

st.success("Data loaded and strategy calculated successfully.")
