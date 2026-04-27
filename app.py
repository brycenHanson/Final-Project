import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf 
import matplotlib.pyplot as plt
from ta.momentum import RSIIndicator

# Optional: set page config
st.set_page_config(page_title="Stock Analytics & Portfolio Dashboard", layout="wide")

st.title("Stock Analytics & Portfolio Dashboard (Streamlit)")
st.write("Explore a single stock and a 5-stock portfolio with performance metrics.")

@st.cache
def fetch_stock_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end, progress=False)
    if 'Close' in data.columns:
        data = data[['Close']].rename(columns={'Close': 'Close'})
        data = data.dropna()
    return data

@st.cache
def fetch_multiple_close(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, progress=False)['Close']
    return data

def compute_part1(data, window1=20, window2=50, rsi_window=14):
    data = data.copy()

    data['20MA'] = data['Close'].rolling(window=window1).mean()
    data['50MA'] = data['Close'].rolling(window=window2).mean()
    data['Return'] = data['Close'].pct_change()
    data['Vol_20d'] = data['Return'].rolling(window=20).std() * np.sqrt(252)

    # --- FIXED RSI SECTION ---
    close_prices = data['Close']

    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    close_prices = pd.Series(close_prices).dropna()

    rsi = RSIIndicator(close=close_prices, window=rsi_window).rsi()
    data['RSI'] = rsi
    # -------------------------

    latest = data.iloc[-1].squeeze()
    close_val = float(latest['Close'])
ma20 = float(latest['20MA']) if pd.notna(latest['20MA']) else np.nan
ma50 = float(latest['50MA']) if pd.notna(latest['50MA']) else np.nan
rsi_val = float(latest['RSI']) if pd.notna(latest['RSI']) else np.nan
vol_val = float(latest['Vol_20d']) if pd.notna(latest['Vol_20d']) else np.nan

    if pd.isna(latest['20MA']) or pd.isna(latest['50MA']):
        trend = 'Insufficient data'
    elif latest['Close'] > latest['20MA'] > latest['50MA']:
        trend = 'Strong Uptrend'
    elif latest['Close'] < latest['20MA'] < latest['50MA']:
        trend = 'Strong Downtrend'
    else:
        trend = 'Mixed Trend'

    signal = 'Hold'
    if not (pd.isna(latest['RSI']) or pd.isna(latest['Vol_20d'])):
        if latest['RSI'] < 30:
            signal = 'Buy'
        elif latest['RSI'] > 70:
            signal = 'Sell'

    latest_summary = {
        'CurrentPrice': float(latest['Close']),
        '20MA': float(latest['20MA']),
        '50MA': float(latest['50MA']),
        'RSI': float(latest['RSI']),
        'Vol20d': float(latest['Vol_20d']),
        'Trend': trend,
        'Signal': signal
    }

    return data, latest_summary

def compute_part2(stocks, weights, benchmark, start, end):
    tickers = stocks + [benchmark]
    prices = fetch_multiple_close(tickers, start, end)
    returns = prices.pct_change().dropna()
    stock_ret = returns[stocks]
    port_ret = (returns[stocks] * weights).sum(axis=1)
    bench_ret = returns[benchmark]
    portfolio_ann = port_ret.mean() * 252
    bench_ann = bench_ret.mean() * 252
    port_vol = port_ret.std() * np.sqrt(252)
    bench_vol = bench_ret.std() * np.sqrt(252)
    port_sharpe = port_ret.mean() / port_ret.std() * np.sqrt(252)
    bench_sharpe = bench_ret.mean() / bench_ret.std() * np.sqrt(252)
    port_cum = (1 + port_ret).iloc[-1] - 1
    bench_cum = (1 + bench_ret).iloc[-1] - 1
    metrics = {
        'Portfolio Annualized Return': portfolio_ann,
        'Benchmark Annualized Return': bench_ann,
        'Portfolio Volatility ( Annualized )': port_vol,
        'Benchmark Volatility ( Annualized )': bench_vol,
        'Portfolio Sharpe': port_sharpe,
        'Outperformance vs Benchmark': portfolio_ann - bench_ann,
        'Portfolio Cumulative Return': port_cum,
        'Benchmark Cumulative Return': bench_cum,
    }
    return prices, returns, metrics

# UI: Part selection
st.sidebar.header("Part selection")
mode = st.sidebar.radio("Choose part to view", ["Part 1 - Individual Stock", "Part 2 - Portfolio"])

if mode == "Part 1 - Individual Stock":
    st.header("Part 1: Individual Stock Analysis")
    ticker = st.text_input("Ticker", value="AAPL")
    days = st.number_input("Data lookback (days, up to ~180)", min_value=60, max_value=365, value=180, step=1)
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.Timedelta(days=int(days))

    data = fetch_stock_data(ticker, start_date, end_date)
    if data is None or data.empty:
        st.error("No data found for ticker.")
    else:
        data, latest = compute_part1(data)
        st.subheader(f"{ticker} Analysis Summary")
        st.write(latest)

        # Charts
        st.line_chart(data[['Close', '20MA', '50MA']].dropna())
        # RSI plot
        fig, ax = plt.subplots()
        ax.plot(data.index, data['RSI'], label='RSI(14)')
        ax.set_title('RSI (14)')
        ax.set_ylabel('RSI')
        ax.legend()
        st.pyplot(fig)

        # Volatility track
        st.line_chart(data['Vol_20d'].dropna())

        # Option: show a small table of recent rows
        st.dataframe(data.tail(5))

elif mode == "Part 2 - Portfolio":
    st.header("Part 2: Portfolio Performance Dashboard")
    # User inputs for portfolio
    tickers = st.text_input("Enter 5 tickers separated by commas", value="AAPL,MSFT,AMZN,GOOGL,TSLA")
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if len(ticker_list) != 5:
        st.error("Please enter exactly 5 tickers.")
    else:
        weights_input = st.text_input("Weights (comma-separated, sum to 1.0)", value="0.20,0.25,0.15,0.25,0.15")
        weights = [float(w.strip()) for w in weights_input.split(",")]
        if len(weights) != 5 or not np.isclose(sum(weights), 1.0):
            st.error("Weights must sum to 1.0 and be 5 values.")
        else:
            benchmark = st.text_input("Benchmark ticker", value="SPY")
            end = pd.Timestamp.today()
            start = end - pd.DateOffset(years=1)

            prices, returns, metrics = compute_part2(ticker_list, np.array(weights), benchmark, start, end)

            st.subheader("Portfolio Metrics")
            st.table(pd.DataFrame([metrics]))

            # Cumulative return chart
            port_cum = (1 + returns[ticker_list].mul(weights, axis=1).sum(axis=1)).cumprod()
            bench_cum = (1 + returns[benchmark]).cumprod()
            fig2, ax2 = plt.subplots(figsize=(10,4))
            ax2.plot(port_cum.index, port_cum, label="Portfolio")
            ax2.plot(bench_cum.index, bench_cum, label="Benchmark")
            ax2.set_ylabel("Cumulative Return")
            ax2.legend()
            st.pyplot(fig2)

            # Optional: show a simple table of final values
            final_vals = pd.DataFrame({
                "Portfolio": port_cum.iloc[-1],
                "Benchmark": bench_cum.iloc[-1]
            }, index=["Cumulative Return"])
            st.write(final_vals)
