import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("📊 Stock Market Dashboard")
st.caption("Enter any stock ticker to explore price trends, volatility, and momentum.")

# --- Inputs ---
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    ticker = st.text_input("Ticker Symbol", value="AAPL", help="e.g. AAPL, MSFT, RY.TO, SHOP.TO").upper()
with col2:
    period = st.selectbox("Time Period", ["6mo", "1y", "2y", "5y"], index=1)
with col3:
    ma_window = st.selectbox("Moving Average", [20, 50, 100, 200], index=1)

# --- Fetch Data ---
@st.cache_data(ttl=300)
def load_data(ticker, period):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    return data

try:
    df = load_data(ticker, period)
    if df.empty:
        st.error("Ticker not found. Try something like AAPL, MSFT, or RY.TO.")
        st.stop()
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

df = df.copy()
df[f"MA{ma_window}"] = df["Close"].rolling(ma_window).mean()
df["Returns"] = df["Close"].pct_change()
df["Volatility"] = df["Returns"].rolling(20).std() * np.sqrt(252) * 100  # annualized %

# --- Summary Metrics ---
latest = df["Close"].iloc[-1]
start_price = df["Close"].iloc[0]
total_return = ((latest - start_price) / start_price) * 100
avg_vol = df["Volatility"].dropna().mean()
max_price = df["Close"].max()
min_price = df["Close"].min()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Current Price", f"${float(latest):.2f}")
m2.metric("Total Return", f"{float(total_return):+.1f}%")
m3.metric("Avg Annualized Volatility", f"{float(avg_vol):.1f}%")
m4.metric(f"52-Week Range", f"${float(min_price):.2f} – ${float(max_price):.2f}")

st.divider()

# --- Charts ---
fig, axes = plt.subplots(3, 1, figsize=(12, 10), facecolor="#0e1117")
fig.subplots_adjust(hspace=0.4)

for ax in axes:
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

# Chart 1: Price + MA
ax1 = axes[0]
ax1.plot(df.index, df["Close"], color="#4FC3F7", linewidth=1.5, label="Close Price")
ax1.plot(df.index, df[f"MA{ma_window}"], color="#FF8A65", linewidth=1.5,
         linestyle="--", label=f"{ma_window}-Day MA")
ax1.fill_between(df.index, df["Close"], alpha=0.08, color="#4FC3F7")
ax1.set_title(f"{ticker} — Price & Moving Average")
ax1.legend(facecolor="#1a1a2e", labelcolor="white")
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

# Chart 2: Daily Returns
ax2 = axes[1]
returns = df["Returns"].dropna()
colors = ["#66BB6A" if r >= 0 else "#EF5350" for r in returns]
ax2.bar(returns.index, returns * 100, color=colors, width=1.5, alpha=0.8)
ax2.axhline(0, color="white", linewidth=0.5)
ax2.set_title("Daily Returns (%)")
ax2.set_ylabel("Return (%)", color="white")

# Chart 3: Rolling Volatility
ax3 = axes[2]
ax3.plot(df.index, df["Volatility"], color="#CE93D8", linewidth=1.5)
ax3.fill_between(df.index, df["Volatility"], alpha=0.15, color="#CE93D8")
ax3.set_title("Annualized Volatility (20-Day Rolling)")
ax3.set_ylabel("Volatility (%)", color="white")
ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))

st.pyplot(fig)
plt.close()

st.divider()

# --- Linear Regression Trend ---
st.subheader("📈 Price Trend (Linear Regression)")

clean = df[["Close"]].dropna().copy()
X = np.arange(len(clean)).reshape(-1, 1)
y = clean["Close"].values.flatten()

model = LinearRegression()
model.fit(X, y)
trend = model.predict(X)
r2 = model.score(X, y)
slope = float(model.coef_[0])

fig2, ax = plt.subplots(figsize=(12, 3.5), facecolor="#0e1117")
ax.set_facecolor("#0e1117")
ax.tick_params(colors="white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
ax.plot(clean.index, y, color="#4FC3F7", linewidth=1, alpha=0.6, label="Actual")
ax.plot(clean.index, trend, color="#FFD54F", linewidth=2, linestyle="--", label="Trend Line")
ax.legend(facecolor="#1a1a2e", labelcolor="white")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
ax.tick_params(colors="white")
st.pyplot(fig2)
plt.close()

c1, c2 = st.columns(2)
c1.metric("Trend Slope", f"${slope:+.4f} / day")
c2.metric("R² Score", f"{r2:.4f}", help="How well the linear trend fits. Closer to 1 = stronger trend.")

st.divider()

# --- Raw Data ---
with st.expander("View Raw Data"):
    st.dataframe(df[["Open", "High", "Low", "Close", "Volume"]].tail(30).round(2))