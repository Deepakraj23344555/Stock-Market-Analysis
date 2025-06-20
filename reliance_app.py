# reliance_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import ta
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Page config
st.set_page_config(layout="wide")
st.title("📈 Reliance Industries Stock Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    df = yf.download("RELIANCE.NS", start="2020-01-01", end="2025-01-01")
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    rsi = ta.momentum.RSIIndicator(close=df['Close'], window=14)
    macd = ta.trend.MACD(close=df['Close'])
    df['RSI'] = rsi.rsi()
    df['MACD'] = macd.macd()
    df['Signal'] = macd.macd_signal()
    df = df.dropna()
    return df

data = load_data()

# --- Sidebar ---
st.sidebar.header("Indicators")
show_sma = st.sidebar.checkbox("Show SMA (20 & 50)", value=True)
show_rsi = st.sidebar.checkbox("Show RSI (14)", value=True)
show_macd = st.sidebar.checkbox("Show MACD", value=True)
st.sidebar.download_button("📥 Download Data as CSV", data.to_csv(), "reliance_data.csv")

# --- Price Chart ---
st.subheader("📊 Closing Price with Indicators")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(data['Close'], label='Close', color='blue')
if show_sma:
    ax.plot(data['SMA_20'], label='SMA 20', color='red', linestyle='--')
    ax.plot(data['SMA_50'], label='SMA 50', color='green', linestyle='--')
ax.set_ylabel("Price (INR)")
ax.set_xlabel("Date")
ax.legend()
st.pyplot(fig)

# --- RSI Chart ---
if show_rsi:
    st.subheader("📉 RSI (Relative Strength Index)")
    fig_rsi, ax_rsi = plt.subplots(figsize=(12, 3))
    ax_rsi.plot(data['RSI'], label='RSI', color='purple')
    ax_rsi.axhline(70, linestyle='--', color='red')
    ax_rsi.axhline(30, linestyle='--', color='green')
    ax_rsi.set_ylabel("RSI")
    ax_rsi.legend()
    st.pyplot(fig_rsi)

# --- MACD Chart ---
if show_macd:
    st.subheader("📈 MACD (Moving Average Convergence Divergence)")
    fig_macd, ax_macd = plt.subplots(figsize=(12, 3))
    ax_macd.plot(data['MACD'], label='MACD', color='blue')
    ax_macd.plot(data['Signal'], label='Signal', color='orange')
    ax_macd.legend()
    st.pyplot(fig_macd)

# --- Machine Learning Prediction ---
st.subheader("🤖 Price Prediction (Linear Regression)")

# Prepare features and target
features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal']
X = data[features]
y = data['Close']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Show chart
fig_pred, ax_pred = plt.subplots(figsize=(12, 5))
ax_pred.plot(y_test.index, y_test, label='Actual', color='black')
ax_pred.plot(y_test.index, y_pred, label='Predicted', linestyle='--', color='orange')
ax_pred.set_title(f"Actual vs Predicted Close Price (MSE = {mse:.2f})")
ax_pred.set_ylabel("Price (INR)")
ax_pred.legend()
st.pyplot(fig_pred)

# --- Show Raw Data ---
with st.expander("📄 Show Raw Data"):
    st.dataframe(data.tail())
