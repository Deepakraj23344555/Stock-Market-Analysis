import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ta

# Page setup
st.set_page_config(page_title="Reliance Stock Analysis", layout="wide")

# Title
st.title("📊 Reliance Industries Stock Analysis App")
st.markdown("Analyze historical performance, technical indicators, and investment signals for **RELIANCE.NS**.")

# Function to load and process data
@st.cache_data
def load_data():
    df = yf.download('RELIANCE.NS', start='2010-01-01', end='2025-01-01')
    df.dropna(inplace=True)

    # Safety checks
    if df.empty or 'Close' not in df.columns:
        return pd.DataFrame()

    # Technical Indicators
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    # RSI
    try:
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
    except Exception as e:
        st.warning(f"⚠️ RSI calculation failed: {e}")
        df['RSI'] = None

    # MACD
    try:
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
    except:
        df['MACD'] = df['MACD_Signal'] = None

    # Bollinger Bands
    try:
        bb = ta.volatility.BollingerBands(close=df['Close'])
        df['BB_High'] = bb.bollinger_hband()
        df['BB_Low'] = bb.bollinger_lband()
    except:
        df['BB_High'] = df['BB_Low'] = None

    df['Daily_Return'] = df['Close'].pct_change()

    return df

# Load the data
df = load_data()

# Stop if data failed to load
if df.empty:
    st.error("❌ Failed to fetch valid data for RELIANCE.NS. Please check your internet or try again later.")
    st.stop()

# --- Section: Latest Data Table ---
st.subheader("📅 Latest Data Snapshot")
st.dataframe(df.tail(10))

# --- Section: Price with SMA/EMA ---
st.subheader("📈 Close Price with SMA & EMA")
fig1, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df['Close'], label='Close Price')
ax1.plot(df['SMA_20'], label='SMA 20')
ax1.plot(df['EMA_20'], label='EMA 20')
ax1.set_title("RELIANCE.NS Price with SMA & EMA")
ax1.legend()
ax1.grid()
st.pyplot(fig1)

# --- Section: RSI Chart ---
if df['RSI'].notnull().any():
    st.subheader("📊 RSI (Relative Strength Index)")
    fig2, ax2 = plt.subplots(figsize=(12, 3))
    ax2.plot(df['RSI'], color='orange')
    ax2.axhline(70, linestyle='--', color='red', label='Overbought')
    ax2.axhline(30, linestyle='--', color='green', label='Oversold')
    ax2.set_title("RSI Indicator")
    ax2.legend()
    ax2.grid()
    st.pyplot(fig2)

# --- Section: MACD Chart ---
if df['MACD'].notnull().any():
    st.subheader("📉 MACD (Moving Average Convergence Divergence)")
    fig3, ax3 = plt.subplots(figsize=(12, 3))
    ax3.plot(df['MACD'], label='MACD', color='blue')
    ax3.plot(df['MACD_Signal'], label='Signal', color='red')
    ax3.set_title("MACD and Signal Line")
    ax3.legend()
    ax3.grid()
    st.pyplot(fig3)

# --- Section: Bollinger Bands ---
if df['BB_High'].notnull().any():
    st.subheader("📊 Bollinger Bands")
    fig4, ax4 = plt.subplots(figsize=(12, 5))
    ax4.plot(df['Close'], label='Close Price', color='blue')
    ax4.plot(df['BB_High'], label='Upper Band', linestyle='--', color='green')
    ax4.plot(df['BB_Low'], label='Lower Band', linestyle='--', color='red')
    ax4.fill_between(df.index, df['BB_High'], df['BB_Low'], alpha=0.1)
    ax4.set_title("Bollinger Bands")
    ax4.legend()
    ax4.grid()
    st.pyplot(fig4)

# --- Section: Return Analysis ---
st.subheader("📈 Daily Return Distribution")
fig5, ax5 = plt.subplots(figsize=(10, 4))
sns.histplot(df['Daily_Return'].dropna(), bins=50, kde=True, ax=ax5)
ax5.set_title("Distribution of Daily Returns")
st.pyplot(fig5)

# --- Section: Investment Summary ---
st.subheader("💡 Investment Summary")
rsi_value = df['RSI'].iloc[-1] if df['RSI'].notnull().any() else None
rsi_signal = (
    "📈 Overbought" if rsi_value and rsi_value > 70 else
    "📉 Oversold" if rsi_value and rsi_value < 30 else
    "🟡 Neutral" if rsi_value else "N/A"
)
trend = "🟢 Bullish" if df['Close'].iloc[-1] > df['SMA_20'].iloc[-1] else "🔴 Bearish"

summary = f"""
- **Latest Close Price**: ₹{df['Close'].iloc[-1]:.2f}  
- **Average Daily Return**: {df['Daily_Return'].mean():.4f}  
- **Volatility (Std Dev)**: {df['Daily_Return'].std():.4f}  
- **RSI Value**: {rsi_value:.2f if rsi_value else 'N/A'} → {rsi_signal}  
- **Trend Signal**: {trend}
"""
st.markdown(summary)
