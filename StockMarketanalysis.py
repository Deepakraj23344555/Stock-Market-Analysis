"""
Stock Analysis Tool for AAPL

Author: Deepak Raj
Description:
    This Python script performs technical analysis on historical stock data using Pandas and Matplotlib.
    It includes moving averages, Bollinger Bands, RSI, MACD, and AI-style interpretation.

Usage:
    - Save this file as stock_analysis.py
    - Place it in the same directory as AAPL.csv
    - Run it in a Python 3 environment
"""

# ===================== #
#      LIBRARIES        #
# ===================== #
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===================== #
#      LOAD DATA        #
# ===================== #
df = pd.read_csv("AAPL.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.sort_index(inplace=True)

# ============================== #
#  TECHNICAL INDICATORS CALC     #
# ============================== #

# Simple & Exponential Moving Averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

# Bollinger Bands
rolling_std = df['Close'].rolling(window=20).std()
df['Bollinger_Upper'] = df['EMA_20'] + (2 * rolling_std)
df['Bollinger_Lower'] = df['EMA_20'] - (2 * rolling_std)

# RSI
delta = df['Close'].diff()
gain = np.where(delta > 0, delta, 0)
loss = np.where(delta < 0, -delta, 0)
avg_gain = pd.Series(gain).rolling(window=14).mean()
avg_loss = pd.Series(loss).rolling(window=14).mean()
rs = avg_gain / avg_loss
df['RSI'] = 100 - (100 / (1 + rs))

# MACD
ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = ema_12 - ema_26
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

# Daily Return
df['Daily_Return'] = df['Close'].pct_change()

# Signal Generation (SMA vs EMA)
df['Signal'] = 0
df.loc[df['SMA_50'] > df['EMA_20'], 'Signal'] = 1
df.loc[df['SMA_50'] < df['EMA_20'], 'Signal'] = -1

# Drop NaNs
df.dropna(inplace=True)

# ===================== #
#       PLOTS           #
# ===================== #

# Plot 1: Price, SMA, EMA
plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close Price')
plt.plot(df['SMA_50'], label='SMA 50', color='green')
plt.plot(df['EMA_20'], label='EMA 20', color='orange')
plt.title('Close Price with SMA & EMA')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Bollinger Bands
plt.figure(figsize=(14,6))
plt.plot(df['Close'], label='Close')
plt.plot(df['Bollinger_Upper'], label='Upper Band', linestyle='--', color='red')
plt.plot(df['Bollinger_Lower'], label='Lower Band', linestyle='--', color='green')
plt.fill_between(df.index, df['Bollinger_Lower'], df['Bollinger_Upper'], color='gray', alpha=0.1)
plt.title("Bollinger Bands")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: RSI
plt.figure(figsize=(12,3))
plt.plot(df['RSI'], label='RSI', color='purple')
plt.axhline(70, linestyle='--', color='red', label='Overbought')
plt.axhline(30, linestyle='--', color='green', label='Oversold')
plt.title("RSI Indicator")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 4: MACD
plt.figure(figsize=(12,4))
plt.plot(df['MACD'], label='MACD', color='blue')
plt.plot(df['Signal_Line'], label='Signal Line', color='orange')
plt.title("MACD & Signal Line")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 5: Daily Returns
plt.figure(figsize=(12,4))
plt.plot(df['Daily_Return'], label='Daily Return', color='black')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Daily Returns Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ============================= #
#    AI-STYLE SUMMARY OUTPUT    #
# ============================= #
latest = df.iloc[-1]
summary = f"""
📊 Investment Insight Summary (AAPL)
----------------------------------------
Latest Close Price: ${latest['Close']:.2f}
Daily Return: {latest['Daily_Return']*100:.2f}%
RSI: {latest['RSI']:.2f} → {'Overbought' if latest['RSI'] > 70 else 'Oversold' if latest['RSI'] < 30 else 'Neutral'}
MACD: {latest['MACD']:.2f} vs Signal: {latest['Signal_Line']:.2f} → {"Uptrend" if latest['MACD'] > latest['Signal_Line'] else "Downtrend"}
SMA vs EMA: {"Buy Signal" if latest['Signal'] == 1 else "Sell Signal" if latest['Signal'] == -1 else "Hold"}
Bollinger Bands: {'Above Upper Band' if latest['Close'] > latest['Bollinger_Upper'] else 'Below Lower Band' if latest['Close'] < latest['Bollinger_Lower'] else 'Within Bands'}
"""

print(summary)
