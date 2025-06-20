# 📦 Import libraries
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import ta

# 📊 Set seaborn style
sns.set(style="darkgrid")

# 📥 Download stock data
ticker = 'RELIANCE.NS'  # Use '500325.BO' for BSE
data = yf.download(ticker, start='2020-01-01', end='2025-01-01', interval='1d')

# 🧮 Technical indicators
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

rsi = ta.momentum.RSIIndicator(close=data['Close'], window=14)
data['RSI'] = rsi.rsi()

macd = ta.trend.MACD(close=data['Close'])
data['MACD'] = macd.macd()
data['Signal'] = macd.macd_signal()

# 🖼️ Plot closing price + SMAs
plt.figure(figsize=(14, 6))
plt.plot(data['Close'], label='Close Price', color='blue')
plt.plot(data['SMA_20'], label='20-Day SMA', color='red')
plt.plot(data['SMA_50'], label='50-Day SMA', color='green')
plt.title('Reliance Daily Close Price with SMAs')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

# 🖼️ Plot RSI
plt.figure(figsize=(14, 4))
plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(70, color='red', linestyle='--')
plt.axhline(30, color='green', linestyle='--')
plt.title('RSI (14-day)')
plt.xlabel('Date')
plt.ylabel('RSI')
plt.legend()
plt.tight_layout()
plt.show()

# 🖼️ Plot MACD
plt.figure(figsize=(14, 4))
plt.plot(data['MACD'], label='MACD', color='blue')
plt.plot(data['Signal'], label='Signal Line', color='orange')
plt.title('MACD')
plt.xlabel('Date')
plt.ylabel('MACD Value')
plt.legend()
plt.tight_layout()
plt.show()

# 💾 Drop missing values for model training
data = data.dropna()

# 🎯 ML: Prepare data
features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'Signal']
X = data[features]
y = data['Close']

# 🧪 Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 🤖 Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 📈 Predict
y_pred = model.predict(X_test)

# 📉 Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f'\n📊 Mean Squared Error: {mse:.2f}\n')

# 🖼️ Plot actual vs predicted prices
plt.figure(figsize=(14, 6))
plt.plot(y_test.index, y_test, label='Actual', color='black')
plt.plot(y_test.index, y_pred, label='Predicted', color='orange', linestyle='--')
plt.title('Actual vs Predicted Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.tight_layout()
plt.show()

# 💾 Export to CSV
data.to_csv('reliance_stock_analysis.csv')
print("✅ Data exported to 'reliance_stock_analysis.csv'")
