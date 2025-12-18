import yfinance as yf
import pandas as pd
import numpy as np
from src.config import RAW_DATA_DIR


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def download_data(ticker="AAPL"):
    print(f"--- DOWNLOADING MULTIVARIATE DATA FOR {ticker} ---")

    # Download 5 years to ensure we have enough buffer for SMA calculations
    df = yf.download(ticker, period="5y", interval="1d")

    # Flatten MultiIndex if necessary
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # 1. Feature Engineering
    # SMA 50
    df['SMA_50'] = df['Close'].rolling(window=50).mean()

    # RSI 14
    df['RSI'] = calculate_rsi(df['Close'], period=14)

    # 2. Clean Data
    # Drop rows with NaN (the first 50 rows will be empty due to SMA)
    df.dropna(inplace=True)

    # 3. Select Columns
    final_df = df[['Close', 'Volume', 'SMA_50', 'RSI']]

    csv_path = RAW_DATA_DIR / "stock_data.csv"
    final_df.to_csv(csv_path)

    print(f"✅ Saved clean multivariate data to {csv_path}")
    print(f"   Shape: {final_df.shape}")
    print(f"   Columns: {final_df.columns.tolist()}")


if __name__ == "__main__":
    download_data()