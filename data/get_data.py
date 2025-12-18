import yfinance as yf
from src.config import RAW_DATA_DIR


def download_data(ticker="AAPL"):
    print(f"--- DOWNLOADING DATA FOR {ticker} ---")
    df = yf.download(ticker, period="2y", interval="1d")

    # --- FIX: FLATTEN MULTI-LEVEL COLUMNS ---
    # If yfinance returns headers like ("Close", "AAPL"), flatten them to just "Close"
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # OPTIONAL: Save ONLY the 'Close' column to keep the file tiny and clean
    df = df[['Close']]

    csv_path = RAW_DATA_DIR / "stock_data.csv"
    df.to_csv(csv_path)
    print(f"✅ Saved clean data to {csv_path}")


if __name__ == "__main__":
    import pandas as pd  # Ensure pandas is imported

    download_data()