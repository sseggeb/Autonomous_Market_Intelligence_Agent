import yfinance as yf
import torch
import ast
import joblib
import numpy as np
from langchain.tools import tool
from langchain_chroma import Chroma

# Import settings from your centralized config
# This ensures we use the correct Embedding Model (OpenAI vs Local) and DB Path
from src.ml_engine.architecture import MarketLSTM
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, ML_CONFIG, MODEL_DIR, PROCESSED_DATA_DIR

# Initialize the Vector DB connection once (Global)
print(f"[INIT] Loading Vector DB from {CHROMA_DB_PATH}...")
vector_db = Chroma(
    persist_directory=str(CHROMA_DB_PATH),
    embedding_function=EMBEDDING_MODEL
)

# We load the model when the app starts, so we don't reload it for every user request
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ML_CONFIG["model_path"]

# Load Model
model = None
print(f"[INIT] Loading LSTM Model from {MODEL_PATH}...")
try:
    # 1. Initialize the Architecture (Must match training!)
    model = MarketLSTM(
        input_size=ML_CONFIG["input_size"],
        hidden_size=ML_CONFIG["hidden_size"],
        output_size=ML_CONFIG["output_size"],
        num_layers=ML_CONFIG["num_layers"],
        dropout=ML_CONFIG["dropout"]
    )
    # 2. Load the Weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode (freezes dropout/batchnorm)
    print("[INIT] Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load model ({e}). Inference tool will fail.")

# Load scalers
feature_scaler = None
target_scaler = None
try:
    print(f"[INIT] Loading Scalers from {PROCESSED_DATA_DIR}...")
    feature_scaler = joblib.load(PROCESSED_DATA_DIR / "feature_scaler.pkl")
    target_scaler = joblib.load(PROCESSED_DATA_DIR / "target_scaler.pkl")
    print("[INIT] Scalers loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load scaler(s) ({e}). Predictions will be wildly wrong (unscaled).")

@tool
def get_current_stock_price(ticker: str) -> str:
    """
    Fetches the latest closing price for a given stock ticker (e.g., 'AAPL', 'NVDA').
    Returns a string message with the price.
    """
    print(f"[TOOL] Fetching price for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        # Fetch 1 day of history
        data = stock.history(period="1d")

        if data.empty:
            return f"Error: Could not fetch data for ticker '{ticker}'. Is it valid?"

        # Get the 'Close' price of the most recent data point
        price = data["Close"].iloc[-1]
        return f"{price:.2f}"

    except Exception as e:
        return f"Error fetching price for {ticker}: {str(e)}"

@tool
def retrieve_financial_context(query: str) -> str:
    """
    Searches the internal knowledge base (SEC filings, news) for relevant context.
    Use this to find specific financial risks or news mentioned in reports.
    """
    print(f"[TOOL] RAG Search for: '{query}'...")

    # Search the Vector DB (Top 3 most relevant chunks)
    results = vector_db.similarity_search(query, k=3)

    if not results:
        return "No relevant financial documents found in the database."

    # Combine the chunks into a single text block
    context_text = "\n---\n".join([doc.page_content for doc in results])
    return context_text

@tool
def predict_future_price(recent_data_str: str) -> str:
    """
    Args:
        recent_data_str: String of list of lists e.g. "[[150, 10000, 145, 60], ...]"
    """
    # ... checks ...

    try:
        # 1. Parse Data
        recent_data = ast.literal_eval(recent_data_str)

        # 2. Preprocess
        # Shape: (60, 4)
        input_array = np.array(recent_data)

        # Scale inputs using feature_scaler
        scaled_inputs = feature_scaler.transform(input_array)

        # To Tensor: (1, 60, 4)
        input_tensor = torch.tensor(scaled_inputs).float().view(1, -1, 4).to(DEVICE)

        # 3. Inference
        with torch.no_grad():
            prediction = model(input_tensor)
            scaled_pred_val = prediction.cpu().numpy()  # Shape (1, 1)

        # 4. Inverse Transform (Target only!)
        # Use target_scaler, which expects shape (N, 1)
        real_price = target_scaler.inverse_transform(scaled_pred_val)
        final_price = real_price[0][0]

        # Calculate change based on the LAST CLOSE price (Column 0 of the last row)
        last_close = recent_data[-1][0]
        change = ((final_price - last_close) / last_close) * 100
        direction = "UP" if change > 0 else "DOWN"

        return (f"Based on Price, Volume, SMA, and RSI, the model predicts "
                f"{direction} to ${final_price:.2f} ({change:+.2f}%).")

    except Exception as e:
        return f"Error: {str(e)}"

# --- TEST BLOCK ---
if __name__ == "__main__":
    # Test 1: Stock Price
    print("\n--- Testing Stock Price ---")
    price = get_current_stock_price.invoke("AAPL")
    print(f"Result: {price}")

    # Test 2: RAG (Only works if you ran ingest.py)
    print("\n--- Testing RAG ---")
    try:
        context = retrieve_financial_context.invoke("revenue")
        print(f"Result: {context[:100]}...")  # Print first 100 chars
    except Exception as e:
        print(f"RAG Test Failed (Expected if DB is empty): {e}")