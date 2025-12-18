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
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, ML_CONFIG, MODEL_DIR, RAW_DATA_DIR

# Initialize the Vector DB connection once (Global)
print(f"[INIT] Loading Vector DB from {CHROMA_DB_PATH}...")
vector_db = Chroma(
    persist_directory=str(CHROMA_DB_PATH),
    embedding_function=EMBEDDING_MODEL
)

# We load the model when the app starts, so we don't reload it for every user request
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ML_CONFIG["model_path"]
SCALER_PATH = RAW_DATA_DIR / "scaler.pkl"

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

# Load scaler
scaler = None
try:
    print(f"[INIT] Loading Scaler from {SCALER_PATH}...")
    scaler = joblib.load(SCALER_PATH)
    print("[INIT] Scaler loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load scaler ({e}). Predictions will be wildly wrong (unscaled).")

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
def predict_future_price(recent_prices_str: str) -> str:
    """
    Uses the PyTorch LSTM model to forecast the price.
    Args:
        recent_prices_str (str): A string list of recent prices (e.g., "[150.1, 152.3]").
    """
    if model is None:
        return "Error: Prediction model is not loaded."
    if scaler is None:
        return "Error: Data Scaler is not loaded (cannot normalize inputs)."

    print(f"[TOOL] Running PyTorch Inference...")

    try:
        # A. Parse Input (String -> List)
        recent_prices = ast.literal_eval(recent_prices_str)

        # Validation: We need at least 'window_size' data points
        # If we have too few, the model will crash or give garbage.
        required_window = ML_CONFIG.get("window_size", 10)
        if len(recent_prices) < required_window:
            return f"Error: Model requires {required_window} days of data, but got {len(recent_prices)}."

        # Keep only the most recent 'window_size' items
        recent_prices = recent_prices[-required_window:]

        # B. PREPROCESS (Apply the Scaler)
        # 1. Convert to numpy array (Shape: [N, 1])
        input_array = np.array(recent_prices).reshape(-1, 1)

        # 2. Scale values to 0-1 range (using the logic from training)
        scaled_inputs = scaler.transform(input_array)

        # 3. Convert to Tensor (Batch=1, Seq=Window, Feature=1)
        input_tensor = torch.tensor(scaled_inputs).float().view(1, -1, 1).to(DEVICE)

        # C. INFERENCE
        with torch.no_grad():
            prediction_tensor = model(input_tensor)
            # Output is a single scaled float (e.g., 0.65)
            scaled_prediction = prediction_tensor.cpu().numpy()

        # D. POST-PROCESS (Inverse Transform)
        # We must reshape to (1, 1) for the scaler to accept it
        real_prediction = scaler.inverse_transform(scaled_prediction.reshape(-1, 1))
        predicted_price = real_prediction[0][0]

        # Calculate context for the user
        last_price = recent_prices[-1]
        change = ((predicted_price - last_price) / last_price) * 100
        direction = "UP" if change > 0 else "DOWN"

        return (f"The LSTM model predicts the price will move {direction} to "
                f"${predicted_price:.2f} ({change:+.2f}%).")

    except Exception as e:
        return f"Error running inference: {str(e)}"


# --- TEST BLOCK ---
# Run this file directly to test if yfinance works!
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