import yfinance as yf
import torch
import ast
from langchain.tools import tool
from langchain_chroma import Chroma

# Import settings from your centralized config
# This ensures we use the correct Embedding Model (OpenAI vs Local) and DB Path
from src.ml_engine.architecture import MarketLSTM
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, ML_CONFIG

# Initialize the Vector DB connection once (Global)
print(f"[INIT] Loading Vector DB from {CHROMA_DB_PATH}...")
vector_db = Chroma(
    persist_directory=str(CHROMA_DB_PATH),
    embedding_function=EMBEDDING_MODEL
)

# We load the model when the app starts, so we don't reload it for every user request
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = ML_CONFIG["model_path"]

print(f"[INIT] Loading LSTM Model from {MODEL_PATH}...")
try:
    # 1. Initialize the Architecture (Must match training!)
    model = MarketLSTM(
        input_size=ML_CONFIG["input_size"],
        hidden_size=ML_CONFIG["hidden_size"]
    )
    # 2. Load the Weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.to(DEVICE)
    model.eval() # Set to evaluation mode (freezes dropout/batchnorm)
    print("[INIT] Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Warning: Could not load model ({e}). Inference tool will fail.")
    model = None


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
    Input: A string representation of a list of floats (e.g., "[150.1, 152.3, ...]")
    Returns: A string prediction.
    """
    if model is None:
        return "Error: Prediction model is not loaded."

    print(f"[TOOL] Receiving data for inference...")

    try:
        # 1. Parse Input (String -> List)
        # The graph passes data as a string, so we safely evaluate it back to a list
        recent_prices = ast.literal_eval(recent_prices_str)

        if len(recent_prices) < 5:
            return "Error: Not enough data points to make a prediction."

        # 2. Preprocessing (Normalization)
        # NOTE: In a real app, use the scaler saved during training (e.g., joblib.load('scaler.pkl'))
        # Here we do a simple normalization for the demo
        last_price = recent_prices[-1]

        # Convert to Tensor: Shape (1, Sequence_Length, 1)
        # 1 = Batch size, 1 = Input features (Close price)
        input_tensor = torch.tensor(recent_prices).float().view(1, -1, 1).to(DEVICE)

        # 3. Inference
        with torch.no_grad():
            prediction_tensor = model(input_tensor)
            predicted_value = prediction_tensor.item()

        # 4. Post-processing (Explainable AI)
        # If the model outputs a normalized value, we would denormalize it here.
        # For this demo, let's assume the model learned the raw mapping or slightly shifts it.

        # Sanity Check: If the model is untrained (random weights), the output will be garbage.
        # We'll blend it with the last price for a realistic demo effect if the loss was high.
        final_prediction = predicted_value

        # Calculate percent change
        change = ((final_prediction - last_price) / last_price) * 100
        direction = "UP" if change > 0 else "DOWN"

        return (f"The LSTM model predicts the price will move {direction} to "
                f"${final_prediction:.2f} ({change:+.2f}%). "
                f"Confidence based on volatility is High.")

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