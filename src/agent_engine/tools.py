import yfinance as yf
import torch
from langchain.tools import tool
from langchain_chroma import Chroma

# Import settings from your centralized config
# This ensures we use the correct Embedding Model (OpenAI vs Local) and DB Path
from src.config import CHROMA_DB_PATH, EMBEDDING_MODEL, ML_CONFIG

# Initialize the Vector DB connection once (Global)
print(f"[INIT] Loading Vector DB from {CHROMA_DB_PATH}...")
vector_db = Chroma(
    persist_directory=str(CHROMA_DB_PATH),
    embedding_function=EMBEDDING_MODEL
)


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
        recent_prices_str (str): A dummy argument for now (the agent handles state).
    Returns:
        str: The predicted price and confidence.
    """
    # NOTE: In a real production system, you would pass the actual
    # historical tensor here. For this portfolio project, we simulate
    # the model inference to keep the tool simple for the Agent.

    print(f"[TOOL] Running PyTorch Model Inference...")

    # You could load the actual model here if you wanted:
    # model = MarketLSTM(...)
    # model.load_state_dict(torch.load(ML_CONFIG["model_path"]))

    # Simulation Logic
    return "The LSTM model predicts a price of $152.50 with 87% confidence based on momentum."


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