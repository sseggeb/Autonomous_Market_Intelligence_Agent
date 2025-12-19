from typing import TypedDict, List
import yfinance as yf
from langgraph.graph import StateGraph, END

# Import the tools/logic we built earlier
# use the 'predict_future_price' tool logic inside the graph
from src.agent_engine.tools import retrieve_financial_context, predict_future_price
from src.config import ML_CONFIG


# DEFINE STATE
class AgentState(TypedDict):
    ticker: str
    price_history: List[float]  # The last N days of closing prices
    news_documents: str  # Summarized news context
    forecast_report: str  # The ML prediction text
    final_report: str  # The final output to the user

# DEFINE NODES

def fetch_market_data(state: AgentState):
    ticker = state["ticker"]
    print(f"--- [GRAPH] FETCHING & PROCESSING DATA FOR {ticker} ---")

    try:
        stock = yf.Ticker(ticker)
        # Fetch 1 year to be safe for 50-day SMA + 60-day window
        hist = stock.history(period="1y")

        if hist.empty:
            return {"price_history": []}

        # --- LIVE FEATURE ENGINEERING ---
        # 1. Calculate SMA 50
        hist['SMA_50'] = hist['Close'].rolling(window=50).mean()

        # 2. Calculate RSI 14
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        hist['RSI'] = 100 - (100 / (1 + rs))

        # 3. Drop NaNs
        hist.dropna(inplace=True)

        # 4. Check length
        window = ML_CONFIG.get("window_size", 60)
        if len(hist) < window:
            print("Not enough data after calculating indicators.")
            return {"price_history": []}

        # 5. Extract the 4 columns
        # a list of lists: [[Close, Vol, SMA, RSI], [Close, Vol, ...], ...]
        final_data = hist[['Close', 'Volume', 'SMA_50', 'RSI']].tail(window)

        # Convert to a format we can pass to the tool
        # .values.tolist() creates [[...], [...]]
        history_list = final_data.values.tolist()

        return {"price_history": history_list}

    except Exception as e:
        print(f"Error: {e}")
        return {"price_history": []}

def run_prediction_node(state: AgentState):
    """
    Node 2: Prediction.
    Runs the ML model (or simulation) on the fetched data.
    """
    print("--- [GRAPH] RUNNING PREDICTION MODEL ---")
    history = state["price_history"]

    if not history:
        return {"forecast_report": "Error: Insufficient data for prediction."}

    # We reuse the logic from our tools.py
    # In a real app, you'd pass the list[float] directly to the model.
    # Here, we pass it to our tool wrapper which returns a string string.
    result_text = predict_future_price.invoke(str(history))

    return {"forecast_report": result_text}

def research_news_node(state: AgentState):
    """
    Node 3: Contextualization.
    If the user asks for analysis, we fetch news/RAG context.
    """
    print("--- [GRAPH] SEARCHING NEWS (RAG) ---")
    ticker = state["ticker"]

    # Search for the ticker name in the vector DB
    # You might want to expand this query (e.g., "Apple revenue risks")
    context = retrieve_financial_context.invoke(ticker)

    return {"news_documents": context}

def generate_report_node(state: AgentState):
    """
    Node 4: Synthesis.
    Combines ML stats + RAG news into the final answer.
    """
    print("--- [GRAPH] GENERATING FINAL REPORT ---")

    ticker = state["ticker"]
    prediction = state["forecast_report"]
    news = state["news_documents"]

    # Simple template for the final output
    report = f"""
    ANALYSIS REPORT FOR: {ticker}
    --------------------------------------
    1. MARKET FORECAST (Quantitative):
    {prediction}

    2. RELEVANT NEWS/CONTEXT (Qualitative):
    {news[:500]}... (truncated for brevity)
    --------------------------------------
    """
    return {"final_report": report}

# 3. BUILD GRAPH

workflow = StateGraph(AgentState)

# Add Nodes
workflow.add_node("fetch_data", fetch_market_data)
workflow.add_node("predict", run_prediction_node)
workflow.add_node("research", research_news_node)
workflow.add_node("report", generate_report_node)

# Set Entry Point
workflow.set_entry_point("fetch_data")

# Define Edges (The Flow)
# 1. Start -> Fetch Data
# 2. Fetch Data -> Predict Price
workflow.add_edge("fetch_data", "predict")

# 3. Predict Price -> Research News
# (In a complex agent, you could add a conditional edge here:
#  "If confidence is low, do more research")
workflow.add_edge("predict", "research")

# 4. Research -> Report -> End
workflow.add_edge("research", "report")
workflow.add_edge("report", END)

# Compile
app = workflow.compile()