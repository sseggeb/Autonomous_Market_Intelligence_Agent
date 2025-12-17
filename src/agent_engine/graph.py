from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END

# 1. Define the State
# This acts as the memory that gets passed between nodes
class AgentState(TypedDict):
    ticker: str
    price_history: List[float]
    news_documents: List[str]
    forecast_value: float
    forecast_confidence: float
    final_report: str

# 2. Define the Nodes (The actual work)

def fetch_market_data(state: AgentState):
    print(f"--- FETCHING DATA FOR {state['ticker']} ---")
    # Logic to pull API data (e.g., yfinance)
    state['price_history'] = api_call(...)
    return state


def run_pytorch_forecast(state: AgentState):
    print("--- RUNNING DEEP LEARNING MODEL ---")
    # Load your PyTorch model from src/ml_engine/
    model = load_model("models/price_predictor.pth")
    prediction = model(state['price_history'])

    # Simulate a result
    # state['forecast_value'] = 150.00
    # state['forecast_confidence'] = 0.85  # 85% confident
    return state


def consult_rag_system(state: AgentState):
    print("--- RETRIEVING CONTEXT VIA RAG ---")
    # Logic to query Vector DB (Chroma/Pinecone)
    docs = vector_store.similarity_search(state['ticker'])
    # state['news_documents'] = ["SEC Filing: Revenue up 10%...", "News: CEO steps down..."]
    return state


def generate_report(state: AgentState):
    print("--- GENERATING FINAL REPORT ---")
    # Use LangChain/OpenAI here
    prompt = f"The LSTM model predicts {state['forecast_value']}. Context: {state['news_documents']}."
    response = llm.invoke(prompt)
    state['final_report'] = "Based on the LSTM model and recent CEO news, the outlook is..."
    return state


# 3. Define the Edges (The Logic/Routing)
def confidence_router(state: AgentState):
    # If the ML model is unsure, force the agent to read the news first
    if state['forecast_confidence'] < 0.90:
        return "consult_rag"
    else:
        return "generate_report"


# 4. Build the Graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("fetch_data", fetch_market_data)
workflow.add_node("predict_price", run_pytorch_forecast)
workflow.add_node("research_news", consult_rag_system)
workflow.add_node("write_report", generate_report)

# Set entry point
workflow.set_entry_point("fetch_data")

# Add edges
workflow.add_edge("fetch_data", "predict_price")

# Conditional edge: After prediction, decide whether to Research or Write Report
workflow.add_conditional_edges(
    "predict_price",
    confidence_router,
    {
        "consult_rag": "research_news",
        "generate_report": "write_report"
    }
)

workflow.add_edge("research_news", "write_report")
workflow.add_edge("write_report", END)

# 5. Compile
app = workflow.compile()