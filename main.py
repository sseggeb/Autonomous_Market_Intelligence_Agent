import os
import sys
from dotenv import load_dotenv

# Import the compiled LangGraph app we built in src/agent_engine/graph.py
# Note: Ensure your directory structure is set so python can find 'src'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.agent_engine.graph import app

# --- CONFIGURATION ---
load_dotenv()  # Loads OPENAI_API_KEY from .env file


def print_banner():
    print("""
    =================================================
       AUTONOMOUS MARKET INTELLIGENCE AGENT (AMIA)
    =================================================
    Powered by: LangGraph, PyTorch, ChromaDB, OpenAI
    """)


def save_graph_image():
    """
    Optional: Visualizes the agent's logic flow as a PNG image.
    """
    try:
        graph_image = app.get_graph().draw_mermaid_png()
        with open("agent_workflow.png", "wb") as f:
            f.write(graph_image)
        print(" [INFO] Saved agent workflow diagram to 'agent_workflow.png'")
    except Exception as e:
        print(f" [WARN] Could not save graph image (requires graphviz): {e}")


def main():
    print_banner()

    # 1. (Optional) Save the graph visualization
    save_graph_image()

    while True:
        # 2. Get User Input
        user_ticker = input("\nEnter Stock Ticker (or 'q' to quit): ").strip().upper()

        if user_ticker == 'Q':
            print("Exiting...")
            break

        if not user_ticker:
            continue

        print(f"\n--- 🤖 AGENT ACTIVATED FOR {user_ticker} ---\n")

        # 3. Initialize State
        # We only need to provide the input 'ticker'. The nodes will fill in the rest.
        initial_state = {
            "ticker": user_ticker,
            "price_history": [],
            "news_documents": [],
            "forecast_value": 0.0,
            "forecast_confidence": 0.0,
            "final_report": ""
        }

        # 4. Invoke the Graph
        # We use .invoke() to run the graph from Start -> End
        try:
            result = app.invoke(initial_state)

            # 5. Display Results
            print("\n" + "=" * 40)
            print("       💰 FINAL STRATEGIC REPORT       ")
            print("=" * 40)
            print(result["final_report"])
            print("\n[Debug Data]")
            print(f"Forecast: ${result['forecast_value']} (Confidence: {result['forecast_confidence']:.2f})")

        except Exception as e:
            print(f"❌ An error occurred: {e}")


if __name__ == "__main__":
    main()