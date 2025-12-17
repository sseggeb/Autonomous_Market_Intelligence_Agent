import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Load Environment Variables
# This automatically looks for a .env file in the root directory
load_dotenv()

# 2. Project Paths
# using pathlib makes this OS-agnostic (works on Windows/Mac/Linux)
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure crucial directories exist immediately
for path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOGS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# 3. Vector Database Settings (RAG)
CHROMA_DB_PATH = PROCESSED_DATA_DIR / "chroma_db"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 4. Machine Learning Hyperparameters
# Centralizing these makes it easy to run experiments
ML_CONFIG = {
    "input_size": 1,        # Number of features (e.g., Close price)
    "hidden_size": 64,      # LSTM hidden neurons
    "output_size": 1,       # Prediction horizon
    "num_layers": 2,        # Stacked LSTM layers
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 1,
    "window_size": 10,      # Lookback period (how many days to look behind)
    "model_path": MODEL_DIR / "price_predictor.pth"
}

# 5. API Keys & Secrets
# NEVER hardcode keys. Always fetch from env.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("⚠️  OPENAI_API_KEY not found in environment variables!")

CHROMA_API_KEY = os.getenv("CHROMA_API_KEY")

if not CHROMA_API_KEY:
    print("⚠️  Warning: CHROMA_API_KEY not found. (Ignore this if running locally without auth)")

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")

# 6. Global Constants
TICKER_SYMBOL = "AAPL" # Default ticker for testing