import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.config import ML_CONFIG, MLFLOW_TRACKING_URI, RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.ml_engine.architecture import MarketLSTM


# Helper function: sliding window
def create_sequence(data, seq_length):
    """
    Turns a list of prices into:
    X: [[10,11],[11,12],[12,13],[13,14],[14,15],[15,16]]
    y: [15,16,17]
    (assuming seq_length=2)
    :param data:
    :param seq_length:
    :return:
    """
    xs, ys = [],[]
    for i in range(len(data) - seq_length):
        x = data[i : i+seq_length]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Market_Price_Forecaster")

    print(f"--- STARTING TRAINING ---")
    # lOAD .csv
    csv_path = RAW_DATA_DIR / "stock_data.csv"

    if not csv_path.exists():
        print(f"Error: csv file not found at {csv_path}")
        print(" Please create a csv with a 'Close' column.")
        return

    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # Preprocessing
    # filter Close column
    data = df.filter(['Close']).values

    # Scale data to (o,1) for the LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Save the scaler for use in tools.py
    scaler_path = RAW_DATA_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Create sequences
    seq_length = ML_CONFIG["window_size"]
    X_numpy, y_numpy = create_sequence(scaled_data, seq_length)

    # Convert to Float32 for pytorch
    X = torch.tensor(X_numpy).float()
    y = torch.tensor(y_numpy).float()

    print(f"training data shape: {X.shape}")
    # expected: (Samples, window_size, 1)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=ML_CONFIG["batch_size"], shuffle=True)

    # 2. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the architecture imported from src.ml_engine.architecture
    model = MarketLSTM(
        input_size=ML_CONFIG["input_size"],
        hidden_size=ML_CONFIG["hidden_size"],
        output_size=ML_CONFIG["output_size"],
        num_layers=ML_CONFIG["num_layers"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ML_CONFIG["learning_rate"])

    print(f"Training on device: {device}")

    # --- 3. TRAINING LOOP WITH MLFLOW ---
    with mlflow.start_run():
        # Log params to compare experiments later
        mlflow.log_params(ML_CONFIG)

        model.train()
        for epoch in range(ML_CONFIG["epochs"]):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)

            # Log metrics to the MLflow dashboard
            mlflow.log_metric("mse_loss", avg_loss, step=epoch)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch [{epoch + 1}/{ML_CONFIG['epochs']}], Loss: {avg_loss:.4f}")

        # --- 4. SAVING ARTIFACTS ---
        # A. Save for MLflow (Metadata + Dependencies)
        # Create a sample input so MLflow knows the shape (Signature)
        input_example = X_numpy[:1].astype(np.float32)
        mlflow.pytorch.log_model(model, name="model", input_example=input_example)

        # B. Save for the Agent (The raw .pth file)
        # We save ONLY the state_dict (weights), not the whole class
        torch.save(model.state_dict(), ML_CONFIG["model_path"])
        print(f"✅ Model weights saved locally to: {ML_CONFIG['model_path']}")


if __name__ == "__main__":
    train_model()