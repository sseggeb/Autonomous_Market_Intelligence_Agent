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
def create_sequences(data, target, seq_length):
    """
    data: (N, 4) array (Features)
    target: (N, 1) array (Price)
    """
    xs, ys = [],[]
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Market_Price_Forecaster_Multi")

    print(f"--- STARTING MULTIVARIATE TRAINING ---")

    # 1. Load Data
    csv_path = RAW_DATA_DIR / "stock_data.csv"
    df = pd.read_csv(csv_path)

    # Ensure correct column order
    features = df[['Close', 'Volume', 'SMA_50', 'RSI']].values
    target = df[['Close']].values

    # 2. Scale Data
    # Feature Scaler (for Inputs)
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = feature_scaler.fit_transform(features)

    # Target Scaler (for Output/Prediction)
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_target = target_scaler.fit_transform(target)

    # Save Scalers
    joblib.dump(feature_scaler, PROCESSED_DATA_DIR / "feature_scaler.pkl")
    joblib.dump(target_scaler, PROCESSED_DATA_DIR / "target_scaler.pkl")
    print("Scalers saved.")

    # 3. Create Sequences
    seq_length = ML_CONFIG["window_size"]
    X_numpy, y_numpy = create_sequences(scaled_features, scaled_target, seq_length)

    X = torch.tensor(X_numpy).float()
    y = torch.tensor(y_numpy).float()

    print(f"Training Shape: {X.shape}")  # (Samples, 60, 4)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=ML_CONFIG["batch_size"], shuffle=True)

    # 4. Model Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarketLSTM(
        input_size=ML_CONFIG["input_size"],  # Now 4
        hidden_size=ML_CONFIG["hidden_size"],
        output_size=ML_CONFIG["output_size"],
        num_layers=ML_CONFIG["num_layers"],
        dropout=ML_CONFIG["dropout"]
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=ML_CONFIG["learning_rate"])

    # 5. Train
    with mlflow.start_run():
        mlflow.log_params(ML_CONFIG)
        model.train()
        for epoch in range(ML_CONFIG["epochs"]):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}, Loss: {loss.item():.6f}")

        # Save
        input_example = X_numpy[:1].astype(np.float32)
        mlflow.pytorch.log_model(model, name="model", input_example=input_example)
        torch.save(model.state_dict(), ML_CONFIG["model_path"])
        print(f"✅ Model saved to {ML_CONFIG['model_path']}")

if __name__ == "__main__":
    train_model()