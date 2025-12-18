import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.config import ML_CONFIG, MLFLOW_TRACKING_URI
from src.ml_engine.architecture import MarketLSTM


def train_model():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("Market_Price_Forecaster")

    print(f"--- STARTING TRAINING ---")
    print(f"Config: {ML_CONFIG}")

    # --- 1. DATA GENERATION (Simulated) ---
    # In a real app, you would load a .csv here using pandas
    # We create dummy data: 1000 samples, Sequence length=window_size, Features=input_size
    num_samples = 1000
    seq_length = ML_CONFIG["window_size"]
    input_dim = ML_CONFIG["input_size"]

    # FORCE FLOAT32: This prevents the "Double vs Float"
    X_numpy = np.random.randn(num_samples, seq_length, input_dim).astype(np.float32)
    y_numpy = np.random.randn(num_samples, 1).astype(np.float32)

    # Convert to PyTorch Tensors
    X = torch.tensor(X_numpy)
    y = torch.tensor(y_numpy)

    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=ML_CONFIG["batch_size"], shuffle=True)

    # --- 2. MODEL INITIALIZATION ---
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
        input_example = X_numpy[:1]
        mlflow.pytorch.log_model(model, name="model", input_example=input_example)

        # B. Save for the Agent (The raw .pth file)
        # We save ONLY the state_dict (weights), not the whole class
        torch.save(model.state_dict(), ML_CONFIG["model_path"])
        print(f"✅ Model weights saved locally to: {ML_CONFIG['model_path']}")


if __name__ == "__main__":
    train_model()