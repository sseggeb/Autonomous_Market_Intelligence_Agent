import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
import mlflow.pytorch
from torch.utils.data import DataLoader, TensorDataset
from src.config import ML_CONFIG, MLFLOW_TRACKING_URI


# 1. Define a dummy LSTM Architecture (for demonstration)
# In a real app, you would import this from src.ml_engine.architecture
class MarketLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super(MarketLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out


def train_model():
    # --- CONFIGURATION (Hyperparameters) ---
    hidden_size = ML_CONFIG["hidden_size"]
    lr = ML_CONFIG["learning_rate"]

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # We define these explicitly so MLflow can track them
    params = {
        "epochs": 10,
        "batch_size": 32,
        "optimizer": "Adam"
    }

    # --- DUMMY DATA GENERATION ---
    # Simulating market data (1000 samples, sequence length 10, 1 feature)
    X = torch.randn(1000, 10, 1)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=params["batch_size"])

    # --- MODEL SETUP ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MarketLSTM(hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Starting training on {device}...")

    # --- MLFLOW EXPERIMENT TRACKING ---
    # This is the "Productivity Multiplier"
    mlflow.set_experiment("Market_Price_Forecaster")

    with mlflow.start_run():
        # A. Log Hyperparameters (Input configs)
        mlflow.log_params(params)

        # B. Training Loop
        model.train()
        for epoch in range(params["epochs"]):
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

            # C. Log Metrics (Output performance)
            # This draws the line chart in the MLflow UI
            mlflow.log_metric("mse_loss", avg_loss, step=epoch)

            print(f"Epoch [{epoch + 1}/{params['epochs']}], Loss: {avg_loss:.4f}")

        # D. Log the Model Artifact
        # Saves the actual file so you can load it later in LangGraph
        input_example = torch.randn(1, 10, 1).numpy()
        mlflow.pytorch.log_model(
            model,
            "model",
            input_example=input_example
        )
        print("Training complete. Model logged to MLflow.")


if __name__ == "__main__":
    train_model()