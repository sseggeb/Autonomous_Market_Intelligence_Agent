import torch
import torch.nn as nn


class MarketLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=1, dropout=0.2):
        super(MarketLSTM, self).__init__()

        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout= dropout if num_layers > 1 else 0.0,
        )

        # Fully Connected Layer (Output)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # Forward propagate LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x)

        # Decode the hidden state of the last time step
        # out[:, -1, :] gets the last output for every item in the batch
        out = self.fc(out[:, -1, :])
        return out