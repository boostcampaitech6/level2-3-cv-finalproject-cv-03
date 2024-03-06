import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def __str__(self):
        return "GRU"

    def forward(self, x):
        x, _ = self.gru(x)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=False,
        )

    def __str__(self):
        return "LSTM"

    def forward(self, x):
        x, _ = self.lstm(x)
        return x
