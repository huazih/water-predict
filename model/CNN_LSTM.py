import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, cnn_channels=16,h=231,w=271):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Conv2d(input_size, cnn_channels, kernel_size=(3, 3), padding=(1, 1))
        self.lstm = nn.LSTM(cnn_channels*h*w, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        x = x.view(-1, channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, seq_length, -1)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :]).squeeze()
        return x