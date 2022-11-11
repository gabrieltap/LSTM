import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.device = device

        self.lstm = nn.LSTM(input_size,
                            self.hidden_size,
                            num_layers=2,
                            dropout=dropout)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, input_, hidden):
        out, hidden = self.lstm(input_, hidden)
        out = self.linear(out)

        return out, hidden

    def init_hidden(self):
        # We need two hidden layers because of our two layered lstm!
        # Your model should be able to use this implementation of initHidden()
        return (torch.zeros(2, self.hidden_size).to(self.device),
                torch.zeros(2, self.hidden_size).to(self.device))
