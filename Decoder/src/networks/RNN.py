import torch
import torch.nn as nn
from ..Main import Decoder


class RNN(Decoder, nn.Module):
    def __init__(self):
        super(RNN, self).__init__()

        # Define the LSTM layer
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity='tanh', dropout=self.dropout, bias=self.bias)

        # Define the output layer
        self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=self.bias)

    def forward(self, input_data, hidden):
        rnn_out, hidden = self.rnn(input_data, hidden)
        output = self.h2o(rnn_out[-1, :, :])

        return output, hidden

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)
