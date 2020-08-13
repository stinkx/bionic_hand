import torch
import torch.nn as nn
from Decoder.src.Main import Decoder


class GRU(Decoder, nn.Module):

    def __init__(self):
        super(GRU, self).__init__()
        super(Decoder, self).__init__()

        # Define the GRU layer
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, bias=self.bias)

        # Define the output layer
        self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=self.bias)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device)

    def forward(self, input_data, hidden):
        gru_out, hidden = self.gru(input_data, hidden)
        output = self.h2o(gru_out[-1, :, :])

        return output, hidden
