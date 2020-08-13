import torch
import torch.nn as nn
from Decoder.src.Main import Decoder


class LSTM(Decoder, nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()
        super(Decoder, self).__init__()

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=self.dropout, bias=self.bias)

        # Define the output layer
        self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=self.bias)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(self.device))  # for hidden and cell state (h_t, C_t)

    def forward(self, input_data, hidden):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, hidden = self.lstm(input_data, hidden)

        # Only take the output from the final time step
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.h2o(lstm_out[-1, :, :])

        # output = torch.sigmoid(self.h2o(hidden))
        return y_pred, hidden
