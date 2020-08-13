import torch
import torch.nn as nn
from Decoder.src.Main import Decoder


class Custom(Decoder, nn.Module):

    def __init__(self):
        super(Custom, self).__init__()
        super(Decoder, self).__init__()

        self.i2h = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.h2o = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input_data, hidden):
        combined = torch.cat((input_data, hidden), 1)
        hidden = torch.tanh(self.i2h(combined))
        output = torch.sigmoid(self.h2o(hidden))

        return output, hidden

    def init_hidden(self):
        hidden = torch.randn(1, self.hidden_size).to(self.device)
        return hidden
