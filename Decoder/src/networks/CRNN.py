import torch
import torch.nn as nn
from Decoder.src.Main import Decoder


class CRNN(Decoder, nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        super(Decoder, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv1d(self.input_size, 128, kernel_size=23, stride=1, padding=1, bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(p=self.dropout))
        self.layer2 = nn.Sequential(
            nn.Conv1d(128, 192, kernel_size=13, stride=1, padding=1, bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout())
        self.layer3 = nn.Sequential(
            nn.Conv1d(192, 192, kernel_size=11, stride=1, padding=1, bias=self.bias),
            nn.ReLU(),
            nn.BatchNorm1d(192),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout())
        self.layer4 = nn.Sequential(
            nn.Conv1d(192, 256, kernel_size=10, stride=1, padding=0, bias=self.bias),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(256, 128, bias=self.bias),  # 256, 128
            nn.ReLU())
        self.fc2 = nn.Linear(128, self.input_size, bias=self.bias)

        # Define the RNN layer
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, nonlinearity='tanh', dropout=self.dropout, bias=self.bias)

        # Define the output layer
        self.h2o = nn.Linear(self.hidden_size, self.output_size, bias=self.bias)

    def init_hidden(self):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

    def forward(self, input_data, hidden):
        # input size (batch x channel x window length)
        if len(input_data.size()) == 4:
            input_data = input_data[0, :, :, :]
        out = self.layer1(input_data)  # one layer worked best so far
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.fc1(out[:, :, 0])
        out = self.fc2(out)
        out, hidden = self.rnn(out.view(1, 1, -1), hidden)
        output = torch.sigmoid(self.h2o(out[-1, :, :]))

        return output, hidden