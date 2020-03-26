import torch.nn as nn
import torch


def get_model(model_name, input_size, output_size, hidden_size, batch_size, num_layers, dropout, bias):
    # check for Cuda device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if model_name == 'RNN':
        class RNN(nn.Module):
            def __init__(self):
                super(RNN, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.batch_size = batch_size
                self.num_layers = num_layers

                # Define the LSTM layer
                self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', dropout=dropout, bias=bias)

                # Define the output layer
                self.h2o = nn.Linear(self.hidden_size, output_size, bias=bias)

            def forward(self, input_data, hidden):
                rnn_out, hidden = self.rnn(input_data, hidden)
                #output = torch.sigmoid(self.h2o(rnn_out[-1, :, :]))  # only use last sequence for prediction
                output = self.h2o(rnn_out[-1, :, :])

                return output, hidden

            def initHidden(self):
                return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

        return RNN()

    # elif model_name == 'RNN_custom':
    #     class RNN(nn.Module):
    #         def __init__(self):
    #             super(RNN, self).__init__()
    #
    #             self.hidden_size = hidden_size
    #
    #             self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
    #             self.h2o = nn.Linear(hidden_size, output_size)
    #
    #         def forward(self, input, hidden):
    #             combined = torch.cat((input, hidden), 1)
    #             hidden = torch.tanh(self.i2h(combined))
    #             output = torch.sigmoid(self.h2o(hidden))
    #
    #             return output, hidden
    #
    #         def initHidden(self):
    #             hidden = torch.randn(1, self.hidden_size).to(device)
    #             # hidden = torch.zeros(1, self.hidden_size).to(device)
    #             return hidden
    #
    #     return RNN()

    elif model_name == 'LSTM':
        class LSTM(nn.Module):

            def __init__(self):
                super(LSTM, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.batch_size = batch_size
                self.num_layers = num_layers

                # Define the LSTM layer
                self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, dropout=dropout, bias=bias)

                # Define the output layer
                self.h2o = nn.Linear(self.hidden_size, output_size, bias=bias)

            def initHidden(self):
                return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                        torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))  # for hidden and cell state (h_t, C_t)

            def forward(self, input_data, hidden):
                # Forward pass through LSTM layer
                # shape of lstm_out: [input_size, batch_size, hidden_dim]
                # shape of self.hidden: (a, b), where a and b both
                # have shape (num_layers, batch_size, hidden_dim).
                #lstm_out, hidden = self.lstm(input.view(len(input), self.batch_size, -1))
                lstm_out, hidden = self.lstm(input_data, hidden)

                # Only take the output from the final timetep
                # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
                #y_pred = self.h2o(lstm_out[-1].view(self.batch_size, -1))
                #y_pred = self.h2o(lstm_out)
                #y_pred = torch.sigmoid(self.h2o(lstm_out[-1, :, :]))  # only use last sequence for prediction
                y_pred = self.h2o(lstm_out[-1, :, :])

                #output = torch.sigmoid(self.h2o(hidden))
                return y_pred, hidden

        return LSTM()

    elif model_name == 'GRU':
        class GRU(nn.Module):

            def __init__(self):
                super(GRU, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.batch_size = batch_size
                self.num_layers = num_layers

                # Define the GRU layer
                self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers, dropout=dropout, bias=bias)

                # Define the output layer
                self.h2o = nn.Linear(self.hidden_size, output_size, bias=bias)

            def initHidden(self):
                return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

            def forward(self, input_data, hidden):
                gru_out, hidden = self.gru(input_data, hidden)
                #output = torch.sigmoid(self.h2o(gru_out[-1, :, :]))  # only use last sequence for prediction TODO: is it correct to use last sequence?
                output = self.h2o(gru_out[-1, :, :])

                return output, hidden

        return GRU()

    elif model_name == 'GBRT':  # Alazrai 2016
        class GBRT(nn.Module):

            def __init__(self):
                super(GBRT, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.batch_size = batch_size
                self.num_layers = num_layers

                self.i2o = nn.Linear(input_size, output_size, bias=bias)

            def initHidden(self):
                return torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device)

            def forward(self, input_data, hidden):
                output = self.i2o(input_data[-1, :, :])
                #output = torch.sigmoid(self.h2o(gru_out[-1, :, :]))  # only use last sequence for prediction
                #output = self.h2o(gru_out[-1, :, :])

                return output, hidden

        return GBRT()

    elif model_name == 'CNN':
        class CNN(nn.Module):
            def __init__(self):
                super(CNN, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.batch_size = batch_size
                self.num_layers = num_layers
                # TODO: decide whether to use ReLU or LeakyReLU
                self.layer1 = nn.Sequential(
                    nn.Conv1d(input_size, 128, kernel_size=23, stride=1, padding=1, bias=bias),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(128),
                    nn.MaxPool1d(kernel_size=8, stride=8),
                    nn.Dropout(p=dropout))
                self.layer2 = nn.Sequential(
                    nn.Conv1d(128, 192, kernel_size=13, stride=1, padding=1, bias=bias),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(192),
                    nn.MaxPool1d(kernel_size=4, stride=4),
                    nn.Dropout())
                self.layer3 = nn.Sequential(
                    nn.Conv1d(192, 192, kernel_size=11, stride=1, padding=1, bias=bias),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(192),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout())
                self.layer4 = nn.Sequential(
                    nn.Conv1d(192, 256, kernel_size=10, stride=1, padding=0, bias=bias),
                    nn.LeakyReLU())
                self.fc1 = nn.Sequential(
                    nn.Linear(256, 128, bias=bias),  # 256, 128
                    nn.LeakyReLU())
                self.fc2 = nn.Linear(128, output_size, bias=bias)

            def initHidden(self):
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
                output = self.fc2(out)

                return output, hidden

        return CNN()

    elif model_name == 'CRNN':
        class CRNN(nn.Module):
            def __init__(self):
                super(CRNN, self).__init__()
                self.input_size = input_size
                self.hidden_size = hidden_size
                self.batch_size = batch_size
                self.num_layers = num_layers
                self.layer1 = nn.Sequential(
                    nn.Conv1d(input_size, 128, kernel_size=23, stride=1, padding=1, bias=bias),
                    nn.ReLU(),
                    nn.BatchNorm1d(128),
                    nn.MaxPool1d(kernel_size=8, stride=8),
                    nn.Dropout(p=dropout))
                self.layer2 = nn.Sequential(
                    nn.Conv1d(128, 192, kernel_size=13, stride=1, padding=1, bias=bias),
                    nn.ReLU(),
                    nn.BatchNorm1d(192),
                    nn.MaxPool1d(kernel_size=4, stride=4),
                    nn.Dropout())
                self.layer3 = nn.Sequential(
                    nn.Conv1d(192, 192, kernel_size=11, stride=1, padding=1, bias=bias),
                    nn.ReLU(),
                    nn.BatchNorm1d(192),
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.Dropout())
                self.layer4 = nn.Sequential(
                    nn.Conv1d(192, 256, kernel_size=10, stride=1, padding=0, bias=bias),
                    nn.ReLU())
                self.fc1 = nn.Sequential(
                    nn.Linear(256, 128, bias=bias),  # 256, 128
                    nn.ReLU())
                self.fc2 = nn.Linear(128, input_size, bias=bias)

                # Define the LSTM layer
                self.rnn = nn.RNN(input_size, hidden_size, num_layers, nonlinearity='tanh', dropout=dropout, bias=bias)

                # Define the output layer
                self.h2o = nn.Linear(self.hidden_size, output_size, bias=bias)

            def initHidden(self):
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

        return CRNN()

    else:
        model = False
        print('Invalid model!')
        return model
