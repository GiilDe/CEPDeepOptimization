import torch
import torch.nn as nn
import constants


class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear((constants.window_limit[0]*2-1)*5, 40)
        self.b_norm1 = nn.BatchNorm1d(40)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(40, 20)
        self.b_norm2 = nn.BatchNorm1d(20)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(20, 7)
        self.b_norm3 = nn.BatchNorm1d(7)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout()
        self.fc4 = nn.Linear(7, 1)
        self.modules = [self.fc1, self.b_norm1, self.relu1, self.fc2, self.b_norm2, self.relu2,
                        self.fc3, self.b_norm3, self.relu3, self.fc4]

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x


class FCNetForRNN(nn.Module):
    def __init__(self):
        super(FCNetForRNN, self).__init__()
        self.fc1 = nn.Linear(Constants.RNN_hidden_size * 2, 40)
        self.b_norm1 = nn.BatchNorm1d(40)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(40, 20)
        self.b_norm2 = nn.BatchNorm1d(20)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        self.fc3 = nn.Linear(20, 7)
        self.b_norm3 = nn.BatchNorm1d(7)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout()
        self.fc4 = nn.Linear(7, 1)
        self.modules = [self.fc1, self.b_norm1, self.relu1, self.fc2, self.b_norm2, self.relu2,
                        self.fc3, self.b_norm3, self.relu3, self.fc4]

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x


class BiDirectRNNSingleOutput(nn.Module):
    def __init__(self):
        super(BiDirectRNNSingleOutput, self).__init__()
        self.gru = torch.nn.GRU(input_size=6, hidden_size=Constants.RNN_hidden_size,
                                num_layers=1, batch_first=False, bidirectional=False)
        self.reversed_gru = torch.nn.GRU(input_size=6, hidden_size=Constants.RNN_hidden_size,
                                         num_layers=1, batch_first=False, bidirectional=False)
        self.compressing_net = FCNetForRNN()

    def forward(self, x):
        current_event_index = Constants.max_seq_RNN
        seq_before_current = x[:current_event_index + 1]
        seq_after_current = torch.cat((x[current_event_index + 1:], x[current_event_index].reshape(1, -1, 6)))
        y1, _ = self.gru(seq_before_current)
        y1 = y1[-1]
        y2, _ = self.reversed_gru(seq_after_current)
        y2 = y2[-1]
        combined = torch.cat((y1, y2), dim=-1)
        return self.compressing_net(combined)
