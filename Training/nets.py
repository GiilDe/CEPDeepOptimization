import torch.nn as nn
from constants import constants
import torch
import numpy as np
from itertools import product


class NeuralCombOptLinearNet(nn.Module):
    def __init__(self, batch_size, use_cuda):
        super(NeuralCombOptLinearNet, self).__init__()
        self.actor_net = WindowToFiltersReinforce(batch_size, use_cuda)

    def forward(self, x):
        return self.actor_net.forward(x)


class WindowToFiltersReinforce(nn.Module):
    def __init__(self, batch_size, use_cuda):
        super(WindowToFiltersReinforce, self).__init__()
        self.probs_net = WindowToFiltersFC(batch_size)
        if use_cuda:
            self.probs_net = self.probs_net.cuda()
        self.batch_size = batch_size
        self.use_cuda = use_cuda

    def forward(self, events):
        # events: [batch_size, event_size, window_size]
        # events_probs: [batch_size, window_size]
        events_probs = self.probs_net(events)
        selections = torch.empty_like(events_probs).bool()
        actions = [[] for _ in range(self.batch_size)]
        for batch_i, event_i in product(range(self.batch_size), range(constants['window_size'])):
            choice = np.random.choice([0, 1], p=[1 - events_probs[batch_i, event_i].item(),
                                                 events_probs[batch_i, event_i].item()])
            selections[batch_i, event_i] = choice
            if choice == 1:
                actions[batch_i].append(event_i + 1)

        log_probs = torch.log(events_probs)
        masked_log_probs = log_probs * selections.int()
        log_probs = torch.sum(masked_log_probs, dim=1)

        for batch_actions in actions:
            batch_actions += [0]*(constants['window_size'] + 1 - len(batch_actions))

        actions_idxs = []
        for i in range(constants['window_size'] + 1):
            selection = torch.tensor([batch_actions[i] for batch_actions in actions])
            if self.use_cuda:
                selection = selection.cuda()
            actions_idxs.append(selection)

        return log_probs, None, actions_idxs


class WindowToFiltersFC(nn.Module):
    def __init__(self, batch_size):
        super(WindowToFiltersFC, self).__init__()
        fc1 = nn.Linear(constants['event_size']*constants['window_size'], 50)
        b_norm1 = nn.BatchNorm1d(50)
        relu1 = nn.ReLU()
        drop1 = nn.Dropout()
        fc2 = nn.Linear(50, 30)
        b_norm2 = nn.BatchNorm1d(30)
        relu2 = nn.ReLU()
        drop2 = nn.Dropout()
        fc3 = nn.Linear(30, 20)
        b_norm3 = nn.BatchNorm1d(20)
        relu3 = nn.ReLU()
        drop3 = nn.Dropout()
        fc4 = nn.Linear(20, 15)
        relu4 = nn.ReLU()
        drop4 = nn.Dropout()
        fc5 = nn.Linear(15, constants['window_size'])
        sigmoid = nn.Sigmoid()
        modules = [fc1, b_norm1, relu1, fc2, b_norm2, relu2, fc3, b_norm3, relu3, fc4, relu4, fc5, sigmoid]
        self.network = nn.Sequential(*modules)
        self.network.double()
        self.batch_size = batch_size

    def forward(self, events):
        # events dimensions: (batch_size, window_size, event_size)
        # output dimensions: (batch_size, window_size)
        x = events.reshape((self.batch_size, constants['window_size'] * constants['event_size']))
        return self.network(x)
