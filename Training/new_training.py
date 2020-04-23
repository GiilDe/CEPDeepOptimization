import torch
import torch.functional
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pandas as pd
import constants
import nets
import numpy as np
import logging

loss_function_type = torch.nn.MSELoss


def get_x(events):
    events_converted = []
    for i in range(constants.window_limit):
        event_type = events.iat[i, 0]
        events_converted = events_converted + event_type + list(events.iloc[i, 1:])
    x = Variable(torch.tensor(events_converted))
    return x


previous_leftover_x = None
x_start_count = 0
x_end_count = 0


def get_batch_x(X):
    global previous_leftover_x, x_start_count, x_end_count
    try:
        next_x = next(X)
        next_x.iloc[:, 0] = next_x.iloc[:, 0].apply(lambda x: constants.type_to_vec[x])
        batch = pd.concat([previous_leftover_x, next_x]) if previous_leftover_x is not None else next_x
        batch_x = []
        end_index = len(batch) - constants.window_limit
        x_start_count = batch.index[0]
        x_end_count = batch.index[end_index]
        for i in range(0, end_index):
            events = batch.iloc[i:i + constants.window_limit]
            batch_x.append(get_x(events))
        batch_x = torch.stack(batch_x)
        previous_leftover_x = batch.iloc[end_index:]

        return batch_x
    except StopIteration:
        previous_leftover_x = None
        return None


previous_leftover_y = None


def get_batch_y(Y):
    global previous_leftover_y, x_start_count, x_end_count
    try:
        next_y = next(Y)
        batch = pd.concat([previous_leftover_y, next_y]) if previous_leftover_y is not None else next_y
        end_index = len(batch) - constants.window_limit
        batch_y = torch.tensor(batch.iloc[:end_index].to_numpy(), dtype=torch.float)
        previous_leftover_y = batch.iloc[end_index:]
        y_start_count = batch.index[0]
        y_end_count = batch.index[end_index]
        if y_start_count != x_start_count or y_end_count != x_end_count:
            logging.error("x,y indices not compatible")
        return batch_y
    except StopIteration:
        previous_leftover_y = None
        return None


def get_batch(X, Y):
    return get_batch_x(X), get_batch_y(Y)


def net_train(epochs, batch_interval, batch_size, epoch_interval=1):
    net = nets.FCNetBestSubset()
    net.to(device=constants.device)
    criterion = loss_function_type()
    criterion.to(device=constants.device)
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)

    epochs_train_losses = []
    epochs_test_losses = []

    report_file = open(constants.training_report_path, 'w')
    for epoch in range(epochs):
        current_epoch_losses = []
        X = pd.read_csv(constants.train_file_path, chunksize=batch_size, header=None, usecols=[0, 1])
        Y = pd.read_csv(constants.bestsubset_train_labels, chunksize=batch_size, header=None, usecols=[5])
        global previous_leftover_x, previous_leftover_y
        previous_leftover_x = None
        previous_leftover_y = None
        net.train()
        processed_events = 0
        x, y = get_batch(X, Y)
        while x is not None and y is not None:
            if x.shape[0] == y.shape[0]:
                optimizer.zero_grad()
                y_hat = net.forward(x)
                loss = criterion(y_hat, y)
                loss.backward()
                optimizer.step()
                batch_loss = loss.data.item()
                current_epoch_losses.append(batch_loss)
                if processed_events % batch_interval == 0:
                    print("Epoch " + str(epoch) + ": Processed " + str(processed_events) + " out of "
                          + str(constants.train_size) + " sampled loss of " + str(batch_loss))
            else:
                logging.error("Shapes unaligned")
            processed_events += batch_size
            x, y = get_batch(X, Y)

        if epoch % epoch_interval == 0:
            epochs_train_losses.append(np.average(current_epoch_losses))
            test_loss = net_test(net, batch_size)
            epochs_test_losses.append(test_loss)
            train_losses = "Epochs train losses: " + str(epochs_train_losses)
            test_losses = "Epochs test losses: " + str(epochs_test_losses)
            print(train_losses)
            print(test_losses)
            report_file.write(train_losses + "\n")
            report_file.write(test_losses + "\n")
        torch.save(net, constants.model_path + "_" + str(epoch))


def net_test(net, batch_size):
    global previous_leftover_x, previous_leftover_y
    previous_leftover_x = None
    previous_leftover_y = None
    criterion = loss_function_type()
    criterion.to(device=constants.device)
    epoch_losses = []
    last_predictions = open(constants.last_predictions_file_path, 'w')
    last_predictions.write("y y_hat\n")
    net.eval()
    batch_idx = 0
    X = pd.read_csv(constants.test_file_path, chunksize=batch_size, header=None, usecols=[0, 1])
    Y = pd.read_csv(constants.bestsubset_test_labels, chunksize=batch_size, header=None, usecols=[5])
    i = 0
    with torch.no_grad():
        x, y = get_batch(X, Y)
        while x is not None and y is not None:
            if x.shape[0] == y.shape[0]:
                y_hat = net.forward(x)
                loss = criterion(y_hat, y)
                batch_loss = loss.data.item()
                epoch_losses.append(batch_loss)
                batch_idx += 1
                for spec_y, spec_y_hat in zip(y, y_hat):
                    last_predictions.write(str(spec_y.numpy()) + " " + str(spec_y_hat.numpy()) + "\n")
            else:
                logging.error("Shapes unaligned")
            if i % 1000 == 0:
                print(i)
            i += batch_size
            x, y = get_batch(X, Y)

    last_predictions.close()
    test_loss = np.average(epoch_losses)
    return test_loss


if __name__ == "__main__":
    net_train(100, 1000, 32)
