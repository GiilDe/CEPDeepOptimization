import torch
import torch.functional
import torch.optim as optim
import pandas as pd
import nets
import numpy as np
from constants import constants
import scipy.stats as stats

event_size = 5

allow_gpu = False
dev = "cuda" if allow_gpu and torch.cuda.is_available() else "cpu"

device = torch.device(dev)

loss_function_type = torch.nn.MSELoss
batch_size = 32
model_path = "training_data/second_net"
training_report_path = "training_data/report.txt"
last_predictions_file_path = "training_data/last predictions.txt"


def get_batch_matches(M, is_train=True):
    try:
        batch = next(M)
        return batch
    except (StopIteration, RuntimeError):
        return None


def get_batch_events(X, is_train=True):
    def get_dummies(x):
        padding = pd.DataFrame([['A', -1], ['B', -1], ['C', -1], ['D', -1]])
        x = padding.append(x, ignore_index=True)
        x = pd.get_dummies(x)
        x = x.drop(axis=0, labels=[0, 1, 2, 3])
        return x

    try:
        batch = next(X)
        batch = get_dummies(batch)
        batch = torch.tensor(batch.to_numpy(), dtype=torch.float, requires_grad=is_train) \
            .reshape((batch_size, constants['window_size'], event_size))
        return batch
    except (StopIteration, RuntimeError):
        return None


def get_batch_y(Y, is_train=True):
    try:
        batch = next(Y).to_numpy()
        batch = torch.tensor(batch, dtype=torch.float, requires_grad=is_train).reshape((batch_size, 1))
        return batch
    except (StopIteration, RuntimeError):
        return None


def initialize_data_y(is_train=True):
    Y = pd.read_csv(constants['train_scores_file_path'] if is_train else constants['test_scores_file_path'],
                    chunksize=batch_size, header=None)
    return Y


def window_to_score_initializer(is_train):
    def initialize_data_x(is_train=True):
        X = pd.read_csv(constants['train_stream_path_repeat'] if is_train else constants['test_stream_path_repeat'],
                        chunksize=batch_size * constants['window_size'], header=None, usecols=[0, 1])
        return X

    return initialize_data_x(is_train), initialize_data_y(is_train), get_batch_events, get_batch_y


def filtered_window_to_score_initializer(is_train):
    def get_batch_f(F, is_train=True):
        try:
            batch = next(F).to_numpy()
            batch = torch.tensor(batch, dtype=torch.float, requires_grad=is_train) \
                .reshape((batch_size, constants['window_size'], 1))
            return batch
        except (StopIteration, RuntimeError):
            return None

    def get_batch_x(X_F, is_train=True):
        X = X_F[0]
        F = X_F[1]
        events = get_batch_events(X, is_train)
        filterings = get_batch_f(F, is_train)
        return (events, filterings) if events is not None else None

    def initialize_data_x(is_train):
        X = pd.read_csv(constants['train_stream_path_repeat'] if is_train else constants['test_stream_path_repeat'],
                        chunksize=batch_size * constants['window_size'], header=None, usecols=[0, 1])
        F = pd.read_csv(constants['train_filters_file_path'] if is_train else constants['test_filters_file_path'],
                        chunksize=batch_size * constants['window_size'], header=None)
        return X, F

    return initialize_data_x(is_train), initialize_data_y(is_train), get_batch_x, get_batch_y


def net_train(epochs, batch_interval, net, initializer, epoch_interval=1, load_path=None):
    loss_function = loss_function_type()
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9)
    epoch = 0

    epochs_train_losses = []
    epochs_test_losses = []
    epochs_test_pearson_corrs = []

    if load_path is not None:
        checkpoint = torch.load(load_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        loss_function = checkpoint['loss_function']
        epoch = checkpoint['epoch']
        epochs_train_losses = checkpoint['epochs_train_losses']
        epochs_test_losses = checkpoint['epochs_test_losses']
        epochs_test_pearson_corrs = checkpoint['epochs_test_pearson_corrs']

    loss_function.to(device=device)
    net.to(device=device)
    report_file = open(training_report_path, 'w')
    while epoch < epochs:
        current_epoch_losses = []
        X, Y, get_batch_x, get_batch_y = initializer(True)
        net.train()
        processed_events = 0
        assert net.network[3].training is True
        x, y = get_batch_x(X), get_batch_y(Y)
        while x is not None and y is not None:
            optimizer.zero_grad()
            y_hat = net.forward(x)
            loss = loss_function(y_hat, y)
            loss.backward()
            optimizer.step()
            batch_loss = loss.data.item()
            current_epoch_losses.append(batch_loss)
            if processed_events % batch_interval == 0:
                print("Epoch " + str(epoch) + ": Processed " + str(processed_events) + " out of "
                      + str(constants['train_size']) + " sampled loss_function of " + str(batch_loss))
            processed_events += batch_size
            x, y = get_batch_x(X), get_batch_y(Y)

        if epoch % epoch_interval == 0:
            epochs_train_losses.append(np.average(current_epoch_losses))
            test_loss, pearson_corr = net_test(net, initializer, loss_function)
            epochs_test_losses.append(test_loss)
            epochs_test_pearson_corrs.append(pearson_corr)
            train_losses = "Epochs train losses: " + str(epochs_train_losses)
            test_losses = "Epochs test losses: " + str(epochs_test_losses)
            pearson_corr = "Epochs test pearson corrs: " + str(epochs_test_pearson_corrs)
            print(train_losses)
            print(test_losses)
            print(pearson_corr)
            report_file.write(train_losses + "\n")
            report_file.write(test_losses + "\n")
            report_file.write(pearson_corr + "\n")

        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_function': loss_function,
            'epochs_train_losses': epochs_train_losses,
            'epochs_test_losses': epochs_test_losses,
            'epochs_test_pearson_corrs': epochs_test_pearson_corrs
        }, model_path + "_" + str(epoch))
        epoch += 1


def net_test(net, initializer, loss_function):
    epoch_losses = []
    last_predictions = open(last_predictions_file_path, 'w')
    last_predictions.write("y y_hat\n")
    net.eval()
    assert net.network[3].training is False
    batch_idx = 0
    X, Y, get_batch_x, get_batch_y = initializer(False)
    i = 0
    all_y = []
    all_y_hat = []
    with torch.no_grad():
        x, y = get_batch_x(X, False), get_batch_y(Y, False)
        while x is not None and y is not None:
            all_y += [row.data.item() for row in y]
            y_hat = net.forward(x)
            all_y_hat += [row.data.item() for row in y_hat]
            loss = loss_function(y_hat, y)
            batch_loss = loss.data.item()
            epoch_losses.append(batch_loss)
            batch_idx += 1
            for spec_y, spec_y_hat in zip(y, y_hat):
                last_predictions.write(str(spec_y.numpy()) + " " + str(spec_y_hat.numpy()) + "\n")
            if i % 1000 == 0:
                print(i)
            i += batch_size
            x, y = get_batch_x(X, False), get_batch_y(Y, False)

    pearson_corr = stats.pearsonr(all_y, all_y_hat)[0]
    last_predictions.close()
    test_loss = np.average(epoch_losses)
    return test_loss, pearson_corr


if __name__ == "__main__":
    net_train(100, 1000, nets.FilteredWindowToScoreFC(keep_filterings=False, automatic_filtering=False), filtered_window_to_score_initializer)
