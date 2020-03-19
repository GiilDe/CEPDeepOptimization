import processing_utilities
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import nets, constants
import numpy as np
import scipy.stats as stats
import sklearn.metrics as metrics
import torch.functional
import pickle
import processing_utilities

loss_function_type = torch.nn.MSELoss


def get_batch(sequences, labels, batch_size, normalize):
    batch_x = []
    batch_y = []
    finished = False
    i = 0
    while i < batch_size:
        x_line, y_line = sequences.readline(), labels.readline()
        if x_line in {"$", "$\n", " ", "\n", ""}:
            finished = True
            break
        x, y = getXY(x_line, y_line, normalize)
        batch_x.append(x)
        batch_y.append(y)
        i += 1

    batch_x = torch.stack(batch_x) if batch_x != [] else "-1"
    batch_y = torch.stack(batch_y) if batch_y != [] else "-1"
    return batch_x, batch_y, finished


def getXY(x_line, y_line, normalize=False):
    x_temp = x_line.split(";")[:-1]
    events = [processing_utilities.convert_event(processing_utilities.get_event_from_str(event, *constants.event_format))
              for event in x_temp]
    x = Variable(torch.stack(events), requires_grad=True).reshape(-1)
    y = Variable(torch.tensor([float(y_line.split("\n")[0])]))
    if normalize:
        y = torch.tanh(y)
    return x, y


def net_train(epochs, batch_interval, batch_size, normalize=False, epoch_interval=1, report_path="data/report.txt"):
    net = nets.FCNet()
    criterion = loss_function_type()
    optimizer = optim.SGD(net.parameters(), lr=0.00001, momentum=0.9)
    epochs_train_losses = []
    epochs_test_losses = []
    epochs_test_pearson_corrs = []
    epochs_test_spearman_corrs = []
    epochs_test_mutual_infos = []
    seqs_file = constants.train_file_path_sequences
    labels_file = constants.train_file_path_labels
    report_file = open(report_path, 'w')
    for epoch in range(epochs):
        current_epoch_losses = []
        sequences = open(seqs_file, 'r')
        labels = open(labels_file, 'r')
        net.train()
        batch_idx = 0
        for _ in range(constants.window_limit[0]):
            sequences.readline()
            labels.readline()
        x, y, finished = get_batch(sequences, labels, batch_size, normalize)
        while not finished and type(x) != str:
            optimizer.zero_grad()
            y_hat = net.forward(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            batch_loss = loss.data.item()
            current_epoch_losses.append(batch_loss)
            if batch_idx % batch_interval == 0:
                print('Train Epoch: {} \tBatch Loss: {:.6f}'.format(
                    epoch, batch_loss))
            batch_idx += 1
            x, y, finished = get_batch(sequences, labels, batch_size, normalize)
        sequences.close()
        labels.close()
        if epoch % epoch_interval == 0:
            epochs_train_losses.append(np.average(current_epoch_losses))
            test_loss, pearson_corr, mutual_info, spearman_corr = net_test(net, batch_size, normalize)
            epochs_test_losses.append(test_loss)
            epochs_test_pearson_corrs.append(pearson_corr)
            epochs_test_mutual_infos.append(mutual_info)
            epochs_test_spearman_corrs.append(spearman_corr)
            train_losses = "Epochs train losses: " + str(epochs_train_losses)
            test_losses = "Epochs test losses: " + str(epochs_test_losses)
            pearson_corr = "Epochs test pearson corrs: " + str(epochs_test_pearson_corrs)
            print(train_losses)
            print(test_losses)
            print(pearson_corr)
            report_file.write(train_losses + "\n")
            report_file.write(test_losses + "\n")
            report_file.write(pearson_corr + "\n")
    torch.save(net, constants.model_path)


def dump_lengthes():
    y_to_quantity = dict()
    labels = open(constants.test_file_path_labels, 'r')
    for line in labels:
        _, y = getXY("1;", line)
        y = y.item()
        if y not in y_to_quantity:
            y_to_quantity[y] = 1
        else:
            y_to_quantity[y] += 1

    m = sorted(y_to_quantity.values())[6]
    all_quantity = 0
    for value in y_to_quantity.values():
        all_quantity += value
    y_to_quantity['all_quantity'] = all_quantity
    y_to_quantity['min_quantity'] = m

    pickle.dump(y_to_quantity, open('y_to_quantity.obj', 'wb'))


def net_test(net, batch_size, normalize=False, loss_per_label=False):
    criterion = nn.MSELoss()
    epoch_losses = []
    sequences = open(constants.test_file_path_sequences, 'r')
    labels = open(constants.test_file_path_labels, 'r')
    for _ in range(constants.window_limit[0]):
        sequences.readline()
        labels.readline()
    last_predictions = open(constants.last_predictions_file_path, 'w')
    last_predictions.write("y y_hat\n")
    net.eval()
    batch_idx = 0
    all_y = []
    all_y_hat = []
    x, y, finished = get_batch(sequences, labels, batch_size, normalize)
    while not finished and type(x) != str:
        all_y += [row.data.item() for row in y]
        y_hat = net.forward(x)
        all_y_hat += [row.data.item() for row in y_hat]
        loss = criterion(y_hat, y)
        batch_loss = loss.data.item()
        epoch_losses.append(batch_loss)
        batch_idx += 1
        for spec_y, spec_y_hat in zip(y, y_hat):
            last_predictions.write(str(spec_y.data.item()) + " " + str(spec_y_hat.data.item()) + "\n")
        x, y, finished = get_batch(sequences, labels, batch_size, normalize)

    pearson_corr = stats.pearsonr(all_y, all_y_hat)[0]
    spearman_corr = stats.spearmanr(all_y, all_y_hat)[0]
    mutual_info = metrics.mutual_info_score(all_y, all_y_hat)
    if loss_per_label:
        loss_per_label = dict()
        mse_loss = torch.nn.MSELoss()
        for y, y_hat in zip(all_y, all_y_hat):
            if y not in loss_per_label:
                loss_per_label[y] = []
            variable1 = Variable(torch.tensor(y), requires_grad=False)
            variable2 = Variable(torch.tensor(y_hat), requires_grad=False)
            loss = mse_loss(variable1, variable2).data.item()
            loss_per_label[y].append(loss)
        last_predictions.write("Label    Avg    Std\n")
        loss_per_label_sorted = dict(sorted(loss_per_label.items()))
        for y in loss_per_label_sorted:
            y_errors = loss_per_label_sorted[y]
            avg = np.average(y_errors)
            std = np.std(y_errors)
            last_predictions.write(str(y) + "    " + str(avg) + "    " + str(std) + "\n")
    sequences.close()
    labels.close()
    test_loss = np.average(epoch_losses)
    return test_loss, pearson_corr, mutual_info, spearman_corr


if __name__ == "__main__":
    net_train(1, 1000, 32, report_path="data/report.txt")


