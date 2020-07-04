import torch
import torch.functional
import torch.optim as optim
from constants import constants
import dataset
from matches_finder_pointer_net import PointerNetwork

tanh_exploration = 10
use_tanh = True
hidden_dim = 64


train_size = int(constants['train_size'] / constants['window_size'])
test_size = int(constants['test_size'] / constants['window_size'])

checkpoint_path = "training_data/checkpoint"

batch_interval = 300


beta = 0.9
decay_step = 5000
decay_rate = 0.96
max_grad_norm = 2.0

loss_func = torch.nn.CrossEntropyLoss()


def get_pointer_net_optimizer(net):
    lr = 0.0001
    return optim.Adam(net.parameters(), lr=lr), lr


def print_interval(actions_idxs, m, loss, epoch, processed_events, size, net_time=None, cep_time=None,
                   first_wind_net_time=None, first_wind_cep_time=None, is_validation=False):
    first_actions_idxs = torch.stack(actions_idxs).transpose(0, 1)[0, :]
    first_m = m[0, :]
    if processed_events % batch_interval == 0:
        print("------------")
        if not is_validation:
            print("Epoch " + str(epoch) + ": Processed " + str(processed_events) + " out of " +
                  str(size))
        else:
            print("~Validation~ epoch " + str(epoch) + ": Processed " + str(processed_events) + " out of " +
                  str(size))
        print("average loss: " + str(loss))

        print("sampled matches from net: " + str(first_actions_idxs))
        print("real matches are: " + str(first_m))

        if is_validation:

            time_ratio = str(net_time / cep_time)
            net_time_ = "batched net to cep time ratio: " + time_ratio
            print(net_time_)

            time_ratio = str((first_wind_net_time) / first_wind_cep_time)

            net_time_ = "non batched net to cep time ratio: " + time_ratio
            print(net_time_)


def net_train(epochs, net, load_path=None):
    global test_losses

    def save_checkpoint(_prev_i=None):
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses': epochs_losses,
            'test_losses': test_losses,
            'prev_i': _prev_i,
            'epoch_avg_loss': epoch_average_loss
        }, checkpoint_path + "_" + str(net) + "_" + str(epoch) + (("_" + str(_prev_i)) if _prev_i is not None else ""))

    optimizer, learning_rate = get_pointer_net_optimizer(net)

    epoch = 0
    epochs_losses = []

    prev_i = None
    if load_path is not None:
        checkpoint = torch.load(load_path, map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device=dataset.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        epochs_losses = checkpoint['losses']
        test_losses = checkpoint['test_losses']
        prev_i = checkpoint['prev_i']
        epoch_average_loss = checkpoint['epoch_avg_loss']

    details = "starting training with:\n"
    details += "lr = " + str(learning_rate) + "\n"
    print(details)

    net.to(device=dataset.device)
    while epoch < epochs:
        X, M = dataset.initialize_data_x(True), dataset.initialize_data_matches(True)

        net.train()
        i = -1
        processed_events = 0
        if prev_i is None:
            epoch_average_loss = 0
            x, m = dataset.get_batch_events(X), dataset.get_batch_matches(M)
        else:
            while i != prev_i:
                x, m = dataset.get_batch_events(X), dataset.get_batch_matches(M)
                i += 1
                processed_events += dataset.batch_size
            prev_i = None
        while x is not None and m is not None:
            if i != 0 and i % 1500 == 0:
                save_checkpoint(i)

            batches_lengths = torch.sum(m.bool().int(), dim=1)
            probs, actions_idxs = net.forward(x, batches_lengths)

            probs = probs.sort()[0]
            net_matches_len = int(probs.size(0)/dataset.batch_size)
            if net_matches_len - m.size(1) > 0:
                zeros = torch.zeros((dataset.batch_size, net_matches_len - m.size(1))).long()
                m = torch.cat((m, zeros), dim=1)

            loss = loss_func(probs, m.view(-1))

            optimizer.zero_grad()
            epoch_average_loss += loss.item()
            loss.backward()
            optimizer.step()

            print_interval(actions_idxs, m, loss.item(), epoch, processed_events, train_size)

            processed_events += dataset.batch_size
            x, m = dataset.get_batch_events(X), dataset.get_batch_matches(M)
            i += 1

        epoch_average_loss = epoch_average_loss / i
        epochs_losses.append(epoch_average_loss)

        print("train rewards: " + str(epochs_losses))
        net_test(net, epoch)
        print("test rewards: " + str(test_losses))

        epoch += 1

        save_checkpoint()


test_losses = []


def net_test(net, epoch):
    global test_losses
    epoch_average_reward = 0
    X, M, E = dataset.initialize_data_x(False), dataset.initialize_data_matches(False), dataset.initialize_data_x(False)
    net.eval()
    processed_events = 0
    print("\n~validation~\n")
    x, m, e = dataset.get_batch_events(X), dataset.get_batch_matches(M), \
              dataset.get_batch_events_as_events(E)
    i = -1
    while x is not None and m is not None:
        events_probs, net_time = net.forward(x)
        chosen_events = net.sample_events(events_probs)
        rewards, found_matches_portions, found_matches_portion, denominator, b_whole_time, \
        b_filtered_time, f_whole_time, f_filtered_time = dataset.get_rewards(m, chosen_events, e, is_train=False)
        epoch_average_reward += rewards.mean().item()
        _, first_window_net_time = net.forward(x[0, :, :].unsqueeze(0), 1)
        print_interval(chosen_events, epoch, found_matches_portion, found_matches_portions, processed_events,
                       rewards, test_size, denominator, is_validation=True, net_time=net_time,
                       cep_whole_time=b_whole_time, cep_filtered_time=b_filtered_time,
                       first_wind_net_time=first_window_net_time, first_wind_cep_time=f_whole_time,
                       first_wind_filt_cep_time=f_filtered_time)
        processed_events += dataset.batch_size
        x, m, e = dataset.get_batch_events(X), dataset.get_batch_matches(M), \
                  dataset.get_batch_events_as_events(E)
        i += 1

    epoch_average_reward = epoch_average_reward / i
    test_losses.append(epoch_average_reward)


if __name__ == "__main__":
    pointer_net = PointerNetwork(
        embedding_dim=constants['event_size'],
        hidden_dim=hidden_dim,
        max_decoding_len=constants['window_size'] + 1,
        n_glimpses=2,
        tanh_exploration=tanh_exploration,
        use_tanh=use_tanh,
        use_cuda=True if dataset.dev == "cuda" else False,
        encoder_bi_directional=True,
        encoder_num_layers=2,
        padding_value=-1
    )
    net_train(100, pointer_net)
