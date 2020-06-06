import torch
import torch.functional
import torch.optim as optim
from constants import constants
from torch.optim import lr_scheduler
import dataset
import numpy as np
from neural_combinatorial_rl import NeuralCombOptNet, CriticNetwork
from nets import *

use_time_ratio = True if dataset.time_calc_index in {0, 1} else False
tanh_exploration = 10
use_tanh = True
hidden_dim = 64

steps = 1

train_size = int(constants['train_size'] * steps / constants['window_size'])
test_size = int(constants['test_size'] * steps / constants['window_size'])

model_path = "training_data/net"
batch_interval = 1000

batch_interval = int(batch_interval / dataset.batch_size) * dataset.batch_size

beta = 0.9
decay_step = 5000
decay_rate = 0.96
max_grad_norm = 2.0

critic_mse = torch.nn.MSELoss()


def get_pointer_net_optimizer(net):
    lr = 0.0001
    return optim.Adam(net.parameters(), lr=lr), lr


def get_linear_net_optimizer(net):
    lr = 0.0002
    return optim.Adam(net.parameters(), lr=lr), lr


def net_train(epochs, net, load_path=None, critic_net=None):
    global test_rewards
    optimizer, learning_rate = get_pointer_net_optimizer(net) if type(net) == NeuralCombOptNet else \
        get_linear_net_optimizer(net)

    # scheduler = lr_scheduler.MultiStepLR(optimizer, range(decay_step, decay_step * 1000, decay_step),
    #                                      gamma=decay_rate)

    if critic_net is not None:
        critic_optim = optim.Adam(critic_net.parameters(), lr=learning_rate)
        critic_scheduler = lr_scheduler.MultiStepLR(critic_optim, range(decay_step, decay_step * 1000, decay_step),
                                                    gamma=decay_rate)
    else:
        critic_exp_mvg_avg = torch.zeros(1, device=dataset.device)

    epoch = 0
    epochs_rewards = []

    log_file = open(constants['train_log_file'], "w") if load_path is None else open(constants['train_log_file'], "a")

    if load_path is not None:
        checkpoint = torch.load(load_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        net.to(device=dataset.device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        epochs_rewards = checkpoint['rewards']
        test_rewards = checkpoint['test_rewards']
        critic_exp_mvg_avg = checkpoint['critic']

    details = "starting training with:\n"
    details += "penalty = " + str(dataset.UNFOUND_MATCHES_PENALTY) + "\n"
    details += "lr = " + str(learning_rate) + "\n"
    details += "required matches portion = " + str(dataset.REQUIRED_MATCHES_PORTION) + "\n"
    details += "using critic net? " + ("yes" if critic_net is not None else "no (using moving average)") + "\n"
    details += "net type: " + str(type(net)) + "\n"
    details += "steps = " + str(steps) + "\n"
    details += "time calculation: " + dataset.time_calc_types[dataset.time_calc_index] + "\n"
    details += "------------"
    print(details)
    log_file.write(details)

    net.to(device=dataset.device)
    while epoch < epochs:
        epoch_average_reward = 0
        X, M = dataset.initialize_data_x(True), dataset.initialize_data_matches(True)
        if use_time_ratio:
            E = dataset.initialize_data_x(True)
        net.train()
        processed_events = 0
        x, m, e = dataset.get_batch_events(X), dataset.get_batch_matches(M), None
        if use_time_ratio:
            e = dataset.get_batch_events_as_events(E)
        i = -1
        # while x is not None and m is not None:
        while False:
            if steps != 1:
                print("\n~new batch~\n")
            for _ in range(steps):
                chosen_events, log_probs, net_time = net.forward(x)
                rewards, batches_chosen_events_num, found_matches_portions, found_matches_portion, denominator = \
                    dataset.get_rewards(m, chosen_events, e if use_time_ratio else None)
                epoch_average_reward += rewards.mean().item()
                chosen_events_num = np.mean(batches_chosen_events_num)
                if critic_net is not None:
                    critic_out = critic_net(x.detach())
                else:
                    if processed_events == 0:
                        critic_exp_mvg_avg = rewards.mean()
                    else:
                        critic_exp_mvg_avg = (critic_exp_mvg_avg * beta) + ((1. - beta) * rewards.mean())

                normalizer = critic_out if critic_net is not None else critic_exp_mvg_avg
                advantage = rewards - normalizer

                optimizer.zero_grad()
                losses = (-1) * log_probs * advantage.detach()
                losses = losses.mean()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm, norm_type=2)
                optimizer.step()
                # scheduler.step()

                if critic_net is not None:
                    rewards = rewards.detach()
                    critic_optim.zero_grad()
                    critic_loss = critic_mse(critic_out, rewards)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic_net.parameters(), max_grad_norm, norm_type=2)
                    critic_optim.step()
                    critic_scheduler.step()
                else:
                    critic_exp_mvg_avg = critic_exp_mvg_avg.detach()

                print_interval(batches_chosen_events_num, chosen_events, chosen_events_num, epoch,
                               found_matches_portion, found_matches_portions, log_file, processed_events, rewards,
                               train_size, denominator)

                processed_events += dataset.batch_size
            x, m, e = dataset.get_batch_events(X), dataset.get_batch_matches(M), None
            if use_time_ratio:
                e = dataset.get_batch_events_as_events(E)

            i += 1

        epoch_average_reward = epoch_average_reward / i
        epochs_rewards.append(epoch_average_reward)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rewards': epochs_rewards,
            'critic': critic_exp_mvg_avg,
            'test_rewards': test_rewards
        }, model_path + "_" + str(epoch))

        print("train rewards: " + str(epochs_rewards))
        net_test(net, epoch, log_file)
        print("test rewards: " + str(test_rewards))

        epoch += 1

    log_file.close()


def print_interval(batches_chosen_events_num, chosen_events, chosen_events_num, epoch, found_matches_portion,
                   found_matches_portions, log_file, processed_events, rewards, size, denominator, is_validation=False,
                   net_time=None, cep_whole_time=None, cep_filtered_time=None, first_wind_net_time=None,
                   first_wind_cep_time=None, first_wind_filt_cep_time=None):
    if processed_events % batch_interval == 0:
        if not is_validation:
            print("Epoch " + str(epoch) + ": Processed " + str(processed_events) + " out of " +
                  str(size))
        else:
            print("~Validation~ epoch " + str(epoch) + ": Processed " + str(processed_events) + " out of " +
                  str(size))
        print("average reward: " + str(rewards.mean().item()))
        print("sampled chosen events " + str(chosen_events[np.random.choice(range(dataset.batch_size))]))
        print("found matches portion: " + str(found_matches_portion) +
              ", chosen events num: " + str(chosen_events_num))
        print("matches portion to chosen events = " +
              str(found_matches_portion / (chosen_events_num / constants['window_size'])))
        print("time metric portion = " + str(denominator))

        log_file.write("chosen events:\n" + str(batches_chosen_events_num) + "\n")
        log_file.write("found matches portions:\n" + str(found_matches_portions) + "\n")
        log_file.write("rewards:\n" + str(rewards.tolist()) + "\n")
        log_file.write("average reward: " + str(rewards.mean().item()) + "\n")
        log_file.write("matches portion to chosen events = " +
                       str(found_matches_portion / (chosen_events_num / constants['window_size'])))

        if is_validation:
            time_ = "actual time portion = " + str(cep_filtered_time / cep_whole_time)

            print(time_)
            log_file.write(time_)
            time_ratio = str((cep_filtered_time + net_time) / cep_whole_time)
            net_time_ = "batches actual time portion with net = " + time_ratio
            print(net_time_)
            log_file.write(net_time_)

            time_ratio = str((first_wind_filt_cep_time + first_wind_net_time) / first_wind_cep_time)
            net_time_ = "non batched actual time portion with net = " + time_ratio + " (first window has + " \
                        + str(batches_chosen_events_num[0]) + " events" + ")"
            print(net_time_)
            log_file.write(net_time_)

    log_file.write("-------------------\n")


test_rewards = []


def net_test(net, epoch, log_file):
    global test_rewards
    net.to(device=dataset.device)
    epoch_average_reward = 0
    X, M, E = dataset.initialize_data_x(False), dataset.initialize_data_matches(False), dataset.initialize_data_x(False)
    net.eval()
    processed_events = 0
    log_file.write("\n~validation~\n")
    print("\n~validation~\n")
    x, m, e = dataset.get_batch_events(X), dataset.get_batch_matches(M), \
        dataset.get_batch_events_as_events(E)
    i = -1
    while x is not None and m is not None:
        chosen_events, log_probs, net_time = net.forward(x)
        rewards, batches_chosen_events_num, found_matches_portions, found_matches_portion, denominator, b_whole_time, \
            b_filtered_time, f_whole_time, f_filtered_time = dataset.get_rewards(m, chosen_events, e, is_train=False)
        chosen_events_num = np.mean(batches_chosen_events_num)
        epoch_average_reward += rewards.mean().item()
        _, _, first_window_net_time = net.forward(x[0, :, :].unsqueeze(0), 1)
        print_interval(batches_chosen_events_num, chosen_events, chosen_events_num, epoch, found_matches_portion,
                       found_matches_portions, log_file, processed_events, rewards, test_size, denominator,
                       is_validation=True, net_time=net_time, cep_whole_time=b_whole_time,
                       cep_filtered_time=b_filtered_time, first_wind_net_time=first_window_net_time,
                       first_wind_cep_time=f_whole_time, first_wind_filt_cep_time=f_filtered_time)
        processed_events += dataset.batch_size
        x, m, e = dataset.get_batch_events(X), dataset.get_batch_matches(M), \
            dataset.get_batch_events_as_events(E)
        i += 1

    epoch_average_reward = epoch_average_reward / i
    test_rewards.append(epoch_average_reward)


if __name__ == "__main__":
    pointer_net = NeuralCombOptNet(
        input_dim=constants['event_size'],
        embedding_dim=None,
        hidden_dim=hidden_dim,
        max_decoding_len=constants['window_size'] + 1,
        n_glimpses=2,
        tanh_exploration=tanh_exploration,
        use_tanh=use_tanh,
        is_train=True,
        use_cuda=True if dataset.dev == "cuda" else False,
        encoder_bi_directional=False,
        encoder_num_layers=1,
        padding_value=-1
    )
    linear_model = LinearWindowToFilters(dataset.batch_size)
    network = CriticNetwork(
        input_dim=constants['event_size'],
        hidden_dim=hidden_dim,
        tanh_exploration=tanh_exploration,
        use_tanh=use_tanh,
        use_cuda=True if dataset.dev == "cuda" else False,
        n_process_block_iters=3
    )
    conv_model = ConvWindowToFilters(dataset.batch_size, False)
    net_train(100, conv_model, load_path="training_data/net_0", critic_net=None)
