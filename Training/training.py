import torch
import torch.functional
import torch.optim as optim
import pandas as pd
import nets
from constants import constants

steps = 1
event_size = 5

allow_gpu = False
dev = "cuda" if allow_gpu and torch.cuda.is_available() else "cpu"

device = torch.device(dev)

loss_function_type = torch.nn.MSELoss
model_path = "training_data/reinforce"
training_report_path = "training_data/report.txt"
batch_size = 50

UNFOUND_MATCHES_PENALTY = 1
REQUIRED_MATCHES_PORTION = 0.8

FULL_WINDOW_COMPLEXITY = \
    2**(constants['pattern_window_size'])*(constants['window_size'] - constants['pattern_window_size'] + 1)

test_rewards = []


def get_batch_matches(M):
    try:
        batches = []
        for _ in range(batch_size):
            batch = M.readline()
            if batch == "":
                raise StopIteration
            batch = batch.split(",")
            for i, obj_i in enumerate(batch):
                batch[i] = int(obj_i)
            batches.append(batch)
        return batches
    except (StopIteration, RuntimeError):
        return None


def get_batch_events(X):
    def get_dummies(x):
        padding = pd.DataFrame([['A', -1], ['B', -1], ['C', -1], ['D', -1]])
        x = padding.append(x, ignore_index=True)
        x = pd.get_dummies(x)
        x = x.drop(axis=0, labels=[0, 1, 2, 3])
        return x
    try:
        batch = next(X)
        batch = get_dummies(batch)
        batch = torch.tensor(batch.to_numpy(), dtype=torch.float64, requires_grad=True) \
            .reshape((batch_size, constants['window_size'], event_size))
        return batch
    except (StopIteration, RuntimeError):
        return None


def initialize_data_x(is_train):
    X = pd.read_csv(constants['train_stream_path'] if is_train else constants['test_stream_path'],
                    chunksize=batch_size * constants['window_size'], header=None, usecols=[0, 1])
    return X


def initialize_data_matches(is_train):
    # M = pd.read_csv(constants['train_matches'], chunksize=batch_size, header=None)
    M = open(constants['train_matches'] if is_train else constants['test_matches'], "r")
    return M


epsilon = 1e-9

matches_sum = 0
found_matches_sum = 0
found_matches_portion = 0


def get_rewards(matches, chosen_events):
    def get_window_complexity_ratio(i):
        batch_chosen_events = chosen_events[i]
        pattern_window_size = constants['pattern_window_size']
        window_complexity = 0
        pattern_window_selected = 0

        for i in range(pattern_window_size):
            pattern_window_selected += batch_chosen_events[i]

        window_complexity += 2**pattern_window_selected

        for i in range(constants['window_size'] - pattern_window_size):
            pattern_window_selected -= batch_chosen_events[i]
            pattern_window_selected += batch_chosen_events[i + pattern_window_size]
            window_complexity += 2**pattern_window_selected

        return window_complexity/FULL_WINDOW_COMPLEXITY

    def get_window_matches(i):
        global matches_sum, found_matches_sum, found_matches_portion
        batch_matches = matches[i]
        batch_chosen_events = chosen_events[i]
        if len(batch_matches) == 1:
            return 0, 0
        else:
            non_chosen = set()
            for i in range(constants['window_size']):
                if batch_chosen_events[i] == 0:
                    non_chosen.add(i)
            matches_ = []
            for i in range(0, len(batch_matches), constants['match_size']):
                match = set(batch_matches[i:i + constants['match_size']])
                matches_.append(match)

            matches_num = len(matches_)
            filtered = 0
            for match in matches_:
                for count in match:
                    if count in non_chosen:
                        filtered += 1
                        break

            found_matches_num = matches_num - filtered
            found_matches_sum += found_matches_num
            matches_sum += matches_num
            found_matches_portion = found_matches_sum/matches_sum
            return matches_num, found_matches_num

    def unfound_match_penalty(matches_ratio):
        # gap = max(REQUIRED_MATCHES_PORTION * matches_num - found_matches_num, 0)
        # return UNFOUND_MATCHES_PENALTY * gap
        return UNFOUND_MATCHES_PENALTY * max(REQUIRED_MATCHES_PORTION - matches_ratio, 0)

    rewards = []
    for i in range(batch_size):
        window_complexity_ratio = max(get_window_complexity_ratio(i), 0.005)
        matches_num, filtered_matches_num = get_window_matches(i)

        matches_ratio = filtered_matches_num/matches_num if matches_num != 0 else 1

        ratio = matches_ratio/window_complexity_ratio
        penalty = unfound_match_penalty(matches_ratio)

        reward = ratio - penalty
        rewards.append(reward)

    rewards = torch.tensor(rewards)
    # return rewards, (rewards - rewards.mean()) / (rewards.std() + epsilon)
    return rewards, rewards


def net_train(epochs, batch_interval, net, load_path=None):
    global test_rewards
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    epoch = 0
    epochs_rewards = []

    if load_path is not None:
        checkpoint = torch.load(load_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        epochs_rewards = checkpoint['rewards']

    net.to(device=device)
    while epoch < epochs:
        epoch_average_reward = 0
        X, M = initialize_data_x(True), initialize_data_matches(True)
        net.train()
        processed_events = 0
        x, m = get_batch_events(X), get_batch_matches(M)
        while x is not None and m is not None:
            for _ in range(steps):
                optimizer.zero_grad()
                chosen_events, log_probs = net.forward(x)
                rewards, rewards_normalized = get_rewards(m, chosen_events)
                rewards_mean = rewards.mean().item()
                epoch_average_reward += rewards_mean
                losses = -log_probs*rewards_normalized
                losses.sum().backward()
                optimizer.step()
            if processed_events % batch_interval == 0:
                print("Epoch " + str(epoch) + ": Processed " + str(processed_events) + " out of "
                      + str(constants['train_size']) + " sampled reward of " + str(rewards_mean) + "\n" +
                      "and chosen events " + str(chosen_events[0]) + "\n" + "found matches portion: "
                      + str(found_matches_portion))
            processed_events += batch_size
            x, m = get_batch_events(X), get_batch_matches(M)

        epoch_average_reward = epoch_average_reward/(steps*constants['train_size'])
        epochs_rewards.append(epoch_average_reward)
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'rewards': epochs_rewards,
        }, model_path + "_" + str(epoch))
        epoch += 1

        print("train losses: " + str(epochs_rewards))
        net_test(net)
        print("test losses: " + str(test_rewards))


def net_test(net):
    global test_rewards

    net.to(device=device)
    epoch_average_reward = 0
    X, M = initialize_data_x(False), initialize_data_matches(False)
    net.eval()
    processed_events = 0
    i = 0
    x, m = get_batch_events(X), get_batch_matches(M)
    while x is not None and m is not None:
        chosen_events, _ = net.forward(x)
        rewards, _ = get_rewards(m, chosen_events)
        epoch_average_reward += rewards.mean().item()
        processed_events += batch_size
        x, m = get_batch_events(X), get_batch_matches(M)
        if i % 1000 == 0:
            print(i)
        i += batch_size

    epoch_average_reward = epoch_average_reward/(steps*constants['test_size'])
    test_rewards.append(epoch_average_reward)


if __name__ == "__main__":
    net_train(100, 1000, nets.WindowToFiltersReinforce(batch_size))
