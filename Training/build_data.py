from Training import processing_utilities
from Training import constants
import random


def write_non_zero_rows(file_path):
    data_stream = open(file_path, 'r')
    non_zero = 0
    for line in data_stream:
        quantity = int(line.split("\n")[0])
        if quantity > 0:
            non_zero += 1
    data_stream.close()
    data_stream = open(file_path, 'a')
    data_stream.write("Relevant events amount: " + str(non_zero))
    data_stream.close()


def build_sequences(is_test):
    file_path = constants.test_file_path if is_test else constants.train_file_path
    file_path_seqs = constants.test_file_path_sequences if is_test \
        else constants.train_file_path_sequences
    data_stream = open(file_path, 'r')
    seqs_file = open(file_path_seqs, 'w')
    events = [processing_utilities.get_event_from_str(line, *constants.event_format) for line in data_stream]
    window_size = constants.window_limit[0]
    relevant_events = events[window_size - 1:len(events) - window_size]
    for i, event in enumerate(relevant_events):
        relevant_events = events[i - (window_size - 1):i + window_size]
        for event in relevant_events:
            seqs_file.write(str(event) + ";")
        seqs_file.write("\n")
    data_stream.close()
    seqs_file.close()


def build_sequences_RNN(is_test):
    def get_before_events(i, events):
        event = events[i]
        res = []
        time = event.get_time()
        j = i - 1
        while True:
            if 0 <= j <= len(events) - 1 and events[j].get_time() + constants.time_limit[0] >= time:
                res.append(events[j])
            else:
                break
            j -= 1
        res.reverse()
        return res

    def get_after_events(i, events):
        event = events[i]
        res = []
        time = event.get_time()
        j = i + 1
        while True:
            if 0 <= j <= len(events) - 1 and events[j].get_time() - constants.time_limit[0] <= time:
                res.append(events[j])
            else:
                break
            j += 1
        return res

    file_path = constants.test_file_path_RNN if is_test else constants.train_file_path_RNN
    file_path_seqs = constants.test_file_path_sequences_RNN if is_test else constants.train_file_path_sequences_RNN
    data_stream = open(file_path, 'r')
    seqs_file = open(file_path_seqs, 'w')
    events = [processing_utilities.get_event_from_str(line, **constants.event_format) for line in data_stream]
    stock_types_with_indices = dict(zip(constants.pattern, range(len(constants.pattern))))
    for i, event in enumerate(events):
        if event.get_type() in stock_types_with_indices.keys():
            index = stock_types_with_indices[event.get_type()]
            if index == 0:
                before_events = []
                after_events = get_after_events(i, events)
            elif index == len(constants.stock_types) - 1:
                before_events = get_before_events(i, events)
                after_events = []
            else:
                before_events = get_before_events(i, events)
                after_events = get_after_events(i, events)

            before_padding = ["P" for _ in range(constants.max_seq_RNN - len(before_events))]
            after_padding = ["P" for _ in range(constants.max_seq_RNN - len(after_events))]
            relevant_events = before_padding + before_events + [event] + after_events + after_padding
            for event in relevant_events:
                seqs_file.write(str(event) + ";")
            seqs_file.write("\n")
        else:
            seqs_file.write("-1;\n")
    data_stream.close()
    seqs_file.close()


def build_data_with_values():
    def write_event(output_file, types, counter):
        name = str(random.choice(types))
        s_counter = str(counter)
        s = name + "," + str(random.random()) + ", " + s_counter + "\n"
        output_file.write(s)

    train_output_file = open(constants.train_file_path, "w")
    test_output_file = open(constants.test_file_path, "w")
    types = ['A', 'A', 'A', 'B', 'B', 'C', 'D']
    counter = 0
    for _ in range(constants.train_size):
        write_event(train_output_file, types, counter)
        counter += 1
    counter = 0
    for _ in range(constants.test_size):
        write_event(test_output_file, types, counter)
        counter += 1
    train_output_file.close()
    test_output_file.close()


def build_data_for_RNN(is_test):
    output_file = open(constants.test_file_path_RNN, "w") if is_test else open(constants.train_file_path_RNN, "w")
    types = ['A', 'A', 'A', 'B', 'B', 'C', 'D']
    time = 0
    possible_time_lengths = range(constants.max_seq_RNN - 4, constants.max_seq_RNN + 1)
    time_length = random.choice(possible_time_lengths)
    size = constants.test_size if is_test else constants.train_size
    counter = 0
    for _ in range(size):
        if time_length == 0:
            time_length = random.choice(possible_time_lengths)
            time += 1
        s = str(random.choice(types)) + "," + str(time) + "," + str(random.random()) + "," + str(random.random()) + \
            "," + str(counter) + "\n"
        output_file.write(s)
        time_length -= 1
        counter += 1

    output_file.close()


