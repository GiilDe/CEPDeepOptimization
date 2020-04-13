import constants
import random
import pandas as pd
import processing_utilities


def build_sequences(is_test):
    file_path = constants.test_file_path if is_test else constants.train_file_path
    file_path_seqs = constants.test_file_path_sequences if is_test else constants.train_file_path_sequences
    data_stream = pd.read_csv(file_path, header=None)
    window_size = constants.window_limit
    for i in range(window_size - 1, len(data_stream) - window_size + 1):
        window = data_stream.loc[i - (window_size - 1):i + window_size - 1]
        window = pd.DataFrame(window.values.flatten())
        window = window.T
        window.to_csv(file_path_seqs, index=False, header=False, mode='a')
        if i % 1000 == 0:
            print(i)


def old_build_sequences(is_test):
    file_path = constants.test_file_path if is_test else constants.train_file_path
    file_path_seqs = constants.test_file_path_sequences if is_test \
        else constants.train_file_path_sequences
    data_stream = open(file_path, 'r')
    seqs_file = open(file_path_seqs, 'w')
    events = [processing_utilities.get_event_from_str(line, *constants.event_format) for line in data_stream]
    window_size = constants.window_limit
    for i in range(window_size - 1, len(events) - window_size):
        window = events[i - (window_size - 1):i + window_size]
        for event in window:
            seqs_file.write(str(event) + ";")
        seqs_file.write("\n")
        if i % 1000 == 0:
            print(i)
    data_stream.close()
    seqs_file.close()


def build_data_stream():
    types = ['A', 'A', 'A', 'B', 'B', 'C', 'D']
    chunk_size = 10**4
    for size, path in zip([constants.train_size, constants.test_size], [constants.train_file_path, constants.test_size]):
        counter = 0
        i = 0
        chunk = pd.DataFrame()
        for _ in range(size):
            name = str(random.choice(types))
            value = random.random()
            s_counter = str(counter)
            chunk.append(pd.Series(data=[name, value, s_counter]))
            if i % chunk_size:
                chunk.to_csv(path, index=False, header=False, mode='a')
                chunk = pd.DataFrame()
            counter += 1


if __name__ == "__main__":
    build_sequences(False)
