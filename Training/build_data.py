import constants
import random
import pandas as pd


def build_sequences(is_test):
    file_path = constants.test_file_path if is_test else constants.train_file_path
    file_path_seqs = constants.test_file_path_sequences if is_test else constants.train_file_path_sequences
    data_stream = pd.read_csv(file_path, header=None)
    window_size = constants.window_limit[0]
    for i in range(window_size - 1, len(data_stream) - window_size + 1):
        window = data_stream.loc[i - (window_size - 1):i + window_size - 1]
        window = pd.DataFrame(window.values.flatten())
        window = window.T
        window.to_csv(file_path_seqs, index=False, header=False, mode='a')
        if i % 1000 == 0:
            print(i)


def build_data_stream():
    types = ['A', 'A', 'A', 'B', 'B', 'C', 'D']
    chunk_size = 10**4
    for size, path in zip([constants.train_size, constants.test_size], [constants.train_file_path, constants.test_size]):
        done = False
        counter = 0
        while not done:
            for _ in range(chunk_size):
                if counter == size:
                    done = True
                    break
                chunk = pd.DataFrame()
                name = str(random.choice(types))
                value = random.random()
                s_counter = str(counter)
                chunk.append(pd.Series(data=[name, value, s_counter]))
                chunk.to_csv(path, index=False, header=False, mode='a')
                counter += 1


if __name__ == "__main__":
    build_sequences(False)
