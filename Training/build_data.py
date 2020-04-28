from constants import constants
import random
import pandas as pd
import numpy as np
import os


def build_labels(is_train):
    file = open(constants['train_filters_file_path'] if is_train else constants['test_filters_file_path'], "w")
    stream_size = constants['train_size'] if is_train else constants['test_size']
    for i in range(int(stream_size * constants['window_repeat'])):
        file.write(str(np.random.uniform(0, 1)) + "\n")
        if i % 1000 == 0:
            print(i)
    file.close()


def build_data_stream_with_repeat(is_train):
    read_file = open(constants['train_stream_path'] if is_train else constants['test_stream_path'], "r")
    write_file = open(constants['train_stream_path_repeat'] if is_train else constants['test_stream_path_repeat'], "w")
    i = 0
    while True:
        window = []
        for _ in range(constants['window_size']):
            line = read_file.readline()
            if line != "":
                window.append(line)
            else:
                read_file.close()
                write_file.close()
                return
        for _ in range(constants['window_repeat']):
            for line in window:
                write_file.write(line)
                if i % 1000 == 0:
                    print(i)
                i += 1


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


def build_data_from_stream():
    build_labels(True)
    build_labels(False)
    build_data_stream_with_repeat(True)
    build_data_stream_with_repeat(False)
    os.chdir(os.path.dirname("../"))
    os.system("java -jar Application.jar")


if __name__ == "__main__":
    build_data_from_stream()
