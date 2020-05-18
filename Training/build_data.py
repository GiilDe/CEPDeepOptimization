from constants import constants
import random
import pandas as pd
import numpy as np
import os
import shutil


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


def pad_matches():
    def pad_matches_(is_train):
        def find_maximum_line():
            f = open(constants['train_matches'] if is_train else constants['test_matches'], "r")
            max_size = 0
            i = 0
            for line in f:
                line = line.split(",")
                line[-1] = line[-1].split("\n")[0]
                this_size = len(line)
                if this_size > max_size:
                    max_size = this_size
                if i % 1000 == 0:
                    print(i)
                i += 1
            f.close()
            return max_size

        max_size = find_maximum_line()
        read_f = open(constants['train_matches'] if is_train else constants['test_matches'], "r")
        temp_file = "temp.txt"
        write_f = open(temp_file, "w")
        i = 0
        for line in read_f:
            line = line.split(",")
            line[-1] = line[-1].split("\n")[0]
            this_size = len(line)
            for _ in range(max_size - this_size):
                line += ["-1"]
            line = ','.join(line)
            line += "\n"
            write_f.write(line)
            if i % 1000 == 0:
                print(i)
            i += 1
        read_f.close()
        write_f.close()
        shutil.move(temp_file, constants['train_matches'] if is_train else constants['test_matches'])

    pad_matches_(True)
    pad_matches_(False)


def build_data_from_stream():
    build_labels(True)
    build_labels(False)
    build_data_stream_with_repeat(True)
    build_data_stream_with_repeat(False)
    os.chdir(os.path.dirname("../"))
    os.system("java -jar Application.jar")


if __name__ == "__main__":
    pad_matches()
