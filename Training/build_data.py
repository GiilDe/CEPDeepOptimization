from constants import constants
import random
import pandas as pd
import shutil


def build_data_stream():
    for size, path in zip([constants['train_size'], constants['test_size']],
                          [constants['train_stream_path'], constants['test_stream_path']]):
        file = open(path, "w")
        counter = 0
        for _ in range(size):
            name = str(random.choice(constants['event_types']))
            value = str(random.random())
            s_counter = str(counter)
            event = ','.join([name, value, s_counter]) + "\n"
            file.write(event)
            if counter % 1000 == 0:
                print(str(counter))
            counter += 1
        file.close()


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


if __name__ == "__main__":
    build_data_stream()
