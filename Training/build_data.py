from constants import constants
import random
import shutil


def build_data_stream():
    types = ['A', 'B', 'C', 'D']
    for size, path in zip([constants['train_size'], constants['test_size']],
                          [constants['train_stream_path'], constants['test_stream_path']]):
        i = 0
        with open(path, "w") as f:
            for _ in range(size):
                name = str(random.choice(types))
                value = str(random.random())
                line = ",".join([name, value]) + "\n"
                f.write(line)
                if i % 1000 == 0:
                    print(i)
                i += 1


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


def label_no_matched_events():


if __name__ == "__main__":
    pad_matches()
