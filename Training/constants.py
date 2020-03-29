import torch

model_path = "training data/FCnet"
last_predictions_file_path = "training data/last predictions.txt"
scores_path = "../Data/scores.txt"

train_file_path = "../Data/train data stream.txt"
test_file_path = "../Data/test data stream.txt"


train_file_path_labels = "training data/FOR_TRAIN_LABELS.txt"
train_file_path_sequences = "training data/FOR_TRAIN_SEQS_8.txt"

test_file_path_labels = "training data/FOR_TEST_LABELS.txt"
test_file_path_sequences = "training data/FOR_TEST_SEQS_8.txt"

training_report_path = "training data/report.txt"

event_format = ['type', 'value', 'counter'], 2, 0
window_limit = 8


stop_lines = {"$", "$\n", " ", "\n", "", "-1", "-1\n"}

chunk_size = 10**4

# if torch.cuda.is_available():
#     dev = "cuda"
# else:
dev = "cpu"

device = torch.device(dev)

test_size = 1000000
train_size = 6077571

type_to_vec = {'A': [0, 0, 0, 1], 'B': [0, 0, 1, 0], 'C': [0, 1, 0, 0], 'D': [1, 0, 0, 0], 0: [0, 0, 0, 0]}
