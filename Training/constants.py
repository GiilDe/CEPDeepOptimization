import torch

model_path = "training data/FCnet"
last_predictions_file_path = "training data/last predictions.txt"
scores_path = "../Data/scores.txt"

train_file_path = "../Data/train data stream.txt"
test_file_path = "../Data/test data stream.txt"


train_file_path_labels = "training data/FOR_TRAIN_LABELS.txt"
train_file_path_sequences = "training data/FOR_TRAIN_SEQS.txt"

test_file_path_labels = "training data/FOR_TEST_LABELS.txt"
test_file_path_sequences = "training data/FOR_TEST_SEQS.txt"


train_file_path_labels_RNN = "training data/FOR_TRAIN_LABELS_RNN.txt"
train_file_path_sequences_RNN = "training data/FOR_TRAIN_SEQS_RNN.txt"

test_file_path_labels_RNN = "training data/FOR_TEST_LABELS_RNN.txt"
test_file_path_sequences_RNN = "training data/FOR_TEST_SEQS_RNN.txt"


training_report_path = "training data/report.txt"

event_format = ['type', 'value', 'counter'], 2, 0
window_limit = (8, True)


stop_lines = {"$", "$\n", " ", "\n", "", "-1", "-1\n"}


# if torch.cuda.is_available():
#     dev = "cuda"
# else:
dev = "cpu"

device = torch.device(dev)