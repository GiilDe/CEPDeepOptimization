import torch
import constants
import pandas as pd
import new_training


relevant_window_size = (constants.window_limit * 2 - 1)


def infer(batch_size=32):
    net = torch.load(constants.model_path)
    net.to(device=constants.device)
    net.eval()
    X = pd.read_csv(constants.test_file_path, chunksize=batch_size, header=None, usecols=[0, 1])
    scores = pd.DataFrame()
    for _ in range(constants.window_limit - 1):
        scores = scores.append(pd.Series(-1), ignore_index=True)
    i = 0
    x = new_training.get_batch_x(X)
    with torch.no_grad():
        while x is not None:
            y_hat = net.forward(x).data.numpy().reshape(-1)
            for row in y_hat:
                scores = scores.append(pd.Series(row.item()), ignore_index=True)
            if i % 1000 == 0:
                print(i)
            i += batch_size
            x = new_training.get_batch_x(X)
    for _ in range(constants.window_limit - 1):
        scores = scores.append(pd.Series(-1), ignore_index=True)
    scores.to_csv(constants.scores_path, index=False, header=False)


if __name__ == "__main__":
    infer(16)

