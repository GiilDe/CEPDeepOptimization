import os
import matplotlib.pyplot as plt
import pandas as pd

threshold_details = "../output data/details/"
topk_details = "../output data/details/"
scores_path = "../../Data/scores.txt"
all_events_num = 1000000
all_matches_num = 124603
window_size = 8
windows_num = all_events_num - window_size
threshold_maximumk = 0.0


def events_num_topk(k):
    return (k/window_size)*all_events_num


def topk_complexity(k):
    complexity = windows_num*(2**k)
    return complexity


def maximum_k_complexity(threshold, k):
    scores = open(scores_path, "r")
    scores = [float(line.split("\n")[0]) for line in scores]
    complexity = 0
    for i in range(0, all_events_num - window_size):
        windows_scores = scores[i:i + window_size]
        windows_scores = sorted(windows_scores, reverse=True)
        unfiltered = 0
        for j in range(k):
            score = windows_scores[j]
            if score >= threshold or score == -1:
                unfiltered += 1
        window_complexity = 2**unfiltered if unfiltered != 0 else 0
        complexity += window_complexity
    return complexity


def threshold_complexity(threshold):
    scores = open(scores_path, "r")
    scores = [float(line.split("\n")[0]) for line in scores]
    complexity = 0
    for i in range(0, all_events_num - window_size):
        windows_scores = scores[i:i + window_size]
        unfiltered = 0
        for score in windows_scores:
            if score >= threshold or score == -1:
                unfiltered += 1
        window_complexity = 2**unfiltered if unfiltered != 0 else 0
        complexity += window_complexity
    return complexity


def maximum_k_graphs():
    all = []
    for file_name in os.listdir(topk_details):
        file = open(topk_details + file_name, 'r')
        lines = file.readlines()
        k = int(lines[0].split("\n")[0])
        matches_num = float(lines[2].split("\n")[0])
        all.append((k, matches_num))

    ks = [k for k, _ in all]
    matches_found_ratio_maximum_k = [matches_num / all_matches_num for _, matches_num in all]
    ks_complexity = [maximum_k_complexity(threshold_maximumk, k) for k in ks]

    plt.plot(ks, matches_found_ratio_maximum_k, marker='o')
    plt.title("K value to matches found ratio with threshold: " + str(threshold_maximumk))
    plt.xlabel("k value")
    plt.ylabel("matches found ratio")
    plt.show()

    plt.plot(ks_complexity, matches_found_ratio_maximum_k, marker='o')
    plt.title("Maximum-k: complexity to matches found ratio with threshold: " + str(threshold_maximumk))
    plt.xlabel("complexity")
    plt.ylabel("matches found ratio")
    plt.show()

    plt.title("Threshold and Maximum-k: complexity to matches found ratio")
    plt.xlabel("complexity")
    plt.ylabel("matches found ratio")
    plt.plot(ks_complexity, matches_found_ratio_maximum_k, color='green', label='maximum-k', marker='o')
    plt.plot(thresholds_complexity, matches_found_ratio, color='blue', label='threshold', marker='o')
    plt.legend()
    plt.show()


def threshold_graphs():
    all = []
    for file_name in os.listdir(threshold_details):
        file = pd.read_csv(threshold_details + file_name, header=None)
        threshold = file.at[0, 0]
        events_num = file.at[1, 0]
        matches_num = file.at[2, 0]
        all.append((threshold, events_num, matches_num))

    all = sorted(all, key=lambda x: x[0])
    events_used_ratio = [events_num/all_events_num for _, events_num, _ in all]
    matches_found_ratio = [matches_found/all_matches_num for _, _, matches_found in all]
    thresholds = [threshold for threshold, _, _ in all]
    thresholds_complexity = [threshold_complexity(threshold) for threshold in thresholds]
    plt.plot(thresholds, events_used_ratio, marker='o')
    plt.title("Threshold to events used ratio")
    plt.xlabel("threshold")
    plt.ylabel("events used ratio")
    plt.show()

    plt.plot(thresholds, matches_found_ratio, marker='o')
    plt.title("Threshold to matches found ratio")
    plt.xlabel("threshold")
    plt.ylabel("matches found ratio")
    plt.show()

    plt.plot(events_used_ratio, matches_found_ratio, marker='o')
    plt.title("Threshold: events used ratio to matches found ratio")
    plt.xlabel("events used ratio")
    plt.ylabel("matches found ratio")
    plt.show()

    plt.plot(thresholds_complexity, matches_found_ratio, marker='o')
    plt.title("Threshold: complexity to matches found ratio")
    plt.xlabel("complexity")
    plt.ylabel("matches found ratio")
    plt.show()


if __name__ == "__main__":
    threshold_graphs()


