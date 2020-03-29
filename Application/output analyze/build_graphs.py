import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

threshold_details = "../output data/details/threshold details/"
maximumk_details = "../output data/details/maximumk details/"
scores_path = "../../Data/scores.txt"
all_events_num = 1000000
all_matches_num = 124603
window_size = 8
windows_num = all_events_num - window_size
threshold_maximumk = 0.12

graphs_path = "../output data/graphs/"


def events_num_topk(k):
    return (k/window_size)*all_events_num


def topk_complexity(k):
    complexity = windows_num*(2**k)
    return complexity


def maximum_k_complexities(threshold, ks):
    scores = pd.read_csv(scores_path, header=None)
    N = len(ks)
    complexities = np.zeros(N)
    windows_scores = [scores.at[j, 0] for j in range(0, window_size)]
    for i in range(window_size, all_events_num):
        windows_scores_sorted = sorted(windows_scores, reverse=True)
        unfiltered = np.zeros(N, dtype=int)
        for t in range(N):
            k = ks[t]
            for j in range(k):
                score = windows_scores_sorted[j]
                if score >= threshold or score == -1:
                    unfiltered[t] += 1
        windows_complexities = [2 ** u if u != 0 else 0 for u in unfiltered]
        complexities += windows_complexities
        del windows_scores[0]
        windows_scores.append(scores.at[i, 0])
        if i % 1000 == 0:
            print(i)
    return complexities


def threshold_complexities(thresholds):
    scores = pd.read_csv(scores_path, header=None)
    N = len(thresholds)
    complexities = np.zeros(N)
    for i in range(0, all_events_num - window_size):
        unfiltered = np.zeros(N, dtype=int)
        for j in range(i, i + window_size):
            score = scores.at[j, 0]
            for t in range(N):
                if score >= thresholds[t] or score == -1:
                    unfiltered[t] += 1
        windows_complexities = [2 ** u if u != 0 else 0 for u in unfiltered]
        complexities += windows_complexities
        if i % 1000 == 0:
            print(i)
    return complexities


def maximum_k_graphs():
    all = []
    for file_name in os.listdir(maximumk_details):
        file = pd.read_csv(maximumk_details + file_name, header=None)
        k = int(file.at[0, 0])
        matches_num = float(file.at[2, 0])
        all.append((k, matches_num))

    ks = [k for k, _ in all]
    matches_found = [matches_num / all_matches_num for _, matches_num in all]
    complexity = maximum_k_complexities(threshold_maximumk, ks)

    plt.plot(ks, matches_found, marker='o')
    title = "K value to matches found ratio with threshold " + str(threshold_maximumk) + ".png"
    plt.title(title)
    plt.xlabel("k value")
    plt.ylabel("matches found ratio")
    plt.show()
    plt.savefig(graphs_path + title)

    plt.plot(complexity, matches_found, marker='o')
    title = "Maximum-k complexity to matches found ratio with threshold " + str(threshold_maximumk) + ".png"
    plt.title(title)
    plt.xlabel("complexity")
    plt.ylabel("matches found ratio")
    plt.show()
    plt.savefig(graphs_path + title)

    return complexity, matches_found


def maximumk_to_threshold(ks_complexity, matches_found_ratio_maximum_k, thresholds_complexity,
                          matches_found_ratio_threshold):
    title = "Threshold and Maximum-k complexity to matches found ratio"
    plt.title(title)
    plt.xlabel("complexity")
    plt.ylabel("matches found ratio")
    plt.plot(ks_complexity, matches_found_ratio_maximum_k, color='green', label='maximum-k', marker='o')
    plt.plot(thresholds_complexity, matches_found_ratio_threshold, color='blue', label='threshold', marker='o')
    plt.legend()
    plt.show()
    plt.savefig(graphs_path + title)


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
    complexity = threshold_complexities(thresholds)

    plt.plot(thresholds, events_used_ratio, marker='o')
    title = "Threshold to events used ratio"
    plt.title(title)
    plt.xlabel("threshold")
    plt.ylabel("events used ratio")
    plt.show()
    plt.savefig(graphs_path + title)

    plt.plot(thresholds, matches_found_ratio, marker='o')
    title = "Threshold to matches found ratio"
    plt.title(title)
    plt.xlabel("threshold")
    plt.ylabel("matches found ratio")
    plt.show()
    plt.savefig(graphs_path + title)

    plt.plot(events_used_ratio, matches_found_ratio, marker='o')
    title = "Threshold events used ratio to matches found ratio"
    plt.title(title)
    plt.xlabel("events used ratio")
    plt.ylabel("matches found ratio")
    plt.show()
    plt.savefig(graphs_path + title)

    plt.plot(complexity, matches_found_ratio, marker='o')
    title = "Threshold complexity to matches found ratio"
    plt.title(title)
    plt.xlabel("complexity")
    plt.ylabel("matches found ratio")
    plt.savefig(graphs_path + title)
    plt.show()

    return complexity, matches_found_ratio


if __name__ == "__main__":
    t_complexity, t_matches = threshold_graphs()
    k_complexity, k_matches = maximum_k_graphs()
    maximumk_to_threshold(k_complexity, k_matches, t_complexity, t_matches)

