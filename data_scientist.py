# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.imp

import ast
import math
from collections import Counter

import numpy
import pandas as pd


def calculate_similarity(focal: pd.DataFrame, cluster: pd.DataFrame) -> float:
    """
    Computes cosine similarity between two DataFrames.
    Parameters:
        focal : The histogram.
        cluster : The cluster histogram to compare.
    Variables:
        intersect_keys (set): Set of common keys between the two histograms.
        focal_size (float): Sum of the 'c' values in the focal DataFrame.
        cluster_size (float): Sum of the 'c' values in the cluster DataFrame.
        hist_sum (float): Product sum of corresponding 'c' values in the focal and cluster DataFrames.
    Returns:
        float: Cosine similarity between the two dataframes.
    """
    # find the common keys between the histograms
    print(f"Shapes: Focal:{focal.shape[0]}, Cluster:{cluster.shape[0]}")
    intersect_keys = set(focal['k']).intersection(cluster['k'])
    print(f"Intersection Size: {len(intersect_keys)}")
    # if there are no common keys, return 0
    if len(intersect_keys) < 1:
        return 0

    focal_size = focal['c'].sum()
    cluster_size = cluster['c'].sum()
    print(f"Sizes: Focal:{focal_size}, Cluster:{cluster_size}")

    if focal_size == 0 or cluster_size == 0:
        return 0

    focal = focal[focal['k'].isin(intersect_keys)]
    cluster = cluster[cluster['k'].isin(intersect_keys)]

    hist_sum = (focal.set_index('k')['c'] * cluster.set_index('k')['c']).sum()
    print(f"Intersection Sum: {hist_sum}")
    hist_sum /= (focal_size * cluster_size)
    print(f"Intersection Normalized Sum: {hist_sum}")
    return max(min(hist_sum, 1), 0)


def calculate_distance(focal: pd.DataFrame, cluster: pd.DataFrame) -> float:
    """
    Computes cosine distance between two DataFrames.
    Parameters:
        focal (pd.DataFrame): The focal DataFrame to compare.
        cluster (pd.DataFrame): The cluster DataFrame to compare.
    Variables:
        similarity (float): Cosine similarity between the two dataframes.
    Returns:
        float: Cosine distance between the two dataframes.
    """
    focal = pd.DataFrame(focal)
    cluster = pd.DataFrame(cluster)
    similarity = calculate_similarity(focal=focal, cluster=cluster)
    # if similarity is 0 or 1, return the opposite (cosine)
    if similarity in [0, 1]:
        return abs(similarity - 1)

    return 2 * math.acos(similarity) / math.pi


def load_hist_df(df) -> dict:
    """"Load the histogram CSV and create a dict with the number as key and the histogram as value """
    out = {}
    i = 0
    for index, row in df.iterrows():
        out[i] = ast.literal_eval(row['hist'])
        i = i + 1
    return out


def get_key_value(hist) -> dict:
    """ get a cluster or session histogram and return adict with the Key values in that histogram"""
    out = {}
    pairs = hist['k'].split('\t')
    for pair in pairs:
        pair = pair.split('=')
        if len(pair) > 1:
            out[pair[0]] = pair[1]
    return out


def get_dist(ses_dict):
    """get the distrebution of key values in all the histograms """
    dist = {}
    for k in ses_dict.keys():
        for hist in ses_dict.get(k):
            pairs = get_key_value(hist)
            for k1, v in pairs.items():
                c = dist.get(k1)
                if c:
                    c.update({v: 1})
                else:
                    dist[k1] = Counter({v: 1})
    return dist


def run():
    # load the filed
    df_ses = pd.read_csv(r"sessions_histogram.csv")
    df_clu = pd.read_csv(r"clusters_histogram.csv")
    dist_sess = numpy.load(r"sessions_distances.npy")
    dist_clus = numpy.load(r"cluster_distances.npy")

    # convert the session snd cluster history to a python form
    ses_dict = load_hist_df(df_ses)
    clu_dict = load_hist_df(df_clu)

    # some example on how to parse the histograms to key value
    dist = get_dist(ses_dict)

    # for k in dist.keys():
    #     print(k)
    #     print(dist[k])
    # print('*' * 100)
    # use the calculated distance and print it with the distance matrix to compare
    # print(calculate_distance(ses_dict[0], clu_dict[1]))
    # print(dist_sess[0][1])
    for i in range(10):
        for j in range(10):
            print(f"Session {i}, Cluster {j}")
            distance = calculate_distance(ses_dict[i], clu_dict[j])
            print(f"Distance: {distance}")
    j = 1


if __name__ == '__main__':
    run()
