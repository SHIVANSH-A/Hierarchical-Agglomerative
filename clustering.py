import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import os

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

def hierarchical_clustering(data):
    clusters = {i: [i] for i in range(len(data))}
    distances = {}
    steps = []

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distances[(i, j)] = euclidean_distance(data[i], data[j])

    linkage_matrix = []

    while len(clusters) > 1:
        min_pair = min(distances, key=distances.get)
        cluster1, cluster2 = min_pair
        new_cluster = clusters[cluster1] + clusters[cluster2]
        new_cluster_idx = max(clusters.keys()) + 1

        linkage_matrix.append([cluster1, cluster2, distances[min_pair], len(new_cluster)])
        steps.append(f"Merging clusters {cluster1} and {cluster2} with distance {distances[min_pair]:.2f}")

        del clusters[cluster1]
        del clusters[cluster2]
        clusters[new_cluster_idx] = new_cluster

        new_distances = {}
        for i in clusters:
            if i != new_cluster_idx:
                min_dist = np.mean([euclidean_distance(data[p1], data[p2])
                                    for p1 in new_cluster for p2 in clusters[i]])
                new_distances[(min(i, new_cluster_idx), max(i, new_cluster_idx))] = min_dist

        distances = new_distances

    return np.array(linkage_matrix), steps

def plot_dendrogram(linkage_matrix):
    plt.figure(figsize=(10, 6))
    dendrogram(linkage_matrix)
    plt.title("Hierarchical Clustering Dendrogram")
    plt.savefig("static/dendrogram.png")
    plt.close()

def run_clustering(csv_path):
    df = pd.read_csv(csv_path)
    df_numeric = df.select_dtypes(include=[np.number])  # ignore non-numeric columns
    data = df_numeric.values
    linkage_matrix, steps = hierarchical_clustering(data)
    plot_dendrogram(linkage_matrix)
    return linkage_matrix, steps
