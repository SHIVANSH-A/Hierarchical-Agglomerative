import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def load_data():
    """ Load the Iris dataset manually (without sklearn) """
    df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", 
                     header=None, names=["sepal_length", "sepal_width", "petal_length", "petal_width", "class"])
    df.drop("class", axis=1, inplace=True)  # Drop class labels for clustering
    return df.values

def euclidean_distance(point1, point2):
    """ Compute Euclidean distance between two points """
    return np.sqrt(np.sum((point1 - point2) ** 2))

def hierarchical_clustering(data):
    """ Custom implementation of Hierarchical Agglomerative Clustering (HAC) """

    # Initialize clusters (each point is its own cluster)
    clusters = {i: [i] for i in range(len(data))}  # Dictionary to track cluster indices
    distances = {}

    # Compute initial distances between all pairs
    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            distances[(i, j)] = euclidean_distance(data[i], data[j])

    linkage_matrix = []
    
    while len(clusters) > 1:
        # Find the two closest clusters
        min_pair = min(distances, key=distances.get)
        cluster1, cluster2 = min_pair

        # Merge clusters
        new_cluster = clusters[cluster1] + clusters[cluster2]
        new_cluster_idx = max(clusters.keys()) + 1  # Assign new cluster index

        # Update linkage matrix
        linkage_matrix.append([cluster1, cluster2, distances[min_pair], len(new_cluster)])

        # Remove old clusters and update
        del clusters[cluster1]
        del clusters[cluster2]
        clusters[new_cluster_idx] = new_cluster

        # Update distances (single-linkage method)
        new_distances = {}
        for i in clusters:
            if i != new_cluster_idx:
                min_dist = np.mean([euclidean_distance(data[p1], data[p2]) 
                                    for p1 in new_cluster for p2 in clusters[i]])
                new_distances[(i, new_cluster_idx)] = min_dist

        distances = new_distances  # Replace old distances

    return np.array(linkage_matrix)

def plot_dendrogram(linkage_matrix):
    """ Generate a dendrogram from the linkage matrix """
    plt.figure(figsize=(8, 5))
    dendrogram(linkage_matrix)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.savefig("static/dendrogram.png")  # Save for frontend
    plt.close()

def run_clustering():
    """ Load data, perform clustering, and generate dendrogram """
    data = load_data()
    linkage_matrix = hierarchical_clustering(data)
    plot_dendrogram(linkage_matrix)
    return "Clustering completed"
