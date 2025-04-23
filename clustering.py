import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
    plt.tight_layout()
    plt.savefig("static/dendrogram.png")
    plt.close()

def preprocess_data(df):
    # Drop columns with all missing values
    df = df.dropna(axis=1, how='all')
    
    # Drop rows with all missing values
    df = df.dropna(axis=0, how='all')

    # Fill remaining missing values with column mean (numeric) or mode (categorical)
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # Detect numeric and categorical columns
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Create transformer
    transformer = ColumnTransformer(transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

    pipeline = Pipeline(steps=[('transform', transformer)])
    data_transformed = pipeline.fit_transform(df)
    
    return data_transformed

def run_clustering(csv_path):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("The uploaded CSV file is empty or invalid.")
    
    data = preprocess_data(df)
    
    if data.shape[0] < 2:
        raise ValueError("Need at least 2 samples for clustering.")

    linkage_matrix, steps = hierarchical_clustering(data.toarray() if hasattr(data, 'toarray') else data)
    plot_dendrogram(linkage_matrix)
    return linkage_matrix, steps
