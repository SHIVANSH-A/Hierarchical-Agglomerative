import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from io import BytesIO
import base64

def hierarchical_clustering(data):
    steps = []
    intermediate_plots = []

    # Perform hierarchical clustering using scipy
    linkage_matrix = linkage(data, method='ward')  # You can change method

    # Create intermediate dendrograms at each step if needed
    # (not practical for large data, so simulate with a few steps here)
    # For now, just simulate with one intermediate plot
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(linkage_matrix, ax=ax)
    img_path = os.path.join("static", "intermediate1.png")
    plt.savefig(img_path)
    plt.close()
    intermediate_plots.append(img_path)

    steps.append("Performed hierarchical clustering using Ward's method.")
    return linkage_matrix, steps, intermediate_plots

def plot_dendrogram(linkage_matrix):
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix)
    plt.title("Final Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    plt.savefig("static/dendrogram.png")
    plt.close()
