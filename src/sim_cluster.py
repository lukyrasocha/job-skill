from networkx.algorithms import community
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
import numpy as np

def points_graph(num_nodes, scores, threshold):
    # Create a graph
    G = nx.Graph()

    # Add nodes (text points)
    G.add_nodes_from(range(num_nodes))

    # Add edges based on similarity scores (you can adjust the threshold)
    for idx, idy in scores:
        if scores[(idx,idy)] > threshold :
            G.add_edge(idx, idy)
    return G

def girvan_cluster(G):
    # Use Girvan-Newman algorithm to detect communities
    comp = community.girvan_newman(G)

    # Convert the communities to a list for easier access
    communities = list(tuple(c) for c in next(comp))

    # Create a DataFrame for the communities
    clusters = {}
    for i, node in enumerate(communities):
        clusters[i] = node
    
    return clusters

def sim_DBSAN(N, scores):
    similarity_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                similarity_matrix[i][j] = 1.0
            elif i > j:
                similarity_matrix[i][j] = scores[(j, i)]
            else:
                similarity_matrix[i][j] = scores[(i, j)]

    # DBSCAN clustering
    epsilon = 0.2  # Adjust as needed
    min_samples = 2  # Adjust as needed
    dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric="precomputed")
    labels = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance

    # Retrieve the cluster assignments
    clusters = {}
    for i, node in zip(labels, range(N)):
        if i in clusters:
            clusters[i].append(node)
        else:
            clusters[i] = [node]

    return clusters

def sim_AHC(N, scores):
    similarity_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                similarity_matrix[i][j] = 1.0
            elif i > j:
                similarity_matrix[i][j] = scores[(j, i)]
            else:
                similarity_matrix[i][j] = scores[(i, j)]

    # Convert similarity values to a condensed distance matrix
    distance_matrix = 1 - similarity_matrix
    condensed_distance = squareform(distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method='average')  # Adjust method as needed

    # Define a threshold to cut the dendrogram into clusters
    threshold = 0.2  # Adjust as needed
    labels = fcluster(linkage_matrix, threshold, criterion='distance')

    # Create clusters and print the results
    clusters = {}
    for label, idx in zip(labels, range(N)):
        if label in clusters:
            clusters[label].append(idx)
        else:
            clusters[label] = [idx]
    return clusters