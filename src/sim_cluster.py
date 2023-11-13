from networkx.algorithms import community
from sklearn.cluster import DBSCAN
from sklearn.metrics import davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import networkx as nx
import numpy as np

def girvan_cluster(num_nodes, scores):
    # Create a graph
    G = nx.Graph()

    # Add nodes (text points)
    G.add_nodes_from(range(num_nodes))

    # Add edges based on similarity scores (you can adjust the threshold)
    for idx, idy in scores:
        G.add_edge(idx, idy, weight = scores[(idx,idy)])

    # Use Girvan-Newman algorithm to detect communities
    communities = community.louvain_communities(G)
    
    # Create a DataFrame for the communities
    clusters = {}
    for label, nodes in enumerate(communities):
        for idx in nodes:
            clusters[idx] = label

    # clusters = {}
    # for label, nodes in enumerate(communities):
    #     clusters[label] = nodes
        
    return clusters


def sim_DBSAN(N, scores):
    similarity_matrix = convert_matrix(N,scores)

    labels, dbi  = choose_best_parameter_based_on_davies_bouldin_index_DBSCAN(similarity_matrix)
    print(dbi)

    # Retrieve the cluster assignments
    clusters = {}
    for label, idx in zip(labels, range(N)):
        
        clusters[idx] = label

    return clusters

def choose_best_parameter_based_on_davies_bouldin_index_DBSCAN(similarity_matrix):
    """
    Determines the optimal number of
    clusters for a given similarity matrix using the Davies-Bouldin index.
    The minimum score is zero, with lower values indicating better clustering
    
    Args:
      similarity_matrix
      label

    Returns:
      the best parameter and the corresponding Davies-Bouldin index.
    
    """

    best_dbi = float("inf")  # Initialize with a high value
    best_labels = None
    for min_samples in range(2, 6):
        for epsilon in np.arange(0.01, 0.5, 0.01):
            # DBSCAN clustering
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples, metric="precomputed")
            labels = dbscan.fit_predict(1-similarity_matrix)  # Convert similarity to distance

            dbi = davies_bouldin_score(1-similarity_matrix, labels)

            if dbi < best_dbi:
                best_labels = labels
                best_dbi = dbi

    return best_labels, best_dbi

def sim_AHC(N, scores):
    similarity_matrix = convert_matrix(N,scores)

    # Convert similarity values to a condensed distance matrix
    distance_matrix = 1 - similarity_matrix
    condensed_distance = squareform(distance_matrix)

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_distance, method='average')  # Adjust method as needed

    # Define a threshold to cut the dendrogram into clusters
    threshold = 0.2  # Adjust as needed
    labels = fcluster(linkage_matrix, threshold, criterion='distance')

    # # Create clusters and print the results
    # clusters = {}
    # for label, idx in zip(labels, range(N)):
    #     if label in clusters:
    #         clusters[label].append(idx)
    #     else:
    #         clusters[label] = [idx]
    # Retrieve the cluster assignments
    clusters = {}
    for label, idx in zip(labels, range(N)):
        
        clusters[idx] = label
    return clusters