import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast
import re
from similarity import (
    find_sim, 
    louvain_cluster, 
    kmean_cluster,
    clusterings_eval)
from utils import load_data

def remove_words_with_numbers(word_list_str):
    """
    Takes a string representation of a list of words as input,
    removes any special characters from the words, and then removes any words that contain numbers.

    Args:
      word_list_str: A string representation of a list of words.

    Returns:
      The function `remove_words_with_numbers` returns a list of words without any special characters or
    numbers.
    """
    word_list = ast.literal_eval(word_list_str)
    word_list_without_special = [
        re.sub(r"[^a-zA-Z0-9\s]", "", word) for word in word_list
    ]
    word_list_without_numbers = [
        word for word in word_list_without_special if not re.search(r"\d", word)
    ]
    return word_list_without_numbers

def main():

    
    # Load the data
    df  = load_data(kind="processed")
    df["description"] = df["description"].apply(
            lambda x: remove_words_with_numbers(x)
        )
    ground = pd.read_csv('../csv_files/feature_clustering_id.csv', sep=',', header=0, names=['id', 'ground'])
    df = pd.merge(df, ground, on='id')
    
    # Number of jobs
    N = len(df)
    # Give q & seeds for hash to find similarity for each job's descriptions
    # q = number of singles ( k = 2 or 3 for small documents such as emails)
    q = 2
    seeds = 100
    scores = find_sim(df['description'],q,seeds)

    # Plot the network based on similarity and find community based on graph
    # To evaluate the functionaly of cluster, calculate the dbi(The minimum 
    # score is zero, with lower values indicating better clustering)
    # and measure rand index between feature label ground truth and prediction 
    # (similarity score between 0.0 and 1.0, inclusive, 1.0 stands for perfect match)
    cluster_graph, dbi_graph = louvain_cluster(N, scores)
    df['cluster_graph'] = cluster_graph
    cluster_kmean, dbi_kmean  = kmean_cluster(N, scores, 30)
    df['cluster_kmean'] = cluster_kmean

    print("The DBindex value using graph:", dbi_graph , " Rand index comparing ground truth:", clusterings_eval(df['cluster_graph'], df['ground'], "RAND_INDEX"))
    print("The DBindex value using kmean:", dbi_kmean, " Rand index comparing ground truth:", clusterings_eval(df['cluster_kmean'], df['ground'], "RAND_INDEX"))
    df[["id", "cluster_graph", "cluster_kmean", 'ground']].to_csv("../csv_files/similarity.csv", index=False)
    
if __name__ == "__main__":
    main()
