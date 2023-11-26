"""
EVALUATION SCRIPT
This script is the main entrypoint for all of our experiments evaluation. 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import rand_score
from logger import success, working_on, winner, error


def load_clustering_methods(paths):
  """Loads multiple clusterings from specified file paths."""
  clustering_method = {}
  for name, path in paths.items():
    clustering_method[name] = pd.read_csv(path)
  return clustering_method


def compare_clusters_nmi(clusters):
  """Compares multiple sets of clusters using Normalized Mutual Information (NMI)."""
  nmi_matrix = pd.DataFrame(index=clusters.keys(), columns=clusters.keys())
  for name1, data1 in clusters.items():
    for name2, data2 in clusters.items():
      if name1 != name2:
        merged_data = pd.merge(data1, data2, on='id', suffixes=('_1', '_2'))
        nmi_score = normalized_mutual_info_score(
            merged_data['cluster_1'], merged_data['cluster_2'])
        nmi_matrix.loc[name1, name2] = nmi_score
      else:
        nmi_matrix.loc[name1, name2] = 1.0  # Same clustering method comparison
  return nmi_matrix


def compare_clusters_rand_index(clusters):
  """Compares multiple sets of clusters using Rand Index."""
  rand_matrix = pd.DataFrame(index=clusters.keys(), columns=clusters.keys())
  for name1, data1 in clusters.items():
    for name2, data2 in clusters.items():
      if name1 != name2:
        merged_data = pd.merge(data1, data2, on='id', suffixes=('_1', '_2'))
        rand = rand_score(
            merged_data["cluster_1"], merged_data["cluster_2"])
        rand_matrix.loc[name1, name2] = rand
      else:
        # Same clustering method comparison
        rand_matrix.loc[name1, name2] = 1.0
  return rand_matrix


def evaluation():
  paths = {
      # 'ground_truth': 'clusters/ground_truth_onehot.csv',
      # 'ground_truth': 'clusters/ground_truth_keywords.csv',
      'ground_truth_gpt': 'clusters/ground_truth_gpt.csv',
      'word2vec': 'clusters/word2vec_clusters.csv',
      'tfidf_text': 'clusters/tf_idf_clusters_job_desc.csv',
      'tfidf_industries': "clusters/tfidf_industries_and_functions_clusters.csv",
      'tfidf_nouns': 'clusters/tfidf_noun_clusters.csv',
      'tfidf_adj': 'clusters/tfidf_adj_clusters.csv',
      'tfidf_verbs': 'clusters/tfidf_verb_clusters.csv',
      'similarity_community_disc': 'clusters/sim_community_discovery_clusters.csv',
      'similarity_kmeans': 'clusters/sim_kmeans_clusters.csv',
      'doc2vec_gmm': 'clusters/doc2vec_gmm_clusters.csv',
      'doc2vec_kmeans': 'clusters/doc2vec_kmeans_clusters.csv',
  }

  # If path does not exist throw error
  for name, path in paths.items():
    try:
      with open(path, 'r'):
        pass
    except FileNotFoundError:
      error(f"File {path} not found!")
      FileNotFoundError(f"File {path} not found!")

  # Load the datasets
  working_on("Comparing clusters ...")
  cluster_methods = load_clustering_methods(paths)

  # Compare the clusters and get NMI matrix
  nmi_matrix = compare_clusters_nmi(cluster_methods)
  rand_index_matrix = compare_clusters_rand_index(cluster_methods)

  success("Normalized Mutual Information matrix:")
  # Dataframe to string
  print(nmi_matrix.to_string(index=False))

  success("Rand Index matrix:")
  print(rand_index_matrix.to_string(index=False))

  ground_truth_nmi = nmi_matrix['ground_truth_gpt'].drop('ground_truth_gpt')
  ground_truth_rand = rand_index_matrix['ground_truth_gpt'].drop(
      'ground_truth_gpt')

  # Select the best clustering method based on NMI and Rand Index
  best_nmi = ground_truth_nmi.idxmax()
  best_rand = ground_truth_rand.idxmax()

  winner(f"Best clustering method based on NMI: {best_nmi}")
  print(f"NMI SCORE: {round(ground_truth_nmi[best_nmi],3)}")

  winner(f"Best clustering method based on Rand Index: {best_rand}")
  print(f"RAND SCORE: {round(ground_truth_rand[best_rand],3)}")

  # Plot the NMI and Rand Index

  # Plotting NMI for each clustering method
  plt.figure(figsize=(10, 5))
  plt.bar(ground_truth_nmi.index, ground_truth_nmi.values, color='dodgerblue')
  plt.xticks(rotation=25)
  plt.title("NMI for each method compared to ground truth",
            fontsize=14, fontweight='bold')
  plt.xlabel("Clustering method", fontsize=12)
  plt.ylabel("NMI", fontsize=12)
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()

  # Save the first plot
  plt.savefig('figures/nmi_plot.png')
  plt.close()

  # Plotting Rand Index for each clustering method
  plt.figure(figsize=(10, 5))
  plt.bar(ground_truth_rand.index,
          ground_truth_rand.values, color='mediumslateblue')
  plt.xticks(rotation=25)
  plt.title("Rand Index for each method compared to ground truth",
            fontsize=14, fontweight='bold')
  plt.xlabel("Clustering method", fontsize=12)
  plt.ylabel("Rand Index", fontsize=12)
  plt.grid(axis='y', linestyle='--', alpha=0.7)
  plt.tight_layout()

  # Save the second plot
  plt.savefig('figures/rand_plot.png')
  plt.close()


if __name__ == "__main__":
  evaluation()
