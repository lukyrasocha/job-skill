import pandas as pd
from similarity import (
    find_sim,
    louvain_cluster,
    kmean_cluster)
from utils import load_data
from logger import working_on


def main():

  # Load the data
  df = load_data(kind="processed")

  # Number of jobs
  N = len(df)
  # Give q & seeds for hash to find similarity for each job's descriptions
  # q = number of singles ( k = 2 or 3 for small documents such as emails)
  q = 2
  seeds = 100
  working_on("Finding similarity ...")
  scores = find_sim(df['description'], q, seeds)

  # Plot the network based on similarity and find community based on graph
  # To evaluate the functionaly of cluster, calculate the dbi(The minimum
  # score is zero, with lower values indicating better clustering)
  # and measure rand index between feature label ground truth and prediction
  # (similarity score between 0.0 and 1.0, inclusive, 1.0 stands for perfect match)

  working_on("Clustering based on community discovery...")
  cluster_graph, dbi_graph = louvain_cluster(N, scores)
  df['cluster_graph'] = cluster_graph

  working_on("Clustering based on kmean...")
  cluster_kmean, dbi_kmean = kmean_cluster(N, scores)
  df['cluster_kmean'] = cluster_kmean

  working_on("Saving clusters ...")
  graph_clusters = df[["id", "cluster_graph"]]
  graph_clusters = graph_clusters.rename(columns={"cluster_graph": "cluster"})

  kmean_clusters = df[["id", "cluster_kmean"]]
  kmean_clusters = kmean_clusters.rename(columns={"cluster_kmean": "cluster"})

  graph_clusters.to_csv(
      "clusters/sim_community_discovery_clusters.csv", index=False)
  kmean_clusters.to_csv("clusters/sim_kmeans_clusters.csv", index=False)


if __name__ == "__main__":
  main()
