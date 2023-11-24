from similarity import (
    find_sim,
    louvain_cluster,
    kmean_cluster,
    convert_matrix)
from utils import load_data
from logger import working_on


def similarity_cluster(data, save_clusters=True, q=2, seeds=100, n_clusters=20):
  """
  data : pandas dataframe (cleaned jobs)
  save_clusters : Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  q : number of singles ( k = 2 or 3 for small documents such as emails)
  seeds : number of seeds to generate
  n_clusters : number of clusters to generate
  """

  df = data

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

  working_on("Clustering based on community discovery ...")
  cluster_graph, dbi_graph = louvain_cluster(N, scores)
  df['cluster_graph'] = cluster_graph

  working_on("Clustering based on kmean ...")
  cluster_kmean, dbi_kmean = kmean_cluster(N, scores, n_clusters=n_clusters)
  df['cluster_kmean'] = cluster_kmean

  graph_clusters = df[["id", "cluster_graph"]]
  graph_clusters = graph_clusters.rename(columns={"cluster_graph": "cluster"})

  kmean_clusters = df[["id", "cluster_kmean"]]
  kmean_clusters = kmean_clusters.rename(columns={"cluster_kmean": "cluster"})

  if save_clusters:
    working_on("Saving clusters ...")
    graph_clusters.to_csv(
        "clusters/sim_community_discovery_clusters.csv", index=False)
    kmean_clusters.to_csv("clusters/sim_kmeans_clusters.csv", index=False)

  sim_matrix = convert_matrix(N, scores)

  return graph_clusters, kmean_clusters, sim_matrix


if __name__ == "__main__":
  data = load_data(kind="processed")
  similarity_cluster(data, save_clusters=True, q=2, seeds=100, n_clusters=20)
