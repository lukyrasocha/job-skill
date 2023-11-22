import ast
from sklearn.metrics import davies_bouldin_score

from utils import (
    load_data,
    apply_tftidf,
    apply_kmeans,
    words2sentence,
)
from logger import success


def TFIDF_cluster(data, save_clusters=True):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  """

  data["description"] = data["description"].apply(words2sentence)
  tfidf_matrix = apply_tftidf(data["description"])

  data["cluster"] = apply_kmeans(tfidf_matrix, k=20)

  if save_clusters:
    data[["id", "cluster"]].to_csv("clusters/tfidf_clusters.csv", index=False)

  dbs = round(davies_bouldin_score(tfidf_matrix.toarray(), data["cluster"]), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], tfidf_matrix.toarray()

def TFIDF_industries_and_functions_cluster(data, save_clusters=False):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  """

  data["industries"] = data['function'] + ', ' + data['industries']
  data['industries'] = data['industries'].str.replace(',,', ',', regex=False)

  tfidf_matrix = apply_tftidf(data["industries"])

  data["cluster"] = apply_kmeans(tfidf_matrix, k=20)

  if save_clusters:
    data[["id", "cluster"]].to_csv("clusters/tfidf_industries_and_functions_clusters.csv", index=False)

  dbs = round(davies_bouldin_score(tfidf_matrix.toarray(), data["cluster"]), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], tfidf_matrix.toarray()


if __name__ == "__main__":
  data = load_data(kind="processed")
  data["description"] = data["description"].apply(ast.literal_eval)
  clusters_text = TFIDF_cluster(data, save=False)
  clusters_industries = TFIDF_industries_and_functions_cluster(data, save_clusters=False)

  print(clusters_text, clusters_industries)
