from gensim.models import Word2Vec
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

from src.helper.utils import load_data
from src.helper.logger import success, working_on


def description_to_vector(model, description):
  # Filter out words not in the model's vocabulary
  valid_words = [word for word in description if word in model.wv.key_to_index]
  if valid_words:
    # Average the vectors of the words in the description
    return np.mean(model.wv[valid_words], axis=0)
  else:
    # If no valid words, return a zero vector
    return np.zeros(model.vector_size)


def word2vec_cluster(data,
                     save_clusters=True,
                     vector_size=100,
                     window=5,
                     min_count=1,
                     workers=4,
                     n_clusters=20):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  vector_size: Dimensionality of the feature vectors
  window: Maximum distance between the current and predicted word within a sentence
  min_count: Ignores all words with total frequency lower than this
  workers: Use these many worker threads to train the model
  n_clusters: Number of clusters to cluster the embeddings into
  """

  ######### Word2Vec #########
  model = Word2Vec(sentences=data['description'],
                   vector_size=vector_size, window=window, min_count=min_count, workers=workers)

  working_on("Creating vectors ...")
  data['vector'] = data['description'].apply(
      lambda x: description_to_vector(model, x))

  vectors = np.array(data['vector'].tolist())

  kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
  data['cluster'] = kmeans.fit_predict(vectors)

  df_id_and_cluster = data[["id", "cluster"]].sort_values(
      by="cluster", ascending=True
  )

  if save_clusters:
    df_id_and_cluster.to_csv("clusters/word2vec_clusters.csv", index=False)

  dbs = round(davies_bouldin_score(vectors, data["cluster"]), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], vectors


if __name__ == "__main__":
  data = load_data(kind="processed")
  word2vec_cluster(data, save_clusters=False)
