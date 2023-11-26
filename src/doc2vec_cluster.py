from src.doc2vec import Doc2VecWrapper
from src.utils import load_data
from src.logger import working_on, success
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

import numpy as np


def doc2vec_cluster(data,
                    save_clusters=True,
                    method="kmeans",
                    vector_size=100,
                    alpha=0.025,
                    min_alpha=0.00025,
                    min_count=10,
                    epochs=500,
                    n_clusters=20):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  vector_size: Dimensionality of the feature vectors
  alpha: The initial learning rate
  min_alpha: Learning rate will linearly drop to min_alpha as training progresses
  min_count: Ignores all words with total frequency lower than this
  epochs: Number of iterations (epochs) over the corpus
  n_clusters: Number of clusters to cluster the embeddings into
  """

  jobs_descriptions = data['description'].tolist()

  working_on("Training doc2vec ...")
  doc2vec = Doc2VecWrapper()
  doc2vec.init(vector_size=vector_size, alpha=alpha,
               min_alpha=min_alpha, min_count=min_count, epochs=epochs)
  doc2vec.fit(jobs_descriptions)
  doc2vec.train()

  # Cluster similar documents
  vectors = [doc2vec.model.dv[i] for i in range(len(doc2vec.model.dv))]
  vectors = np.array(vectors)

  # Put the embeddings into the original dataframe
  data['embeddings'] = vectors.tolist()

  # Cluster the embeddings

  if method == "kmeans":
    kmeans = KMeans(n_clusters=n_clusters, random_state=42,
                    n_init=10).fit(data['embeddings'].tolist())
    data['cluster'] = kmeans.labels_.tolist()
  elif method == "gmm":
    gmm = GaussianMixture(n_components=n_clusters, random_state=42).fit(
        data['embeddings'].tolist())
    data['cluster'] = gmm.predict(data['embeddings'].tolist())

  if save_clusters:
    data[["id", "cluster"]].to_csv(
        f"clusters/doc2vec_{method}_clusters.csv", index=False)
    success(
        f"Clusters saved to clusters/doc2vec_{method}_clusters.csv")

  return data[["id", "cluster"]], np.array(data['embeddings'].tolist())


if __name__ == "__main__":
  data = load_data(kind="processed")
  doc2vec_cluster(data, save_clusters=False, method="gmm")
