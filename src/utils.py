import pandas as pd
import numpy as np

from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def load_data(kind="processed"):
  """
  Load the data from the data folder.
  args:
    kind: "raw" or "processed"
  """
  if kind == "raw":
    df = pd.read_csv('data/raw/jobs.csv', sep=';')
  elif kind == "processed":
    df = pd.read_csv('data/processed/cleaned_jobs.csv', sep=';')
  elif kind == "ground_truth":
    df = pd.read_csv('clusters/ground_truth_gpt.csv')
  elif kind == "skills":
    df = pd.read_csv('extracted_skills/skills_extracted_gpt3.csv')
  return df


def is_english(text):
  try:
    return detect(text) == 'en'
  except:
    return False


def apply_kmeans(tfidf_matrix, k=5):
  kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
  return kmeans.fit_predict(tfidf_matrix.toarray())


def words2sentence(word_list):
  return " ".join(word_list)


def apply_tftidf(data):
  vectorizer = TfidfVectorizer()
  return vectorizer.fit_transform(data)


def visualize_cluster(data,
                      cluster,
                      reduce_dim=True,
                      savefig=False,
                      filename="cluster.png",
                      name="Cluster method"):
  """
  Visualize the clusters
  Data: 2d numpy array of the individual data points that were used for clustering
  cluster: 1d numpy array of the cluster labels
  reduced_dim: Boolean, if True, perform pca to 2 dimensions
  """

  if reduce_dim:
    pca = PCA(n_components=2)
    data = pca.fit_transform(data)

  plt.figure(figsize=(10, 6))
  plt.scatter(data[:, 0], data[:, 1], c=cluster,
              cmap='tab20', edgecolor='black', alpha=0.7, s=100)
  plt.title(name, fontsize=16, fontweight='bold')
  plt.xlabel("PCA 1", fontsize=14)
  plt.ylabel("PCA 2", fontsize=14)
  plt.colorbar()
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  if savefig:
    plt.savefig(f"figures/{filename}")
  plt.show()


def visualize_ground_truth(gt, savefig=False, filename="ground_truth.png"):
  plt.figure(figsize=(10, 6))
  plt.bar(gt["category"].value_counts().index,
          gt["category"].value_counts().values, color='dodgerblue')

  plt.xticks(rotation=75)
  plt.title("Ground truth distribution", fontsize=16, fontweight='bold')
  plt.xlabel("Category", fontsize=14)
  plt.ylabel("Count", fontsize=14)
  plt.grid(True, linestyle='--', alpha=0.5)
  plt.tight_layout()
  if savefig:
    plt.savefig(f"figures/{filename}")
  plt.show()


def skill_cleanup(data):

  # skills is a list of strings, connect them into one string

  data["skills_string"] = data["skills"].apply(lambda x: ' '.join(x))

  print(data.head())
  return data
