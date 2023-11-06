import pandas as pd
import numpy as np

from langdetect import detect
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score

import matplotlib.pyplot as plt


def load_data(kind="processed"):
    """
    Load the data from the data folder.
    args:
      kind: "raw" or "processed"
    """
    if kind == "raw":
        df = pd.read_csv("data/raw/jobs.csv", sep=";")
    elif kind == "processed":
        df = pd.read_csv("data/processed/cleaned_jobs.csv", sep=";")
    return df


def is_english(text):
    try:
        return detect(text) == "en"
    except:
        return False


def find_best_k(data, max_k):
    best_score = -1
    best_k = 0

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k)
        cluster_labels = kmeans.fit_predict(data)
        score = silhouette_score(data, cluster_labels)

        if score > best_score:
            best_score = score
            best_k = k
    print(f"Best k: {best_k}")
    return best_k


def choose_best_k_based_on_davies_bouldin_index(
    data, columns_to_predict, k_range=(2, 30)
):
    best_k = None
    best_dbi = float("inf")

    for k in range(k_range[0], k_range[1] + 1):
        kmeans = KMeans(n_clusters=k, random_state=0)
        data["cluster"] = kmeans.fit_predict(data[columns_to_predict])

        dbi = davies_bouldin_score(data[columns_to_predict], data["cluster"])

        if dbi < best_dbi:
            best_k = k
            best_dbi = dbi

    return best_k, best_dbi


def apply_dbscan(tfidf_matrix, eps=0.5, min_samples=2):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(tfidf_matrix.toarray())


def apply_kmeans(tfidf_matrix, k_max=5):
    kmeans = KMeans(n_clusters=k_max, random_state=0)
    return kmeans.fit_predict(tfidf_matrix.toarray())


def apply_tftidf(data):
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(data)


def apply_word2vec(data):
    model = Word2Vec(data, vector_size=100, window=5, min_count=1, sg=0)

    # Generate word embeddings for each document by averaging the embeddings of all words in the document
    embeddings = []
    for row in data:
        doc_vector = np.mean([model.wv[word] for word in row], axis=0)
        embeddings.append(doc_vector)

    return np.array(embeddings)


def apply_pca(data, n_components):
    pca = PCA(n_components)
    return pca.fit_transform(data.toarray())


def save_unique_entities(data, column_name):
    file_name = f"unique_{column_name}.txt"
    data[column_name] = (
        data[column_name]
        .astype(str)
        .apply(preprocess_and_tokenize)
        .apply(lambda token_list: " ".join(token_list))
    )
    data_array = data[column_name].unique()
    with open(file_name, "w") as file:
        # Write each element of the array as a line in the text file
        for item in data_array:
            file.write(f"{item}\n")


def combine_text(row):
    return " ".join(str(val) for val in row)


def visualize(data, tfidf_reduced, method, n_components):
    # Plotting in 2D
    fig = plt.figure(figsize=(8, 6))

    unique_clusters = data.unique()
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_clusters)))

    if n_components == 3:
        ax = fig.add_subplot(111, projection="3d")
        for cluster_id, color in zip(unique_clusters, colors):
            cluster_data = tfidf_reduced[data == cluster_id]
            ax.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                cluster_data[:, 2],
                c=color,
                label=f"Cluster {cluster_id}",
            )
        ax.set_title(f"{method} Clustering Visualization in 3D")
    else:
        for cluster_id, color in zip(unique_clusters, colors):
            cluster_data = tfidf_reduced[data == cluster_id]
            plt.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                c=color,
                label=f"Cluster {cluster_id}",
            )
        plt.title(f"{method} Clustering Visualization in 2D")

    plt.legend()
    plt.show()
    plt.savefig(f"./figures/{method}_clustering_{n_components}_components.png")


def save_csv(data, method, n_components):
    file_name = f"./csv_files/{method}_clustering_{n_components}_components.csv"
    data.to_csv(file_name, index=False)
