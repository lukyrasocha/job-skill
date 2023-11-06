from utils import (
    load_data,
    apply_tftidf,
    apply_dbscan,
    apply_kmeans,
    apply_pca,
    visualize,
    save_csv,
)
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import re
import ast


# 2D or 3D
PCA_COMPONENTS = 3
# K-MEANS or DBSCAN
METHOD = "K-MEANS"


def choose_best_k_based_on_davies_bouldin_index_tfidf(tfidf_matrix, k_max=30):
    """
    Determines the optimal number of
    clusters (k) for a given TF-IDF matrix using the Davies-Bouldin index.

    Args:
      tfidf_matrix: The tfidf_matrix is a matrix representation of the TF-IDF (Term Frequency-Inverse
    Document Frequency) values of a set of documents. It is typically a sparse matrix where each row
    represents a document and each column represents a term, with the cell values indicating the TF-IDF
    score for that term
      k_max: The parameter `k_max` represents the maximum number of clusters to consider. In the given
    code, it is set to 30, which means the function will iterate over values of `k` from 2 to 30
    (inclusive) to find the best value of `k` based on the. Defaults to 30

    Returns:
      the best value of k (number of clusters) and the corresponding Davies-Bouldin index.
    """

    best_k = None
    best_dbi = float("inf")  # Initialize with a high value

    for k in range(2, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)

        dbi = davies_bouldin_score(tfidf_matrix.toarray(), cluster_labels)

        if dbi < best_dbi:
            best_k = k
            best_dbi = dbi

        print(k, dbi)
    return best_k, best_dbi


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


def words_to_sentence(word_list):
    return " ".join(word_list)


def main():

    data = load_data(kind="processed")

    # Apply the function to the 'words' column of the DataFrame
    data["description"] = data["description"].apply(
        lambda x: remove_words_with_numbers(x)
    )
    data["description"] = data["description"].apply(words_to_sentence)
    tfidf_matrix = apply_tftidf(data["description"])

    # TODO try latent semantic indexing (LSI) for reduction of dimensions

    if METHOD == "DBSCAN":
        data["cluster"] = apply_dbscan(tfidf_matrix)
    elif METHOD == "K-MEANS":
        # best_k = find_best_k(tfidf_matrix, max_k=30)
        data["cluster"] = apply_kmeans(tfidf_matrix, k_max=20)
    else:
        print("Method unavailable")

    df_sorted_by_cluster = data[
        ["title", "function", "industries", "cluster"]
    ].sort_values(by="cluster", ascending=True)
    df_sorted_by_cluster.to_csv("./csv_files/text_clustering.csv", index=False)


    # Saving below df is for later comparison of text and feature clustering.
    df_id_and_cluster = data[["id", "cluster"]].sort_values(
        by="cluster", ascending=True
    )
    df_id_and_cluster.to_csv("./csv_files/text_clustering_id.csv", index=False)

    # Dimensionality reduction using PCA
    # tfidf_reduced = apply_pca(data = tfidf_matrix, n_components=PCA_COMPONENTS)

    # visualize(data['cluster'], tfidf_reduced, method = METHOD, n_components=PCA_COMPONENTS)
    # df_sorted_by_cluster  = data[['title','function','industries','cluster']].sort_values(by='cluster', ascending=True)
    # save_csv(df_sorted_by_cluster, method = METHOD, n_components=PCA_COMPONENTS)


if __name__ == "__main__":
    main()
