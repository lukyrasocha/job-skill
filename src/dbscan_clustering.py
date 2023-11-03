# from utils import tfidf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt

# df = pd.read_csv("data/processed/cleaned_jobs_without_lemm_and_punct.csv", sep=';')
df = pd.read_csv("data/processed/cleaned_jobs.csv", sep=';')
documents = df['description']

# Initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the documents
tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

# Get the feature names (words in the vocabulary)
feature_names = tfidf_vectorizer.get_feature_names_out()

# Convert the TF-IDF matrix to a dense array
tfidf_matrix_array = tfidf_matrix.toarray()

tfidf_df = pd.DataFrame(tfidf_matrix_array, columns=feature_names)

# Print the TF-IDF DataFrame
# print(tfidf_df)

# Specify the number of clusters (k)
# max_k = 15  # Maximum number of clusters to consider
# best_score = -1
# best_k = 0

# for k in range(2, max_k + 1):
#     kmeans = KMeans(n_clusters=k)
#     cluster_labels = kmeans.fit_predict(tfidf_matrix)
#     score = silhouette_score(tfidf_matrix, cluster_labels)
    
#     if score > best_score:
#         best_score = score
#         best_k = k

# print(f"Best number of clusters (k): {best_k}")

n_components = 2  # Number of components for 2D visualization
svd = TruncatedSVD(n_components=n_components)
tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# Specify the number of clusters (k)
max_k = 10  # Maximum number of clusters to consider
best_score = -1
best_silhouette_k = 0

for k in range(2, max_k + 1):
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(tfidf_matrix_reduced)
    score = silhouette_score(tfidf_matrix_reduced, cluster_labels)
    
    if score > best_score:
        best_score = score
        best_silhouette_k = k

print(f"Best number of clusters (k) according to silhouette score: {best_silhouette_k}")


# distortions = []  # To store the values of the distortions (inertia)
# K = range(1, 10)  # Try different values of k

# for k in K:
#     kmeanModel = KMeans(n_clusters=k)
#     kmeanModel.fit(tfidf_matrix_reduced)
#     distortions.append(kmeanModel.inertia_)

# # Plot the Elbow Method graph
# plt.figure(figsize=(8, 6))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k (Number of Clusters)')
# plt.ylabel('Distortion (Inertia)')
# plt.title('The Elbow Method for Optimal k')
# plt.show()


# k = 5
kmeans = KMeans(n_clusters=best_silhouette_k, random_state=0)

# # Fit the TF-IDF matrix to the clustering algorithm
clusters = kmeans.fit_predict(tfidf_matrix)


# n_components = 2  # Number of components for 2D visualization
# svd = TruncatedSVD(n_components=n_components)
# tfidf_matrix_reduced = svd.fit_transform(tfidf_matrix)

# Scatter plot the clusters
plt.figure(figsize=(8, 6))
for cluster in range(best_silhouette_k):
    plt.scatter(
        tfidf_matrix_reduced[clusters == cluster, 0],
        tfidf_matrix_reduced[clusters == cluster, 1],
        label=f'Cluster {cluster}',
    )

plt.title('TF-IDF Clustering Visualization')
plt.xlabel(f'Component 1 (Explained Variance: {svd.explained_variance_ratio_[0]:.2f})')
plt.ylabel(f'Component 2 (Explained Variance: {svd.explained_variance_ratio_[1]:.2f})')
plt.legend()
plt.savefig('./figures/cluster_k_means_cleaned_jobs.png')
# plt.savefig('./figures/cluster_k_means_cleaned_jobs_without_lemm_and_punct.png')
