from utils import load_data, apply_tftidf, apply_dbscan, apply_kmeans, apply_pca, visualize, save_csv

# 2D or 3D
PCA_COMPONENTS = 3

# K-MEANS or DBSCAN
METHOD = 'K-MEANS'


def main():

    data = load_data(kind="processed")
    # breakpoint()
    tfidf_matrix = apply_tftidf(data['description'])

    #TODO try latent semantic indexing (LSI) for reduction of dimensions

    if METHOD=='DBSCAN':
        data['cluster'] = apply_dbscan(tfidf_matrix)
    elif METHOD=='K-MEANS':
        # best_k = find_best_k(tfidf_matrix, max_k=30)
        data['cluster'] = apply_kmeans(tfidf_matrix, k_max=14)
    else:
        print("Method unavailable")

    # Dimensionality reduction using PCA
    tfidf_reduced = apply_pca(data = tfidf_matrix, n_components=PCA_COMPONENTS)

    visualize(data['cluster'], tfidf_reduced, method = METHOD, n_components=PCA_COMPONENTS)
    # df_sorted_by_cluster  = data[['title','function','industries','cluster']].sort_values(by='cluster', ascending=True)
    # save_csv(df_sorted_by_cluster, method = METHOD, n_components=PCA_COMPONENTS)

if __name__ == "__main__":
    main()
