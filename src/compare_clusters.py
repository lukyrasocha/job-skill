

import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

def load_datasets(paths):
    """Loads multiple datasets from specified file paths."""
    datasets = {}
    for name, path in paths.items():
        datasets[name] = pd.read_csv(path)
    return datasets

def compare_clusters(datasets):
    """Compares multiple sets of clusters using Normalized Mutual Information (NMI)."""
    nmi_matrix = pd.DataFrame(index=datasets.keys(), columns=datasets.keys())
    for name1, data1 in datasets.items():
        for name2, data2 in datasets.items():
            if name1 != name2:
                merged_data = pd.merge(data1, data2, on='id', suffixes=('_1', '_2'))
                nmi_score = normalized_mutual_info_score(merged_data['cluster_1'], merged_data['cluster_2'])
                nmi_matrix.loc[name1, name2] = nmi_score
            else:
                nmi_matrix.loc[name1, name2] = 1.0  # Same dataset comparison
    return nmi_matrix

def main():
    # File paths for the datasets
    paths = {
        'text_word2vec': 'csv_files/word2wev_clustering_id.csv',
        'text_tfidf': 'csv_files/text_clustering_id.csv',
        'features': 'csv_files/feature_clustering_id.csv'
    }

    # Load the datasets
    datasets = load_datasets(paths)

    # Compare the clusters and get NMI matrix
    nmi_matrix = compare_clusters(datasets)
    print("Normalized Mutual Information matrix:")
    print(nmi_matrix)

if __name__ == "__main__":
    main()



