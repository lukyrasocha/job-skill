import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from utils import (
    find_best_k,
    apply_dbscan,
    apply_kmeans,
    load_data,
    combine_text,
    choose_best_k_based_on_davies_bouldin_index,
)
from gensim.models import Word2Vec
from preprocessing import text_preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN


def preprocess_and_tokenize(text):
    """
    Preprocesses and tokenizes a given text by removing non-alphabetic characters,
    converting tokens to lowercase, removing stop words, and lemmatizing the tokens.

    Args:
      text: The `text` parameter is the input text that you want to preprocess and tokenize.

    Returns:
      a list of preprocessed and tokenized words from the input text.
    """
    tokens = word_tokenize(text)

    tokens = [token.lower() for token in tokens if token.isalpha()]

    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    return tokens


def map_to_general_category(industry, mapping):
    """
    Takes an industry and a mapping of categories to industries,
    and returns the general categories that the industry belongs to.

    Args:
      industry: The industry parameter is the specific industry that you want to map to a general
    category. It could be a string representing the industry name.
      mapping: The `mapping` parameter is a dictionary where the keys are general categories and the
    values are lists of industries that fall under each category.

    Returns:
      a string that represents the general category of the given industry. If the industry is found in
    the mapping, the function returns a comma-separated string of the categories that the industry
    belongs to. If the industry is not found in the mapping, the function returns the string 'Other'.
    """
    categories = set()  # Use a set to keep unique categories
    for category, industries in mapping.items():
        if industry in industries:
            categories.add(category)
    return ", ".join(categories) if categories else "Other"


def main():

    df = load_data(kind="processed")
    breakpoint()
    data = df[["id", "title", "function", "industries"]].fillna("")

    # Removing outliers (where industries is whole description of offer)
    data["industries_length"] = data["industries"].str.split().apply(len)
    condition = data["industries_length"] > 15
    data = data[~condition]

    data["industries"] = data["industries"].str.replace(" and ", ",")
    data["function"] = data["function"].str.replace(" and ", ",")
    data["industries"] = data["industries"].str.replace("/", ",")
    data["function"] = data["function"].str.replace("/", ",")

    data["industries"] = data["industries"].str.replace(r",,|, ,", ",")
    data["function"] = data["function"].str.replace(r",,|, ,", ",")

    # data["industries"] = data["industries"].str.replace(r'\s*,\s*', ',', regex=True)
    # data["function"] = data["function"].str.replace(r'\s*,\s*', ',', regex=True)

    with open("industries.yaml", "r") as yaml_file:
        industry_mapping = yaml.safe_load(yaml_file)
    with open("functions.yaml", "r") as yaml_file:
        function_mapping = yaml.safe_load(yaml_file)

    industry_categories = list(industry_mapping.keys())
    function_categories = list(function_mapping.keys())

    data["industries"] = data["industries"].apply(
        lambda x: ",".join([s.strip() for s in x.split(",")])
    )
    data["industry_group"] = data["industries"].apply(
        lambda x: ", ".join(
            map_to_general_category(ind, industry_mapping) for ind in x.split(",")
        )
    )
    data["industry_group"] = data["industry_group"].apply(
        lambda x: ", ".join(sorted(set(x.split(", "))))
    )

    data["function"] = data["function"].apply(
        lambda x: ",".join([s.strip() for s in x.split(",")])
    )
    data["function_group"] = data["function"].apply(
        lambda x: ", ".join(
            map_to_general_category(ind, function_mapping) for ind in x.split(",")
        )
    )
    data["function_group"] = data["function_group"].apply(
        lambda x: ", ".join(sorted(set(x.split(", "))))
    )

    # Create a new column for each general category and initialize with 0
    for category in industry_categories:
        data[category] = 0

    for category in function_categories:
        data[category] = 0

    # Iterate through the 'industry_group' column and set corresponding columns to 1
    for idx, row in data.iterrows():
        industry_groups = row["industry_group"].split(", ")
        function_groups = row["function_group"].split(", ")

        for category in industry_groups:
            if category in industry_categories:
                data.at[idx, category] = 1

        for category in function_groups:
            if category in function_categories:
                data.at[idx, category] = 1

    industries_and_functions = [
        "Technology and Information",
        "Manufacturing",
        "Financial and Business Services",
        "Transportation and Logistics",
        "Healthcare and Pharmaceuticals",
        "Retail and Consumer Goods",
        "Education and Non-profit",
        "Real Estate and Construction",
        "Energy and Environment",
        "Aerospace and Defense",
        "Food and Beverage",
        "Services and Miscellaneous",
        "Management and Leadership",
        "Manufacturing and Engineering",
        "Information Technology",
        "Sales and Marketing",
        "Administrative and Support",
        "Writing, Editing, and Creative",
        "Customer Service",
        "Legal and Finance",
        "Research and Analysis",
        "Human Resources and People Management",
        "Purchasing and Supply Chain",
        "Healthcare and Science",
        "Education and Training",
    ]

    # Example usage:
    # best_k, best_dbi = choose_best_k_based_on_davies_bouldin_index(data, industries_and_functions)
    # print(f"The best K value based on Davies-Bouldin Index is {best_k} with a DBI of {best_dbi}")

    num_clusters = 20
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    data["cluster"] = kmeans.fit_predict(data[industries_and_functions])

    df_sorted_by_cluster = data[
        [
            "title",
            "function",
            "function_group",
            "industries",
            "industry_group",
            "cluster",
        ]
    ].sort_values(by="cluster", ascending=True)
    df_sorted_by_cluster.to_csv("./csv_files/feature_clustering.csv", index=False)
    data.to_csv("./csv_files/csv_with_one_hot.csv", index=False)

    df_id_and_cluster = data[["id", "cluster"]].sort_values(
        by="cluster", ascending=True
    )
    df_id_and_cluster.to_csv("./csv_files/feature_clustering_id.csv", index=False)

    # pca = PCA(n_components=2)
    # subset_data_pca = pca.fit_transform(one_hot_scaled)

    # # Plot the data in the second dimension
    # plt.scatter(subset_data_pca[:, 0], subset_data_pca[:, 1], s=5, alpha=0.5)
    # plt.title('Data in the Second Dimension')
    # plt.xlabel('Principal Component 1')
    # plt.ylabel('Principal Component 2')
    # plt.show()


if __name__ == "__main__":
    main()
