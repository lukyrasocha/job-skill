
from logger import success, working_on
from utils import (
    load_data,
    apply_tftidf,
    apply_kmeans,
    words2sentence,
)
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import davies_bouldin_score
from nltk.corpus import wordnet
import nltk

# WARNING: Uncomment the following lines if you get an error when running the script
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')


def TFIDF_cluster(data, save_clusters=True, n_clusters=20):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  n_clusters: Number of clusters 
  """

  data["description"] = data["description"].apply(words2sentence)
  tfidf_matrix = apply_tftidf(data["description"])

  data["cluster"] = apply_kmeans(tfidf_matrix, k=n_clusters)

  if save_clusters:
    data[["id", "cluster"]].to_csv(
        "clusters/tfidf_clusters_job_desc.csv", index=False)
    success("Clusters saved to clusters/tfidf_clusters_job_desc.csv")

  dbs = round(davies_bouldin_score(tfidf_matrix.toarray(), data["cluster"]), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], tfidf_matrix.toarray()


def TFIDF_industries_and_functions_cluster(data, save_clusters=False, n_clusters=20):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  n_clusters: Number of clusters
  """

  data["industries"] = data['function'] + ', ' + data['industries']
  data['industries'] = data['industries'].str.replace(',,', ',', regex=False)

  tfidf_matrix = apply_tftidf(data["industries"])

  data["cluster"] = apply_kmeans(tfidf_matrix, k=n_clusters)

  if save_clusters:
    data[["id", "cluster"]].to_csv(
        "clusters/tfidf_industries_and_functions_clusters.csv", index=False)
    success("Clusters saved to clusters/tfidf_industries_and_functions_clusters.csv")

  dbs = round(davies_bouldin_score(tfidf_matrix.toarray(), data["cluster"]), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], tfidf_matrix.toarray()


# Define the pos_tagger function
def pos_tagger(nltk_tag):
  if nltk_tag.startswith('J'):
    return wordnet.ADJ
  elif nltk_tag.startswith('V'):
    return wordnet.VERB
  elif nltk_tag.startswith('N'):
    return wordnet.NOUN
  elif nltk_tag.startswith('R'):
    return wordnet.ADV
  else:
    return None


def get_tfidf_vectors(description_type):
  """
  Get TF-IDF vectors and keywords for each description.

  Parameters:
  - description_type: List of text descriptions.

  Returns:
  - vectors: TF-IDF vectors.
  - all_keywords: List of keywords for each description.
  """
  # create a TF-IDF vectorizer
  vectorizer = TfidfVectorizer()

  # fit and transform the text descriptions using the TF-IDF vectorizer
  vectors = vectorizer.fit_transform(description_type)

  # get the feature names (words) corresponding to the TF-IDF vectors
  feature_names = vectorizer.get_feature_names_out()

  # convert the sparse TF-IDF matrix to a dense representation
  dense = vectors.todense()
  denselist = dense.tolist()

  # initialize a list to store keywords for each description
  all_keywords = []

  # iterate through each description in the dense representation
  for description in denselist:
    # extract keywords (feature names) where the TF-IDF value is greater than 0
    keywords = [feature_names[i]
                for i, word in enumerate(description) if word > 0]

    # append the keywords for the current description to the list
    all_keywords.append(keywords)

  return vectors, all_keywords, vectorizer


def TFIDF_verbs_cluster(data, save_clusters=True, n_clusters=20):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  n_clusters: Number of clusters 
  """

  data = data[['id', 'description']]

  data = data.copy()
  working_on("Extracting verbs from descriptions ...")
  data.loc[:, 'description_verb'] = data['description'].apply(lambda x: [word for word, pos_tag in [(word, pos_tagger(tag[1][0].upper(
  ))) for word, tag in zip(eval(x), nltk.pos_tag(eval(x))) if pos_tagger(tag[1][0].upper()) == wordnet.VERB] if pos_tag is not None])

  # data['description_verb'] = data['description'].apply(lambda x: [word for word, pos_tag in [(word, pos_tagger(tag[1][0].upper(
  # ))) for word, tag in zip(eval(x), nltk.pos_tag(eval(x))) if pos_tagger(tag[1][0].upper()) == wordnet.VERB] if pos_tag is not None])

  description_verb = data['description_verb']

  description_verb_strings = [' '.join(description)
                              for description in description_verb]

  working_on("Clustering verbs ...")
  model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10)

  vectors, all_keywords, vectorizer = get_tfidf_vectors(
      description_verb_strings)

  # Fit the model to the TF-IDF vectors
  model.fit(vectors)

  # Get the indices that sort the cluster centers in descending order
  order_centroids = model.cluster_centers_.argsort()[:, ::-1]

  # Get the feature names (words) from the TF-IDF vectorizer
  terms = vectorizer.get_feature_names_out()

  # Write the top terms for each cluster to a text file
  with open("results/kmeans_verb_results.txt", "w", encoding="utf-8") as f:
    for i in range(n_clusters):
      f.write(f"Cluster {i+1}")
      f.write("\n")
      for ind in order_centroids[i, :10]:
        f.write(' %s' % terms[ind],)
        f.write("\n")
      f.write("\n")
      f.write("\n")

  # Predict cluster assignments for each document in the TF-IDF vectors
  kmean_indicates = model.fit_predict(vectors)

  # Add a 'cluster' column to the 'data' DataFrame
  # data['cluster_verb'] = kmean_indicates
  data.loc[:, 'cluster_verb'] = kmean_indicates

  data = data.rename(columns={'cluster_verb': 'cluster'})

  # Save the results to a CSV file with 'id' and 'cluster' columns
  if save_clusters:
    result_df = data[['id', 'cluster']]
    result_df.to_csv('clusters/tfidf_verb_clusters.csv', index=False)
    success("Clusters saved to clusters/tfidf_verb_clusters.csv")

  dbs = round(davies_bouldin_score(vectors.toarray(), kmean_indicates), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], vectors.toarray()


def TFIDF_nouns_cluster(data,  save_clusters=True, n_clusters=20):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  n_clusters: Number of clusters 
  """

  data = data[['id', 'description']]

  data = data.copy()
  working_on("Extracting nouns from descriptions ...")
  data.loc[:, 'description_noun'] = data['description'].apply(
      # use a lambda function to extract nouns using POS tagging
      lambda x: [
          word  # extract the word
          for word, pos_tag in [
              (word, pos_tagger(tag[1][0].upper()))  # POS tag each word
              # pair each word with its POS tag
              for word, tag in zip(eval(x), nltk.pos_tag(eval(x)))
              # filter for nouns
              if pos_tagger(tag[1][0].upper()) == wordnet.NOUN
          ]
          if pos_tag is not None  # exclude words with undefined POS tags
      ]
  )
  description_noun = data['description_noun']

  description_noun_strings = [' '.join(description)
                              for description in description_noun]

  working_on("Clustering nouns ...")
  model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10)

  vectors, all_keywords, vectorizer = get_tfidf_vectors(
      description_noun_strings)

  # Fit the model to the TF-IDF vectors
  model.fit(vectors)

  # Get the indices that sort the cluster centers in descending order
  order_centroids = model.cluster_centers_.argsort()[:, ::-1]

  # Get the feature names (words) from the TF-IDF vectorizer
  terms = vectorizer.get_feature_names_out()

  # Write the top terms for each cluster to a text file
  with open("results/kmeans_noun_results.txt", "w", encoding="utf-8") as f:
    for i in range(n_clusters):
      f.write(f"Cluster {i+1}")
      f.write("\n")
      for ind in order_centroids[i, :10]:
        f.write(' %s' % terms[ind],)
        f.write("\n")
      f.write("\n")
      f.write("\n")

  # Predict cluster assignments for each document in the TF-IDF vectors
  kmean_indicates = model.fit_predict(vectors)

  # Add a 'cluster' column to the 'data' DataFrame
  # data['cluster_noun'] = kmean_indicates
  data.loc[:, 'cluster_noun'] = kmean_indicates

  data = data.rename(columns={'cluster_noun': 'cluster'})

  # Save the results to a

  if save_clusters:
    result_df = data[['id', 'cluster']]
    result_df.to_csv('clusters/tfidf_noun_clusters.csv', index=False)
    success("Clusters saved to clusters/tfidf_noun_clusters.csv")

  dbs = round(davies_bouldin_score(vectors.toarray(), kmean_indicates), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], vectors.toarray()


def TFIDF_adjectives_cluster(data, save_clusters=True, n_clusters=20):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  n_clusters: Number of clusters 
  """

  data = data[['id', 'description']]

  data = data.copy()
  working_on("Extracting adjectives from descriptions ...")
  data.loc[:, 'description_adj'] = data['description'].apply(
      lambda x: [word for word, pos_tag in [(word, pos_tagger(tag[1][0].upper())) for word, tag in zip(
          eval(x), nltk.pos_tag(eval(x))) if pos_tagger(tag[1][0].upper()) == wordnet.ADJ] if pos_tag is not None]
  )
  description_adj = data['description_adj']

  description_adj_strings = [' '.join(description)
                             for description in description_adj]

  working_on("Clustering adjectives ...")
  model = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10)

  vectors, all_keywords, vectorizer = get_tfidf_vectors(
      description_adj_strings)

  # Fit the model to the TF-IDF vectors
  model.fit(vectors)

  # Get the indices that sort the cluster centers in descending order
  order_centroids = model.cluster_centers_.argsort()[:, ::-1]

  # Get the feature names (words) from the TF-IDF vectorizer
  terms = vectorizer.get_feature_names_out()

  # Write the top terms for each cluster to a text file
  with open("results/kmeans_adj_results.txt", "w", encoding="utf-8") as f:
    for i in range(n_clusters):
      f.write(f"Cluster {i+1}")
      f.write("\n")
      for ind in order_centroids[i, :10]:
        f.write(' %s' % terms[ind],)
        f.write("\n")
      f.write("\n")
      f.write("\n")

  # Predict cluster assignments for each document in the TF-IDF vectors
  kmean_indicates = model.fit_predict(vectors)

  # Add a 'cluster' column to the 'data' DataFrame

  data.loc[:, 'cluster_adj'] = kmean_indicates

  data = data.rename(columns={'cluster_adj': 'cluster'})

  # Save the results to a CSV file with 'id' and 'cluster' columns

  if save_clusters:
    result_df = data[['id', 'cluster']]
    result_df.to_csv('clusters/tfidf_adj_clusters.csv', index=False)
    success("Clusters saved to clusters/tfidf_adj_clusters.csv")

  dbs = round(davies_bouldin_score(vectors.toarray(), kmean_indicates), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], vectors.toarray()


if __name__ == "__main__":
  data = load_data(kind="processed")
  # data["description"] = data["description"].apply(ast.literal_eval)
  # clusters_text = TFIDF_cluster(data, save_clusters=False)
  # clusters_industries = TFIDF_industries_and_functions_cluster(
  #    data, save_clusters=False)
  # clusters_verbs = TFIDF_verbs_cluster(data, save_clusters=False)
  # clusters_nouns = TFIDF_nouns_cluster(data, save_clusters=False)
  cluster_adj = TFIDF_adjectives_cluster(data, save_clusters=False)
