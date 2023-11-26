"""
MAIN ENTRYPOINT OF THE ENTIRE PROJECT
"""

import ast
import os
from utils import load_data, visualize_cluster, visualize_ground_truth, skill_cleanup
from scraper import LinkedinScraper
from preprocessing import preprocess
from tfidf_cluster import TFIDF_cluster, TFIDF_industries_and_functions_cluster, TFIDF_verbs_cluster, TFIDF_nouns_cluster, TFIDF_adjectives_cluster
from word2vec_cluster import word2vec_cluster
from onehot_cluster import onehot_cluster
from doc2vec_cluster import doc2vec_cluster
from similarity_cluster import similarity_cluster
from evaluation import evaluation
from skill_extraction import skill_extraction
from logger import working_on, success, info, warning, error


def main():
  """SCRAPING"""
  q = input("💼 Do you want to scrape new job posts? (y/n) ")

  if q == "y":
    loc = input("💼 Choose location (e.g. Denmark): ")
    keywords = input(
        "💼 What keywords (e.g. Software), type 'all' for all jobs: ")
    amount = input(
        "💼 How many jobs do you want to scrape? (max 1000 at a time): ")

    if keywords == "all":
      keywords = None

    scraper = LinkedinScraper(location=loc,
                              keywords=keywords,
                              amount=int(amount))
    scraper.scrape()
    success("Job posts saved to 'data/raw/jobs.csv'")

  """PREPROCESSING"""
  q = input("🧹 Do you want to preprocess the raw data? (y/n) ")

  if q == "y":
    working_on("Preprocessing")
    preprocess()
    success("Data preprocessed and saved to 'data/processed/cleaned_jobs.csv'")

  """LOAD CLEAN DATA"""
  working_on("Loading data")
  data = load_data(kind="processed")
  data["description"] = data["description"].apply(ast.literal_eval)
  success("Data loaded")

  """CLUSTERING METHODS"""
  save_clusters = input(
      "📝 The clustering will now be performed based on various different techniques, do you wish to save the final clusters? (y/n) ")
  if save_clusters == "y":
    info("The clusters will be saved to 'clusters/'")
    save_clusters = True
  else:
    save_clusters = False

  """TF IDF CLUSTERING BASED ON JOB DESCRIPTIONS"""
  q = input(
      "🧪 Do you want to perform TFIDF clustering (based on job descriptions)? (y/n) ")
  if q == "y":
    working_on("TFIDF Clustering (job descriptions)")
    tfidf_clusters, tfidf_matrix = TFIDF_cluster(data,
                                                 save_clusters=save_clusters,
                                                 n_clusters=20)
    visualize_cluster(tfidf_matrix,
                      tfidf_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="tfidf_clusters.png",
                      name="TFIDF Clustering")
    tfidf_matrix = 0

  """TF IDF CLUSTERING BASED ON INDUSTRIES AND FUNCTIONS"""
  q = input(
      "🧪 Do you want to perform TFIDF clustering (based on industries and functions)? (y/n) ")

  if q == "y":
    working_on("TFIDF Clustering (industries and functions)")
    tfidf_clusters, tfidf_matrix = TFIDF_industries_and_functions_cluster(data,
                                                                          save_clusters=save_clusters,
                                                                          n_clusters=20)
    visualize_cluster(tfidf_matrix,
                      tfidf_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="tfidf_industries_and_functions_clusters.png",
                      name="TFIDF Clustering")
    tfidf_matrix = 0

  """TF IDF CLUSTERING BASED ON VERBS FROM THE JOB DESCRIPTION"""
  q = input(
      "🧪 Do you want to perform TFIDF clustering (based on verbs from the job descriptions)? (y/n) ")
  if q == "y":
    working_on("TFIDF Clustering (verbs)")
    data_string_desc = load_data(kind="processed")
    tfidf_clusters, tfidf_matrix = TFIDF_verbs_cluster(data_string_desc,
                                                       save_clusters=save_clusters,
                                                       n_clusters=20)
    visualize_cluster(tfidf_matrix,
                      tfidf_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="tfidf_verbs_clusters.png",
                      name="TFIDF Clustering (verbs)")
    tfidf_matrix = 0
  """TF IDF CLUSTERING BASED ON NOUNS FROM THE JOB DESCRIPTION"""
  q = input(
      "🧪 Do you want to perform TFIDF clustering (based on nouns from the job descriptions)? (y/n) ")
  if q == "y":
    working_on("TFIDF Clustering (nouns)")
    data = load_data(kind="processed")
    tfidf_clusters, tfidf_matrix = TFIDF_nouns_cluster(data_string_desc,
                                                       save_clusters=save_clusters,
                                                       n_clusters=20)
    visualize_cluster(tfidf_matrix,
                      tfidf_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="tfidf_nouns_clusters.png",
                      name="TFIDF Clustering (nouns)")
    tfidf_matrix = 0

  """TF IDF CLUSTERING BASED ON ADJECTIVES FROM THE JOB DESCRIPTION"""
  q = input(
      "🧪 Do you want to perform TFIDF clustering (based on adjectives from the job descriptions)? (y/n) ")
  if q == "y":
    working_on("TFIDF Clustering (adjectives)")
    data = load_data(kind="processed")
    tfidf_clusters, tfidf_matrix = TFIDF_adjectives_cluster(data_string_desc,
                                                            save_clusters=save_clusters,
                                                            n_clusters=20)
    visualize_cluster(tfidf_matrix,
                      tfidf_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="tfidf_adjectives_clusters.png",
                      name="TFIDF Clustering (adjectives)")
    tfidf_matrix = 0

  """WORD2VEC CLUSTERING"""
  q = input("🧪 Do you want to perform Word2Vec clustering? (y/n) ")
  if q == "y":
    working_on("Word2Vec Clustering")
    word2vec_clusters, word2vec_vectors = word2vec_cluster(data,
                                                           save_clusters=save_clusters,
                                                           vector_size=100,
                                                           window=5,
                                                           min_count=1,
                                                           workers=4,
                                                           n_clusters=20)
    visualize_cluster(word2vec_vectors,
                      word2vec_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="word2vec_clusters.png",
                      name="Word2Vec Clustering")

  """FEATURE CLUSTERING (ONE HOT ENCODED FUNCTIONS AND INDUSTRIES)"""
  q = input(
      "🧪 Do you want to perform one hot clustering (based on industries and functions)? (y/n) ")
  if q == "y":
    working_on("One hot clustering (industries and functions)")
    onehot_clusters, onehot_features = onehot_cluster(data,
                                                      save_clusters=save_clusters,
                                                      n_clusters=20)
    visualize_cluster(onehot_features,
                      onehot_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="onehot_clusters.png",
                      name="One hot clustering")

  """DOC2VEC CLUSTERING"""
  q = input("🧪 Do you want to perform Doc2Vec clustering? (y/n) ")
  if q == "y":
    working_on("Doc2Vec Clustering")
    doc2vec_clusters, doc2vec_vectors = doc2vec_cluster(data,
                                                        save_clusters=save_clusters,
                                                        method="kmeans",  # or "gmm"
                                                        vector_size=100,
                                                        alpha=0.025,
                                                        min_alpha=0.00025,
                                                        min_count=10,
                                                        epochs=300,
                                                        n_clusters=20)
    visualize_cluster(doc2vec_vectors,
                      doc2vec_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="doc2vec_clusters.png",
                      name="Doc2Vec Clustering")

  """SIMILARITY CLUSTERING"""
  q = input("🧪 Do you want to perform similarity clustering (based on community discovery and kmeans)? (y/n) ")
  if q == "y":
    working_on("Similarity Clustering")
    sim_community_discovery_clusters, sim_kmeans_clusters, sim_matrix = similarity_cluster(data,
                                                                                           save_clusters=save_clusters,
                                                                                           q=2,
                                                                                           seeds=100,
                                                                                           n_clusters=20)
    visualize_cluster(sim_matrix,
                      sim_community_discovery_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="sim_community_discovery_clusters.png",
                      name="Similarity Clustering (Community Discovery)")

    visualize_cluster(sim_matrix,
                      sim_kmeans_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="sim_kmeans_clusters.png",
                      name="Similarity Clustering (KMeans)")

  success("All clustering methods performed")

  """GROUND TRUTH INFERENCE"""
  if not os.path.exists("clusters/ground_truth_gpt.csv"):
    error("Ground truth not found: missing clusters/ground_truth_gpt.csv")
    print("First you need to infer the ground truth. For this we used GPT. You can reproduce the ground truth by first setting the environment variable OPEN_AI_KEY to your OpenAI key. Then you can run the following command in the terminal: python src/ground_truth_gpt.py")
    return

  gt = load_data(kind="ground_truth")
  visualize_ground_truth(gt, savefig=True, filename="ground_truth.png")

  """EVALUATION OF CLUSTERS COMPARED TO GROUND TRUTH"""
  evaluation()

  """SKILL EXTRACTION"""
  q = input("🧠 Do you want to extract skills from the job descriptions (you need to set an env variable OPENAI_API_KEY) ? (y/n) ")
  if q == "y":
    working_on("Skill extraction")
    skill_extraction(save_skills=True)
    success(
        "Skills saved to 'extracted_skills/skills_extracted_gpt3.csv'")

  """SKILL ANALYSIS"""
  # TINGHUI's CODE


if __name__ == "__main__":
  main()
