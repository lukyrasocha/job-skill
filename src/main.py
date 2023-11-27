"""
MAIN ENTRYPOINT OF THE ENTIRE PROJECT
"""

import ast
import os
import pandas as pd

# Custom scripts
from src.helper.utils import load_data, visualize_cluster, visualize_ground_truth
from src.helper.scraper import LinkedinScraper
from src.helper.preprocessing import preprocess
from src.clustering.tfidf_cluster import TFIDF_cluster, TFIDF_industries_and_functions_cluster, TFIDF_verbs_cluster, TFIDF_nouns_cluster, TFIDF_adjectives_cluster
from src.clustering.word2vec_cluster import word2vec_cluster
from src.ground_truth.ground_truth_onehot import ground_truth_onehot
from src.ground_truth.ground_truth_keywords import ground_truth_keywords
from src.clustering.doc2vec_cluster import doc2vec_cluster
from src.clustering.similarity_cluster import similarity_cluster
from src.evaluation import evaluation
from src.skill_extract.skill_extraction_gpt import skill_extraction as skill_extraction_gpt
from src.skill_extract.skill_extraction_hugging_face import skill_extraction as skill_extraction_hugging_face
from src.skill_extract.skill_cluster_plot import skill_analysis
from src.helper.logger import working_on, success, info, warning, error


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
    data_string_desc = load_data(kind="processed")
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
    data_string_desc = load_data(kind="processed")
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
    data_string_desc = load_data(kind="processed")
    sim_community_discovery_clusters, sim_kmeans_clusters, sim_matrix = similarity_cluster(data_string_desc,
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

  success("Clustering completed")

  """GROUND TRUTH INFERENCE"""
  print(150*"-")
  info("To infer the ground truth, we used three different methods. The first one is a keyword based method, where we used a list of keywords for each category. The second one is a one hot method, where we used the industries and functions of the job posts. The third one is a GPT based method, where the ground truth was infered by a general purpose language model.")
  warning("If you want to reproduce the ground truth based on GPT, you need to set the environment variable OPENAI_API_KEY to your OpenAI key. Then you can run the following command in the terminal: python src/ground_truth/ground_truth_gpt.py. The ground truth will be saved to clusters/ground_truth_gpt.csv")
  print(150*"-")
  q = input(
      "🧠 Do you want to infer the ground truth based on one hot encoded Industries and Functions? (y/n) ")
  if q == "y":
    working_on("One hot clustering (industries and functions)")
    onehot_clusters, onehot_features = ground_truth_onehot(data,
                                                           save_clusters=save_clusters,
                                                           n_clusters=20)
    visualize_cluster(onehot_features,
                      onehot_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="onehot_clusters.png",
                      name="One hot clustering")

  q = input(
      "🧠 Do you want to infer the ground truth based on keywords? (y/n) ")
  if q == "y":
    working_on("Keyword based clustering")
    data_string_desc = load_data(kind="processed")
    keywords_clusters = ground_truth_keywords(data_string_desc,
                                              save_clusters=save_clusters)

  if not os.path.exists("clusters/ground_truth_gpt.csv"):
    error("Ground truth not found: missing clusters/ground_truth_gpt.csv")
    print("First you need to infer the ground truth. For this we used GPT. You can reproduce the ground truth by first setting the environment variable OPEN_AI_KEY to your OpenAI key. Then you can run the following command in the terminal: python src/ground_truth/ground_truth_gpt.py")
    return

  working_on("Loading GPT-3.5 ground truth")
  gt = load_data(kind="ground_truth_gpt")
  visualize_ground_truth(gt, savefig=True, filename="ground_truth.png")
  success("Ground truth loaded")

  """EVALUATION OF CLUSTERS COMPARED TO GROUND TRUTH"""
  evaluation()

  """SKILL EXTRACTION"""
  print(150*"-")
  info("To extract skills from job descriptions, we used two different Machine Learning models. The first one is a model from Hugging Face, which is a pre-trained model which was trained to identify hard and soft skills from text. The second one is a model from OpenAI (GPT-3.5-turbo) which is a general purpose language model.")
  print(150*"-")
  q = input("🧠 Do you want to extract skills from the job descriptions using GPT-3.5-turbo (you need to set an env variable OPENAI_API_KEY) ? (y/n) ")
  if q == "y":
    working_on("Skill extraction using GPT-3.5-turbo")
    skill_extraction_gpt(save_skills=True)
    success(
        "Skills saved to 'extracted_skills/skills_extracted_gpt3.csv'")

  q = input(
      "🧠 Do you want to extract skills from the job descriptions using Hugging Face? (y/n) ")
  if q == "y":
    working_on("Skill extraction using Hugging Face")
    skill_extraction_hugging_face(save_skills=True)
    success(
        "Skills saved to 'extracted_skills/huggleface_skills.csv'")

  """SKILL ANALYSIS"""
  q = input(
      "🧠 Do you want to analyze the skills for the clustering based on TFIDF whole job descriptions? (y/n) ")
  if q == "y":
    compare = pd.read_csv('clusters/tfidf_clusters_job_desc.csv')
    skill_analysis(compare, gt)
    success("Saved to figures/.")

  success("Done")


if __name__ == "__main__":
  main()
