"""
MAIN ENTRYPOINT OF THE ENTIRE PROJECT
"""

import argparse
import ast
from utils import load_data, visualize_cluster, visualize_ground_truth, skill_cleanup
from tfidf_cluster import TFIDF_cluster
from logger import working_on, success, info

parser = argparse.ArgumentParser()
parser.add_argument("--scrape", help="Scrape data",
                    action="store_true", default=False)
args = parser.parse_args()


def main():
  """SCRAPING"""
  if args.scrape:
    print("true")
    pass

  """PREPROCESSING"""

  """LOAD CLEAN DATA"""
  working_on("Loading data")
  data = load_data(kind="processed")
  data["description"] = data["description"].apply(ast.literal_eval)
  success("Data loaded")

  """TF IDF CLUSTERING"""
  q = input("🧪 Do you want to perform TFIDF clustering? (y/n) ")
  if q == "y":
    working_on("TFIDF Clustering")
    tfidf_clusters, tfidf_matrix = TFIDF_cluster(data, save_clusters=False)
    visualize_cluster(tfidf_matrix,
                      tfidf_clusters["cluster"].to_numpy(),
                      savefig=True,
                      filename="tfidf_clusters.png",
                      name="TFIDF Clustering")
    tfidf_matrix = 0

  """WORD2VEC CLUSTERING"""
  # Cluster
  # Save clusters

  """FEATURE CLUSTERING (ONE HOT ENCODED FUNCTIONS AND INDUSTRIES)"""
  # Cluster
  # Save clusters

  """DOC2VEC CLUSTERING"""
  # Cluster
  # Save clusters

  """SIMILARITY CLUSTERING COMMUNITY DISCOVERY"""
  # Cluster
  # Save clusters

  """SIMILARITY CLUSTERING KMEANS"""
  # Cluster
  # Save clusters

  """EVALUATION"""
  # Load clusters
  # Load ground truth
  # Evaluate
  # Select best clustering
  # Calculate DBINDEX
  # Visualize clusters

  """GROUND TRUTH INFERENCE"""
  # SET ENV VARIABLE OPEN_AI_KEY TO YOUR OPEN AI KEY
  # SAVE GROUND TRUTH

  gt = load_data(kind="ground_truth")
  # visualize_ground_truth(gt, savefig=True, filename="ground_truth.png")

  """SKILL EXTRACTION"""
  # SET ENV VARIABLE OPEN_AI_KEY TO YOUR OPEN AI KEY
  # SAVE SKILLS FOR EACH CLUSTERK

  skills = load_data(kind="skills")
  skills["skills"] = skills["skills"].apply(ast.literal_eval)
  cleaned = skill_cleanup(skills)

  for i in range(len(skills)):
    print(cleaned.iloc[i]["skills_string"])


if __name__ == "__main__":
  main()
