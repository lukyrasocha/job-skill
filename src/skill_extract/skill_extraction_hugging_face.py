import pandas as pd
from transformers import pipeline
import textwrap
import ast
import re
from src.helper.utils import load_data
from src.helper.logger import success


def words_to_sentence(word_list):
  return " ".join(ast.literal_eval(word_list))


def extract_skills_hugging_face(merged):
  """
  Take a dictionary including the id and their cluster label. Extract skills and knowledge from the 
  raw job description using pretrained model: https://huggingface.co/spaces/jjzha/skill_extraction_demo

  Args:
    cluster_dict: A dictionary with a key of id and a value of its corresponding cluster label 

  Returns:
    returns a dictionary with a key of cluster label and a value of the extracted skills.
  """
  # Load the data

  # Use a pipeline as a high-level helper

  token_skill_classifier = pipeline(
      "token-classification", model="jjzha/jobbert_skill_extraction")
  token_knowledge_classifier = pipeline(
      "token-classification", model="jjzha/jobbert_knowledge_extraction")

  # Or use download the model and use it directly
  # token_skill_classifier = pipeline(
  #    "token-classification", model="src/model/jobbert_skill_extraction", aggregation_strategy="first")
  # token_knowledge_classifier = pipeline(
  #    "token-classification", model="src/model/jobbert_knowledge_extraction", aggregation_strategy="first")

  N = len(merged)
  count = 0
  extracted_skills = {"id": [], "skills": [], "description_raw": []}

  for _, row in merged.iterrows():
    skills = []
    job_description = row['description_raw']
    job_description = job_description.replace("\n", " ")
    pattern = r'(?<=[a-z])(?=[A-Z])'
    job_description = re.sub(pattern, ' ', job_description)
    # Remove the last 56 trash characters
    job_description = job_description[:-56]
    # print(job_description)

    lines = textwrap.wrap(job_description, 500, break_long_words=False)

    [skills.extend(skill) for skill in token_skill_classifier(lines)]
    [skills.extend(knowledge)
     for knowledge in token_knowledge_classifier(lines)]
    skills = [entry['word'] for entry in skills]
    _id = row['id']

    extracted_skills["id"].append(_id)
    extracted_skills["skills"].append(skills)
    extracted_skills["description_raw"].append(job_description)

    count += 1

    # Print progress in place
    print(f"\r💬 Skills for {_id} extracted! Progress: {count}/{N}", end="")

  return extracted_skills


def skill_extraction(save_skills=False):

  df_clean = load_data("processed")
  df_raw = load_data("raw")

  df_clean = df_clean[['id', 'description']]
  df_raw = df_raw[['id', 'description']]

  # Obtain the original unprocessed job descriptions from the jobs that appear in the clean dataset
  merged = pd.merge(df_clean, df_raw, on='id', how="left",
                    suffixes=('_clean', '_raw'))

  # Drop duplicates based on id
  merged = merged.drop_duplicates(subset=['id'])

  skills = extract_skills_hugging_face(merged)

  extracted_skills_df = pd.DataFrame(skills)
  success("Skills extracted")
  # Save results
  if save_skills:
    name = "huggleface_skills.csv"
    extracted_skills_df.to_csv(
        f"extracted_skills/{name}", index=False)
    success(f"Skills saved to extracted_skills/{name}")


if __name__ == "__main__":
  skill_extraction(save_skills=False)
