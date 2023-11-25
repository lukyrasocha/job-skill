import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import textwrap
import ast
import re
from src.utils import load_data
from logger import success

def words_to_sentence(word_list):
    return " ".join(ast.literal_eval(word_list))



def skill_extract＿lm_ner(job_description):
    """
    Take a dictionary including the id and their cluster label. Extract skills and knowledge from the 
    raw job description using pretrained model: https://huggingface.co/algiraldohe/lm-ner-linkedin-skills-recognition

    Args:
      cluster_dict: A dictionary with a key of id and a value of its corresponding cluster label 

    Returns:
      returns a dictionary with a key of cluster label and a value of the extracted skills.
    """
    # Load the data
    df  = load_data(kind="processed")
    df["description"] = df["description"].apply(words_to_sentence)
    pipe = pipeline("token-classification", model="src/model/lm-ner-linkedin-skills-recognition")
    cluster_skill = {}
    for label in tqdm(set(cluster_dict.values())):
        skills = []
        ids = [key for key, value in cluster_dict.items() if value == label]
        for id in ids:
            description = df.loc[df['id'] == id]['description'].values[0]
            skills.extend(pipe(description)) 
        cluster_skill[label] = count_words(entry['word'] for entry in skills)
    return cluster_skill

 
def skill_extract(merged):
    """
    Take a dictionary including the id and their cluster label. Extract skills and knowledge from the 
    raw job description using pretrained model: https://huggingface.co/spaces/jjzha/skill_extraction_demo

    Args:
      cluster_dict: A dictionary with a key of id and a value of its corresponding cluster label 

    Returns:
      returns a dictionary with a key of cluster label and a value of the extracted skills.
    """
    # Load the data

    token_skill_classifier = pipeline("token-classification", model="src/model/jobbert_skill_extraction", aggregation_strategy="first")
    token_knowledge_classifier = pipeline("token-classification", model="src/model/jobbert_knowledge_extraction", aggregation_strategy="first")
    
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
        #print(job_description)
        
        lines = textwrap.wrap(job_description, 500, break_long_words=False)
        
        [skills.extend(skill) for skill in token_skill_classifier(lines)]
        [skills.extend(knowledge) for knowledge in token_knowledge_classifier(lines)]
        skills = [entry['word'] for entry in skills]
        _id = row['id']

        extracted_skills["id"].append(_id)
        extracted_skills["skills"].append(skills)
        extracted_skills["description_raw"].append(job_description)

        count += 1

        # Print progress in place
        print(f"\r💬 Skills for {_id} extracted! Progress: {count}/{N}", end="")

            
    
    return extracted_skills

def main(save_skills=False):

    df_clean = load_data("processed")
    df_raw = load_data("raw")

    df_clean = df_clean[['id', 'description']]
    df_raw = df_raw[['id', 'description']]

    # Obtain the original unprocessed job descriptions from the jobs that appear in the clean dataset
    merged = pd.merge(df_clean, df_raw, on='id', how="left",
                        suffixes=('_clean', '_raw'))

    # Drop duplicates based on id
    merged = merged.drop_duplicates(subset=['id'])

    

    

    skills = skill_extract(merged)
    
    # ground = pd.read_csv('clusters/ground_truth.csv', sep=',')
    # # Given a dictionary in the format of "id": "cluster label", and return a dictionary 
    # # of "cluster label": "skills"
    # skills = skill_extract(dict(zip(ground["id"], ground["cluster"])))

    extracted_skills_df = pd.DataFrame(skills)
    success("Skills extracted")
    # Save results
    if save_skills:
        name = "huggleface_skills.csv"
        extracted_skills_df.to_csv(
        f"extracted_skills/{name}", index=False)
        success(f"Skills saved to extracted_skills/{name}")
    

    
if __name__ == "__main__":
    main(save_skills=False)
