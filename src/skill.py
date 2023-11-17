import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import textwrap
import re
import ast
from collections import Counter
import spacy


# ======================================== These codes are from text_clustering.py in Tomasz's code =========================================
def remove_words_with_numbers(word_list_str):
    """
    Takes a string representation of a list of words as input,
    removes any special characters from the words, and then removes any words that contain numbers.

    Args:
      word_list_str: A string representation of a list of words.

    Returns:
      The function `remove_words_with_numbers` returns a list of words without any special characters or
    numbers.
    """
    word_list = ast.literal_eval(word_list_str)
    word_list_without_special = [
        re.sub(r"[^a-zA-Z0-9\s]", "", word) for word in word_list
    ]
    word_list_without_numbers = [
        word for word in word_list_without_special if not re.search(r"\d", word)
    ]
    return word_list_without_numbers

def words_to_sentence(word_list):
    return " ".join(word_list)
# ============================================================================================================================================

def count_words(words):
    nlp = spacy.load("en_core_web_sm")
    words = [token.lemma_ for token in nlp(" ".join(words))]
    word_counts = Counter(words)
    return dict(word_counts)
        
def skill_extract(cluster_dict):
    """
    Take a dictionary including the id and their cluster label. Extract skills and knowledge from the 
    raw job description using pretrained model: https://huggingface.co/spaces/jjzha/skill_extraction_demo

    Args:
      cluster_dict: A dictionary with a key of id and a value of its corresponding cluster label 

    Returns:
      returns a dictionary with a key of cluster label and a value of the extracted skills.
    """
    # Load the data
    df  = pd.read_csv('data/processed/cleaned_jobs.csv', sep=';')
    df["description"] = df["description"].apply(
        lambda x: remove_words_with_numbers(x)
    )
    df["description"] = df["description"].apply(words_to_sentence)
    token_skill_classifier = pipeline("token-classification", model="src/model/jobbert_skill_extraction", aggregation_strategy="first")
    token_knowledge_classifier = pipeline("token-classification", model="src/model/jobbert_knowledge_extraction", aggregation_strategy="first")
    
    cluster_skill = {}
    for label in tqdm(set(cluster_dict.values())):


        skills = []
        ids = [key for key, value in cluster_dict.items() if value == label]
        for id in ids:
            description = df.loc[df['id'] == id]['description'].values[0]
            lines = textwrap.wrap(description, 500, break_long_words=False)
        
            [skills.extend(skill) for skill in token_skill_classifier(lines)]
            [skills.extend(knowledge) for knowledge in token_knowledge_classifier(lines)]
            
        cluster_skill[label] = count_words(entry['word'] for entry in skills)
    return cluster_skill

