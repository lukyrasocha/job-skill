import pandas as pd
import ast
import re
from transformers import  pipeline
from tqdm import tqdm

def words_to_sentence(word_list):
    return " ".join(word_list)
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

# Load the data
df  = pd.read_csv('./data/processed/cleaned_jobs.csv', sep=';')
cluster = pd.read_csv('./csv_files/similarity.csv', sep=',')
pipe = pipeline("token-classification", model="lm-ner-linkedin-skills-recognition")

df["description"] = df["description"].apply(
            lambda x: remove_words_with_numbers(x)
        )
df["description"] = df["description"].apply(words_to_sentence)



cluster_skill = {}
for label in tqdm(sorted(cluster['cluster_graph'].unique())):
    skills = []
    for id in cluster.loc[cluster['cluster_graph'] == label]['id']:
        skills.extend(pipe(df.loc[df['id'] == id]['description'].values[0]))
        
    skill_extract = [{'entity': entry['entity'], 'word': entry['word']} for entry in skills]
    cluster_skill[label] = skill_extract


pd.DataFrame.from_dict(cluster_skill.items()).to_csv("./csv_files/skills.csv", index=False)