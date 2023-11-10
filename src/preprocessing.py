
# Note: you might need to download the nltk packages
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm

from src.utils import load_data, is_english
from src.logger import working_on


def convert_date_posted(date_str, date_scraped):
  try:
    days_ago = int(date_str.split(' ')[0])
    actual_date = pd.to_datetime(date_scraped) - pd.Timedelta(days=days_ago)
    return actual_date
  except:
    return date_scraped  # If the format is not "x days ago", use the scraped date


def split_combined_words(text):
  """
  Since during the scraping, some words are combined, e.g. "requirementsYou're" or "offerings.If" we need to split them
  Splits words at:
  1. Punctuation marks followed by capital letters.
  2. Lowercase letters followed by uppercase letters.
  """
  # 1. split
  text = re.sub(r'([!?,.;:])([A-Z])', r'\1 \2', text)

  # 2. split
  text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

  return text


def text_preprocessing(text):
  """
  Preprocesses text by:
    - Splitting combined words
    - Tokenizing
    - Removing stopwords
    - Remove punctuation
    - Lemmatizing
  """

  text = split_combined_words(text)

  tokens = word_tokenize(text)

  stop_words = set(stopwords.words('english'))
  tokens = [w for w in tokens if not w in stop_words]

  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(w) for w in tokens]

  punctuation = {'!', ',', '.', ';', ':', '?', '(', ')', '[', ']', '-','+','"','*', '—','•', '’', '‘', '“', '”', '``'}

  tokens = [w.lower() for w in tokens if w not in punctuation]

  # Remove last 3 words since they are always the same (scraped buttons from the website)
  tokens = tokens[:-3]

  return tokens


def main():
  """
  Main function of the preprocessing module.
  Loads the raw data and does the following:
  - Checks for english language
  - Removes rows with missing descriptions
  - Inferes the date posted
  - Preprocesses the description
  - Saves the preprocessed data to data/processed/cleaned_jobs.csv
  """

  working_on("Loading data")
  df = load_data(kind="raw")

  # Remove duplicates
  df.drop_duplicates(subset=['id'], inplace=True)
  df.drop_duplicates(subset=['description'], inplace=True)
  # Filter out jobs with missing descriptions
  df = df[df['description'].notna()]

  working_on("Filtering out non-english descriptions ...")
  for index, row in df.iterrows():
    if not is_english(row['description'][:100]):
      df.drop(index, inplace=True)

  working_on("Infering dates ...")
  df['date_posted'] = df.apply(lambda x: convert_date_posted(
      x['date_posted'], x['date_scraped']), axis=1)

  # Lower case all text
  df['title'] = df['title'].str.lower()
  df['function'] = df['function'].str.lower()
  df['industries'] = df['industries'].str.lower()
  df['industries'] = df['industries'].str.replace('\n', ' ')

  tqdm.pandas(desc="🐼 Preprocessing description", ascii=True, colour="#0077B5")

  df['description'] = df['description'].progress_apply(text_preprocessing)

  # Remove rows with empty descriptions or descriptions containing less than 3 words
  df = df[df['description'].map(len) > 3]

  working_on("Saving preprocessed data ...")
  df.to_csv('data/processed/cleaned_jobs.csv', index=False, sep=';')


if __name__ == "__main__":
  main()
  df = load_data(kind="processed")
  #print(df.iloc[970]['description'])
  #print(df[df["id"]==3733315884])
  print(df)