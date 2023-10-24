
# Note: you might need to download the nltk packages
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from src.utils import load_data, is_english


def convert_date_posted(date_str, date_scraped):
  try:
    days_ago = int(date_str.split(' ')[0])
    actual_date = pd.to_datetime(date_scraped) - pd.Timedelta(days=days_ago)
    return actual_date
  except:
    return date_scraped  # If the format is not "x days ago", use the scraped date


def text_preprocessing(text):
  """
  Preprocesses text by tokenizing, removing stopwords and lemmatizing.
  """
  tokens = word_tokenize(text)

  stop_words = set(stopwords.words('english'))
  tokens = [w for w in tokens if not w in stop_words]

  lemmatizer = WordNetLemmatizer()
  tokens = [lemmatizer.lemmatize(w) for w in tokens]

  return tokens


def main():
  """
  Main function of the preprocessing module.
  Loads the raw data and does the following:
  - Checks for english language
  - Removes rows with missing descriptions
  """

  df = load_data(kind="raw")

  # Filter out jobs with missing descriptions
  df = df[df['description'].notna()]

  # Filter out non-english descriptions
  for index, row in df.iterrows():
    if not is_english(row['description'][:100]):
      df.drop(index, inplace=True)

  # Convert date_posted to a uniform format

  df['date_posted'] = df.apply(lambda x: convert_date_posted(
      x['date_posted'], x['date_scraped']), axis=1)

  # Lower case all text
  df['description'] = df['description'].str.lower()
  df['title'] = df['title'].str.lower()
  df['function'] = df['function'].str.lower()
  df['industries'] = df['industries'].str.lower()

  df['description'] = df['description'].apply(text_preprocessing)

  df.to_csv('data/processed/cleaned_jobs.csv', index=False, sep=';')


if __name__ == "__main__":
  # main()
  df = load_data(kind="processed")
  print(df.iloc[500]["description"])
