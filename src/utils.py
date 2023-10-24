import pandas as pd
from langdetect import detect


def load_data(kind="processed"):
  """
  Load the data from the data folder.
  args:
    kind: "raw" or "processed"
  """
  if kind == "raw":
    df = pd.read_csv('data/raw/jobs.csv', sep=';')
  elif kind == "processed":
    df = pd.read_csv('data/processed/cleaned_jobs.csv', sep=';')
  return df


def is_english(text):
  try:
    return detect(text) == 'en'
  except:
    return False
