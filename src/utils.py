import pandas as pd
from langdetect import detect


def load_raw_data():
  df = pd.read_csv('data/raw/jobs.csv', sep=';')
  return df


def is_english(text):
  try:
    return detect(text) == 'en'
  except:
    return False
