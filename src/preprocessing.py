from src.utils import load_raw_data, is_english


def main():
  """
  Main function of the preprocessing module.
  Loads the raw data and does the following:
  - Checks for english language
  - Removes rows with missing descriptions
  """

  df = load_raw_data()

  # Filter out jobs with missing descriptions
  df = df[df['description'].notna()]

  # Check for non-english descriptions
  for index, row in df.iterrows():
    if not is_english(row['description'][:100]):
      df.drop(index, inplace=True)

  print(df)


if __name__ == "__main__":
  main()
