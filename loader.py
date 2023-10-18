import pandas as pd


def load_data():
  df = pd.read_csv('jobs.csv', sep=';')
  return df


df = load_data()

print(len(set(df["id"])))
print(df.iloc[1500])
