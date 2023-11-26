from src.helper.utils import load_data
from src.helper.logger import success
from sklearn.metrics import davies_bouldin_score
import yaml
from sklearn.cluster import KMeans


def map_to_general_category(industry, mapping):
  """
  Takes an industry and a mapping of categories to industries,
  and returns the general categories that the industry belongs to.

  Args:
    industry: The industry parameter is the specific industry that you want to map to a general
  category. It could be a string representing the industry name.
    mapping: The `mapping` parameter is a dictionary where the keys are general categories and the
  values are lists of industries that fall under each category.

  Returns:
    a string that represents the general category of the given industry. If the industry is found in
  the mapping, the function returns a comma-separated string of the categories that the industry
  belongs to. If the industry is not found in the mapping, the function returns the string 'Other'.
  """
  categories = set()  # Use a set to keep unique categories
  for category, industries in mapping.items():
    if industry in industries:
      categories.add(category)
  return ", ".join(categories) if categories else "Other"


def ground_truth_onehot(data, save_clusters=True, n_clusters=20):
  """
  data: pandas dataframe (cleaned jobs)
  save_clusters: Boolean, if True, save the clusters to a csv file in a format "id, cluster"
  n_clusters: Number of clusters
  """

  data = data[["id", "title", "function", "industries"]].fillna("")

  with open("src/ground_truth/industries.yaml", "r") as yaml_file:
    industry_mapping = yaml.safe_load(yaml_file)
  with open("src/ground_truth/functions.yaml", "r") as yaml_file:
    function_mapping = yaml.safe_load(yaml_file)

  industry_categories = list(industry_mapping.keys())
  function_categories = list(function_mapping.keys())

  data["industries"] = data["industries"].apply(
      lambda x: ",".join([s.strip() for s in x.split(",")])
  )
  data["industry_group"] = data["industries"].apply(
      lambda x: ", ".join(
          map_to_general_category(ind, industry_mapping) for ind in x.split(",")
      )
  )
  data["industry_group"] = data["industry_group"].apply(
      lambda x: ", ".join(sorted(set(x.split(", "))))
  )

  data["function"] = data["function"].apply(
      lambda x: ",".join([s.strip() for s in x.split(",")])
  )
  data["function_group"] = data["function"].apply(
      lambda x: ", ".join(
          map_to_general_category(ind, function_mapping) for ind in x.split(",")
      )
  )
  data["function_group"] = data["function_group"].apply(
      lambda x: ", ".join(sorted(set(x.split(", "))))
  )

  # Create a new column for each general category and initialize with 0
  for category in industry_categories:
    data[category] = 0

  for category in function_categories:
    data[category] = 0

  # Iterate through the 'industry_group' column and set corresponding columns to 1
  for idx, row in data.iterrows():
    industry_groups = row["industry_group"].split(", ")
    function_groups = row["function_group"].split(", ")

    for category in industry_groups:
      if category in industry_categories:
        data.at[idx, category] = 1

    for category in function_groups:
      if category in function_categories:
        data.at[idx, category] = 1

  industries_and_functions = [
      "Technology and Information",
      "Manufacturing",
      "Financial and Business Services",
      "Transportation and Logistics",
      "Healthcare and Pharmaceuticals",
      "Retail and Consumer Goods",
      "Education and Non-profit",
      "Real Estate and Construction",
      "Energy and Environment",
      "Aerospace and Defense",
      "Food and Beverage",
      "Services and Miscellaneous",
      "Management and Leadership",
      "Manufacturing and Engineering",
      "Information Technology",
      "Sales and Marketing",
      "Administrative and Support",
      "Writing, Editing, and Creative",
      "Customer Service",
      "Legal and Finance",
      "Research and Analysis",
      "Human Resources and People Management",
      "Purchasing and Supply Chain",
      "Healthcare and Science",
      "Education and Training",
  ]

  kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
  data["cluster"] = kmeans.fit_predict(data[industries_and_functions])

  # Saving below df is for later comparison of text and feature clustering.
  df_id_and_cluster = data[["id", "cluster"]].sort_values(
      by="cluster", ascending=True
  )

  if save_clusters:
    df_id_and_cluster.to_csv(
        "clusters/ground_truth_onehot.csv", index=False)
    success("Clusters saved to clusters/ground_truth_onehot.csv")

  dbs = round(davies_bouldin_score(
      data[industries_and_functions], data["cluster"]), 3)

  success("David Bouldin score: " + str(dbs))

  return data[["id", "cluster"]], data[industries_and_functions]


if __name__ == "__main__":
  data = load_data(kind="processed")
  ground_truth_onehot(data, save_clusters=False)
