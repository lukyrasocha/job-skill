import pandas as pd
from src.logger import success, working_on
from src.utils import load_data

keywords = {
    'Software & IT': ['software', 'it support', 'network administration', 'cybersecurity', 'system analysis'],
    'Healthcare & Medicine': ['medical', 'healthcare', 'nursing', 'doctor', 'clinical'],
    'Education & Training': ['teaching', 'academic', 'education administration', 'training', 'curriculum development'],
    'Engineering & Manufacturing': ['mechanical engineering', 'civil engineering', 'electrical engineering', 'manufacturing', 'quality control'],
    'Finance & Accounting': ['accounting', 'financial', 'auditing', 'banking', 'investment'],
    'Sales & Marketing': ['sales', 'social media', 'digital marketing', 'public relations', 'brand strategy'],
    'Creative Arts & Design': ['graphic design', 'fashion', 'photography', 'graphics', 'creative'],
    'Hospitality & Tourism': ['hotel', 'travel consulting', 'event', 'culinary arts', 'tourism'],
    'Construction & Real Estate': ['construction', 'architecture', 'urban planning', 'real estate', 'building design'],
    'Legal & Compliance': ['legal', 'compliance', 'law', 'regulatory affairs', 'legal advisory'],
    'Science & Research': ['research', 'laboratory', 'scientific', 'study', 'experimental'],
    'Human Resources & Recruitment': ['employee management', 'recruitment', 'training development', 'organizational', 'workforce planning'],
    'Transportation & Logistics': ['transportation', 'supply chain', 'logistics', 'fleet', 'shipping coordination'],
    'Agriculture & Environmental': ['farming', 'environmental', 'resource management', 'agricultural', 'sustainable'],
    'Retail & Consumer Goods': ['retail', 'goods', 'consumer', 'product marketing', 'merchandising'],
    'Media & Communications': ['journalism', 'broadcasting', 'content', 'communication', 'media production'],
    'Government & Public Sector': ['public administration', 'policy', 'service', 'government', 'civil'],
    'Non-Profit & Social Services': ['non-profit', 'social', 'community', 'charity', 'volunteer'],
    'Energy & Utilities': ['energy', 'renewable', 'utility management', 'energy conservation', 'sustainability'],
    'Arts & Entertainment': ['acting', 'music performance', 'event coordination', 'entertainment', 'art']
}


def transform_string(s):
  return s[1:-1].replace("'", "").replace(", ", " ")


def refined_keyword_based_categorize_job(row):
  # Combine the function, industries, and description into a single string for analysis
  combined_info = f"{row['function']} {row['industries']} {row['description']}".lower(
  )

  # Count matching keywords for each category
  keyword_counts = {category: sum(keyword in combined_info for keyword in keywords)
                    for category, keywords in keywords.items()}

  # Determine the category with the most matching keywords
  best_category = max(keyword_counts, key=keyword_counts.get)

  # If the best category has 0 matches and there's an 'Other' category, assign 'Other'
  if keyword_counts[best_category] == 0 and 'Other' in keywords:
    return 'Other'
  return best_category


def keywords_ground_truth(data, save_clusters=False):

  df_jobs = data
  df_jobs['description'] = df_jobs['description'].apply(transform_string)

  working_on("Categorizing jobs ...")
  # Apply the refined keyword-based categorization function to each row
  df_jobs['category'] = df_jobs.apply(
      refined_keyword_based_categorize_job, axis=1)

  # Display the updated dataframe with the new 'refined_keyword_based_general_cluster' column
  df_jobs[['title', 'function', 'industries', 'description', 'category']].head()

  print("Number of jobs in each category based on keywords ground truth:")
  for key in keywords.keys():
    print(key, sum(df_jobs.category == key))

  # Turn to categories
  df_jobs['category'] = pd.Categorical(df_jobs['category'])
  df_jobs['cluster'] = df_jobs['category'].cat.codes
  df_id_and_cluster = df_jobs[["id", "category", "cluster"]].sort_values(
      by="cluster", ascending=True
  )

  if save_clusters:
    df_id_and_cluster.to_csv("clusters/ground_truth_keywords.csv", index=False)
    success("Clusters saved to clusters/ground_truth_keywords.csv")


if __name__ == "__main__":
  data = load_data(kind="processed")
  keywords_ground_truth(data, save_clusters=False)
