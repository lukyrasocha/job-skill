import pandas as pd
import os
import yaml

df_jobs = pd.read_csv("data/processed/cleaned_jobs.csv", delimiter=';')

yaml_file = 'ground_truth.yaml'
if os.path.exists(yaml_file):
    with open(yaml_file, 'r') as file:
        yaml_content = yaml.safe_load(file) or {}


ground_truth_df = pd.DataFrame.from_dict(yaml_content, orient='index', columns=['category'])
ground_truth_df.index.name = 'id'
ground_truth_df.reset_index(inplace=True)

mapping_rules = {
    'Software & IT': 'Software & IT',
    'Creative Arts & Design': 'Creative Arts & Design',
    'Engineering & Manufacturing': 'Engineering & Manufacturing',
    'Manufacturing': 'Engineering & Manufacturing',
    'Human Resources & Recruitment': 'Human Resources & Recruitment',
    'Energy & Utilities': 'Energy & Utilities',
    'Sales & Marketing': 'Sales & Marketing',
    'Consumer Goods': 'Retail & Consumer Goods',
    'Transportation & Logistics': 'Transportation & Logistics',
    'Finance & Accounting': 'Finance & Accounting',
    'Information Technology & Services': 'Software & IT',
    'IT & Software': 'Software & IT',
    'Non-Profit & Social Services': 'Non-Profit & Social Services',
    'Media & Communications': 'Media & Communications',
    'Technology': 'Software & IT',
    'Hospitality & Tourism': 'Hospitality & Tourism',
    'Retail & Consumer Goods': 'Retail & Consumer Goods',
    'Technology & Information': 'Software & IT',
    'Legal & Compliance': 'Legal & Compliance',
    'Healthcare & Medicine': 'Healthcare & Medicine',
    'Science & Research': 'Science & Research',
    'Information Technology': 'Software & IT',
    'Education & Training': 'Education & Training',
    'Business & Entrepreneurship': 'Finance & Accounting',
    'Logistics & Supply Chain': 'Transportation & Logistics',
    'Construction & Real Estate': 'Construction & Real Estate',
    'Arts & Entertainment': 'Arts & Entertainment',
    'Agriculture & Environmental': 'Agriculture & Environmental',
    'Staffing & Recruiting': 'Human Resources & Recruitment',
    'Maritime & Transportation': 'Transportation & Logistics',
    'Technology & IT': 'Software & IT',
    'Public Relations & Communications': 'Media & Communications',
    'Customer Service': 'Human Resources & Recruitment',
    'Information Technology (IT)': 'Software & IT',
    'Manufacturing & Engineering': 'Engineering & Manufacturing',
    'Renewable energy': 'Energy & Utilities',
    'Government & Public Sector': 'Government & Public Sector',
    'Customer Success': 'Sales & Marketing',
    'Insurance & Risk Management': 'Finance & Accounting',
    'Human Resources': 'Human Resources & Recruitment',
    'Marketing & Advertising': 'Sales & Marketing',
    'Pharmaceutical & Healthcare': 'Healthcare & Medicine',
    'Retail': 'Retail & Consumer Goods',
    'Environmental & Sustainability': 'Agriculture & Environmental',
    'Real Estate & Construction': 'Construction & Real Estate',
    'Aerospace & Defense': 'Engineering & Manufacturing',
    'Public Relations': 'Media & Communications',
    'Event Planning & Management': 'Hospitality & Tourism',
    'Sports & Recreation': 'Arts & Entertainment',
    'Medical equipment manufacturing': 'Healthcare & Medicine',
    'Renewable Energy': 'Energy & Utilities',
    'Technology & Internet': 'Software & IT',
    'Technology & Information Technology': 'Software & IT',
    'Administration & Office Support': 'Human Resources & Recruitment',
    'Information & Technology': 'Software & IT',
    'Administration': 'Human Resources & Recruitment',
    'Technology & Telecommunications': 'Software & IT',
    'Insurance': 'Finance & Accounting',
    'Insurance & Financial Services': 'Finance & Accounting',
    'Logistics & Supply Chain Management': 'Transportation & Logistics',
    'Market Research': 'Sales & Marketing'
}

ground_truth_df['category'] = ground_truth_df['category'].map(mapping_rules)

ground_truth_df['category'] = pd.Categorical(ground_truth_df['category'])
ground_truth_df['cluster'] = ground_truth_df['category'].cat.codes
df_id_and_cluster = ground_truth_df[["id", "category", "cluster"]].sort_values(
    by="cluster", ascending=True
)

df_id_and_cluster.to_csv("./csv_files/ground_truth_gpt.csv", index=False)

