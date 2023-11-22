from openai import OpenAI
import pandas as pd
import openai
import os
import yaml

openai.organization = ""
openai.api_key = os.getenv("OPENAI_API_KEY")

def load_data(kind="processed"):
    """
    Load the data from the data folder.
    args:
      kind: "raw" or "processed"
    """
    if kind == "raw":
        df = pd.read_csv("data/raw/jobs.csv", sep=";")
    elif kind == "processed":
        df = pd.read_csv("data/processed/cleaned_jobs.csv", sep=";")
    return df

def transform_string(s):
    return s[1:-1].replace("'", "").replace(", ", " ")

df = load_data()
df['description'] = df['description'].apply(transform_string)
job_descriptions = df[['id','description']][:100]


client = OpenAI()

yaml_file = 'extracted_skills.yaml'
if os.path.exists(yaml_file):
    with open(yaml_file, 'r') as file:
        extracted_skills = yaml.safe_load(file) or {}
else:
    extracted_skills = {}

for index, offer in job_descriptions.iterrows():
    # Make the API call

    if offer['id'] in extracted_skills:
        continue  # Skip this id if already processed

    response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a professional job recruiter. Your task is to extract 7 most relevant skills required for a job position and present them in a raw list format: [skill1, skill2, ... skill7]."},
        {"role": "user", "content": "Extract 7 most relevant skills. Here is the job description: '''would like part ryanair group amazing cabin crew family k crew customer oriented love delivering great service want fast track career opportunity would delighted hear experience required bag enthusiasm team spirit europe largest airline carrying k guest daily flight looking next generation cabin crew join u brand new copenhagen base flying board ryanair group aircraft amazing perk including discounted staff travel destination across ryanair network fixed roster pattern free training industry leading pay journey becoming qualified cabin crew member start week training course learn fundamental skill require part day day role delivering top class safety customer service experience guest course required study exam taking place regular interval training culminates supernumerary flight followed cabin crew wing member ryanair group cabin crew family immersed culture day one career opportunity endless including becoming number base supervisor european base manager regional manager aspire becoming director inflight life cabin crew fun rewarding however demanding position safety number priority required operate early late shift report duty early morning early roster return home midnight afternoon roster morning person think twice applying requirement bag enthusiasm customer serviceoriented background ie previous experience working bar restaurant shop etc applicant must demonstrate legal entitlement work unrestricted basis across euyou must cm cm height must able swim meter unaided help hardworking flexible outgoing friendly personality adaptable happy work shift roster enjoy dealing public ability provide excellent customer service attitude comfortable speaking writing english ease passion travelling meeting new people benefit free cabin crew training course adventure experience lifetime within cabin crew network explore new culture city colleague day flexible day day staff roster unlimited highly discounted staff travel rate sale bonus free uniform year security working financially stable airline daily per diem provided whilst training direct employment contract highly competitive salary package click link start new exciting career sky'''"},
        {"role": "assistant", "content": "[Customer Service Orientation, Flexibility and Adaptability, Communication Skills, Teamwork, Safety Awareness, Physical Fitness, Interpersonal Skills]"},
        {"role": "user", "content": f"Extract 7 most relevant skills. Here is the job description: '''{offer['description']}'''"},
    ]
    )

    # Extract skills from the response
    skills = response.choices[0].message.content

    # Store the results
    extracted_skills[offer['id']] = skills

    # Write the results to a YAML file
    with open('extracted_skills.yaml', 'w') as file:
        yaml.dump(extracted_skills, file, default_flow_style=False)

    print(f"Saving extracted skills for offer ID: {offer['id']}")
