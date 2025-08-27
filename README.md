# Identifying key skills in job markets 💼

Written paper can be accessed [here](https://github.com/lukyrasocha/job-skill/blob/main/handin/report.pdf)

### Authors
- Ting-Hui Cheng (s232855)
- Tomasz Truszkowski (s223219)
- Lukas Rasocha (s233498)
- Henrietta Domokos (s233107)

### 🔁 Reproducibility

#### Steps to Reproduce the Results

```bash
# Clone the repository
git clone https://github.com/lukyrasocha/02807-comp-tools.git

# Navigate to the directory
cd 02807-comp-tools

# Create a Python environment (Version 3.10.4; other versions not tested)
conda create -n "comp-ds" python="3.10.4"

# Install dependencies
pip install -r requirements.txt

# Set the PYTHONPATH to include our project directory
export PYTHONPATH="$PWD"

# Run the project pipeline
python src/main.py
```

```python
## If you don't have them already (you get errors), you might need to instal NLTK packages

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## 📝 Introduction and Project Overview
In this project we aim to identity key skills in job markets through clustering analysis. The main motivation is to help job seekers understand the ever evolving job market by allowing them to view and compare the most prominent skills for each cluster of job descriptions. Using our approach, a job seeker is able to analyse and compare trending skills for different clusters, or could cluster a new job description and get back a set of relevant skills based on other similar jobs in the same cluster.

Project Breakdown:

🔎 **1. Data Scraping:**

Developed a LinkedIn Scraper to gather job postings, fetching details like job title, industry, number of applicants, posting date, and company.

🧹 **2. Data Preprocessing:**

Applied text preprocessing methods such as lemmatization, tokenization, and stop word removal to clean the job descriptions.

📐 **3. Clustering Analysis:**

Employed various techniques to convert textual job descriptions into numerical formats for clustering, including:
- TFIDF (Total Frequency-Inverse Document Frequency) 
  - Using different parts of speech (nouns, verbs, adjectives).
  - Using the whole job description.
- Word2Vec and Doc2Vec for embedding generation.
- Similarity-based vector representation.
- Explored clustering algorithms like K-Means, DBSCAN, and Gaussian Mixture Model.

📋 **4. Establishing Ground Truth:**

Investigated methods to determine a baseline 'ground truth' for comparison, utilizing techniques like one-hot encoding, keyword inference, and categorization through OpenAI's GPT-3.5-turbo model.

📈 **5. Evaluation:**

Selected the optimal clustering approach based on its closeness to the ground truth, quantified using the Normalized Mutual Information (NMI) score.

🔬 **6. Skill Extraction:**

Extracted top skills from each cluster using machine learning models. We used an opensource Hugging Face model trained for extracting hard and soft skills from text and general purpose Large Language Model (LLM) GPT-3.5-turbo prompted to extract skills from text.

📊 **7. Skill Analysis:**

Visualized prominent skills per cluster through word clouds and bar charts, focusing on the frequency of skills within each cluster.
