### Project Title: Skillset Clustering in Job Markets
#### TODO:


- [x] **Web Scraping**: Scrape the job posting data from **LinkedIn**
- [x] **Text Preprocessing**: Clean and preprocess the text data (tokenization, stemming, and removing stop words)
- [ ] **Feature Extraction**: Use Tf.idf measure to convert the text data into numerical format.
- [ ] **Clustering**: Apply clustering algorithms like K-means or DBSCAN to group similar job postings together.
- [ ] **Skill Extraction**: Use Named Entity Recognition (NER) or keyword extraction to identify the skills mentioned in each cluster.
- [ ] **Visualization**: Use visualization tools to represent the clusters and the top skills.
- [ ] **Association Rule Mining**: Using A-Priori Algorithm, identify patterns like: "If a job requires skill A, it's also likely to require skill B."
- [ ] **Implement Word2Vec** or other word embedding techniques (not directly taught in the lectures) to capture the context and semantic meaning of words in job descriptions.
- [ ] How do skills differ based on location, industry or employment type

#### Motivation
With the rapidly evolving job market, there's a constant need for job seekers, companies, and educational institutions to understand which skills are in demand. By analyzing job descriptions from LinkedIn, we can gain insights into which skills are trending and how job requirements differ by location, company, and employment type. 

#### Objective 1:

To scrape job postings from LinkedIn, cluster them into distinct categories based on the job descriptions, and identify the top 5 skills required for each cluster. This can help job seekers understand the current market trends and prepare accordingly.

#### Objective 2:

Determine which skills and attributes are trending in the job market based on the job descriptions and understand how these trends vary either by location, company, or employment type.

#### Objective 3:
Integrate machine learning models to predict the popularity or demand for certain skills in the future, based on historical and current data trends. Utilize time-series forecasting models or trends analysis to achieve this.

#### Tools and Techniques:

- **Web Scraping**: Scrape the job posting data from **LinkedIn**
- **Text Preprocessing**: Clean and preprocess the text data (tokenization, stemming, and removing stop words)
- **Feature Extraction**: Use Tf.idf measure to convert the text data into numerical format.
- **Clustering**: Apply clustering algorithms like K-means or DBSCAN to group similar job postings together.
- **Skill Extraction**: Use Named Entity Recognition (NER) or keyword extraction to identify the skills mentioned in each cluster.
- **Visualization**: Use visualization tools to represent the clusters and the top skills.
- **Association Rule Mining**: Using A-Priori Algorithm, identify patterns like: "If a job requires skill A, it's also likely to require skill B."
- **Implement Word2Vec** or other word embedding techniques (not directly taught in the lectures) to capture the context and semantic meaning of words in job descriptions.

#### Steps:

1. **Data Collection**:

   - Use web scraping tools to extract job postings (title, description, location, etc.) from LinkedIn.
   - Store the collected data in a structured format for analysis.

2. **Data Preprocessing**:

   - Clean the text data by removing special characters, numbers, and stop words.
   - Apply stemming or lemmatization to reduce words to their base form.

3. **Feature Extraction**:

   - Apply the **Tf.idf** measure to convert text data into numerical format, capturing the importance of different terms.

4. **Clustering**:

   - Use **K-means, DBSCAN**, or other clustering algorithms to group similar job postings.
   - Optimize the number of clusters using techniques like the Elbow Method
   - Calculate the quality of the cluster using **Davies-Bouldin index**

5. **Skill Extraction**:

   - Apply NER or keyword extraction to identify the skills in each cluster.
   - Rank the skills based on their frequency or importance.

6. **Visualization**:

   - Visualize the clusters and the top 5 skills for each cluster using bar graphs, word clouds, or other visualization methods.

7. **Evaluation**:
   - Evaluate the quality of clusters and the relevance of extracted skills.
   - Provide insights and recommendations based on the analysis.

#### Report Outline:

- **Introduction**: Introduce the problem and its relevance in the current job market.
- **Methodology**: Explain the tools and techniques used, including web scraping, text preprocessing, clustering, and skill extraction.
- **Results**: Present the clusters and the top 5 skills for each cluster, along with visualizations.
- **Evaluation**: Discuss the quality of clusters, the relevance of skills extracted, and the insights gained from the analysis.
- **Conclusion**: Conclude with the project’s findings, potential applications, and suggestions for future work.
