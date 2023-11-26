### Project Title: Skillset Clustering in Job Markets

#### Purpose of our method (inference in production)

- We have evaluated various different clustering approaches and compared them to their ground truth (based on different ways one can view what the ground truth is - e.g. is it based on countries, job function, industries?). From these clusters we then extract the most important skills that charactarize each cluster. One could argue, that when you already know what the ground truth is (when looking at linkedin job posts, that information is available) why don't you just extract the skills from there? The use case becomes aparent when not being presented with ground truth - there is a lot of different job portals, where you have only the actual job descriptions, using our method one can easily place the job with respect to all the other similar jobs (similarity can mean different things - determined by the selected clustering approach). One could further argue, if I have the job description, can I not just extract the skills directly from there? Definitely. Our method, however, allows you to also extract skills from very poorly,vague job descriptions that do not explicitly mention what skills you should have, or what type of person you should be. For instance the job description: "Sales Assistant, looking for a sales assistant to our little cozy team in the middle of Copenhagen, relevant education required". To a person with no idea what is expected from a "sales assistant" role, one could use our system to input the job description, which will then be put in a relevant cluster and the skills required will be given based on other most similar job descriptions.

- Our system could then look like a web app which shows all the clums of clusters and an input field, after entering the job description, the relevant cluster will be highlihted and the skills shown in some format.

#### TODO:


- [x] **Web Scraping**: Scrape the job posting data from **LinkedIn**
- [x] **Text Preprocessing**: Clean and preprocess the text data (tokenization, stemming, and removing stop words)
- [x] **Feature Extraction**: Use Tf.idf measure to convert the text data into numerical format.
- [ ] **Clustering**: Apply clustering algorithms like K-means or DBSCAN to group similar job postings together.

      a) Based on TFIDF (Thomasz, Henrietta)
      b) Based on doc2vec (Lukas)
      c) Based on similarity values (Tinghui)
- [ ] **Skill Extraction**: Use Named Entity Recognition (NER) or keyword extraction to identify the skills mentioned in each cluster.
- [ ] **Visualization**: Use visualization tools to represent the clusters and the top skills.
- [ ] **Association Rule Mining**: Using A-Priori Algorithm, identify patterns like: "If a job requires skill A, it's also likely to require skill B."
- [ ] **Implement Word2Vec** or other word embedding techniques (not directly taught in the lectures) to capture the context and semantic meaning of words in job descriptions.
- [ ] How do skills differ based on location, industry or employment type
- [ ] Measure job popularities, by looking at DATE_SCRAPED, DATE_POSTED and number of applications it received

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
