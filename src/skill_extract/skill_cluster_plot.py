from collections import Counter
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.helper.logger import working_on
from src.helper.utils import load_data
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def count_words(words):

  # Use spacy to reduce words
  # nlp = spacy.load("en_core_web_sm")
  # word = [ re.sub(r"[^a-zA-Z0-9\s]", "", w) for w in words]
  # word = [token.lemma_ for token in nlp(" ".join(word))]

  # Use WordNet lemmatizer to reduce words
  lemmatizer = WordNetLemmatizer()
  # Apply lemmatization to the words
  word = [lemmatizer.lemmatize(word.lower()) for word in words]

  word_counts = Counter(word)
  return dict(word_counts.most_common())


def group_skill_cluster(skills, file):
  cluster_skill = {}
  for label in set(file["cluster"]):
    skill_group = []
    for id in file.loc[file["cluster"] == label]["id"]:
      word_list = [word.strip() for word in skills.loc[skills['id']
                                                       == id]['skills'].values[0][1:-1].split(',')]
      skill_group.extend(word_list)
    cluster_skill[label] = count_words(entry for entry in skill_group)
  return cluster_skill


def plot_top(words, key, ax, top_n):
  sorted_word_frequency = sorted(
      words.items(), key=lambda x: x[1], reverse=True)
  top_words = dict(sorted_word_frequency[:top_n])

  words = list(top_words.keys())
  frequencies = list(top_words.values())

  ax.bar(words, frequencies, color='dodgerblue')
  ax.set_title(f'Top {top_n} Word Frequency')
  ax.set_title(key)
  return ax


def plot_wordcloud(words, key, ax):
  wordcloud = WordCloud(width=800, height=400,
                        background_color='white').generate_from_frequencies(words)
  ax.imshow(wordcloud, interpolation='bilinear')
  ax.axis('off')
  ax.set_title(key)
  return ax


def plot_frequency_histogram(skills, filename, top_n=5):

  N = len(skills)
  num_cols = 2  # Number of columns in the subplot grid
  num_rows = N // num_cols  # Ceiling division to determine the number of rows

  # Create subplots
  fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(N, 2*N))
  fig.suptitle('Word Frequency in each cluster', fontsize=20)

  # Flatten the axs array for easier indexing
  axs = axs.flatten()

  # Plot each dictionary on a subplot
  for i, (key, word_frequency) in enumerate(skills.items()):
    axs[i] = plot_top(word_frequency, 'Cluster {}'.format(key), axs[i], top_n)

  # Adjust layout
  plt.tight_layout(rect=[0.04, 0, 1, 0.96])
  plt.figtext(0.025, 0.5, 'frequency', ha="center",
              va="top", fontsize=20, rotation='vertical')
  # Show the plot
  fig.savefig('figures/'+filename+'_histogram.jpg')

  # Create subplots
  fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(N, 2*N))
  fig.suptitle('Word Cloud Subplots', fontsize=16)

  # Flatten the axs array for easier indexing
  axs = axs.flatten()
  # Generate word clouds for each inner dictionary and plot on a subplot
  for i, (key, word_frequency) in enumerate(skills.items()):
    axs[i] = plot_wordcloud(word_frequency,  'Cluster {}'.format(key), axs[i])

  # Remove empty subplots if needed
  for j in range(i + 1, num_rows * num_cols):
    fig.delaxes(axs[j])

  # Adjust layout
  plt.tight_layout(rect=[0.04, 0, 1, 0.96])

  # Show the plot
  fig.savefig('figures/'+filename+'_wordcloud.jpg')


def plot_compared(gt, compare, file, top_n=5):
  num_cols = 2  # Number of columns in the subplot grid
  num_rows = 1  # Ceiling division to determine the number of rows

  # Create subplots
  fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 6))
  fig.suptitle('Word Frequency in {}'.format(file), fontsize=20)

  # Flatten the axs array for easier indexing
  axs = axs.flatten()
  # Plot each dictionary on a subplot
  axs[0] = plot_top(gt, 'Ground Truth', axs[0], top_n)
  axs[1] = plot_top(compare, 'The Comparing Cluster', axs[1], top_n)

  # Adjust layout
  plt.figtext(0.025, 0.6, 'frequency', ha="center",
              va="top", fontsize=15, rotation='vertical')
  plt.tight_layout(rect=[0.04, 0, 1, 0.96])

  # Show the plot
  fig.savefig('figures/compared_histogram_'+ file+ '.jpg')

  # Create subplots
  fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 6))
  fig.suptitle('Word Cloud Subplots in {}'.format(file), fontsize=20)

  # Flatten the axs array for easier indexing
  axs = axs.flatten()
  # Generate word clouds for each inner dictionary and plot on a subplot
  axs[0] = plot_wordcloud(gt,'Ground Truth', axs[0])
  axs[1] = plot_wordcloud(compare, 'The Comparing Cluster', axs[1])

  # Adjust layout
  plt.tight_layout(rect=[0.01, 0, 1, 0.96])
  # Show the plot
  fig.savefig('figures/compared_wordcloud_'+ file+ '.jpg')


def skill_analysis(compare, gt=pd.DataFrame()):

  # Load the datasets
  working_on("Skill analysis ...")

  skills = load_data("skills_gpt")

  if not gt.empty:
    gt_skill = group_skill_cluster(skills, gt)
  if not compare.empty:
    cluster_skill = group_skill_cluster(skills, compare)
    working_on("Plot wordcloud and histogram ...")
    plot_frequency_histogram(cluster_skill, 'cluster', 5)

  if not gt.empty and not compare.empty:
    match_id = {}
    for i in set(gt['cluster']):
        gt_id = gt.loc[gt['cluster']==i]['id']
        
        X = skills[skills['id'].isin(gt_id)]['skills'].values[0][1:-1]
        

        similarity = np.empty(20)
        for j in set(compare['cluster']):
            compare_id = compare.loc[compare['cluster']==j]['id']
            Y= skills[skills['id'].isin(compare_id)]['skills'].values[0][1:-1]
            
        
            # Consine similarity 
            vectorizer = CountVectorizer()
            vec = vectorizer.fit_transform([X,Y])
            similarity[j] = cosine_similarity(vec[0], vec[1])[0][0]
        
        category = gt.loc[gt['cluster']==i]['category'].unique()
        match_id[i] = similarity.argmax()

        if similarity[match_id[i]] > 0.8:
            plot_compared(gt_skill[i], cluster_skill[match_id[i]], category[0])



if __name__ == "__main__":
  gt = load_data(kind="ground_truth_gpt")
  compare = pd.read_csv('clusters/tfidf_noun_clusters.csv')
  skill_analysis(gt, compare)
