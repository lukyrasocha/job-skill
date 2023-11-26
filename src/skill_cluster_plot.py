from collections import Counter
import spacy
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.logger import working_on
from src.utils import load_data
from nltk.stem import WordNetLemmatizer
import re
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
            word_list = [word.strip() for word in skills.loc[skills['id'] == id]['skills'].values[0][1:-1].split(',')]
            skill_group.extend(word_list)
        cluster_skill[label] = count_words(entry for entry in skill_group)
    return cluster_skill
    

def plot_top(words, key, ax, top_n):
    sorted_word_frequency = sorted(words.items(), key=lambda x: x[1], reverse=True)
    top_words = dict(sorted_word_frequency[:top_n])

    words = list(top_words.keys())
    frequencies = list(top_words.values())

    ax.bar(words, frequencies)
    ax.set_title(f'Top {top_n} Word Frequency')
    ax.set_title(f'Cluster{key}')
    return ax

def plot_wordcloud(words, key, ax):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(words)
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f' Cluster {key}')
    return ax

def plot_frequency_histogram(skills,filename,top_n=5):

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
        axs[i] = plot_top(word_frequency, key, axs[i], top_n)

    # Adjust layout
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])
    plt.figtext(0.025,0.5, 'frequency', ha="center", va="top", fontsize=20, rotation='vertical')
    # Show the plot
    fig.savefig('figures/'+filename+'_histogram.jpg')


    # Create subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(N, 2*N))
    fig.suptitle('Word Cloud Subplots', fontsize=16)

    # Flatten the axs array for easier indexing
    axs = axs.flatten()
    # Generate word clouds for each inner dictionary and plot on a subplot
    for i, (key, word_frequency) in enumerate(skills.items()):
        axs[i] = plot_wordcloud(word_frequency, key, axs[i])

    # Remove empty subplots if needed
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])

    # Show the plot
    fig.savefig('figures/'+filename+'_wordcloud.jpg')



def plot_compared(gt, compare, top_n=5):
    N = len(gt)
    num_cols = 2  # Number of columns in the subplot grid
    num_rows = N  # Ceiling division to determine the number of rows

    # Create subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(24,5*N))
    fig.suptitle('Word Frequency in each cluster', fontsize=25)

    # Flatten the axs array for easier indexing
    axs = axs.flatten()
    # Plot each dictionary on a subplot
    i = 0
    for key, word_frequency in gt.items():
        axs[i] = plot_top(word_frequency, key, axs[i], top_n)
        axs[i+1] = plot_top(compare[key], key, axs[i+1], top_n)
        i += 2
        

    # Adjust layout
    plt.figtext(0.3,0.97, "Ground truth", ha="center", va="top", fontsize=20, color="black")
    plt.figtext(0.8,0.97, "The comparing result", ha="center", va="top", fontsize=20, color="black")
    plt.figtext(0.025,0.5, 'frequency', ha="center", va="top", fontsize=20, rotation='vertical')
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])
    
    # Show the plot
    fig.savefig('figures/compared_histogram.jpg')


    # Create subplots
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(24,5*N))
    fig.suptitle('Word Cloud Subplots', fontsize=25)

    # Flatten the axs array for easier indexing
    axs = axs.flatten()
    # Generate word clouds for each inner dictionary and plot on a subplot
    i = 0
    for key, word_frequency in gt.items():
        axs[i] = plot_wordcloud(word_frequency, key, axs[i])
        axs[i+1] = plot_wordcloud(compare[key], key, axs[i+1])
        i += 2

    # Remove empty subplots if needed
    for j in range(i + 1, num_rows * num_cols):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.figtext(0.3,0.97, "Ground truth", ha="center", va="top", fontsize=20, color="black")
    plt.figtext(0.8,0.97, "The comparing result", ha="center", va="top", fontsize=20, color="black")
    plt.tight_layout(rect=[0.04, 0, 1, 0.96])
    # Show the plot
    fig.savefig('figures/compared_wordcloud.jpg')

def skill_analysis(gt,compare):
    
    # Load the datasets
    working_on("Skill analysis ...")
    
    skills = load_data("skills")
    
    if not gt.empty:
        gt_skill = group_skill_cluster(skills, gt)
    if not compare.empty:
        cluster_skill = group_skill_cluster(skills, compare)
        working_on("Plot wordcloud and histogram...")
        plot_frequency_histogram(cluster_skill,'cluster',5)
    if not gt.empty and not compare.empty:
        plot_compared(gt_skill, cluster_skill)
                                                                              
if __name__ == "__main__":
    gt = load_data(kind="ground_truth")
    compare = pd.read_csv('clusters/tfidf_noun_clusters.csv')
    skill_analysis(gt, compare)