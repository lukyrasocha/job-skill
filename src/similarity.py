import numpy as np
import pandas as pd
import mmh3
from tqdm import tqdm
# find the minimum value between hashes
def minhashes(shingles, seeds):
    hashs =[]
    for seed in range(seeds):
        mini = float('inf')
        for shi in shingles:
            # hashes a list of strings
            hash = 0
            for e in shi:
                hash = hash ^ mmh3.hash(e, seed)
            # find the minimum value
            if mini > hash:
                mini = hash
        hashs.append(mini)
    return list(hashs)

# get every signature in data
def signatures(df, k, seeds):
    hash_dic = {}
    for i in range(len(df)):
        # make a description into k-shingles
        shi = []
        for ch in range(len(df[i])-k+1):
            shi.append(df[i][ch:ch+k])
        
        hash_dic[i] = minhashes(list(shi), seeds)
    return hash_dic

def convert_matrix(N, scores):
    similarity_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == j:
                similarity_matrix[i][j] = 1.0
            elif i > j:
                similarity_matrix[i][j] = scores[(j, i)]
            else:
                similarity_matrix[i][j] = scores[(i, j)]
    return similarity_matrix

def find_sim(data, q, seed):
    """
    Finds the similarity between any two job's description for a given dataset using the shingle, minihash 
    and jaccord similarity.

    Args:
      data: The "data" parameter is the dataset that you want to cluster. It should be a 2D array-like
    object, such as a numpy array or a pandas DataFrame, where each row represents a data point and each
    column represents a feature of that data point.
      q: The q parameter represents the number of shingles ( k = 2 or 3 for small documents such as emails)
      seed: The seed parameter represents how mand seeds to use for doing the minihashes

    Returns:
      A dictionary where the keys are pairs of indices, and the values are scores representing the similarity 
      between job descriptions at those indices
    """
    sign = signatures(data, q, seed)

    score_list = {}
    keys =list(sign.keys())
    for k in tqdm(range(len(keys)-1), desc = 'Calculating jaccard similarity', delay=0.1):
        for j in tqdm(range(k+1,len(keys)), delay=0.1):
            # calculate jaccard simiarity and store the score
            score = len(np.intersect1d(sign[keys[k]],sign[keys[j]]))/len(np.union1d(sign[keys[k]],sign[keys[j]]))
            score_list[(keys[k],keys[j])] = score
    return score_list