import numpy as np
import pandas as pd
import mmh3
import ast

# make a description into k-shingles
def shinge(s, k):
    ans = []
    words = ast.literal_eval(s)
    for ch in range(len(words)-k+1):
        ans.append(words[ch:ch+k])
    return list(ans)

# hashes a list of strings
def listhash(l,seed):
	val = 0
	for e in l:
		val = val ^ mmh3.hash(e, seed)
	return val 

# find the minimum value between hashes
def minhashes(shingles, seeds):
    hashs =[]
    for seed in range(seeds):
        mini = float('inf')
        for shi in shingles:
            hash = listhash(shi, seed)
            if mini > hash:
                mini = hash
        hashs.append(mini)
    return list(hashs)

# get every signature in data
def signatures(df, k, seeds):
    hash_dic = {}
    for i in range(len(df)):
        shi = shinge(df[i],k)
        hash_dic[i] = minhashes(shi, seeds)
    return hash_dic

# calculate jaccard simiarity
def jaccard(name_o, name_s, signature_dict):
    first = signature_dict[name_o]
    second = signature_dict[name_s]

    return len(np.intersect1d(first,second))/len(np.union1d(first,second))

# find all of the simiarity between any two job's description
def find_sim(df, q, seed):
    sign = signatures(df, q, seed)

    score_list = {}
    keys =list(sign.keys())
    for k in range(len(keys)-1):
        for j in range(k+1,len(keys)):
            score = jaccard(keys[k],keys[j],sign)
            score_list[(keys[k],keys[j])] = score
    return score_list