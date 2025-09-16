# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently) # test and dev sets are same
    return train_set, train_labels, dev_set, dev_labels
#train_set is already tokenized 
#train_labels are in order of the files [[pos. sub set first] + [neg. subset second]]

"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=.01, pos_prior=0.8, silently=False):
    print_values(laplace,pos_prior)

    def Counts(data, labels):
        count_pos = Counter()
        count_neg = Counter()
        for tag,tokens in zip(labels, data):
            if tag: # tag is either 1 or 0 (ie, True or False)
                count_pos += Counter(tokens)
            else:
                count_neg += Counter(tokens)
        return count_pos, count_neg

    def likelihood_Laplace_smoothing(data_dict, alpha=laplace):
        keys = data_dict.keys()
        values = data_dict.values()
        V = len(keys)
        n = sum(values)
        denominator = n + alpha * (V + 1)
        likelihoods_dict = dict(zip(keys, [(W_count + alpha) / denominator for W_count in values]))
        likelihoods_dict["UNK"] = alpha / denominator
        return Counter(likelihoods_dict)

    count_pos, count_neg = Counts(train_set, train_labels)
    likelihood_pos = likelihood_Laplace_smoothing(count_pos)
    likelihood_neg = likelihood_Laplace_smoothing(count_neg)

    # perform naive bayes on training data

    yhats = [] # With trained values, perform estimations on dev_set and put calculated guesses into yhats

    def Token_Check(likelihood, token):
        if token in likelihood.keys():
            return likelihood[token]
        else:
            return likelihood["UNK"]
        
    neg_prior = 1 - pos_prior
    for doc in tqdm(dev_set, disable=silently):
        pos_posterior = math.log(pos_prior)
        neg_posterior = math.log(neg_prior)
        for token in doc:
            pos_posterior += math.log(Token_Check(likelihood_pos, token))
            neg_posterior += math.log(Token_Check(likelihood_neg, token))
        if pos_posterior > neg_posterior:
            yhats.append(1)
        else:
            yhats.append(0)
    return yhats
