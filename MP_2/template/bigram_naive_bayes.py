# bigram_naive_bayes.py
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
from nltk.util import bigrams


'''
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=1.0, bigram_laplace=1.0, bigram_lambda=1.0, pos_prior=0.5, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)

    def Counts(data, labels, Bigram=False): 
        count_pos = Counter()
        count_neg = Counter()
        if Bigram:
            for tag, doc in zip(labels, data):
                if tag: # tag is either 1 or 0 (ie, True or False)
                    
                    count_pos += Counter(bigrams(doc))
                else:
                    count_neg += Counter(bigrams(doc))
            return count_pos, count_neg
        else:
            for tag, doc in zip(labels, data):
                if tag: # tag is either 1 or 0 (ie, True or False)
                    count_pos += Counter(doc)
                else:
                    count_neg += Counter(doc)
            return count_pos, count_neg
    
    def likelihood_Laplace_smoothing(data_dict, alpha=1.0):
        keys = data_dict.keys()
        values = data_dict.values()
        V = len(keys)
        n = sum(values)
        denominator = n + alpha * (V + 1)
        likelihoods_dict = dict(zip(keys, [(W_count + alpha) / denominator for W_count in values]))
        likelihoods_dict["UNK"] = alpha / denominator
        return Counter(likelihoods_dict)

    count_pos, count_neg = Counts(train_set, train_labels)
    count_pos_bi, count_neg_bi = Counts(train_set, train_labels, Bigram=True) 

    likelihood_pos = likelihood_Laplace_smoothing(count_pos, alpha=unigram_laplace)
    likelihood_neg = likelihood_Laplace_smoothing(count_neg, alpha=unigram_laplace)

    likelihood_pos_bi = likelihood_Laplace_smoothing(count_pos_bi, alpha=bigram_laplace) # the same process for laplace smoothing
    likelihood_neg_bi = likelihood_Laplace_smoothing(count_neg_bi, alpha=bigram_laplace)


    def Token_Check(likelihood, token):
        if token in likelihood.keys():
            return likelihood[token]
        else:
            return likelihood["UNK"]
    
    neg_prior = 1 - pos_prior
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        pos_posterior = math.log(pos_prior) # if you simplify the mixed model the lambda coeff. for the priors,
                                            # they equate to 1 so no need to add those here.
        neg_posterior = math.log(neg_prior)
        bigram_token = list(bigrams(doc))
        
        for token in doc:
            pos_posterior += (1-bigram_lambda) * math.log(Token_Check(likelihood_pos, token))
            neg_posterior += (1-bigram_lambda) * math.log(Token_Check(likelihood_neg, token)) 
        
        for bi_token in bigram_token:
            pos_posterior +=  bigram_lambda * math.log(Token_Check(likelihood_pos_bi, bi_token))
            neg_posterior += bigram_lambda * math.log(Token_Check(likelihood_neg_bi, bi_token))
        
        if pos_posterior > neg_posterior:
            yhats.append(1)
        else:
            yhats.append(0)

    return yhats



