import nltk as nltk

# This is how to use PorterStemmer
from nltk.stem import PorterStemmer

porter = PorterStemmer()

words = ["running", "runner", "runs", "ran"]
stemmed_words = [porter.stem(word) for word in words]

print(stemmed_words)

# This is how to use word_tokenize
from nltk.tokenize import word_tokenize

text = "NLTK is a powerful library for natural language processing."
tokens = word_tokenize(text)
print(tokens)

# this is how the labels are constructed, they are in order of the files
labels = 42 * [1] + 10 * [0]
print(labels)
print(sum(labels)) # total number of pos.(label=1)s
print(len(labels) - sum(labels)) # total number of neg.(label=0)s

# Basics of uisng Counter
from collections import Counter

words = ['the', 'ant', 'is', 'the', 'aunt', 'the']
# wrapping Counter(..).x() in list() avoids the dict_x type problem
keys = list(Counter(words).keys())
print(keys) # the unique objects in a list
values = list(Counter(words).values())
print(values) # the frequency of each unique object
print(len(keys)) # total number of unique entries
print(len(words)) # count of all the words 

dictionary = Counter(words)
print(dictionary) # each word assigned with it's frequency in order of apperance as a dictionary type

prob_dict = dict(zip(keys, [W/len(words) for W in values])) # naive probabilities assigned to each word in a dict
print(prob_dict)

# Multiplying probabilities
import math
probs = list(prob_dict.values())
pos_prior = 0.5
naive_bayes = pos_prior * math.prod(probs)
print(naive_bayes)

# writing simplified naive bayes

# 0. making some data
from nltk.tokenize import RegexpTokenizer # why this one? what makes this better?
tokenizer = RegexpTokenizer(r'\w+')
reviews_pos_train = [["This movie was the best!"], ["This movie was a triumph!"], ["Saw it too many times."], ["Love, love, loved it!"]]
reviews_neg_train = [["Aweful."], ["I would like my time back."], ["My evening is ruined."], ["I hated this!"]] 

reviews_pos_dev = [["Loved it!"], ["I thought it was the best!"], ["Some say it was terrible, but I loved it."]]
reviews_neg_dev = [["Terrible!"], ["Some said this was the best, but I hated it."], ["why would you waste your money on this?"]]

def review_token(files):
    X0 = []
    for review in files:
        text = []
        for line in review:
            text += tokenizer.tokenize(line)
        X0.append(text)
    return X0

train_pos = review_token(reviews_pos_train)
train_neg = review_token(reviews_neg_train)
train_labels = len(train_pos) * [1] + len(train_neg) * [0]
train_data = train_pos + train_neg

dev_pos = review_token(reviews_pos_dev)
dev_neg = review_token(reviews_neg_dev)
dev_data = dev_pos + dev_neg
dev_labels = len(dev_pos) * [1] + len(dev_neg) * [0] 

# 1. create a dictionary of words with their likelihoods

def Counts(data, labels):
    count_pos = Counter()
    count_neg = Counter()
    for tag,tokens in zip(labels, data):
        if tag: # tag is either 1 or 0 (ie, True or False)
            count_pos += Counter(tokens)
        else:
            count_neg += Counter(tokens)
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

count_pos, count_neg = Counts(train_data, train_labels)
likelihood_pos = likelihood_Laplace_smoothing(count_pos)
likelihood_neg = likelihood_Laplace_smoothing(count_neg)

# 2. Estimate posterior probabilities using naive bayes 
# (put calculated guesses into var. called yhat)

pos_prior = .5
neg_prior = 1 - pos_prior

yhat = []
def Token_Check(likelihood, token):
        if token in likelihood.keys():
            return likelihood[token]
        else:
            return likelihood["UNK"]
    
for review in dev_data:
    pos_posterior = math.log(pos_prior)
    neg_posterior = math.log(neg_prior)
    for token in review:
        pos_posterior += math.log(Token_Check(likelihood_pos, token))
        neg_posterior += math.log(Token_Check(likelihood_neg, token))
    if pos_posterior > neg_posterior:
        yhat.append(1)
    else:
        yhat.append(0)

print(yhat, dev_labels)

