import nltk as nltk
from collections import Counter
import math
from nltk.util import bigrams
# writing simplified naive bayes

# 0. making some data
from nltk.tokenize import RegexpTokenizer # why this one? what makes this better?
tokenizer = RegexpTokenizer(r'\w+')
reviews_pos_train = [["This movie was the best!"], ["This movie was a triumph!"], ["Saw it too many times."], ["Love, love, loved it!"]]
reviews_neg_train = [["Aweful."], ["I would like my time back."], ["My evening is ruined."], ["I hated this!"]] 

reviews_pos_dev = [["This movie was best!"], ["Loved it!"], ["I thought it was the best!"], ["Some say it was terrible, but I loved it."]]
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

count_pos, count_neg = Counts(train_data, train_labels)
count_pos_bi, count_neg_bi = Counts(train_data, train_labels, Bigram=True) 

likelihood_pos = likelihood_Laplace_smoothing(count_pos)
likelihood_neg = likelihood_Laplace_smoothing(count_neg)

likelihood_pos_bi = likelihood_Laplace_smoothing(count_pos_bi) # the same process for laplace smoothing
likelihood_neg_bi = likelihood_Laplace_smoothing(count_neg_bi)

# 2. Estimate posterior probabilities using naive bayes 
# (put calculated guesses into var. called yhat)
bigram_lambda = 1

pos_prior = .5
neg_prior = 1 - pos_prior

yhat = []
def Token_Check(likelihood, token):
        if token in likelihood.keys():
            return likelihood[token]
        else:
            return likelihood["UNK"]
    
for doc in dev_data:
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
        yhat.append(1)
    else:
        yhat.append(0)

print(yhat, dev_labels)
