import pandas as pd
import pickle
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
import nltk.data
train = pd.read_csv("data/kaggle/labeledTrainData.tsv", header=0,delimiter="\t",quoting=3)
test = pd.read_csv("data/kaggle/testData.tsv",header=0,delimiter="\t",quoting=3)
unlabeled_train = pd.read_csv("data/kaggle/unlabeledTrainData.tsv",header=0,delimiter="\t",quoting=3)

print((train["review"].size,test["review"].size,unlabeled_train["review"].size))
app_json = pickle.load(open("app.pickle", "rb"))
mobile_json = pickle.load(open("mobile.pickle", "rb"))
mobilepd = pd.DataFrame(mobile_json)
apppd = pd.DataFrame(app_json)
print((mobilepd.size, apppd.size))
print(list(mobilepd.columns.values))
print(list(apppd.columns.values))


def review_to_wordlist(review, remove_stopwords=False):
    review_text = BeautifulSoup(review).get_text()
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences(review, tokenizer, remove_stopwords=False ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( review_to_wordlist( raw_sentence, \
              remove_stopwords ))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []
print("\n\n\n\nParsing sentences from training set\n\n\n\n")
for review in train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("\n\n\n\nParsing sentences from unlabeled set\n\n\n\n")
for review in unlabeled_train["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("\n\n\n\nParsing sentences from mobile set\n\n\n\n")
for review in mobilepd["review"]:
    sentences += review_to_sentences(review, tokenizer)

print("\n\n\n\nParsing sentences from app set\n\n\n\n")
for review in apppd["review"]:
    sentences += review_to_sentences(review, tokenizer)

print(len(sentences))
pickle.dump(sentences, open("sentences.pickle", "wb"))
