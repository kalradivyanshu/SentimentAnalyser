import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import word2vec
from bs4 import BeautifulSoup
import os
import pickle
from time import time
import re
import sys
import math

app_json = pickle.load(open("app.pickle", "rb"))
mobile_json = pickle.load(open("mobile.pickle", "rb"))
mobilepd = pd.DataFrame(mobile_json)
apppd = pd.DataFrame(app_json)
train = pd.read_csv("data/kaggle/labeledTrainData.tsv", header=0,delimiter="\t",quoting=3)
test = pd.read_csv("data/kaggle/testData.tsv",header=0,delimiter="\t",quoting=3)

print("Data Loaded.")

vectordimension = 300
def cleandata(review_text):
	review_text = BeautifulSoup(review_text, "lxml").get_text()
	review_text = re.sub("[^a-zA-Z]"," ", review_text)
	words = review_text.lower().split()
	stops = set(stopwords.words("english"))
	words = [w for w in words if not w in stops]
	return words

def getvector(review):
	review_words = cleandata(review)
	featureVec = np.zeros((vectordimension,),dtype="float32")
	index2word_set = set(model.index2word)
	for word in review_words:
		if word in index2word_set:
			featureVec = np.add(featureVec,model[word])
	return featureVec.reshape(1,vectordimension)

for index, mb in mobile_train.iterrows():
	xinput = getvector(mb['review'])
	yin = []
	neg = int(not mb['sentiment'])
	pos = mb['sentiment']
	yin = [[neg, pos]]