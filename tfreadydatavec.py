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

start = time()
model = word2vec.Word2Vec.load("300features_40minwords_10context")

def divideit(list1, ratio):
	return list1[int(len(appp)*ratio):], list1[:int(len(appp)*ratio)]

print("Loading Data")
app_json = pickle.load(open("app.pickle", "rb"))
mobile_json = pickle.load(open("mobile.pickle", "rb"))
mobilepd = pd.DataFrame(mobile_json)
apppd = pd.DataFrame(app_json)
#app_train, app_test = divideit(apppd, 7/10)
#mobile_train, mobile_test = divideit(mobilepd, 7/10)
train = pd.read_csv("data/kaggle/labeledTrainData.tsv", header=0,delimiter="\t",quoting=3)
test = pd.read_csv("data/kaggle/testData.tsv",header=0,delimiter="\t",quoting=3)
print("Loaded Data")

def cleandata(review_text):
	review_text = BeautifulSoup(review_text, "lxml").get_text()
	review_text = re.sub("[^a-zA-Z]"," ", review_text)
	words = review_text.lower().split()
	'''
	stops = set(stopwords.words("english"))
	words = [w for w in words if not w in stops]
	'''
	return words

def getvector(review):
	review_words = cleandata(review)
	featureVec = np.zeros((300,),dtype="float32")
	index2word_set = set(model.index2word)
	for word in review_words:
		if word in index2word_set:
			featureVec = np.add(featureVec,model[word])
	return featureVec

x = []
y = []
x_test = []
i = 0
#os.system('echo \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\mobiledata\\\\\\\\\\\\\\\\\\\\\\\\\; > log.txt')
if sys.argv[1] != "nomobile":
	print("MobileData")
	length = len(list(mobilepd.itertuples()))
	print(length)
	last_percent_reported = -1.0
	for index, mb in mobilepd.iterrows():
		#os.system('echo '+str(index)+'>> log.txt')
		x.append(getvector(mb['review']))
		y.append(mb['sentiment'])
		i+=1
		percent = int((float(i)/float(length)) * 100.0)
		if percent != last_percent_reported:
			sys.stdout.write("%s%%..." % percent)
			sys.stdout.flush()
			last_percent_reported = percent
	#os.system('echo saving_data >> log.txt')
	pickle.dump(x, open('x_vec.pickle', 'wb'))
	pickle.dump(y, open('y_vec.pickle', 'wb'))


#os.system('echo \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\appdata\\\\\\\\\\\\\\\\\\\\\\\\\; >> log.txt')
print("AppData")
length = len(list(apppd.itertuples()))
last_percent_reported = -1.0
print(length)
j = 0
for index, mb in apppd.iterrows():
	#os.system('echo '+str(index)+'>> log.txt')
	x.append(getvector(mb['review']))
	y.append(mb['sentiment'])
	i+=1
	percent = int((float(j)/float(length)) * 100.0)
	if percent != last_percent_reported:
		sys.stdout.write("%s%%..." % percent)
		sys.stdout.flush()
		last_percent_reported = percent
	j+=1
#os.system('echo saving_data >> log.txt')
pickle.dump(x, open('x_vec.pickle', 'wb'))
pickle.dump(y, open('y_vec.pickle', 'wb'))



#os.system('echo \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\train\\\\\\\\\\\\\\\\\\\\\\\\\; >> log.txt')
print("Train")
length = len(list(train.itertuples()))
print(length)
j = 0
for index, mb in train.iterrows():
	#os.system('echo '+str(index)+'>> log.txt')
	x.append(getvector(mb['review']))
	y.append(mb['sentiment'])
	i+=1
	
	percent = int((float(j)/float(length)) * 100.0)
	if percent != last_percent_reported:
		sys.stdout.write("%s%%..." % percent)
		sys.stdout.flush()
		last_percent_reported = percent
	j+=1
#os.system('echo saving_data >> log.txt')
pickle.dump(x, open('x_vec.pickle', 'wb'))
pickle.dump(y, open('y_vec.pickle', 'wb'))

#os.system("echo \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\"+str(i)+"\\\\\\\\\\\\\\\\\\\\\\\\\; >> log.txt")


#os.system("echo \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\test\\\\\\\\\\\\\\\\\\\\\\\\\; >> log.txt")
print("Test")
length = len(list(test.itertuples()))
j = 0
for index, mb in test.iterrows():
	#os.system('echo '+str(index)+'>> log.txt')
	x_test.append(getvector(mb['review']))
	percent = int((float(j)/float(length)) * 100.0)
	if percent != last_percent_reported:
		sys.stdout.write("%s%%..." % percent)
		sys.stdout.flush()
		last_percent_reported = percent
	j+=1
#os.system('echo saving_data >> log.txt')
pickle.dump(x_test, open('x_test_vec.pickle', 'wb'))
print(time()-start)
#os.system('echo saved >> log.txt')
#os.system("echo 'Done, all good.' >> log.txt")
#os.system("echo '"+str(time()-start)+"' >> log.txt")
