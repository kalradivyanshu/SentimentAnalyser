import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from gensim.models import word2vec
from bs4 import BeautifulSoup
import os
import pickle
from time import time
import re
import tensorflow as tf
import math

sess = tf.InteractiveSession()

start = time()
model = word2vec.Word2Vec.load("300features_40minwords_10context")

def divideit(list1, ratio):
    return list1[int(len(list1)*ratio):], list1[:int(len(list1)*ratio)]


app_json = pickle.load(open("app.pickle", "rb"))
mobile_json = pickle.load(open("mobile.pickle", "rb"))
mobilepd = pd.DataFrame(mobile_json)
apppd = pd.DataFrame(app_json)
app_train, app_test = divideit(apppd, 7/10)
mobile_train, mobile_test = divideit(mobilepd, 7/10)
train = pd.read_csv("data/kaggle/labeledTrainData.tsv", header=0,delimiter="\t",quoting=3)
test = pd.read_csv("data/kaggle/testData.tsv",header=0,delimiter="\t",quoting=3)

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


x = tf.placeholder("float", [None, vectordimension]) #  can we feed directly?
y_ = tf.placeholder("float", [None, 2]) # two output classes

number_hidden_nodes = 350 # 20 outputs to create some room for negatives and positives
W = tf.Variable(tf.truncated_normal([vectordimension, number_hidden_nodes], stddev=1./math.sqrt(2)))
b = tf.Variable(tf.zeros([number_hidden_nodes]))
hidden  = tf.nn.tanh(tf.matmul(x,W) + b) # first layer.

 # the XOR function is the first nontrivial function, for which a two layer network is needed.
W2 = tf.Variable(tf.truncated_normal([number_hidden_nodes, 2], stddev=1./math.sqrt(2)))
b2 = tf.Variable(tf.zeros([2]))
hidden2 = tf.matmul(hidden, W2)+b2

y = hidden2
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(hidden2, y_))
regularization = tf.nn.l2_loss(W) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(b) + tf.nn.l2_loss(b2)
loss += 5e-4*regularization

# Define loss and optimizer
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)


saver = tf.train.Saver()
# Train
tf.initialize_all_variables().run()
print("training using train data")

for index, mb in train.iterrows():
    xinput = getvector(mb['review'])
    yin = []
    neg = int(not mb['sentiment'])
    pos = mb['sentiment']
    yin = [[neg, pos]]
    for step in range(500):
        feed_dict={x: xinput, y_: yin}
        e, a = sess.run([cross_entropy,train_step],feed_dict)

print("Saving Model.")
saver.save(sess, "/model1.ckpt")
print("Model Saved.")

print("training using train data")
for index, mb in mobile_train.iterrows():
    xinput = getvector(mb['review'])
    yin = []
    neg = int(not mb['sentiment'])
    pos = mb['sentiment']
    yin = [[neg, pos]]
    for step in range(500):
        feed_dict={x: xinput, y_: yin}
        e, a = sess.run([cross_entropy,train_step],feed_dict)

print("Saving Model.")
saver.save(sess, "/model2.ckpt")
print("Model Saved.")

print("training using train data")
for index, mb in app_train.iterrows():
    xinput = getvector(mb['review'])
    yin = []
    neg = int(not mb['sentiment'])
    pos = mb['sentiment']
    yin = [[neg, pos]]
    for step in range(500):
        feed_dict={x: xinput, y_: yin}
        e, a = sess.run([cross_entropy,train_step],feed_dict)

print("Saving Model.")
saver.save(sess, "/model3.ckpt")
print("Model Saved.")
