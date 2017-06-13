from __future__ import print_function
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
from six.moves import range
from matplotlib import pyplot
from time import time
from nltk.corpus import stopwords
from gensim.models import word2vec
from bs4 import BeautifulSoup
import re

print("Loading Word2Vec Models...")

model = word2vec.Word2Vec.load("300features_40minwords_10context")

print("Loaded.")

def cleandata(review_text):
	review_text = BeautifulSoup(review_text, "lxml").get_text()
	review_text = re.sub("[^a-zA-Z]"," ", review_text)
	words = review_text.lower().split()
	stops = set(stopwords.words("english"))
	words = [w for w in words if not w in stops]
	return words

def getvector(review):
	review_words = cleandata(review)
	featureVec = np.zeros((300,),dtype="float32")
	index2word_set = set(model.index2word)
	for word in review_words:
		if word in index2word_set:
			featureVec = np.add(featureVec,model[word])
	return featureVec


num_labels = 2

hidden_nodes = 400
hidden_nodes2 = 450
vector_size = 300
graph = tf.Graph()

print("Initalizing Graph...")
with graph.as_default():

	tf_train_dataset = tf.placeholder(tf.float32, shape=(None, vector_size))
	tf_train_labels = tf.placeholder(tf.float32, shape=(None, num_labels))


	weights = tf.Variable(tf.truncated_normal([vector_size, hidden_nodes]))
	biases = tf.Variable(tf.zeros([hidden_nodes]))

	hidden1 = tf.nn.tanh(tf.matmul(tf_train_dataset, weights) + biases)
	#hidden1_dropout = tf.nn.dropout(hidden1, keep_prob)

	hidden_weights = tf.Variable(tf.truncated_normal([hidden_nodes, hidden_nodes2]))
	hidden_biases = tf.Variable(tf.zeros([hidden_nodes2]))

	hidden2 = tf.nn.tanh(tf.matmul(hidden1, hidden_weights) + hidden_biases)

	hidden_weights2 = tf.Variable(tf.truncated_normal([hidden_nodes2, num_labels]))
	hidden_biases2 = tf.Variable(tf.zeros([num_labels]))

	logits = tf.matmul(hidden2, hidden_weights2) + hidden_biases2
	#logits_dropout = tf.nn.dropout(logits, keep_prob)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(hidden_weights) + tf.nn.l2_loss(hidden_weights2) + tf.nn.l2_loss(biases) + tf.nn.l2_loss(hidden_biases) + tf.nn.l2_loss(hidden_biases2)
	loss += 5e-4*regularization


	optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

	train_prediction = tf.nn.softmax(logits)
	saver = tf.train.Saver()

print("Initalized.")

session = tf.Session(graph = graph)
saver.restore(session, './tfmodel.ckpt')

def predict(tweet):
	input_vec = [getvector(tweet)]
	feed_dict = {tf_train_dataset : input_vec}
	predictions = session.run([train_prediction], feed_dict = feed_dict)
	return predictions
