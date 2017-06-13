from __future__ import print_function
from six.moves import cPickle as pickle
import tensorflow as tf
import numpy as np
from six.moves import range
from matplotlib import pyplot
from time import time

start = time()

print("Loading Data...")
data = pickle.load(open('x_vec.pickle', 'rb'))
data_labels = pickle.load(open('y_vec.pickle', 'rb'))

print("Convert Data to numpy...")
#data = np.asarray(data)
data_labels = np.asarray(data_labels, dtype=np.float32)
print("Converted.")

print("Loaded Data.")
length = len(data)
print(length)

print("Dividing Labels...")

train = data[:544556]
train_labels = data_labels[:544556]

valid = data[544556:544556+155587]
valid_labels = data_labels[544556:544556+155587]

test = data[544556+155587:]
test_labels = data_labels[544556+155587:]

print("Divided.")
print((len(test)+len(train)+len(valid))-length)

num_labels = 2

def reformat(labels):
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return labels

print('Reformating...')
train_labels = reformat(train_labels)
valid_labels = reformat(valid_labels)
test_labels = reformat(test_labels)
print('Reformated.')

batch_size = 128
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

num_steps = 3001

print("Initalized.")
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))/predictions.shape[0])

with tf.Session(graph = graph) as session:
	saver = tf.train.Saver()

	tf.initialize_all_variables().run()
	print('initalized!')
	vacc = []
	tacc = []
	testacc = []
	x = []
	steplist = []
	losses = []
	print('Training')
	for i in range(10):
		print("Iteration: %d" % i)
		for step in range(num_steps):


			offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
			batch_data = train[offset : (offset + batch_size)]
			batch_labels = train_labels[offset : (offset + batch_size)]
			feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
			feed_dict_valid = {tf_train_dataset : valid, tf_train_labels : valid_labels}
			feed_dict_test = {tf_train_dataset : test, tf_train_labels : test_labels}
			_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict = feed_dict)

			if step%100 == 0:
				validpredictions = session.run([train_prediction], feed_dict = feed_dict_valid)
				testpredictions = session.run([train_prediction], feed_dict = feed_dict_test)
				losses.append(l)
				print('Loss at step %d: %f' % (step, l))
				accuracy_t = accuracy(predictions, batch_labels)
				print('Training accuracy: %.1f%%' % accuracy_t)
				accuracy_v = accuracy(validpredictions[0], valid_labels)
				print('Validation accuracy: %.1f%%' % accuracy_v)
				vacc.append(accuracy_v)
				tacc.append(accuracy_t)
				accuracy_test = accuracy(testpredictions[0], test_labels)
				print('Test accuracy: %.1f%%' % accuracy_test)
				testacc.append(accuracy_test)

	print("Trained.")
	print("Saving...")
	acc = {"train" : accuracy_t, "validate" : accuracy_v, "test" : accuracy_test, "loss" : losses}
	pickle.dump(acc, open('accuracy.pickle','wb'))
	saver.save(session, "./tfmodel.ckpt")
	pyplot.plot(accuracy_v)
	pyplot.plot(accuracy_t)
	pyplot.plot(accuracy_test)
	pyplot.plot(losses)
	print("Saved.")
	print(time()-start)

	pyplot.show()

