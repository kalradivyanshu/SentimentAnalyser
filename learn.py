import pickle
print("Loading sentences")
sentences = pickle.load(open("sentences.pickle",'rb'))
print(len(sentences))
num_features = 300
min_word_count = 40
num_workers = 4
context = 10
downsampling = 1e-3

from gensim.models import word2vec
print("Training Model...")
model = word2vec.Word2Vec(sentences, workers=num_workers,size = num_features,min_count=min_word_count,window=context, sample=downsampling)
print("Model Trained")
model_name = "300features_40minwords_10context"
print("Saving Model")
model.save(model_name)