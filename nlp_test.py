import pandas as pd
'''
Loading train and test data sets
'''
data = pd.read_csv('./data/hashcode_classification2020_train.csv')
test = pd.read_csv('./data/hashcode_classification2020_test.csv')
'''
Tokenize the train and test data with Okt from Konlpy
'''
from konlpy.tag import Okt

okt = Okt()
train_docs = []
test_docs = []

'''
Normally ruling out punctuation from the corpus is a good idea.
However, punctuation is important feature when it comes to programming languages.
(Accuracy score significantly goes down when punctuation is ruled out.)
'''
def tokenize(doc):
    result = []
    for t in okt.pos(doc, norm=True, stem=True):
        result.append('/'.join(t))
    return result
'''
Using both title and text to train the model.
Same with test set.
Also there is an error when tokenizing one of the data example,
used try and except to rule that one out. (If multiple data were giving errors, data cleaning would been necessary.)
'''
for idx, text in data.iterrows():
    try:
        train_docs.append((tokenize(text[0] + text[1]),text[2]))
    except:
        continue
for idx, text in test.iterrows():
    try:
        test_docs.append(tokenize(text[0]+text[1]))
    except:
        continue

print(train_docs[:3])

'''
Doc2Vec requires the data to be formed in certain way.
Tagging the training document for our corpus model 
'''
from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', 'words tags')
tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs]

print(tagged_train_docs[:3])
'''
dm = 0 means distributed bag of words(doesn't preserve the word order)
min_count = 3 means train model with words that appeared at least 3 times
(it is hard to learn from such a small set of data so min_count should be small)
hs = 0 using negative sample of 5 (negative = 5 by default)
verctor_size = 300 means 300 dimensionality of the feature vectors
sample = 0 is threshold for configuring which higher-frequency words are randomly downsampled.

'''
from gensim.models import doc2vec
import multiprocessing

cores = multiprocessing.cpu_count()
model = doc2vec.Doc2Vec(dm=0, vector_size=300, hs=0, negative = 5, min_count=3, sample = 0, workers=cores, seed=1234)

model.build_vocab(tagged_train_docs)

alpha = 0.025 # the initial learning rate
min_alpha = 0.001 # learning rate will linearly drop to min_alpha as training progresses
num_epochs = 20 # number of iterations over the corpus
alpha_delta = (alpha - min_alpha) / num_epochs
# train the model
for epoch in range(num_epochs):
    print('Now training epoch: ',epoch)
    model.alpha = alpha
    # fix the learning rate, no decay
    model.min_alpha = alpha
    model.train(tagged_train_docs, total_examples=model.corpus_count, epochs=model.epochs)
    # decrease the learning rate
    alpha -= alpha_delta

# Split the data into x,y train and x,y test
x_train = [model.infer_vector(doc.words) for doc in tagged_train_docs]
y_train = [doc.tags[0] for doc in tagged_train_docs]

x_test = [model.infer_vector(doc) for doc in test_docs]

import sklearn.metrics as metrics
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
import numpy as np

'''
The keras to_categorical expects the category value to start from 0.
Had to subtract one since our model starts from 1 - 5.
'''
for i in range(len(y_train)):
    y_train[i] -= 1
y_train = keras.utils.np_utils.to_categorical(y_train, 5)

in_size = x_train[0].shape[0]
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
'''
Added dropout value 0.2 to prevent overfitting.
It drops out both hidden and visible value from the model by probability of 1 - p or p.
'''
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(in_size,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(5, activation='softmax'))

'''
Used RMSprop, Adam, SGD but got best result using RMSprop
RMSprop is good, fast and very popular optimizer.
'''
model.compile(
    loss='categorical_crossentropy',
    optimizer=RMSprop(),
    metrics=['accuracy'])

hist = model.fit(x_train, y_train,
          batch_size=128, 
          epochs=100,
          verbose=1)
'''
make prediction from the model we trained and save as prediction.csv
before saving, add 1 back to the prediction since we subtracted one to fit into keras category
'''
predictions = model.predict_classes(x_test)

for i in range(len(predictions)):
    predictions[i] += 1
pd.DataFrame(predictions, columns=['label']).to_csv('prediction.csv', index=False)