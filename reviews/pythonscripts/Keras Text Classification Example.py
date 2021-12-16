# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 14:20:13 2021
#https://www.techwithtim.net/tutorials/python-neural-networks/text-classification-p3/
@author: 16317
"""

import tensorflow as tf
from tensorflow import keras
import numpy

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)


# A dictionary mapping words to an integer index
_word_index = imdb.get_word_index()

word_index = {k:(v+3) for k,v in _word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
	return " ".join([reverse_word_index.get(i, "?") for i in text])
# this function will return the decoded (human readable) reviews  

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()  # prints a summary of the model

from keras import losses 
from keras import optimizers 
from keras import metrics 
from keras.optimizers import RMSprop 

model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = [metrics.categorical_accuracy])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

results = model.evaluate(test_data, test_labels)
print(results)

