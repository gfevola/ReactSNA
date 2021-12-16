# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:08:52 2021
#https://realpython.com/python-keras-text-classification/
@author: 16317
"""


import keras
from keras.models import Sequential
from keras.layers import LSTM, Flatten, Dense, Dropout, Masking, Embedding
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences

maxlen = 100
embedding_dim = 50
sentences = fulldata['Text'][0:40000]
scores = fulldata['Score'][0:40000]

tokenizer = Tokenizer(num_words=5000)
sent_train, sent_test, y_train, y_test = train_test_split(sentences, scores, test_size=0.25, random_state=1000)
tokenizer.fit_on_texts(sent_train)
vocab_size = len(tokenizer.word_index) + 1

x_train = pad_sequences(tokenizer.texts_to_sequences(sent_train), padding='post',maxlen=maxlen)
x_test = pad_sequences(tokenizer.texts_to_sequences(sent_test), padding='post',maxlen=maxlen)

model = Sequential()
# Embedding layer
model.add(
    Embedding(input_dim=vocab_size,
              input_length = maxlen,
              output_dim=embedding_dim,
              #weights=[embedding_matrix],
              ))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    epochs=20,
                    verbose=False,
                    validation_data=(x_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))

loss, accuracy = model.evaluate(x_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))