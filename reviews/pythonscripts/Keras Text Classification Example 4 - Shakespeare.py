# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 15:31:11 2021
#https://www.askpython.com/python/examples/predict-shakespearean-text
@author: 16317
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

#file 1: shakespeare
shakespeare_url='https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
filepath=keras.utils.get_file('shakespeare.txt',shakespeare_url)
with open(filepath) as f:
    shakespeare_text=f.read()
    
#file 2: neitzche
neitzche = keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with open(neitzche) as f:
    neitzche_text=f.read()

def makeTextTokenizer(corpus):
    #tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=50000, split=' ', char_level=False, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',)
    tokenizer.fit_on_texts(corpus)   
    return tokenizer

def makeTextDataset(corpus, tokenizer):
    batch_size=32
    max_id = len(tokenizer.word_index)
    dataset_size=tokenizer.document_count
    [encoded]=np.array(tokenizer.texts_to_sequences([corpus]))-1
    
    train_size=dataset_size*90//100
    dataset=tf.data.Dataset.from_tensor_slices(encoded[:train_size])

    n_steps=100
    window_length=n_steps+1
    #data
    dataset=dataset.repeat().window(window_length,shift=1,drop_remainder=True)
    dataset=dataset.flat_map(lambda window: window.batch(window_length))
    dataset=dataset.shuffle(10000).batch(batch_size)
    dataset=dataset.map(lambda windows: (windows[:,:-1],windows[:,1:]))
    dataset=dataset.map(lambda X_batch,Y_batch: (tf.one_hot(X_batch,depth=max_id),Y_batch))
    dataset=dataset.prefetch(1)
    
    return(dataset)


def makeTextModel(dataset, tokenizer):
    batch_size=32
    dataset_size=tokenizer.document_count
    train_size=dataset_size*90//100
    
    max_id = len(tokenizer.word_index)
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(128,return_sequences=True,input_shape=[None,max_id]))
    model.add(keras.layers.GRU(128,return_sequences=True))
    model.add(keras.layers.TimeDistributed(keras.layers.Dense(max_id,activation='softmax')))
    model.compile(loss='sparse_categorical_crossentropy',optimizer='adam')
    history=model.fit(dataset,steps_per_epoch=train_size // batch_size,epochs=1)
    return(model)

neitzche_sent = neitzche_text.split(".")
tokenizer_b = makeTextTokenizer(neitzche_sent)
dset = makeTextDataset(neitzche_text, tokenizer_b)
model_b = makeTextModel(dset, tokenizer_b)

#predict
def preprocess(tokenizer, texts):
    max_id = len(tokenizer.word_index)
    X=np.array(tokenizer.texts_to_sequences(texts))-1
    return tf.one_hot(X,max_id)
 
def next_char(text, model, tokenizer,temperature=1):
    X_new=preprocess(tokenizer, [text])
    y_proba=model.predict(X_new)[0,-1:,:]
    rescaled_logits=tf.math.log(y_proba)/temperature
    char_id=tf.random.categorical(rescaled_logits,num_samples=1)+1
    return tokenizer.sequences_to_texts(char_id.numpy())[0]
 
def complete_text(text, model, tokenizer, n_chars=50, temperature=1):
    for _ in range(n_chars):
        text+=next_char(text, model, tokenizer)
    return text


print("Some predicted texts for word 'except' are as follows:\n ")
for i in range(3):
  print(complete_text(['except'], model_b, tokenizer_b))
  print()

#testing
temperature = 1
max_id = len(tokenizer_b.word_index)
text = ['once we their type']
X1 = np.array(tokenizer_b.texts_to_sequences(text))-1
X_new = tf.one_hot(X1,max_id)
y_proba = model_b.predict(X_new)[0,-1:,:]
rescaled_logits = tf.math.log(y_proba)/temperature
char_id=tf.random.categorical(rescaled_logits,num_samples=2, seed=3)+1
tokenizer_b.sequences_to_texts(char_id.numpy())
