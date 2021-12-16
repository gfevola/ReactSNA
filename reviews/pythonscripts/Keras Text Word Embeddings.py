# -*- coding: utf-8 -*-
"""
Created on Sat Nov 27 16:41:55 2021
#https://stackoverflow.com/questions/63312140/how-to-save-my-own-trained-word-embedding-model-using-keras-like-word2vec-and
@author: 16317
"""
#%%

import numpy as np
import pandas as pd
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras import layers
import random
import nltk as nltk

def NgramSample(textlist, N, toss = .3):
    tlen = len(textlist)
    grams = [textlist[a:a+N+1] for a in range(tlen - N)]
    random.shuffle(grams)
    grams = grams[1:round(tlen * (1-toss))]
    splitgram_x = [x[0:N] for x in grams]
    splitgram_y = [y[-1] for y in grams]
    return([splitgram_x, splitgram_y])

def ycategories(y, numwords):
     y_matrix = to_categorical(y, num_classes=numwords+1)
     return(y_matrix)

# encode the sentences
def docstoWordArray(docs, tokenizer, aslist=False):
    encoded_docs = tokenizer.texts_to_sequences(docs)
    if not aslist:
        encoded_docs = ",".join("'{0}'".format(n) for n in encoded_docs)
        encoded_docs = encoded_docs[2:-2].replace("]'",'').replace("'[",'').replace(' ','').split(",")
        encoded_docs = [int(x) for x in encoded_docs if len(x)>0]
    return(encoded_docs)


#%%

docs = neitzche_sent
labels = np.repeat(0, len(neitzche_sent))

# train the tokenizer
vocab_size = 50000
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(docs)

arry = docstoWordArray(docs, tokenizer)

#create sequences from full array
numwords = len(tokenizer.word_docs)
seq_len = 10
[xx,yy] = NgramSample(arry,seq_len,toss=.3)
yy_mat = ycategories(yy,numwords)

#-------------------------
#other docs
amazon = pd.read_excel("C:\\Projects\\completeversion\\Sample Data\\SampleAmazonReviews.xlsx")

vocab_size = 50000
docs = amazon.loc[:,'Text']
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(docs)
arry = docstoWordArray(docs, tokenizer, aslist=True)

numwords = len(tokenizer.word_docs)
seq_len = 10
gramsXY = [NgramSample(x,seq_len,toss=.3) for x in arry]
grams_x = [x for x,y in gramsXY]
grams_y = [y for x,y in gramsXY]
genx = []
geny = []

for gx in grams_x:
    [genx.append(x) for x in gx]
for gy in grams_y:
    [geny.append(y) for y in gy]

xx1 = genx

yy_mat = ycategories(geny,numwords)

#%%

# define the model
model = Sequential()
model.add(layers.Embedding(input_dim = vocab_size, output_dim = 16, input_length=seq_len, name='embeddings'))
model.add(layers.GRU(256, return_sequences=True))
model.add(layers.SimpleRNN(128))
#model.add(Flatten())
model.add(layers.Dense(1, activation='sigmoid'))

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# fit the model
labels = np.repeat(0, len(xx))
#yy = np.asmatrix(yy).transpose()
model.fit(np.asmatrix(xx), yy_mat, epochs=50, verbose=0)


# save embeddings
term = 'licorice'

embeddings = model.get_layer('embeddings').get_weights()[0]
w2v_my = {}
w2v_pos = {}
wordcor = []

for word, index in tokenizer.word_index.items():
    w2v_my[word] = embeddings[index]
    w2v_pos[word] = nltk.pos_tag([word])

embdist = [np.linalg.norm(w2v_my[term] - i) for i in embeddings]

for word, index in tokenizer.word_index.items():
    wordcor.append([word, w2v_pos[word][0][1], np.linalg.norm(w2v_my[term] - embeddings[index])])
    
wordcor = pd.DataFrame(wordcor)

#%%
#---------version 2 --------------

model1 = Sequential()
model1.add(layers.Embedding(vocab_size, 10, input_length=seq_len))
model1.add(layers.LSTM(50))
model1.add(layers.Dense(numwords+1, activation='softmax'))

# compile network
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model1.fit(np.asmatrix(xx1), yy_mat, epochs=50, verbose=1)

def modelwordprediction(lstmodel,words_input, tokenizer):
    x = tokenizer.texts_to_sequences(words_input)
    x = pad_sequences(x,maxlen = 10, padding="pre")
    output = lstmodel.predict(x).round(3).transpose()
    output = pd.DataFrame(output)
    termlist = tokenizer.word_index.items()
    trmlst = pd.Series([x[0] for x in termlist])
    
    results = pd.DataFrame(pd.concat([trmlst,output],axis=1))
    results.columns = ["Term","Score"]
    return(results)

words_input = ["Product arrived labeled as Jumbo Salted Peanuts the peanuts were"]
res = modelwordprediction(model1, words_input, tokenizer)


#%%