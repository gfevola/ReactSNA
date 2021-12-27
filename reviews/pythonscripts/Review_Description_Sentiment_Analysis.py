# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 15:29:10 2021
#based on "Keras Sentiment on Amazon Reviews.py"
@author: gf
"""

def DescriptionSentiment(reviewdata, descField, scoreField, seq_len = 10, margin=5):
    #keras sentiment model from review description to provided score
    #should have already passed through "Prepare_Reviews_NLTK"

    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras import layers
    from keras_preprocessing.text import Tokenizer
    import random
    import pickle
    from nltk import pos_tag
    
    
    #preprocessing functions
    def NgramSample(textlist, yval, N, numwords):
        textlist = [numwords-1]*(N-margin) + textlist
        tlen = len(textlist)
        Nrev = min(N,tlen)
        grams = [textlist[a:a+N+1] for a in range(tlen - Nrev)]
        splitgram_x = [x[0:Nrev] for x in grams]
        splitgram_y = [yval for x in grams]
        return([splitgram_x, splitgram_y])
    
    
    def xcategories(x, numwords):
         seq = [0]*numwords
         for a in x:
            seq[a] += 1
         return(seq)
        
    # encode the sentences
    def docstoWordArray(docs, tokenizer, aslist=False):
        encoded_docs = tokenizer.texts_to_sequences(docs)
        if not aslist:
            encoded_docs = ",".join("'{0}'".format(n) for n in encoded_docs)
            encoded_docs = encoded_docs[2:-2].replace("]'",'').replace("'[",'').replace(' ','').split(",")
            encoded_docs = [int(x) for x in encoded_docs if len(x)>0]
        return(encoded_docs)
    
    
    def sequence_disambiguateX(gramsXY):
        grams = [input1 for input1,output1 in gramsXY]
        gen = []
        for g in grams:
            [gen.append(x) for x in g]
        return(gen)
    
    def sequence_disambiguateY(gramsXY):
        grams = [output1 for input1,output1 in gramsXY]
        gen = []
        indexer = []
        for i, g in enumerate(grams):
            [gen.append(x) for x in g]
            [indexer.append(i) for x in g]
        return([gen,indexer])
    
    #treating texts
    def treat_texts(texts,yvals_norm):
        arry = docstoWordArray(texts, desc_tokenizer, aslist=True)
        gramsXY = [NgramSample(x,yvals_norm[i],seq_len, numwords) for i,x in enumerate(arry)]
        genx = sequence_disambiguateX(gramsXY)
        if len(genx)>0: #force stop
            xx = pd.DataFrame(genx)
            xx1 = [xcategories(w,numwords) for i, w in xx.iterrows()]
            xx2 = pd.DataFrame(xx1)
            [geny, indices]  = sequence_disambiguateY(gramsXY)
            yy = pd.Series(geny)
            return([xx2,yy, indices])  
        else:
            return([[],[],[]])
        
    #-------begin data processing-------------


    #y setup
    yvals  = reviewdata.loc[:,scoreField]
    s_max = np.max(yvals)
    s_min = np.min(yvals)
    yvals_norm = (yvals - s_min)/(s_max - s_min)  
    def norminv(value):
        return value * (s_max - s_min) + s_min
    
    #x setup - tokenizer
    reviewtexts_trimmed = [" ".join(rev['POS_filt'][0].dropna()) for i,rev in reviewdata.iterrows()]
    
    vocablist = [rev['POS_filt'][0].dropna() for i,rev in reviewdata.iterrows()]
    vocablist = pd.concat(vocablist).unique()

    desc_tokenizer = Tokenizer(num_words = len(vocablist))
    desc_tokenizer.fit_on_texts(reviewtexts_trimmed)
    numwords = len(desc_tokenizer.word_docs)+1
    
    
    [xvals, yvals, indices] = treat_texts(reviewtexts_trimmed,yvals_norm)
    
    sample_no = int(np.round(len(indices)*.4,0))
    sample = pd.Series(random.sample(indices,sample_no))
    sample = random.sample(range(len(indices)), sample_no)
    
    xtrain = xvals.loc[xvals.index.isin(sample),:]
    xtest = xvals.loc[~xvals.index.isin(sample),:]

    ytrain = yvals.loc[yvals.index.isin(sample)]
    ytest = yvals.loc[~yvals.index.isin(sample)].reset_index(drop=True)
    
    #---------model create & train ------
    
    descmodel = Sequential()
    # Embedding layer
    descmodel.add(
        layers.Dense(64,input_dim=numwords,activation='relu'))
    descmodel.add(layers.Dropout(.7))
    descmodel.add(layers.Dense(15,activation='relu'))
    descmodel.add(layers.Dense(1,activation='sigmoid'))
    descmodel.compile(optimizer='adam', 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])
    
    descmodel.fit(xtrain, ytrain, epochs=10, verbose=True)
    
    control = np.round(descmodel.predict(xtest),4)
    control = pd.Series(control[:,0])
    difference = np.mean(abs(control - ytest))

    #save model 
    #modelname = "/models/sentmodel.sav"
    #pickle.dump(descmodel,open(modelname,'wb'))

    #full into model
    #iterate every 10000
    fulloutput = pd.DataFrame()
    limit = int(np.round(len(indices)/10000 + .5))
    for i in range(limit):
        v = pd.DataFrame(descmodel.predict(xvals.loc[i*10000:(i+1)*10000-1,:]))
        fulloutput = pd.concat([fulloutput,v])
    
    fulloutput['Indices'] = indices

    outputsummary = pd.DataFrame()   
    wordslists = []
    for i, rev in reviewdata.iterrows():
        summry = fulloutput.loc[fulloutput["Indices"]== i,0]
        summry = [norminv(s) for s in summry]
        #make score link
        sumdf = pd.DataFrame([np.mean(summry),i,",".join("'{0}'".format(round(n,4)) for n in summry)]).transpose()
        outputsummary = pd.concat([outputsummary,sumdf],axis = 0)
        #make word link
        words = [x[0] for i,x in rev['POS_filt'].iterrows()]
        wordslists.append(",".join(words))
        
    reviewdata['Wordlist'] = wordslists
    outputsummary.columns = ["EstimatedSentiment","Index","SentimentString"]
    outputsummary = outputsummary.reset_index(drop=True)
    outputsummary['EstimatedSentiment'] = outputsummary['EstimatedSentiment'].apply(lambda x: np.round(x,4))
    
    reviewdata = pd.concat([reviewdata,outputsummary],axis=1)
    reviewdata['EstimatedSentiment'] = reviewdata['EstimatedSentiment'].fillna(0)
    
    #test individual words
    wordmat = np.identity(numwords)
    wordmat = wordmat[0:numwords-1,:]
    
    wordDF = pd.Series(desc_tokenizer.word_docs).reset_index()
    wordDF['Expected'] = descmodel.predict(wordmat)
    wordDF.columns = ['Term','Count','SentimentScore']
    
    wordDF['SentimentScore'] = wordDF['SentimentScore'].apply(lambda x: np.round(x,4))
    
    wordDF['POS_Tag'] = wordDF['Term'].apply(lambda x: pos_tag([x])[0][1])
    
    return([reviewdata, wordDF])
    

