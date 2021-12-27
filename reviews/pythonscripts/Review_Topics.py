
def TopicCreate(reviewdata,bigramdata, ntopic=20):

    import pandas as pd
    import numpy as np
    import gensim
    from gensim import corpora, models
    from sklearn.manifold import TSNE

    dictionary = gensim.corpora.Dictionary(reviewdata['Patterns'])
    dictionary.filter_extremes(no_below=1, no_above=.2, keep_n=50000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in reviewdata['Patterns']]

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]
    lda_model = gensim.models.LdaMulticore(bow_corpus,num_topics=20,id2word=dictionary,passes=2,workers=2)

    # for idx, topic in lda_model.print_topics(ntopic):
    #     print('Topic :{} \nWords: {}'.format(idx,topic))

    #check score
    def reviewScore(lmodel,corpus,elem):
        score = max([(v,i) for i, v, in lmodel[corpus[elem]]])
        return np.asarray(score)

    def termScore(lmodel, elem):
        score = [max((v,i) for i, v in lmodel[[(elem,1)]])]
        return pd.DataFrame(score,columns=['TopicScore','Topic'])
    
    def termScoresAll(lmodel, elem):
        score = [(v,i,elem) for i, v in lmodel[[(elem,1)]]]
        return pd.DataFrame(score,columns=['TopicScore','Topic','TermNo'])

    #score indices
    svals = pd.DataFrame([reviewScore(lda_model,bow_corpus,i) for i in range(len(bow_corpus))],columns=['TopicScore','Topic'])
    termscores = pd.concat([termScore(lda_model,i) for i in range(len(dictionary))],axis=0).reset_index(drop=True)
    termscoresall = pd.concat([termScoresAll(lda_model,i) for i in range(len(dictionary))],axis=0).reset_index(drop=True)
    
    reviewdata = pd.concat([reviewdata,svals],axis=1)
    dictterms = pd.DataFrame([dictionary[i] for i in range(len(dictionary))],columns=["Term"])
    termvals = pd.concat([dictterms,termscores],axis=1)
    
    #all terms, tsne
    piv = pd.pivot(termscoresall, index="TermNo",columns=["Topic"],values="TopicScore")
    term_tsne = pd.DataFrame(TSNE(n_components=2).fit_transform(piv))
    termvals_tsne = pd.concat([termvals,term_tsne],axis=1)
    
    dictterms = dictterms.reset_index()
    bigramdata_mg = bigramdata.merge(dictterms,how="left",left_on="Term1",right_on="Term")
    bigramdata_mg = bigramdata_mg.merge(dictterms,how="left",left_on="Term2",right_on="Term",suffixes=['A','B'])
    bigramdata_mg = bigramdata_mg.drop(['TermA','TermB'],axis=1)
    
    bigramdata_mg = bigramdata_mg.merge(termvals,how="left",left_on="Term1",right_on="Term")
    bigramdata_mg = bigramdata_mg.merge(termvals,how="left",left_on="Term2",right_on="Term",suffixes=['A','B'])
    bigramdata_mg = bigramdata_mg.loc[~np.isnan(bigramdata_mg['indexA']) & ~np.isnan(bigramdata_mg['indexB']),:]
    bigramdata_mg = bigramdata_mg.drop(['TermA','TermB'],axis=1)
    
    bigramdata_mg['Same'] = bigramdata_mg['TopicA']==bigramdata_mg['TopicB']
    
    def term_distance(df,rownum1,rownum2):
        return round(sum(abs(df.loc[rownum1,:] - df.loc[rownum2,:])),2)

    comp = [term_distance(piv,rw['indexA'],rw['indexB']) for i, rw in bigramdata_mg.iterrows()]
    bigramdata_mg['TopicDist'] = comp

    dictterms = pd.concat([dictterms,termscores],axis=1)
    dictterms['TopicScore'] =dictterms['TopicScore'].apply(lambda x: np.round(x,4))
    
    reviewdata['TopicScore'] = reviewdata['TopicScore'].apply(lambda x: np.round(x,4))
    
    
    return [reviewdata, bigramdata_mg, dictterms]
