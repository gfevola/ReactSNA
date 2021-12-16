
def BigramCreate(reviewdata, productField, scoreField):
    
    import pandas as pd
    import numpy as np
    
    cols=['Term1','Term2','Distance']
    bigramlist = pd.DataFrame(columns=cols)
 
    for i,r in reviewdata.iterrows():
        df = r['XTerms']
        csdf = pd.DataFrame([r.loc[[productField,scoreField]]]).reset_index(drop=True)
        df_full = pd.concat([df,csdf],axis=1)
        bigramlist = pd.concat([bigramlist,df_full],axis=0,ignore_index=True)

    bigramlist['Distance'] = bigramlist['Distance'].astype('float64')    
    
    #bigrams
    bigramlistavgs = bigramlist.groupby(['Term1','Term2']).mean([scoreField,'Distance'])
    bigramlistavgs1 = bigramlist.groupby(['Term1','Term2']).count()[['Distance']]
    bigramlistavgs['N'] = bigramlistavgs1
    bigramlistavgs = bigramlistavgs.reset_index()

    bigramlistcounts1 = bigramlist.groupby(['Term1']).count().iloc[:,:1]
    bigramlistcounts2 = bigramlist.groupby(['Term2']).count().iloc[:,:1]
    bigramlistcounts1.columns=['Count']
    bigramlistcounts2.columns=['Count']
    bigramlistcounts = bigramlistavgs.merge(bigramlistcounts1,how="left",left_on="Term1",right_on="Term1")
    bigramlistcounts = bigramlistcounts.merge(bigramlistcounts2,how="left",left_on="Term2",right_on="Term2")
    bigramlistcounts['order'] = bigramlistcounts['N']/np.sqrt((bigramlistcounts['Count_x']**2 + bigramlistcounts['Count_y']**2))
    bigramlistcounts = bigramlistcounts.sort_values('order',ascending=False)
    bigramlistcounts = bigramlistcounts.loc[bigramlistcounts['Term1']!=bigramlistcounts['Term2'],:]
    bigramlistcounts['pctCount'] =  (bigramlistcounts['N'] / bigramlistcounts['Count_x'])**2 + (bigramlistcounts['N'] / bigramlistcounts['Count_x'])**2

    uniqueTermsFull = pd.DataFrame(pd.concat([bigramlistcounts['Term1'], bigramlistcounts['Term2']],axis=0).unique())

    #filter out nodes that are too common
    limit1 = bigramlistcounts['Count_x']<20
    limit2 = bigramlistcounts['Count_y']<20
    #limit3 = bigramlistcounts['order'] > .20
    #limit3 = bigramlistcounts['order'] > .07
    #limit4 = bigramlistcounts['pctCount'] > .07
    limit5 = bigramlistcounts['Term1'] != bigramlistcounts['Term2']
    
    #bigramlistfilt = bigramlistcounts.loc[ limit5,:]
    bigramlistfilt = bigramlistcounts.loc[limit1 & limit2 & limit5,:]

    uniqueTerms = pd.DataFrame(pd.concat([bigramlistfilt['Term1'], bigramlistfilt['Term2']],axis=0).unique())
    uniqueTerms['Num'] = range(len(uniqueTerms))

    bigramlistfilt = bigramlistfilt.merge(uniqueTerms,how="left",left_on="Term1",right_on=0)
    bigramlistfilt = bigramlistfilt.merge(uniqueTerms,how="left",left_on="Term2",right_on=0)
    bigramlistfilt = bigramlistfilt.drop(["0_x","0_y"],axis=1)
    bigramlistfilt['Score'] = round(bigramlistfilt['Score'],4)
    bigramlistfilt['Distance'] = round(bigramlistfilt['Score'],4)

    #export
    return bigramlistfilt
