
def PrepareReviews(data, descField):
   
    import numpy as np
    import pandas as pd
    import nltk as nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import RegexpTokenizer
    import random
    
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def parse_term(d):
        if pd.isna(d):
            return '0'
        else:
            p = [lemmatizer.lemmatize(x.lower()) for x in tokenizer.tokenize(d) if not x in stop_words]
            return [q for q in p if not q.isdigit()]

    def apply_pos_tag(d):
        return nltk.pos_tag(d)

    def filter_pos_tag(d):
        p = pd.DataFrame(d)
        return p[~p[1].isin(["PRP","DT","CD","CC","PDT"])]
        
    def pos_calculate(d,cols):
        p = pd.DataFrame(d)
        idx = p.index
        termsdf = pd.DataFrame(columns=cols)
        for i,r in p.iterrows():
            try:
                dfrow = pd.DataFrame([[p.iloc[i,0],p.iloc[i+1,0],idx[i+1] - idx[i]]],columns=cols)
                termsdf = pd.concat([termsdf,dfrow],axis=0)
            except:
                pass
        return termsdf
    
    cols=['Term1','Term2','Distance']
    data['Patterns'] = data[descField].apply(lambda x: parse_term(x))
    data['POS'] = data['Patterns'].apply(lambda x: apply_pos_tag(x))
    data['POS_filt'] = data['POS'].apply(lambda x: filter_pos_tag(x))
    data['XTerms'] = data['POS_filt'].apply(lambda x: pos_calculate(x, cols))
    data['UniqueID'] = [f'{i}_{random.randrange(16**7):x}' for i in range(len(data))]
    
    return data