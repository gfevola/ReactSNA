# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 13:40:57 2021

@author: 16317
"""

import numpy as np
import pandas as pd
import json

s_path = 'C:\\Users\\16317\\Documents\\Datasets\\spotifysongs\\'
trackdata = pd.read_csv(s_path + "data.csv")

dislike = pd.read_json(s_path + "dislike.json")
good = pd.read_json(s_path + "good.json")
yes = pd.read_json(s_path + "yes.json")
no = pd.read_json(s_path + "no.json")

#dislikes
def readjsonrows(data):
    df = []
    [df.append(pd.DataFrame.from_dict([data.iloc[x,0]])) for x in range(len(data))]
    df = pd.concat(df)
    return(df)

dislike_df = readjsonrows(dislike)
good_df = readjsonrows(good)
yes_df = readjsonrows(readjsonrows(yes))
no_df = readjsonrows(readjsonrows(no))

datadf = pd.concat([dislike_df, good_df])
lookupdf = pd.concat([yes_df, no_df])

datadf1 = pd.merge(datadf, lookupdf, left_on="id", right_on="id")
