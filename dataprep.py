import numpy as np 
import pandas as pd 
import re
import random
import math
from collections import Counter

import pickle


df = pd.read_csv('all_accepted_comments.csv')

# Chronological sorting of articles
df['date'] = pd.to_datetime(df['article_time'])
df = df.sort_values(by='date')

import random
random.seed(1234)
article_ids = np.array(df.article_id)
articles = np.unique(article_ids)

# Splitting at the halfway point
set1 = articles[:1476]
set2 = articles[1476:]



# Getting row indices for each set of articles in original data
index1 = []
index2 = []

for row in df.itertuples():
    if row.article_id in set1:
        index1.append(row.Index)
    else:
        index2.append(row.Index)
   
# Splitting data in two sets of comments based on articles     
df1 = df.iloc[index1,:] 
df2 = df.iloc[index2,:]

# Save two sets of comments
df1.to_csv('df1.csv')
df2.to_csv('df2.csv')


# Drop unnecessary information
df1 = df1.drop('status', axis=1)
df1 = df1.drop('article_time', axis=1)
df1 = df1.drop('comment_time', axis=1)
df1 = df1.drop('article_id', axis=1)
df1 =df1.drop(df1.columns[0], axis=1)
df1 =df1.drop(df1.columns[0], axis=1)
df1 =df1.drop('date', axis=1)

# Splitting set 1 80/10/10 for training, validation and testing
np.random.seed(556)

train, validate, test = np.split(df1.sample(frac=1), [int(.8*len(df1)), int(.9*len(df1))])

collections.Counter(validate['featured'])

# Saving the 80/10/10 splits
train.to_csv('train.csv')
test.to_csv('test.csv')
validate.to_csv('validate.csv')   

# Downsampling of non-featured posts in train
# Testing on validation was done and 95/5 won, therefore 57418 non_featured posts in train2
train_feat = train[train.featured == True]
train_nonfeat = train[train.featured == False]
train2 = train_nonfeat.sample(57418, random_state=556)  # DEFINE SPLIT
train2 = pd.concat([train_feat, train2])
train_labels = train2.featured

train2.to_csv('train2.csv')
train2 = train2.drop('featured', axis=1)    