# NDCG RF EMB

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import collections
import pandas as pd
from openpyxl import load_workbook
import glob
import math
import argparse 
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import models
from sentence_transformers import SentenceTransformer
at_3 = []
at_5 = []
at_10 = []
used_arts2 = []

import pickle
with open('rf_emb2', 'rb') as f:
    rf1 = pickle.load(f)
    
word_embedding_model = models.Transformer("robbert/featrec_16122022")

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model],device='cpu')

target_names = ['Non_feat', 'Feat']



path = ""

filenames = glob.glob(path + "/*.csv") 
L = filenames

for i in L:
    if i in used_arts2:
        print('already done')
        continue
    nf = []
    f = []
    path = i
    print(i)
    df = pd.read_csv(i)
    df = df.drop(df.columns[0], axis=1)
    text = df.content
    labels = df.featured
    print('got the df', 'featured posts:', labels.sum())
    if labels.sum()==0:
        print('no featured posts in this thread')
        continue
    labels = list(labels)
    print('Got the labels')
    test_emd = model.encode(text)
    test_emd = pd.DataFrame(test_emd)
    df = df.drop('featured', axis=1)
    content = df.content
    df = df.drop('content', axis=1)
    
    df_test = pd.concat([df, test_emd], axis=1)
    df_test.columns = df_test.columns.map(str)
    pred = rf1.predict_proba(df_test)
    print('predicted')
    pred = pd.DataFrame(pred)
    f = []
    nf = []
    pred.rename({0: 'nf', 1: 'f'}, axis=1, inplace=True)
    for row in pred.itertuples():
        if row.nf>0.5:
            nf.append(True)
            f.append(False)
        else:
            nf.append(False)
            f.append(True)
    pred['pred_label']= f
    pred['label'] = labels
    pred = pred.sort_values(by=['f'], ascending=False)
    pred2 = pred['pred_label']
    true2 = pred['label']
    if True not in f:
        continue
    if True not in labels:
        continue
    labs = list(pred['label'])
    for k in [3,5,10]:
        labs2 = labs[:k]
        if k == 3:
            dcg = (labs[0]/np.log2(1+1))+(labs[1]/np.log2(2+1))+(labs[2]/np.log2(3+1))
            sorte = sorted(labs2, reverse=True)
            idcg = (sorte[0]/np.log2(1+1))+(sorte[1]/np.log2(2+1))+(sorte[2]/np.log2(3+1))
            if idcg ==0:
                at_3.append(0)
            else:
                ndcg = dcg/idcg
                at_3.append(ndcg)
        if k ==5:
            dcg = (labs[0]/np.log2(1+1))+(labs[1]/np.log2(2+1))+(labs[2]/np.log2(3+1))+(labs[3]/np.log2(4+1))+(labs[4]/np.log2(5+1))
            sorte = sorted(labs2, reverse=True)
            idcg = (sorte[0]/np.log2(1+1))+(sorte[1]/np.log2(2+1))+(sorte[2]/np.log2(3+1))+(sorte[3]/np.log2(4+1))+(sorte[4]/np.log2(5+1))
            if idcg ==0:
                at_5.append(0)
            else:
                ndcg = dcg/idcg
                at_5.append(ndcg)
        if k ==10:
            dcg = (labs[0]/np.log2(1+1))+(labs[1]/np.log2(2+1))+(labs[2]/np.log2(3+1))+(labs[3]/np.log2(4+1))+(labs[4]/np.log2(5+1))+(labs[5]/np.log2(6+1))+(labs[6]/np.log2(7+1))+(labs[7]/np.log2(8+1))+(labs[8]/np.log2(9+1))+(labs[9]/np.log2(10+1))
            sorte = sorted(labs2, reverse=True)
            idcg = (sorte[0]/np.log2(1+1))+(sorte[1]/np.log2(2+1))+(sorte[2]/np.log2(3+1))+(sorte[3]/np.log2(4+1))+(sorte[4]/np.log2(5+1))+(sorte[5]/np.log2(6+1))+(sorte[6]/np.log2(7+1))+(sorte[7]/np.log2(8+1))+(sorte[8]/np.log2(9+1))+(sorte[9]/np.log2(10+1))
            if idcg ==0:
                at_10.append(0)
            else:
                ndcg = dcg/idcg
                at_10.append(ndcg)
    used_arts2.append(i)


print('NDCG@k: (3,5,10):')
print(s.mean(at_3), s.mean(at_5), s.mean(at_10))

    