# RECALL RF

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
from sklearn.metrics import ndcg_score

plt.style.use('seaborn-whitegrid')


path = ""


filenames = glob.glob(path + "/*.csv") 
L = filenames
at_3 = []
at_5 = []
at_10 = []


for i in L:
    nf = []
    f = []
    path = i
    print(i)
    df = pd.read_csv(i)
    df = df.drop(df.columns[0], axis=1)
    labels = df.featured
    #df = df.drop('featured', axis=1)
    content = df.content
    #df = df.drop('content', axis=1)
    res = df.ratio_featured
    print('got the df', 'featured posts:', labels.sum())
    if labels.sum()==0:
        print('no featured posts in this thread')
        continue
    labels = list(labels)
    print('Got the labels')
    f = []
    
    for i in range(len(res)):
        if res[i] > 0.03:
            f.append(1)
        else:
            f.append(0)
    
    labels2 = []
    for lab in labels:
        if lab == True:
            labels2.append(1)
        else:
            labels2.append(0)
    df['pred_label']= f
    df['label'] = labels2
    if 1 not in f:
        continue
    if 1 not in labels2:
        continue        
    df = df.sort_values(by=['ratio_featured'], ascending=False)
    if len(df)<10:
        continue
    pred2 = df['pred_label']
    true2 = df['label']
    
    labs = list(df['label'])
    for k in [3,5,10]:
        labs2 = labs[:k]
        if k == 3:
            dcg = (labs[0]/np.log2(1+1))+(labs[1]/np.log2(2+1))+(labs[2]/np.log2(3+1))
            sorte = sorted(labs2, reverse=True)
            idcg = (sorte[0]/np.log2(1+1))+(sorte[1]/np.log2(2+1))+(sorte[2]/np.log2(3+1))
            ndcg = dcg/idcg
            at_3.append(ndcg)
        if k ==5:
            dcg = (labs[0]/np.log2(1+1))+(labs[1]/np.log2(2+1))+(labs[2]/np.log2(3+1))+(labs[3]/np.log2(4+1))+(labs[4]/np.log2(5+1))
            sorte = sorted(labs2, reverse=True)
            idcg = (sorte[0]/np.log2(1+1))+(sorte[1]/np.log2(2+1))+(sorte[2]/np.log2(3+1))+(sorte[3]/np.log2(4+1))+(sorte[4]/np.log2(5+1))
            ndcg = dcg/idcg
            at_5.append(ndcg)
        if k ==10:
            dcg = (labs[0]/np.log2(1+1))+(labs[1]/np.log2(2+1))+(labs[2]/np.log2(3+1))+(labs[3]/np.log2(4+1))+(labs[4]/np.log2(5+1))+(labs[5]/np.log2(6+1))+(labs[6]/np.log2(7+1))+(labs[7]/np.log2(8+1))+(labs[8]/np.log2(9+1))+(labs[9]/np.log2(10+1))
            sorte = sorted(labs2, reverse=True)
            idcg = (sorte[0]/np.log2(1+1))+(sorte[1]/np.log2(2+1))+(sorte[2]/np.log2(3+1))+(sorte[3]/np.log2(4+1))+(sorte[4]/np.log2(5+1))+(sorte[5]/np.log2(6+1))+(sorte[6]/np.log2(7+1))+(sorte[7]/np.log2(8+1))+(sorte[8]/np.log2(9+1))+(sorte[9]/np.log2(10+1))
            ndcg = dcg/idcg
            at_10.append(ndcg)


print('NDCG@k: (3,5,10):')
print(s.mean(at_3), s.mean(at_5), s.mean(at_10))    