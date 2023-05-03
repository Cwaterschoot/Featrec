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
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
import numpy as np
import random
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robBERT-base")
at_3 = []
at_5= []
at_10=[]
plt.style.use('seaborn-whitegrid')


target_names = ['Not_feat', 'Feat']

model = RobertaForSequenceClassification.from_pretrained("robbert/featrec_16122022",num_labels=len(target_names))


def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=250, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return probs.tolist()

path = ""

filenames = glob.glob(path + "/*.csv") 
L = filenames
used_arts = []

for i in L:
    if i in used_arts:
        print('already done')
        continue
    nf = []
    f = []
    path = i
    print(i)
    df = pd.read_csv(i)

    text = df.content
    labels = df.featured
    print('got the df', 'featured posts:', labels.sum())
    if labels.sum()==0:
        print('no featured posts in this thread')
        continue
    labels = list(labels)
    for x in range(len(labels)):
        if labels[x]==False:
            labels[x]='Not_feat'
        else:
            labels[x]='Feat'
    print('Got the labels')
    for p in range(len(text)):
        temp = get_prediction(text[p])
        temp = temp[0]
        nf.append(temp[0])
        f.append(temp[1])
        #print(p)
    print('predicted')
    pred = pd.DataFrame(nf)
    pred['featured'] = f
    pred = pred.rename(columns={0: 'Not_featured', 'featured': 'Featured'})
    pred_label = []
    for row in pred.itertuples():
        if row.Not_featured  > row.Featured:
            pred_label.append('Not_Featured')
        else:
            pred_label.append('Featured')
        
    pred['Pred_label'] = pred_label
    pred['Label'] = labels
    pred = pred.sort_values(by=['Featured'], ascending=False)

    if 'Featured' not in pred_label:
        continue
    if 'Feat' not in labels:
        continue
    pred = pred.sort_values(by=['Featured'], ascending=False)
    pred2 = []
    true2 = []
    for row in pred.itertuples():
        if row.Pred_label=='Featured':
            pred2.append(1)
        else:
            pred2.append(0)
        if row.Label=='Feat':
            true2.append(1)
        else:
            true2.append(0)

    labs = true2
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
    used_arts.append(i)

print('NDCG@k: (3,5,10):')
print(s.mean(at_3), s.mean(at_5), s.mean(at_10))