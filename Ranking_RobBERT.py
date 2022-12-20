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
import statistics as s


target_names = ['Not_feat', 'Feat']

model = RobertaForSequenceClassification.from_pretrained("featrec",num_labels=len(target_names))


def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=205, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return probs.tolist()

# FILL IN PATH TO ARTICLES
path = "articles"

filenames = glob.glob(path + "/*.csv") 
L = filenames
rec = []
rec5 = []
rec10= []
rec15=[]
rec20=[]
rec25=[]
rec30=[]

for i in L:
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
    tot = pred['Label'].value_counts()
    tot = tot['Feat']
    r = [5,10,15,20,25,30]
    for l in r :
        pred2 = pred[0:l]
        found = pred2['Label'].value_counts()
        if 'Not_feat' in found:
            if found['Not_feat']==l:
                found = 0
            else:
                found = found['Feat']
        else:
            found=found['Feat']

        if l ==5:
            rec5.append(found/tot)
        elif l==10:
            rec10.append(found/tot)
        elif l==15:
            rec15.append(found/tot)
        elif l==20:
            rec20.append(found/tot)
        elif l==25:
            rec25.append(found/tot)
        else:
            rec30.append(found/tot)

    
    print('Finished:', i)
    

precision1 = []
precision2 = []
precision3 = []
precision4 = []
precision5 = []
precision6 = []
precision7 = []
precision8 = []
precision9 = []
precision10 = []

for i in L:
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

    pred = pred.sort_values(by=['Featured'], ascending=False)
    total_pred = pred['Pred_label'].value_counts()
    r = [1,2,3,4,5,6,7,8,9,10]
    for l in r :
        if l > len(pred):
            continue
        size = l    
        if total_pred < size:
            continue
        pred2 = pred[0:l]
        found = pred2['Label'].value_counts()
        tot = pred2['Pred_label'].value_counts()

        if 'Feat' in found:
            found = found['Feat']
            prec = found / size
        else:
            prec=0
        if l ==1:
            precision1.append(prec)
        elif l==2:
            precision2.append(prec)
        elif l==2:
            precision2.append(prec)
        elif l==3:
            precision3.append(prec)
        elif l==4:
            precision4.append(prec)
        elif l==5:
            precision5.append(prec)
        elif l==6:
            precision6.append(prec)
        elif l==7:
            precision7.append(prec)
        elif l==8:
            precision8.append(prec)
        elif l==9:
            precision9.append(prec)
        else:
            precision10.append(prec)
    print('Finished:', i)

print('Mean recall@k: (5,10,20,25,30):')
print(s.mean(rec5),s.mean(rec10),s.mean(rec15),s.mean(rec20),s.mean(rec25),s.mean(rec30)) 
print('Mean precision@k (1,2,3,4,5,6,7,8,9,10):')  
print(s.mean(precision1),s.mean(precision2),s.mean(precision3),s.mean(precision4),s.mean(precision5),s.mean(precision6),s.mean(precision7),s.mean(precision8),s.mean(precision9),s.mean(precision10)  )       