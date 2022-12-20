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




#############################################
################## RF #######################
#############################################


# Fill in correct path to articles folder
path = "articles" 

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
plt.style.use('seaborn-whitegrid')


path = "/Users/cedricwaterschoot/Desktop/Meertens/docs/featured_rec/articles"


filenames = glob.glob(path + "/*.csv") 
L = filenames
rec = []
rec5 = []
rec10= []
rec15=[]
rec20=[]
rec25=[]
rec30=[]

with open('rf', 'rb') as f:
    rf1 = pickle.load(f)

for i in L:
    nf = []
    f = []
    path = i
    print(i)
    df = pd.read_csv(i)
    df = df.drop(df.columns[0], axis=1)
    labels = df.featured
    df = df.drop('featured', axis=1)
    content = df.content
    df = df.drop('content', axis=1)
    print('got the df', 'featured posts:', labels.sum())
    if labels.sum()==0:
        print('no featured posts in this thread')
        continue
    labels = list(labels)
    print('Got the labels')
    pred = rf1.predict_proba(df)
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
    tot = pred['label'].value_counts()
    tot = tot[True]
    r = [5,10,15,20,25,30]
    for l in r :
        #print(l)
        pred2 = pred[0:l]
        found = pred2['label'].value_counts()
        if False in found:
            if found[False]==l:
                found = 0
            else:
                found = found[True]
        else:
            found=found[True]
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
    print('Finsihed:', i)
    
    
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
    df = df.drop(df.columns[0], axis=1)
    labels = df.featured
    df = df.drop('featured', axis=1)
    content = df.content
    df = df.drop('content', axis=1)
    print('got the df', 'featured posts:', labels.sum())
    if labels.sum()==0:
        print('no featured posts in this thread')
        continue
    labels = list(labels)
    print('Got the labels')
    pred = rf1.predict_proba(df)
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
    total_pred = pred['pred_label'].value_counts()
    if True in total_pred:
        total_pred = total_pred[True]
    else:
        total_pred=0
    print(total_pred)
    r = [1,2,3,4,5,6,7,8,9,10]
    for l in r :
        if l > len(pred):
            continue
        size = l    
        if total_pred <= size:
            continue
        pred2 = pred[0:size]
        found = pred2['label'].value_counts()
        tot = pred2['pred_label'].value_counts()

        if True in found:
            found = found[True]
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
    #precision.append(found/prectot)
    print('Finsihed:', i)

print('Mean recall@k: (5,10,20,25,30):')
print(s.mean(rec5),s.mean(rec10),s.mean(rec15),s.mean(rec20),s.mean(rec25),s.mean(rec30)) 
print('Mean precision@k (1,2,3,4,5,6,7,8,9,10):')  
print(s.mean(precision1),s.mean(precision2),s.mean(precision3),s.mean(precision4),s.mean(precision5),s.mean(precision6),s.mean(precision7),s.mean(precision8),s.mean(precision9),s.mean(precision10)  )