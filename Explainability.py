from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.model_selection import RandomizedSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
import seaborn as sns
from openpyxl import load_workbook
import glob
from treeinterpreter import treeinterpreter as ti

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,6))

# Get correct data
df_train = pd.read_csv('train_bow.csv')
df_test = pd.read_csv('test_bow.csv')

train_labels = df_train.featured
test_labels = df_test.featured
frames = [df_train, df_test]
df = pd.concat(frames)
df = df.drop('featured', axis=1)
df = df.drop('content', axis=1)
df=df.drop('lab', axis=1)
df = df.replace(np. nan,0)
df = df.drop(df.columns[0], axis=1)

feature_names = list(df)

# Get correlations and plot dendrogram + correlation matrix
# With this many features, it does not tell us much. Purely for calculation purposes later on.
corr = spearmanr(df).correlation
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
corr = np.nan_to_num(corr)
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix,checks=False))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=feature_names, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
fig.tight_layout()
plt.show()
plt.savefig('Dendrogram2.pdf') # Save figure 


# DEFINE THRESHOLD, 1.1 USED IN PAPER
# threshold influences number of clusters to keep
cluster_ids = hierarchy.fcluster(dist_linkage, 1.1, criterion="distance")
cluster_id_to_feature_ids = defaultdict(list)

for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
    
clusterfeat = []
for i in range(1,(len(cluster_id_to_feature_ids)+1)):
    temp = cluster_id_to_feature_ids[i]
    temp = list(df.iloc[:,temp])
    clusterfeat.append(temp)
    
# Get feature importances from permutation on whole test set (to pick representatives)
imp2 = pd.read_excel('original_feature_importance.xlsx')

# Gather features to represent clusters using imp2
selected_features = []
names = []
for v in cluster_id_to_feature_ids.values():
    namelist = df.iloc[:,v]
    namelist = list(namelist)
    feats = v
    v_imp = []
    for index in feats:
        name = df.iloc[:, index].name
        temp = imp2.loc[imp2['var'] == name, 0]
        if name in list(imp2['var']):
            v_imp.append(float(temp))
        else:
            v_imp.append(0)
    index_max = max(range(len(v_imp)), key=v_imp.__getitem__)
    selected_features.append(v[index_max])
    names.append(namelist[index_max])

# get representatives from original dataframe
df3 = df.iloc[:,selected_features]


# Quick trick to rename dataframe with whole cluster + representative first
# This gives us a better overview and allows for plotting
for i in range(len(names)):
    for lijst in clusterfeat:
        if names[i] not in lijst:
            continue
        else:
            newname = ''.join(lijst)
            front = str(names[i])
            newname = str(front + str(lijst))
            #print(newname)
            df3.rename({front: front + str(lijst)}, axis=1, inplace=True)


# resplit data: printing to double check numbers
print(len(df_train), len(df_test))
df_train2 = df3.head(60465)
df_test2 = df3.tail(32122)


# RETRAIN RANDOM FOREST WITH REPRESENTATIVES:


n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
criterion = ['gini', 'entropy']
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap, 'criterion': criterion}

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 30, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random = rf_random.fit(df_train2, train_labels)
rf1 = rf_random.best_estimator_

pred_labels = rf1.predict(df_test2)
print('recall on test set (Rf_clustered):',recall_score(test_labels, list(pred_labels), pos_label=True))
print('prec on test set (Rf_clustered):',precision_score(test_labels, list(pred_labels), pos_label=True))
print('f1 on test set (Rf_clustered):',f1_score(test_labels, list(pred_labels), pos_label=True))



# Calculate feature importance
perm_importance = permutation_importance(rf1, df_test2, test_labels,scoring='f1', n_repeats = 3, random_state=1234, n_jobs=-1)
imp = pd.DataFrame(perm_importance.importances_mean)
feature_names = list(df_test2)
imp['var'] = feature_names
imp = imp.sort_values(by=[0], ascending=False)
imp = imp.head(10)
# Print out top 10 most important features
print('Top 10 most important features + MDF1 score:')
print(imp)

# FOR PLOT:
names = ['Comment Statistics','User Information',  'Comment Length', 'User Information (2)', 'Questions + content: why?', 'Content: COVID-19']
imp = imp.head(6)
imp['names'] = names
sns.set_palette('deep')
sns.barplot(x=imp[0], y=imp['names'],  orient='h')


###################################################
##### ERROR ANALYSIS ###############################
###################################################




# Open RF_BoW to obtain errors
with open('rf_bow', 'rb') as f:
    rf1 = pickle.load(f)

# We need one article to get column names
# Does not matter which one
df = pd.read_csv('articles/6098804.csv')
df = df.drop('featured', axis=1)
content = df.content
df =df.drop(df.columns[0], axis=1)
df = df.drop('content', axis=1)
df = df.replace(np. nan,0)
colnames = list(df)
dat_fp = pd.DataFrame(columns=colnames)    
dat_fn = pd.DataFrame(columns=colnames)    
dat_tp = pd.DataFrame(columns=colnames)    
dat_tn = pd.DataFrame(columns=colnames)    

# FILL IN PATH
path = "articles_bow"

filenames = glob.glob(path + "/*.csv") 

L = filenames

vals_fn = []
vals_fp = []
count = 0


# This loop gathers FPs, FNs, TPs and TNs

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
    df =df.drop(df.columns[0], axis=1)
    df = df.drop('featured', axis=1)
    content = df.content
    df = df.drop('content', axis=1)
    df = df.replace(np. nan,0)
    pred = rf1.predict_proba(df)
    pred = pd.DataFrame(pred)
    f = []
    nf = []
    index = list(df.index)
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
    pred['index']= index
    pred = pred.sort_values(by=['f'], ascending=False)
    pred = pred.head(5)
    count +=1
    for row in pred.itertuples():
        if (row.pred_label == False) & (row.label == True):
            val = int(row.index)
            dat_fn = pd.concat([dat_fn, df.loc[[val]]], ignore_index=True)
            vals_fn.append(row.nf)
        if (row.pred_label == True) & (row.label == False):
            val = int(row.index)
            dat_fp = pd.concat([dat_fp, df.loc[[val]]], ignore_index=True)
            vals_fp.append(row.f)
        if (row.pred_label == True) & (row.label ==True):
            val = int(row.index)
            dat_tp = pd.concat([dat_tp, df.loc[[val]]], ignore_index=True)
        if (row.pred_label == False) & (row.label == False):
            val = int(row.index)
            dat_tn = pd.concat([dat_tn, df.loc[[val]]], ignore_index=True)

# IDS for boxplots used in paper
dat_tn['id']= 'True Negative'
dat_fp['id'] = 'False Positive'
dat_tp['id'] = 'True Positive'
dat_fn['id']='False Negative'
dats = pd.concat([dat_tn, dat_fp, dat_tp, dat_fn])

# Boxplots used in paper:
sns.set(font_scale=1.5)
sns.boxplot(data=dats, x="respect_count", y="id",showfliers = False)
sns.boxplot(data=dats, x="ratio_featured", y="id",showfliers = False)

#######################################
# Contributions #######################
#######################################



# FP
selection = list(range(1, len(dat_fp)))
selected_dat = dat_fp
selected_dat = selected_dat.drop('id', axis=1)
dat2 = selected_dat
prediction, bias, contributions = ti.predict(rf1, selected_dat)

features = dat2.columns
cont = pd.DataFrame(features)
cont = cont.rename(columns={0: 'Features'})
for i in range(len(selection)):

        l = sorted(zip(contributions[i],
                                 dat2.columns),key=lambda x: ~abs(x[0].any()))
        vals = []
        for k in range(len(l)):
            ar = l[k]
            temp = ar[0]
            vals.append(temp[1])
        vals = pd.DataFrame(vals)
        vals.rename({0: str(i)}, axis=1, inplace=True) 
        cont = pd.concat([cont, vals], axis=1)
    
cont['mean'] = cont.mean(axis=1)
cont = cont.sort_values(by=['mean'], ascending=False)

# Print Contributions
print('Contribution to FPs:')
print(cont.head(10))


# FN
selection = list(range(1, len(dat_fn)))
selected_dat = dat_fn
selected_dat = selected_dat.drop('id', axis=1)
dat2 = selected_dat
prediction, bias, contributions = ti.predict(rf1, selected_dat)

features = dat2.columns
cont = pd.DataFrame(features)
cont = cont.rename(columns={0: 'Features'})
for i in range(len(selection)):

        l = sorted(zip(contributions[i],
                                 dat2.columns),key=lambda x: ~abs(x[0].any()))
        vals = []
        for k in range(len(l)):
            ar = l[k]
            temp = ar[0]
            vals.append(temp[1])
        vals = pd.DataFrame(vals)
        vals.rename({0: str(i)}, axis=1, inplace=True) 
        cont = pd.concat([cont, vals], axis=1)
    
cont['mean'] = cont.mean(axis=1)
cont = cont.sort_values(by=['mean'], ascending=False)

# Print Contributions
print('Contribution to FNs:')
print(cont.head(10))
