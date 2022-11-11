import numpy as np 
import pandas as pd 
import re
import random
import math
from tqdm.notebook import tqdm
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.inspection import permutation_importance


import pickle
with open('rf_bow', 'rb') as f:
    rf1 = pickle.load(f)  # Get best performing model to calculate feature importance


## Full  data:
df_train = pd.read_csv('df_train_bow.csv')
df_test = pd.read_csv('df_test_bow.csv')
df_test = df_test.drop('status', axis=1)
train_labels = df_train.featured
test_labels = df_test.featured

###########################################
# Calculate feature importance without clusters so that we can rank features within clusters later
df = df_test
df = df.drop('featured', axis=1)
df = df.replace(np. nan,0)
df= df.drop('accepted_count_user', axis=1)
df= df.drop('rejected_count_user', axis=1)
df= df.drop('total_reply_posts_user', axis=1)
df= df.drop('delta_seconds', axis=1)
df= df.drop('ratio_reply', axis=1)
df =df.drop(df.columns[0], axis=1)
print('Getting original importance')
perm_importance = permutation_importance(rf1, df, test_labels, scoring='f1', n_repeats = 3, random_state=1234, n_jobs=-1)
imp2 = pd.DataFrame(perm_importance.importances_mean)
feature_names = list(df)
imp2['var'] = feature_names



##########################################



train_labels = df_train.featured
test_labels = df_test.featured
frames = [df_train, df_test]
df = pd.concat(frames)
df = df.drop('featured', axis=1)
df = df.replace(np. nan,0)
df = df.drop(df.columns[0], axis=1)

# Get Dendrogram of features
feature_names = list(df)
corr = spearmanr(df).correlation
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1)
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(
    dist_linkage, labels=feature_names, ax=ax1, leaf_rotation=90
)
dendro_idx = np.arange(0, len(dendro["ivl"]))

ax2.imshow(corr[dendro["leaves"], :][:, dendro["leaves"]])
ax2.set_xticks(dendro_idx)
ax2.set_yticks(dendro_idx)
ax2.set_xticklabels(dendro["ivl"], rotation="vertical")
ax2.set_yticklabels(dendro["ivl"])
plt.savefig('Dendrogram.pdf')

print('Got dendrogram')

cluster_ids = hierarchy.fcluster(dist_linkage, 1.1, criterion="distance") # Set threshold for clustering
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
clusterfeat = []
for i in range(1,(len(cluster_id_to_feature_ids)+1)):
    temp = cluster_id_to_feature_ids[i]
    temp = list(df.iloc[:,temp])
    clusterfeat.append(temp)
selected_features = []



names = []
for v in cluster_id_to_feature_ids.values():
    namelist = df.iloc[:,v] # Get names of features in this cluster
    namelist = list(namelist)
    feats = v
    v_imp = []
    for index in feats:
        name = df.iloc[:, index].name
        temp = imp2.loc[imp2['var'] == name, 0] # Get value from original importance
        if name in list(imp2['var']): #some variables are not in original importance, they are set to 0
            v_imp.append(float(temp))
        else:
            v_imp.append(0)
    index_max = max(range(len(v_imp)), key=v_imp.__getitem__) # Get index of max importance in original
    selected_features.append(v[index_max])
    names.append(namelist[index_max])
    
    
df3 = df.iloc[:,selected_features] # New dataframe with one feature per cluster
print('Reformatted data with clustered features')
###### Now we ran retrain RF with less features and recalculate importance based on this model

from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt']
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
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random = rf_random.fit(df_train2, train_labels)
rf1 = rf_random.best_estimator_


# Evaluation
# If metrics are too much off in comparison to original models, the cluster threshold was too high
pred_labels = rf1.predict(df_test2)
print('recall on validation set:',recall_score(test_labels, list(pred_labels), pos_label=True))
print('prec on validation set:',precision_score(test_labels, list(pred_labels), pos_label=True))
print('f1 on validation set:',f1_score(test_labels, list(pred_labels), pos_label=True))

# Calculate new feature importance based on clusters
perm_importance = permutation_importance(rf1, df_test2, test_labels,scoring='f1', n_repeats = 3, random_state=1234, n_jobs=-1)
imp = pd.DataFrame(perm_importance.importances_mean)
feature_names = list(df_test2)
imp['var'] = feature_names
imp = imp.sort_values(by=[0], ascending=False)
imp.to_csv('clustered_feature_importance.csv')
imp = imp.head(10)
#sns.set(font_scale=5) # if too small
sns_plot = sns.barplot(x=imp[0], y=imp['var'],  orient='h')
plt.savefig('feature_importance_clusterd.png')
print('Done')



