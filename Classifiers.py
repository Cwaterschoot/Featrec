import numpy as np 
import pandas as pd 
import re
import random
import math
from tqdm.notebook import tqdm
from collections import Counter
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import random
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from transformers import RobertaTokenizer, RobertaModel
from sentence_transformers import models
from sentence_transformers import SentenceTransformer
import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer
from transformers import Trainer, TrainingArguments
import pickle

# Open datasets
train = pd.read_csv('train2.csv')
validate = pd.read_csv('validate.csv')
test = pd.read_csv('test.csv')

# Labels
train_labels = train.featured
test_labels = test.featured
validate_labels = validate.featured

# No text representation at this point
train = train.drop('content', axis=1)
test = test.drop('content', axis=1)
validate = validate.drop('content', axis=1)

#############################################
################## RF #######################
#############################################

# Defining grid search to be used in all RFs
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
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, verbose=2, random_state=42, n_jobs = -1)
rf_random = rf_random.fit(train, train_labels)
rf1 = rf_random.best_estimator_


###### VALIDATION SET, ONLY USED WITH RF #############################
# Drop featured info from validate
validate = validate.drop('featured', axis=1)

pred = rf1.predict(validate)
print('f1 on validation set (RF):',f1_score(validate_labels, list(pred), pos_label=True))
print('prec on validation set (RF):',precision_score(validate_labels, list(pred), pos_label=True))
print('rec on validation set (RF):',recall_score(validate_labels, list(pred), pos_label=True))
######################################################################

# Evaluation on Test set
#test = test.drop('featured', axis=1)
pred = rf1.predict(test)
print('f1 on test set (RF):',f1_score(test_labels, list(pred), pos_label=True))
print('prec on test set (RF):',precision_score(test_labels, list(pred), pos_label=True))
print('rec on test set (RF):',recall_score(test_labels, list(pred), pos_label=True))

# Save RF
with open('rf', 'wb') as a:
    pickle.dump(rf1, a)


#############################################
##############################Baseline ######
#############################################

baseline = list(test['ratio_featured'])
baseline_labels = []
for i in range(len(baseline)):
    if baseline[i] >=0.03:
        baseline_labels.append(True)
    else:
        baseline_labels.append(False)
print('f1 on test set (BASELINE):',f1_score(test_labels, list(baseline_labels), pos_label=True))
print('prec on test set (BASELINE)::',precision_score(test_labels, list(baseline_labels), pos_label=True))
print('rec on test set (BASELINE)::',recall_score(test_labels, list(baseline_labels), pos_label=True))




#############################################
################## RF_BoW ###################
#############################################

# Load BoW representations of train and test
train_bow = pd.read_csv('train_bow.csv')
test_bow = pd.read_csv('test_bow.csv')

train_labels_bow = train_bow.featured
test_labels_bow= test_bow.featured
train_bow = train_bow.drop('featured', axis=1)
train_bow = train_bow.drop('content', axis=1)
test_bow = test_bow.drop('featured', axis=1)
test_bow = test_bow.drop('content', axis=1)
test_bow = test_bow.drop(test_bow.columns[0], axis=1)
train_bow = train_bow.drop(train_bow.columns[0], axis=1)
train_bow = train_bow.drop('lab', axis=1)
test_bow = test_bow.drop('lab', axis=1)


# Training on same grid as before
rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, scoring = 'f1', verbose=2, random_state=42, n_jobs = -1)
rf_random = rf_random.fit(train_bow, train_labels_bow)
rf1 = rf_random.best_estimator_

# Evaluation on Test set
pred = rf1.predict(test_bow)
print('f1 on test set (RF_BoW):',f1_score(test_labels_bow, list(pred), pos_label=True))
print('prec on test set (RF_BoW):',precision_score(test_labels_bow, list(pred), pos_label=True))
print('rec on test set (RF_BoW):',recall_score(test_labels_bow, list(pred), pos_label=True))


# Save rf_BoW
with open('rf_bow', 'wb') as a:
    pickle.dump(rf1, a)


#############################################
################## RobBERT  #################
#############################################

train = pd.read_csv('train2.csv')
validate = pd.read_csv('validate.csv')
test = pd.read_csv('test.csv')

# labels
train_labels = list(train2.featured)
validate_labels = list(validate['featured'])
test_labels = list(test['featured'])

# Get texts only
train_texts = list(train2['content'])
validate_texts = list(validate['content'])
test_texts = list(test['content'])

# Adjust labels for training (False =0, True=1)
train_labels2 = []
for i in range(len(train_labels)):
    if train_labels[i] == False:
        train_labels2.append(0)
    else:
        train_labels2.append(1)
        
for i in range(len(test_labels)):
    if test_labels[i] == False:
        test_labels[i] = 0
    else:
        test_labels[i]=1
        
for i in range(len(validate_labels)):
    if validate_labels[i] == False:
        validate_labels[i] = 0
    else:
        validate_labels[i]=1

# Names
target_names = ['Not_feat', 'Feat']

# Tokenizing
tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robBERT-base")
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=205)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=205)
validate_encodings = tokenizer(validate_texts, truncation=True, padding=True, max_length=205)

# Combining datasets
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)

# convert our tokenized data into a torch Dataset
train_dataset = Dataset(train_encodings, train_labels2)
test_dataset = Dataset(test_encodings, test_labels)
validate_dataset = Dataset(validate_encodings, validate_labels)

# Load base model
model = RobertaForSequenceClassification.from_pretrained("pdelobelle/robBERT-base",num_labels=len(target_names))



def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    
    return {
      'accuracy': acc,
      'f1': f1,  
      'precision': precision,
      'recall': recall  
          }
# Specifics for training
training_args = TrainingArguments(
    output_dir='./results/feat_rec',          
    num_train_epochs=10,              
    per_device_train_batch_size=64,  
    per_device_eval_batch_size=64,   
    warmup_steps=250, 
    learning_rate=5e-5,
    weight_decay=0.01,               
    logging_dir='./logs/feat_rec',            
    load_best_model_at_end=True,    
    metric_for_best_model='f1',
    logging_steps=500,               
    evaluation_strategy="steps",     
)

# Trainer
trainer = Trainer(
    model=model,                         
    args=training_args,                 
    train_dataset=train_dataset,        
    eval_dataset=validate_dataset,         
    compute_metrics=compute_metrics, 
)
# Gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# And training itself
trainer.train()

# Save model
model_path = "featrec"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)


# On test set:
def get_prediction(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=205, return_tensors="pt")
    outputs = model(**inputs)
    probs = outputs[0].softmax(1)
    return probs.tolist()

test_labels2 = []
#print(test_labels)
for i in range(len(test_labels)):
    if test_labels[i]==0:
        test_labels2.append('Non_feat')
    else:
        test_labels2.append('Feat')

nf = []
f = []


for i in range(len(test_texts)):
    temp = get_prediction(test_texts[i])
    temp = temp[0]
    nf.append(temp[0])
    f.append(temp[1])
pred = pd.DataFrame(nf)
pred['featured'] = f
pred = pred.rename(columns={0: 'Non_feat', 'featured': 'Feat'})
pred_label = []
for row in pred.itertuples():
    if row.Non_feat > row.Feat:
        pred_label.append('Non_feat')
    else:
        pred_label.append('Feat')
        
pred['Pred_label'] = pred_label
pred['Label'] = test_labels2


print('prec on test set (RobBERT):', precision_score(pred['Label'], pred['Pred_label'], pos_label='Feat'))
print('recall on test set(RobBERT):', recall_score(pred['Label'], pred['Pred_label'], pos_label='Feat'))
print('f1 on test set(RobBERT):',f1_score(pred['Label'], pred['Pred_label'], pos_label='Feat'))



#############################################
################## RF_emb ###################
#############################################


# Getting CLS tokens from train and test


word_embedding_model = models.Transformer("featrec")

pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=False,
                               pooling_mode_cls_token=True,
                               pooling_mode_max_tokens=False)

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
train = pd.read_csv('train2.csv')
train_texts_emb = list(train.content)
test = pd.read_csv('test.csv')
test_texts_emb = list(test.content)
print('encoding')
tr_emb = []
for i in range(len(train_texts_emb)):
    temp = model.encode(train_texts_emb[i])
    tr_emb.append(temp)
print('train done')
test_emb = model.encode(test_texts_emb)

train_emb= pd.DataFrame(tr_emb)
test_emb = pd.DataFrame(test_emb)

# Combining CLS tokens with non-textual features
test = test.drop(test.columns[0], axis=1)
train = train.drop(train.columns[0], axis=1)
train2 = pd.concat([train, train_emb], axis=1)
test2 = pd.concat([test, test_emb], axis=1)

# Save for replication
train2.to_csv('train_emb_full.csv')
test2.to_csv('test_emb_full.csv')

# Dropping unnecessary features
train_labels_emb = train2.featured
test_labels_emb= test2.featured
train2 = train2.drop('featured', axis=1)
train2 = train2.drop('content', axis=1)
test2 = test2.drop('featured', axis=1)
test2 = test2.drop('content', axis=1)

# Training
train2.columns = train2.columns.map(str)
test2.columns = test2.columns.map(str)

rf = RandomForestClassifier()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, scoring = 'f1', verbose=2, random_state=42, n_jobs = -1)
rf1 = rf_random.fit(train2, train_labels_emb)
pred = rf1.predict(test2)
print('f1 on test set (RF_emb):',f1_score(test_labels_emb, list(pred), pos_label=True))
print('prec on test set (RF_emb):',precision_score(test_labels_emb, list(pred), pos_label=True))
print('rec on test set (RF_emb):',recall_score(test_labels_emb, list(pred), pos_label=True))

# Save RF_EMB
with open('rf_emb', 'wb') as a:
    pickle.dump(rf1, a)
