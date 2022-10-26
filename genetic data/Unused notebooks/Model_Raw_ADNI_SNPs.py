#!/usr/bin/env python
# coding: utf-8

# In[1]:


if False:
    def warn(*args, **kwargs):
        pass
    import warnings
    warnings.warn = warn

import os
import time
import sys

import torch
from torch import nn
from torch.autograd import Variable
import shap
from copy import deepcopy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from datetime import datetime
class simple_model(nn.Module):
    def __init__(self, num_features=1260680*3, num_hidden=0, hidden_dim=32, drop_probab=.5):
        super(simple_model, self).__init__()
        
        ####
        self.drop_probab = drop_probab
        self.num_hidden = num_hidden
        self.dropout0 = nn.Dropout(p=self.drop_probab)
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.dropout1 = nn.Dropout(p=self.drop_probab)
        self.fc_hidden = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(self.num_hidden)])
        self.dropout_hidden = nn.ModuleList([nn.Dropout(p=self.drop_probab) for i in range(self.num_hidden)])
        self.fc2 = nn.Linear(hidden_dim, 8)
        self.dropout2 = nn.Dropout(p=self.drop_probab)
        self.outLayer = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()
        ####

    def forward(self, features):
        print('..>', features.shape, datetime.now())
        features = self.dropout0(features)
        features = self.fc1(features)
        features = self.dropout1(features)
        for i in range(self.num_hidden):
            features = self.fc_hidden[i](features)
            features = self.dropout_hidden[i](features)
        features = self.fc2(features)
        features = self.dropout2(features)
        logit = self.outLayer(features)
#         print(features.shape, features)
        probab = self.sigmoid(logit)
        return probab
    


# In[ ]:





# In[2]:


import numpy as np
import json
import random


# In[3]:


import pickle
snps_to_consider = pickle.load(open('snps_to_consider.pkl', 'rb'))

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(categories='auto', handle_unknown='error')
X = [[0], [1], [2]]
print(enc.fit(X))
print(enc.categories_)
temp = [[0], [1], [0], [2]]
encoded = enc.transform(temp).toarray().astype(np.float32)
print(encoded)


# enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])

feature_names = enc.get_feature_names(['SNP'])
print(feature_names)


# In[4]:


def generate_raw_snp_samples(IID, get_onehot=False):
    f = open(f'./RAW_SNPs/{IID}.snp', 'r')
    line = f.readline()
    line = line.replace('NA', '0').split() 
    FID, IID, PAT,  MAT, SEX, SNPs = line[0], line[1], line[2], line[3], line[4], line[6:]
    SNPs = np.array(SNPs).astype(int)[snps_to_consider]
    if get_onehot:
        SNPs = enc.transform(SNPs.reshape([-1, 1])).toarray().astype(np.float32)
    return FID, IID, PAT,  MAT, SEX, SNPs

# FID, IID, PAT,  MAT, SEX, SNPs = generate_raw_snp_samples(IID='018_S_0633', get_onehot=True)
# FID, IID, PAT,  MAT, SEX, SNPs


# In[5]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc

def epoch(model, optimizer, criterion, is_training, loader):
    pred = []
    true = []
    total_loss = 0.
    
    for batch_idx, (features, label) in enumerate(loader):
#         print('checkpoint');break
        features = torch.autograd.Variable(features.view(features.shape[0], -1).to(DEVICE).float())
        label = torch.autograd.Variable(label.to(DEVICE).float())
#         print(features.shape, label.shape)
        if is_training:
            model.train()
        else:
            model.eval()
        probab = model(features)
        if is_training:  
            loss = criterion(probab, label)
            ## compute gradient and do SGD step 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
#             print(batch_idx, ':', loss) 
        pred += probab.detach().cpu().numpy().tolist()
        true += label.detach().cpu().numpy().tolist()
    
    pred, true, total_loss = np.array(pred).reshape([-1]), np.array(true).reshape([-1]), total_loss
    pred_binary = (pred > .5).astype(float)
#     precision, recall, fscore, support = precision_recall_fscore_support(true, pred_binary)
#     auroc = roc_auc_score(true, pred)
#     p, r, thresholds = precision_recall_curve(true, pred)
#     auprc = auc(r, p)
    acc = (pred_binary==true).mean()
    
#     return precision[1], recall[1], fscore[1], support, auroc, auprc, acc, total_loss, pred, pred_binary, true
    return None, None, None, None, None, None, acc, total_loss, pred, pred_binary, true


# In[6]:


import torch
from torch.utils import data

class dataSet(data.Dataset):
    def __init__(self, samples_list):
        super(dataSet, self).__init__()  
        self.data_len = len(samples_list)
        self.samples_list = samples_list

    def __getitem__(self, index):
        IID = self.samples_list[index][0]
        missing_files=open("missing.txt",'a')

        try:
            FID, IID, PAT,  MAT, SEX, SNPs = generate_raw_snp_samples(IID=IID, get_onehot=True)
            features = torch.from_numpy(SNPs).float()
            label = torch.tensor([float(self.samples_list[index][1])]).float()
            return features, label
        except:
            print(f'file missing : RAW_SNPs/{IID}.snp')
            missing_files.write("RAW_SNPs/{IID}.snp\n")


            return 0,0
        
    def __len__(self):
        return self.data_len


# In[7]:


def generate_datasets(train_indices, test_indices, random_seed):
    if random_seed is not None: 
        random.seed(random_seed * 3)
    random.shuffle(train_indices)
    train_indices = np.array(train_indices)
    split_pos = int(train_indices.shape[0] * 0.8)
    train_indices, val_indices = train_indices[:split_pos], train_indices[split_pos:]
    train_set = dataSet(samples_list=Final_Samples[train_indices])
    val_set = dataSet(samples_list=Final_Samples[val_indices])
    test_set = dataSet(samples_list=Final_Samples[test_indices])
    
    return train_set, val_set, test_set

def generate_loader(train_set, val_set, test_set, num_workers, CUSTOM_BATCH_SIZE=128):
    train_batch_size = CUSTOM_BATCH_SIZE if CUSTOM_BATCH_SIZE else train_set.__len__()
    val_batch_size = CUSTOM_BATCH_SIZE if CUSTOM_BATCH_SIZE else val_set.__len__()
    test_batch_size = CUSTOM_BATCH_SIZE if CUSTOM_BATCH_SIZE else test_set.__len__()
    train_loader = torch.utils.data.DataLoader(train_set,
                                              batch_size=train_batch_size,
                                              shuffle=True,
                                              pin_memory=(torch.cuda.is_available()),
                                              num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set,
                                              batch_size=val_batch_size,
                                              shuffle=False,
                                              pin_memory=(torch.cuda.is_available()),
                                              num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              pin_memory=(torch.cuda.is_available()),
                                              num_workers=num_workers)
    return train_loader, val_loader, test_loader


# In[8]:


Final_Samples = json.load(open('Final_Samples.json', 'r')) 
negative_samples = Final_Samples[370:]
random.seed(7)
random.shuffle(negative_samples)
Final_Samples = Final_Samples[:370] + negative_samples[:370] 
random.seed(0)
random.shuffle(Final_Samples)
# Final_Samples


# In[ ]:


from sklearn.model_selection import KFold
from tqdm import tqdm

Final_Samples = np.array(Final_Samples)


def train_val_test(train_indices, test_indices):
    global accuracies
    global accuracies_val
    global global_best_acc_val
    train_set, val_set, test_set = generate_datasets(train_indices, test_indices, random_seed=None)
    train_loader, val_loader, test_loader = generate_loader(train_set, val_set, test_set, num_workers=0)
    model = simple_model()
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss() 
    
    best_acc_val = [0., None]
    model_best = None
    for epoch_num in range(total_epochs):
        precision, recall, fscore, support, auroc, auprc, acc_train, total_loss, pred, pred_binary, true = epoch(model=model, 
                                                                                 optimizer=optimizer, 
                                                                                 criterion=criterion, is_training=True, 
                                                                                 loader=train_loader)
        precision, recall, fscore, support, auroc, auprc, acc_val, total_loss, pred, pred_binary, true = epoch(model=model, 
                                                                                 optimizer=optimizer, 
                                                                                 criterion=criterion, is_training=False, 
                                                                                 loader=val_loader)
        if acc_val > best_acc_val[0] and True:
            model_best = deepcopy(model)
            best_acc_val[0] = acc_val
            best_acc_val[1] = epoch_num
            if acc_val > global_best_acc_val: global_best_acc_val = acc_val
        print('acc_val:', acc_val, 'best_acc_val:', best_acc_val)
    del model
#     model_best = model_best.to(DEVICE)
    precision, recall, fscore, support, auroc, auprc, acc_test, total_loss, pred, pred_binary, true = epoch(model=model_best, 
                                                                             optimizer=optimizer, 
                                                                             criterion=criterion, is_training=False, 
                                                                             loader=val_loader)
    accuracies += [acc_test]
    accuracies_val += [best_acc_val[0]]
    print(fold_num, ':', accuracies)
    return

    
kf = KFold(n_splits=10)
# kf.get_n_splits(Final_Samples)
print(kf)
global_best_acc_val = 0.
total_epochs = 1
for fold_num, (train_indices, test_indices) in enumerate(kf.split(Final_Samples)):
    accuracies = []
    accuracies_val = []
    train_val_test(train_indices, test_indices)
    print(f'random_seed:{random_seed}:', np.mean(accuracies), np.std(accuracies), 
          np.mean(accuracies_val), np.std(accuracies_val))


# In[ ]:


# from sklearn.model_selection import KFold
# X = Final_Samples
# kf = KFold(n_splits=10)
# print(kf.get_n_splits(X))

# print(kf)
# # print(y_test)
# accuracies = []
# for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     for i in train_index:
#         Final_Samples[i][0]
#     FID, IID, PAT,  MAT, SEX, SNPs


# In[ ]:




