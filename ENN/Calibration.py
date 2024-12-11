import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from torchmetrics.classification import CalibrationError

from datasets import load_dataset
from tqdm.auto import tqdm
import torch.distributions as dist
from sklearn.metrics import roc_curve

import pandas as pd

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
    AdamW, get_linear_schedule_with_warmup, AutoConfig


from ENN.ENN_model import SentencePairClassifier
from edl_function import *
from ENN.ENN_test import test_pred
from ENN.ENN_train import *
from ENN.ENN_eval import evaluate_loss
from utils import *

import random
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

from torchmetrics.classification import CalibrationError
from sklearn.metrics import brier_score_loss

wiki_nq = load_dataset('Tevatron/wikipedia-nq')


device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(torch.__version__)



set_seed(42)

train = wiki_nq['train'][:58622]  # 90 % of the original training data
val = wiki_nq['dev']  # 10 % of the original training data
# Transform data into pandas dataframes
df__train = pd.DataFrame(train)
df__val = pd.DataFrame(val)

all_train_query = []
all_train_doc = []
all_train_label = []

for i in range(len(df__train)):
    query = str(df__train.loc[i, "query"])
    pos_sent = df__train.loc[i, "positive_passages"]
    neg_sent = df__train.loc[i, "negative_passages"]
    for j in range(len(pos_sent)):
        pos_doc = str(pos_sent[j]['text'])
        all_train_query.append(query)
        all_train_doc.append(pos_doc)
        all_train_label.append(1)
    for j in range(len(neg_sent)):
        neg_doc = str(neg_sent[j]['text'])
        all_train_query.append(query)
        all_train_doc.append(neg_doc)
        all_train_label.append(0)

print(len(all_train_query))

train_query = []
train_doc = []
train_label = []

val_query = []
val_doc = []
val_label = []

##
# all254924, pos107667, neg147257
last_query = []
last_doc = []
last_label = []
for i in range(len(df__train)):
    query = str(df__train.loc[i, "query"])
    pos_sent = df__train.loc[i, "positive_passages"]
    neg_sent = df__train.loc[i, "negative_passages"]
    for j in range(len(pos_sent)):
        if j == 5:
            break
        pos_doc = str(pos_sent[j]['text'])
        if 0 <= j <= 2:
            train_query.append(query)
            train_doc.append(pos_doc)
            train_label.append(1)
        else:
            last_query.append(query)
            last_doc.append(pos_doc)
            last_label.append(1)
    if len(neg_sent) >= len(pos_sent):
        for j in range(len(pos_sent)):
            if j == 4:
                break
            neg_doc = str(neg_sent[j]['text'])
            if 0 <= j <= 1:
                train_query.append(query)
                train_doc.append(neg_doc)
                train_label.append(0)
            else:
                last_query.append(query)
                last_doc.append(neg_doc)
                last_label.append(0)
    else:
        for j in range(len(neg_sent)):
            if j == 4:
                break
            neg_doc = str(neg_sent[j]['text'])
            if 0 <= j <= 1:
                train_query.append(query)
                train_doc.append(neg_doc)
                train_label.append(0)
            else:
                last_query.append(query)
                last_doc.append(neg_doc)
                last_label.append(0)

##

#train_query = train_query[:len(train_doc)]
#train_label = train_label[:len(train_doc)]

for index in range(len(df__val)):
    if index % 2 == 0:
        sent1 = str(df__val.loc[index, "query"])
        sent2 = str(df__val.loc[index, 'positive_passages'][0]['text'])
        lb = 1
    else:
        sent1 = str(df__val.loc[index, "query"])
        sent2 = str(df__val.loc[index, 'negative_passages'][0]['text'])
        lb = 0

    val_query.append(sent1)
    val_doc.append(sent2)
    val_label.append(lb)


df_train = pd.DataFrame(columns = ["query", "documents", "label"])
df_val = pd.DataFrame(columns = ["query", "documents", "label"])

df_train['query'] = train_query + last_query
df_train['documents'] = train_doc + last_doc
df_train['label'] = train_label + last_label
zero_one = df_train["label"].value_counts()
print("Num label 0 : ", zero_one[0])
print("Num label 1 : ", zero_one[1])

df_val['query'] = val_query
df_val['documents'] = val_doc
df_val['label'] = val_label
print(len(df_val))

val_set = D_CustomDataset(df_val, maxlen = 128, with_labels=False)
val_loader = DataLoader(val_set, batch_size=32, num_workers=0, shuffle=False, drop_last=True)

bce = CalibrationError(n_bins=10, norm="max")

li = [ ] # Model Path

ac = []
uac = []
p = []
recall = []
maps = []
ece_list = []
briers = []
eces = []
for index, i in enumerate(li):
    print(i)
    print(type(i))
    path_to_model_enn = "models/"+i
    model = SentencePairClassifier()
    #model.load_state_dict(torch.load(path_to_model_enn), strict=False)
    checkpoint = torch.load(path_to_model_enn)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load(path_to_model_enn))
    model.to(device)
    # test_set = Not_BM25_CustomDataset(df_test.reset_index(drop=True), 128)
    # test_loader = DataLoader(test_set, batch_size=64, num_workers = 0, shuffle = False, drop_last = False)
    # true_label = [1 for i in range(1138)] + [0 for j in range(1139)] # 새로 추가
    path_to_output_file = '/content/drive/MyDrive/output_ENN.txt'

    print("Predicting on test data…")
    probs, uncertainty, pred_E, acc, uncertainty_acc, ece, brier = test_pred(net=model, device=device,
                                                                             dataloader=val_loader,
                                                                             num_samples=len(val_set), with_labels=True,
                                                                             # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                                                                             result_file=path_to_output_file)
    print()
    print("ECE :", ece)
    y_true = df_val['label']
    val_prob = [i[1] for i in probs]  # Assuming probs is a list of tuples
    y_true = np.array(y_true[:6464])
    y_pred = np.array(val_prob[:6464])
    prob_true, prob_pred = calibration_curve(y_true, y_pred, pos_label = 1, n_bins=10)
    print(name[index])
    print(prob_true)
    print(prob_pred)
    error = np.abs(np.array(prob_true) - np.array(prob_pred)).mean()
    print(error)
    eces.append(error)


    plt.title(name[index])
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    plt.plot(prob_true, prob_pred, marker='.')
    plt.show()
print(eces)

