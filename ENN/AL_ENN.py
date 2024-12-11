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


from ENN_model import SentencePairClassifier
from edl_function import *
from ENN_test import test_pred
from ENN_train import *
from ENN_eval import evaluate_loss

import random
import os
import copy
import numpy as np
import matplotlib.pyplot as plt



maxlen = 128
bs = 32  # batch size
iters_to_accumulate = 1
lr = 2e-5  # learning rate
epochs = 5  # number of training epochs
sub_epochs = 2
num_active_sample = 10000
per_num_sample = 10000
sample_model = "models/firstpartkldiv_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_0.8631_ep_2_acc_0.8906.pt"
pth = ""
unc_threshold = 0.5
#model_name= "bert-base-uncased"

wiki_nq = load_dataset("Tevatron/wikipedia-nq")

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

t_short_index = []
for i in range(len(df_val)):
    if len(df_val["documents"][i].split(" ")) <= 5:
        t_short_index.append(i)

#df_train = df_train[:245000]
print(len(t_short_index))

"""marco = load_dataset("Tevatron/msmarco-passage")
t_marco = marco["train"]
m_query = []
m_doc = []
m_label = []

for i in range(len(t_marco)):
  if i % 2 == 0:
    m_query.append(t_marco[i]["query"])
    m_doc.append(t_marco[i]["positive_passages"][0]["text"])
    m_label.append(1)
  else:
    m_query.append(t_marco[i]["query"])
    m_doc.append(t_marco[i]["negative_passages"][0]["text"])
    m_label.append(0)

m_data = pd.DataFrame(columns =["query", "document", "label"])
m_data["query"] = m_query
m_data["document"] = m_doc
m_data["label"] = m_label

print(len(m_data))

df_train = m_data[:360000]
df_val = m_data[360000:380000]
df_test = m_data[380000:]
df_val = df_val.reset_index(drop=True)
print("full_train : ", len(df_train))
#df_test = df_train[245000:]
#df_train = df_train[:240000]
print(df_test)"""

print("num_train : ",len(df_train))
print("num_val : ", len(df_val))
#print("num_test : ", len(df_test))

bce = CalibrationError(n_bins=10, norm="max")

device = "cuda" if torch.cuda.is_available() else "cpu"

new_df_train = df_train[254924:]
df_train = df_train[:254924]
new_df_train = new_df_train.reset_index(drop = True)
train_set = D_CustomDataset(df_train, maxlen = maxlen, with_labels=False)
#active_train_set = D_CustomDataset(new_df_train, maxlen = maxlen, with_labels=False)
train_loader = DataLoader(train_set, batch_size=bs, num_workers=0, shuffle=True, drop_last=True)

#train_subset_loader = DataLoader(active_train_set, batch_size=bs, num_workers=0, shuffle=False, drop_last=True)

val_set = D_CustomDataset(df_val, maxlen = maxlen, with_labels=False)
val_loader = DataLoader(val_set, batch_size=bs, num_workers=0, shuffle=False, drop_last=True)

"""test_set = D_CustomDataset(df_train, maxlen = maxlen, with_labels=False)
test_loader = DataLoader(test_set, batch_size=bs, num_workers=0, shuffle=False, drop_last=True)
"""
model = SentencePairClassifier()

device = "cuda" if torch.cuda.is_available() else "cpu"


train_loss = []
train_accuracy = []
validation_loss = []
validation_accuracy = []

train_col_bel = []
train_wrong_bel = []
train_diss = []
train_vac = []

set_seed(42)
#model.load_state_dict(torch.load("./models/late_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_0.8573_ep_2_acc_0.8641__12.pt"))
model.load_state_dict(torch.load(sample_model))
model.to(device)
opti = AdamW(model.parameters(), lr=lr, eps=1e-8)
#opti = torch.optim.Adam(model.parameters(), lr=lr)
#opti = torch.optim.SGD(model.parameters(), lr=lr, momentum= 0.9, nesterov= True, weight_decay = 1e-6)
#opti = AdamW(model.parameters(), lr = lr)
num_warmup_steps = 0  # The number of steps for the warmup phase.
#num_training_steps = epochs * len(train_loader)  # The total number of training steps
t_total = (len(train_loader) // 1) * epochs  # Necessary to take into account Gradient accumulation

#sub_training_steps = sub_epochs * (num_active_sample + len(df_train))
sub_total = (num_active_sample + len(df_train) // 1) * sub_epochs

sub_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=sub_total)

lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=t_total)

criterion = edl_mse_loss


train_bert(model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, 1)

print(train_loss)
print(validation_loss)

print(train_accuracy)
print(validation_accuracy)

print(train_col_bel)
print(train_wrong_bel)
print(train_vac)
print(train_diss)

plt.subplots(constrained_layout = True)
plt.subplot(3, 1, 1)
plt.ylim(0.0, 1.0)
t1 = np.arange(1.0, epochs + 1, 1)
plt.xticks(t1)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(t1, train_loss, label="Train")
plt.plot(t1, validation_loss, 'r-', label="Validation")
plt.legend()

plt.subplot(3, 1, 2)
plt.ylim(0.0, 1.0)
plt.xticks(t1)
plt.title("ACC Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(t1, train_accuracy, label="Train")
plt.plot(t1, validation_accuracy, 'r-', label="Validation")
plt.legend()

plt.subplot(3, 1, 3)
plt.ylim(0.0, 1.0)
plt.xticks(t1)
plt.title("ACC Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(t1, train_col_bel, label="cor_bel")
plt.plot(t1, train_wrong_bel, 'g-', label="wrong_bel")
plt.plot(t1, train_vac, 'r-', label="vacuity")
plt.plot(t1, train_diss, 'o-', label="dissonance")
plt.legend()
plt.show()
