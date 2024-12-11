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


from CustomDataset import D_CustomDataset
from ENN_model import SentencePairClassifier
from edl_function import *
from test import test_pred

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

def subset_train(net, criterion, optim, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, ep):
    global pth
    running_loss = 0.0
    train_correct = 0
    all_loss = 0.0
    count = 0
    sub_batch = 8
    print("Predicting on test data...")

    net.load_state_dict(
        torch.load(sample_model))

    net.to(device)

    all_samples = []

    # Extract samples from your dataset
    for idx, query in enumerate(train['query']):
        for positive_passage in train['positive_passages'][idx]:
            all_samples.append((query, positive_passage['text'], 1))  # Positive passages are labeled 1

        for negative_passage in train['negative_passages'][idx]:
            all_samples.append((query, negative_passage['text'], 0))  # Negative passages are labeled 0

    # Randomly sample from the dataset
    random_samples = all_samples
    # random_samples = random.sample(all_samples, 5000000)  # Specify the desired number of samples
    print("random_samples:", len(random_samples))
    active_df_train = pd.DataFrame(random_samples, columns=['query', 'documents', 'label'])

    # Remove samples already in your training set (if necessary)
    active_df_train = active_df_train[~active_df_train['documents'].isin(df_train['documents'])]
    active_df_train = active_df_train.sample(frac=1)
    active_df_train = active_df_train.reset_index(drop=True)

    # Define the number of samples to be accumulated each iteration
    num_samples_per_iteration = num_active_sample // 10
    # Initialize an empty list to store the accumulated samples
    accumulated_samples = []
    dataframes = []
    pos = 0
    neg = 0
    # Iterate over the dataset, accumulating samples
    for start_index in range(0, len(active_df_train), num_samples_per_iteration):
        subset_df = active_df_train[start_index:start_index + num_samples_per_iteration]
        subset_df = subset_df.reset_index(drop=True)

        active_train_set = D_CustomDataset(subset_df, maxlen=maxlen, with_labels=False)
        train_subset_loader = DataLoader(active_train_set, batch_size=bs, num_workers=0, shuffle=False,
                                         drop_last=False)
        print("active_train_set:", len(active_train_set))
        probs, uncertainty, pred_E, acc, true_labels = test_pred(net=net, device=device,
                                                              dataloader=train_subset_loader,
                                                              num_samples=len(active_train_set),
                                                              with_labels=True)

        pos_samples = subset_df[subset_df['label'] == 1]
        neg_samples = subset_df[subset_df['label'] == 0]

        pos_indices = pos_samples.index.tolist()
        neg_indices = neg_samples.index.tolist()

        selected_pos_indices = [i for i in pos_indices if uncertainty[i][0] >= unc_threshold]
        selected_neg_indices = [i for i in neg_indices if uncertainty[i][0] >= unc_threshold]

        if ep < 5:
            if len(selected_pos_indices) < (per_num_sample * (ep + 1)) / 2:
                pos_indices_to_add = selected_pos_indices
                pos += len(pos_indices_to_add)
            else:
                pos_indices_to_add = random.sample(selected_pos_indices, (per_num_sample * (ep + 1)) // 2)
                pos += len(pos_indices_to_add)

            if len(selected_neg_indices) < (per_num_sample * (ep + 1)) / 2:
                neg_indices_to_add = selected_neg_indices
                neg += len(neg_indices_to_add)
            else:
                neg_indices_to_add = random.sample(selected_neg_indices, (per_num_sample * (ep + 1)) // 2)
                neg += len(neg_indices_to_add)

            accumulated_samples.extend(pos_indices_to_add + neg_indices_to_add)

            if neg >= (num_active_sample / 2):
                sub_df_train = subset_df.loc[pos_indices_to_add].copy()
            else:
                sub_df_train = subset_df.loc[pos_indices_to_add + neg_indices_to_add].copy()
            sub_df_train = sub_df_train.reset_index(drop = True)
            dataframes.append(sub_df_train)

        elif 5 <= ep < 9:
            if len(selected_pos_indices) < (per_num_sample * (ep - 4)) / 2:
                pos_indices_to_add = selected_pos_indices
            else:
                pos_indices_to_add = random.sample(selected_pos_indices, (per_num_sample * (ep - 4)) // 2)

            if len(selected_neg_indices) < (per_num_sample * (ep - 4)) / 2:
                neg_indices_to_add = selected_neg_indices
            else:
                neg_indices_to_add = random.sample(selected_neg_indices, (per_num_sample * (ep - 4)) // 2)

            accumulated_samples.extend(pos_indices_to_add + neg_indices_to_add)

            sub_df_train = subset_df.loc[pos_indices_to_add + neg_indices_to_add].copy()
            sub_df_train = sub_df_train.reset_index(drop=True)
            dataframes.append(sub_df_train)
        concat_dataframes = pd.concat(dataframes, ignore_index=True)
        if len(concat_dataframes) >= num_active_sample:
            #concat_dataframes = concat_dataframes[:num_active_sample]
            print("Completed accumulated_samples : ", len(concat_dataframes))
            break
        else:
            print("accumulated_samples : ", len(concat_dataframes))
    concat_dataframes = pd.concat(dataframes, ignore_index= True)
    concat_dataframes = concat_dataframes[:num_active_sample]
    active_set = pd.concat([df_train, concat_dataframes])
    active_set = active_set.reset_index(drop = True)

    zero_one = concat_dataframes["label"].value_counts()
    print("Num label 0 : ", zero_one[0])
    print("Num label 1 : ", zero_one[1])

    new_train_set = D_CustomDataset(active_set , maxlen=maxlen, with_labels=False)

    print(len(active_set))
    print("Active data is completed...")

    if len(new_train_set):
        selected_dataloader = DataLoader(new_train_set, batch_size=bs, num_workers=0, shuffle=True, drop_last=False)
        best_acc = -np.Inf
        best_ep = 0
        nb_iterations = len(selected_dataloader)
        print_every = nb_iterations // 25  # print the training loss 5 times per epoch
        print("Starting Subset training...")
        # 26/7/10/9

        net = SentencePairClassifier()
        net.to(device)
        net.train()
        opti = AdamW(net.parameters(), lr=lr, eps=1e-8)

        sub_total = (len(selected_dataloader) // 1) * sub_epochs

        sub_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                                        num_training_steps=sub_total)

        for sep in range(sub_epochs):
            print(sep + 1,"epochs")
            for it, (p_ids, p_attn, p_token, p_label) in enumerate(tqdm(selected_dataloader)):

                # q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)

                p_one_hot = one_hot_embedding(p_label, 2)
                p_one_hot = p_one_hot.to(device)
                p_label = p_label.to(device)

                # Obtaining the logits from the model
                logits, alpha = net(p_ids, p_attn, p_token, p_label)
                p = torch.sigmoid(logits)
                #loss = expected_calibration_error(p.detach().cpu().numpy(), p_label.detach().cpu().numpy())
                loss = net.dir_prior_mult_likelihood_loss(p_one_hot, logits, alpha, sep, p_label, sub = True)
                loss = torch.mean(loss)
                #loss += criterion(logits.squeeze(-1), p_one_hot.float(), sep, 2, 10, device)
                # p_predicted = (alpha.view(-1) >= 0.5).int()
                _, p_predicted = torch.max(alpha, 1)
                vac, wrong_bel, cor_bel, diss = net.calc_loss_vac_bel(alpha, p_one_hot)

                """avg_cor_bel += torch.mean(cor_bel)
                avg_wrong_bel += torch.mean(wrong_bel)
                avg_diss += torch.mean(diss)
                avg_vac += torch.mean(vac)"""
                # print(loss)

                # p_label = p_label.to(device)
                train_correct += (p_label == p_predicted).sum().cpu()
                # loss = criterion(logits.squeeze(-1), one_hot.float(), ep, 3, 10, device)

                loss.backward()

                opti.step()
                # Updates the scale for next iteration.
                # Adjust the learning rate based on the number of iterations.
                sub_scheduler.step()
                # Clear gradients
                opti.zero_grad()

                running_loss += loss.item()
                count += 1
                all_loss += loss.item()

                if (it + 1) % print_every == 0:  # Print training loss information
                    print()
                    print("Iteration {}/{} of sub_epoch {} complete. Loss : {}"
                          .format(it + 1, nb_iterations, sep + 1, running_loss / print_every))

                    running_loss = 0.0
            if sep == 0:
                val_loss, val_acc = evaluate_loss(net, val_loader, ep, criterion)  # Compute validation loss
                val_acc = val_acc / (len(df_val))
                if ep < 5:
                    path_to_model = 'models/mse{}thr_{}_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
                        (ep + 1) * per_num_sample, unc_threshold, lr,
                        train_correct / (len(new_train_set)),
                        (sep + 1),
                        val_acc)
                elif 5 <= ep < 10:
                    path_to_model = 'models/{}Div_AllRandom_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
                        (ep - 4) * per_num_sample, lr,
                        train_correct / (len(new_train_set)),
                        (sep + 1),
                        val_acc)

            else:
                val_loss, val_acc = evaluate_loss(net, val_loader, ep, criterion)  # Compute validation loss
                val_acc = val_acc / (len(df_val))
                if ep < 5:
                    path_to_model = 'models/mse{}thr_{}_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
                        (ep + 1) * per_num_sample, unc_threshold, lr,
                        train_correct / (len(new_train_set)),
                        (sep + 1),
                        val_acc)
                elif 5 <= ep < 10:
                    path_to_model = 'models/{}Div_AllRandom_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
                        (ep - 4) * per_num_sample, lr,
                        train_correct / (len(new_train_set)),
                        (sep + 1),
                        val_acc)
            pth = path_to_model
            #torch.save(net.state_dict(), path_to_model)
            torch.save({
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': opti.state_dict(),
                'scheduler_state_dict': sub_scheduler.state_dict()
            }, pth)
            print("The model has been saved in {}".format(path_to_model))
            """best_ep = ep + 1
            print()
            path_to_model = 'models/SSS_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}.pt'.format(
                lr,
                train_correct / (len(df_train)),
                best_ep)
            torch.save(net.state_dict(), path_to_model)"""
def evaluate_loss(net, dataloader, epoch, criterion):
    net.eval()

    mean_loss = 0
    count = 0
    val_correct = 0

    with torch.no_grad():
        for it, (p_ids, p_attn, p_token, p_label) in enumerate(tqdm(dataloader)):
            # q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)

            # Obtaining the logits from the model
            p_one_hot = one_hot_embedding(p_label, 2)
            p_one_hot = p_one_hot.to(device)
            p_label = p_label.to(device)
            logits, alpha = net(p_ids, p_attn, p_token, p_one_hot)

            loss = net.dir_prior_mult_likelihood_loss(p_one_hot, logits, alpha, epoch, p_label)
            mean_loss += torch.mean(loss)
            #vac, wrong_bel, cor_bel, diss = net.calc_loss_vac_bel(alpha, p_one_hot)

            #mean_loss += criterion(logits.squeeze(-1), p_one_hot.float(), epoch, 2, 10, device)
            #mean_loss = torch.mean(loss)

            #p_predicted = (alpha.view(-1) >= 0.5).int()
            _, p_predicted = torch.max(alpha, 1)



            val_correct += (p_label == p_predicted).sum().cpu()
            count += 1

    return mean_loss / count, val_correct


def train_bert(net, criterion, optim, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):
    global num_active_sample
    global pth
    how_iter = 0
    best_acc = -np.Inf
    best_ep = 0
    nb_iterations = len(train_loader)
    print_every = nb_iterations // 25  # print the training loss 5 times per epoch
    for ep in range(epochs):
        net.train()
        running_loss = 0.0
        train_correct = 0
        all_loss = 0.0
        count = 0

        """if ep <= 4:
            for p in net.parameters():
                p.requires_grad = False

            for p in net.cls_layer.parameters():
                p.requires_grad = True
        else:
            for p in net.parameters():
                p.requires_grad = True"""

        """avg_cor_bel = 0
        avg_wrong_bel = 0
        avg_diss = 0
        avg_vac = 0
        print("Starting Main Training")
        for it, (p_ids, p_attn, p_token, p_label) in enumerate(tqdm(train_loader)):
            # q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)

            p_one_hot = one_hot_embedding(p_label, 2)
            p_one_hot = p_one_hot.to(device)
            p_label = p_label.to(device)

            # Obtaining the logits from the model
            logits, alpha = net(p_ids, p_attn, p_token, p_label)
            loss = net.dir_prior_mult_likelihood_loss(p_one_hot, logits, alpha, ep, p_label, sub = False)
            loss = torch.mean(loss)
            loss += criterion(logits.squeeze(-1), p_one_hot.float(), ep, 2, 10, device)
            #p_predicted = (alpha.view(-1) >= 0.5).int()
            _, p_predicted = torch.max(alpha, 1)
            vac, wrong_bel, cor_bel, diss = net.calc_loss_vac_bel(alpha, p_one_hot)

            avg_cor_bel += torch.mean(cor_bel)
            avg_wrong_bel += torch.mean(wrong_bel)
            avg_diss += torch.mean(diss)
            avg_vac += torch.mean(vac)
            # print(loss)

            #p_label = p_label.to(device)
            train_correct += (p_label == p_predicted).sum().cpu()
            # loss = criterion(logits.squeeze(-1), one_hot.float(), ep, 3, 10, device)

            loss.backward()

            optim.step()
            # Updates the scale for next iteration.
            # Adjust the learning rate based on the number of iterations.
            lr_scheduler.step()
            # Clear gradients
            optim.zero_grad()

            running_loss += loss.item()
            count += 1
            all_loss += loss.item()

            if (it + 1) % print_every == 0:  # Print training loss information
                print()
                print("Iteration {}/{} of epoch {} complete. Loss : {}"
                      .format(it + 1, nb_iterations, ep + 1, running_loss / print_every))

                running_loss = 0.0"""
        subset_train(net, criterion, optim, lr, sub_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, ep)
        val_loss, val_acc = evaluate_loss(net, val_loader, ep, criterion)  # Compute validation loss
        #train_loss.append(all_loss / count)
        validation_loss.append(val_loss.item())
        print()
        val_acc = val_acc / (len(df_val))
        train_accuracy.append(train_correct / len(df_train) + num_active_sample)
        validation_accuracy.append(val_acc)

        """train_col_bel.append((avg_cor_bel / count).detach().cpu().numpy())
        train_wrong_bel.append((avg_wrong_bel / count).detach().cpu().numpy())
        train_diss.append((avg_diss / count).detach().cpu().numpy())
        train_vac.append((avg_vac / count).detach().cpu().numpy())
        print("Epoch {} complete! Train ACC : {} Validation Loss : {} ACC : {}".format(ep + 1,
                                                                                       train_correct / (len(df_train)),
                                                                                       val_loss,
                                                                                       val_acc))
"""

        #if val_acc > best_acc:
        best_ep += 1
        print("Best validation acc improved from {} to {}".format(best_acc, val_acc))
        best_acc = val_acc
        print()
        """if ep < 10:
            path_to_model = 'models/{} Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
                20000 + ep * 1000, lr,
                train_correct / (len(df_train)),
                2,
                val_acc)
        elif 10 <= ep < 20:
            path_to_model = 'models/{} Div_AllRandom_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
                20000 + (ep - 10) * 1000, lr,
                train_correct / (len(df_train)),
                2,
                val_acc)
        else:
            path_to_model = 'models/{} Div_highUncRandom_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
                20000 + (ep - 20) * 1000, lr,
                train_correct / (len(df_train)),
                2,
                val_acc)""""""
        if best_ep >= 3:
            best_ep = 0
        torch.save(net.state_dict(), path_to_model)
        print("The model has been saved in {}".format(path_to_model))"""
        # Saving the model
        save_weights(model, "model_weight.txt", ep)
        num_active_sample += per_num_sample
    #del loss
    torch.cuda.empty_cache()

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
