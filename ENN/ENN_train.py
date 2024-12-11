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
from ENN.ENN_eval import evaluate_loss
from utils import *

import random
import os
import copy
import numpy as np
import matplotlib.pyplot as plt

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
