# Import libraries
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

# Set the device to GPU if available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Set random seeds
def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

# Calculate mAP@10 for evaluation
def calculate_map10(test_prob, y_true, y_index):
    """MAP@10 calculation."""
    average_precision_10 = RetrievalMAP(top_k=10)
    map_output = average_precision_10(torch.tensor(test_prob), torch.tensor(y_true), indexes=torch.tensor(y_index))
    return map_output

# Perform predictions during testing and comput related metrics (Probability, Uncertainty, mAP@10)
def test_pred(net, device, dataloader, num_samples, with_labels=True, result_file="results/output.txt"):
    net.eval()
    probs = []
    uncertainties = []
    predss = []
    true_labels = []
    ece_logits = []
    ece_label = []
    ece_loss_list = []
    correct = 0
    unc_acc = 0
    with torch.no_grad():
        if with_labels:
            for q_ids, q_mask, q_token, label in tqdm(dataloader):
                q_ids, q_mask, q_token, true_label = q_ids.to(device), q_mask.to(device), q_token.to(device), label.to(
                    device)
                logits, alpha = net(q_ids, q_mask, q_token, true_label)
                # logits = logits[0]
                # alpha = F.relu(logits) + 1
                uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
                _, preds = torch.max(alpha, 1)
                #prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                prob = torch.sigmoid(logits)
                probs += prob.tolist()
                uncertainties += uncertainty.tolist()
                predss += preds.tolist()
                true_labels += true_label.tolist()
                ece_logits += torch.sigmoid(logits).tolist()
                ece_label += true_label.tolist()
                correct += (true_label == preds).sum().cpu()

        # y_true = true_label
        # correct = sum(1 for a, b in zip(y_true, predss) if a == b)
        # acc = correct / len(predss)
        u = np.array([i[0] for i in uncertainties])
        total_samples = len(u)

        for i in range(total_samples):
            if predss[i] == true_labels[i]:
                if u[i] <= 0.5:
                    unc_acc += 1
            else:
                if u[i] >= 0.5:
                    unc_acc += 1

        unc_accuracy = unc_acc / total_samples
    return probs, uncertainties, predss, correct / num_samples, unc_accuracy

# Definition of a initial model
class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model="bert-base-uncased", freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)
        self.fix_annealing_rate = False
        self.kl_scaling_factor = 1.0


        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False

        self.act = nn.GELU()
        self.dropout = nn.Dropout(p = 0.1)
        self.cls_layer = nn.Linear(768, 2)

    def forward(self, q_ids, q_mask, token, p_label):
        output = self.bert_layer(q_ids.to(device), q_mask.to(device), token.to(device))
        pooled_output = output["pooler_output"]
        #dropout_rate = random.choice(self.rate)
        logits = self.cls_layer(self.dropout(pooled_output))

        """logit_list = []
        for dropout in self.dropout_layer:
            logit = self.cls_layer(dropout(pooled_output))
            logit_list.append(logit)

        logits = sum(logit_list) / len(self.rate)"""
        alpha = F.softplus(logits) + 1
        #logits = self.cls_layer(self.dropout(pooled_output))

        return logits, alpha

    # KL-Divergence
    def KL_flat_dirichlet(self, alpha):
        """
        Calculate Kl divergence between a flat/uniform dirichlet distribution and a passed dirichlet distribution
        i.e. KL(dist1||dist2)
        distribution is a flat dirichlet distribution
        :param alpha: The parameters of dist2 (2nd distribution)
        :return: KL divergence
        """
        num_classes = alpha.shape[1]
        beta = torch.ones(alpha.shape, dtype=torch.float32, device=device)

        dist1 = dist.Dirichlet(beta)
        dist2 = dist.Dirichlet(alpha)

        kl = dist.kl_divergence(dist1, dist2).reshape(-1, 1)
        return kl

    # Loss function
    def dir_prior_mult_likelihood_loss(self, gt, logits, alpha, current_epoch, p_label, sub = False):
        """
        Calculate the loss based on the dirichlet prior and multinomial likelihoood
        :param gt: The ground truth (one hot vector)
        :param alpha: The prior parameters
        :param current_epoch: For the regularization parameter
        :return: loss
        """
        gt = gt.to(device)
        logits = logits.to(device)
        p = torch.sigmoid(logits)
        alpha = alpha.to(device)

        S = torch.sum(alpha, dim=1, keepdim=True)

        first_part_error = torch.sum(gt * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
        annealing_rate = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(current_epoch / 10, dtype=torch.float32)
        )

        probs = alpha / torch.sum(alpha, dim=1, keepdim=True)
        """
        d_ece_loss = expected_calibration_error(p.detach().cpu().numpy(), p_label.detach().cpu().numpy())
        print(d_ece_loss)
        ece_loss = batch_expected_calibration_error(p.detach().cpu().numpy(), p_label.detach().cpu().numpy())
        print(ece_loss)
        print(ece_loss.shape)"""
        """if self.args.fix_annealing_rate:
            annealing_rate = 1
            # print(annealing_rate)"""

        """alpha_new = (alpha - 1) * (1 - gt) + 1
        kl_err = self.args.kl_scaling_factor * annealing_rate * self.KL_flat_dirichlet(alpha_new)"""
        alpha_new = (alpha - 1) * (1 - gt) + 1
        kl_err = annealing_rate * self.KL_flat_dirichlet(alpha_new)
        # return loss

        dirichlet_strength = torch.sum(alpha, dim=1)
        dirichlet_strength = dirichlet_strength.reshape((-1, 1))

        #uncertainty = 2 / torch.sum(alpha, dim = 1, keepdim=True)
        #uncertainty = 1 / uncertainty
        # Belief
        belief = (alpha - 1) / dirichlet_strength
        #print(ece_loss[0])
        inc_belief = belief * (1 - gt)
        #print(inc_belief)
        inc_belief_error = annealing_rate * torch.mean(inc_belief, dim=1, keepdim=True)
        loss = first_part_error + kl_err
        #loss= inc_belief_error

        return loss

    # Not Used
    def get_output_embedding(self, input_ids, attn_mask):
        return self.bert_layer(input_ids, attn_mask)
        
    # Not Used
    def calculate_dissonance_from_belief(self, belief):
        num_classes = len(belief)
        Bal_mat = torch.zeros((num_classes, num_classes)).to(device)
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                if belief[i] == 0 or belief[j] == 0:
                    Bal_mat[i, j] = 0
                else:
                    Bal_mat[i, j] = 1 - torch.abs(belief[i] - belief[j]) / (belief[i] + belief[j])
                Bal_mat[j, i] = Bal_mat[i, j]
        sum_belief = torch.sum(belief).to(device)
        dissonance = 0
        for i in range(num_classes):
            if torch.sum(belief * Bal_mat[i, :]) == 0: continue
            dissonance += belief[i] * torch.sum(belief * Bal_mat[i, :]) / (sum_belief - belief[i])
        return dissonance

    # Not Used
    def calculate_dissonance(self, belief):

        dissonance = torch.zeros(belief.shape[0])
        for i in range(len(belief)):
            dissonance[i] = self.calculate_dissonance_from_belief(belief[i])
        return dissonance

    # Not Used
    def calculate_dissonance_from_belief_vectorized(self, belief):
        # print("belief shape: ", belief.shape)
        sum_bel_mat = torch.transpose(belief, -2, -1) + belief
        sum_bel_mat[sum_bel_mat == 0] = -500
        # print("sum: ", sum_bel_mat)
        diff_bel_mat = torch.abs(torch.transpose(belief, -2, -1) - belief)
        # print("diff bel mat: ", diff_bel_mat)
        div_diff_sum = torch.div(diff_bel_mat, sum_bel_mat)
        # print("div diff sum up: ", div_diff_sum)

        div_diff_sum[div_diff_sum < -0] = 1
        # print("div diff sum: ", div_diff_sum)
        Bal_mat = 1 - div_diff_sum
        # print("Bal Mat vec: ", Bal_mat)
        # import sys
        # sys.exit()
        num_classes = belief.shape[1]
        Bal_mat[torch.eye(num_classes).byte().bool()] = 0  # torch.zeros((num_classes, num_classes))
        # print("BAL mat: ", Bal_mat)

        sum_belief = torch.sum(belief)

        bel_bal_prod = belief * Bal_mat
        # print("Prod: ", bel_bal_prod)
        sum_bel_bal_prod = torch.sum(bel_bal_prod, dim=1, keepdim=True)
        divisor_belief = sum_belief - belief
        scale_belief = belief / divisor_belief
        scale_belief[divisor_belief == 0] = 1
        each_dis = torch.matmul(scale_belief, sum_bel_bal_prod)

        return torch.squeeze(each_dis)

    # Not Used
    def calculate_dissonance_from_belief_vectorized_again(self, belief):
        belief = torch.unsqueeze(belief, dim=1)

        sum_bel_mat = torch.transpose(belief, -2, -1) + belief  # a + b for all a,b in the belief
        diff_bel_mat = torch.abs(torch.transpose(belief, -2, -1) - belief)

        div_diff_sum = torch.div(diff_bel_mat, sum_bel_mat)  # |a-b|/(a+b)

        Bal_mat = 1 - div_diff_sum
        zero_matrix = torch.zeros(sum_bel_mat.shape, dtype=sum_bel_mat.dtype).to(sum_bel_mat.device)
        Bal_mat[sum_bel_mat == zero_matrix] = 0  # remove cases where a=b=0

        diagonal_matrix = torch.ones(Bal_mat.shape[1], Bal_mat.shape[2]).to(sum_bel_mat.device)
        diagonal_matrix.fill_diagonal_(0)  # remove j != k
        Bal_mat = Bal_mat * diagonal_matrix  # The balance matrix

        belief = torch.einsum('bij->bj', belief)
        sum_bel_bal_prod = torch.einsum('bi,bij->bj', belief, Bal_mat)
        sum_belief = torch.sum(belief, dim=1, keepdim=True)
        divisor_belief = sum_belief - belief
        scale_belief = belief / divisor_belief
        scale_belief[divisor_belief == 0] = 1

        each_dis = torch.einsum('bi,bi->b', scale_belief, sum_bel_bal_prod)

        return each_dis

    # Not Used
    def calculate_dissonance2(self, belief):
        dissonance = torch.zeros(belief.shape[0])
        for i in range(len(belief)):
            dissonance[i] = self.calculate_dissonance_from_belief_vectorized(belief[i:i + 1, :])
            # break
        return dissonance

    # Not Used
    def calculate_dissonance3(self, belief):
        # print("belief: ", belief.shape)
        dissonance = self.calculate_dissonance_from_belief_vectorized_again(belief)
        # break
        return dissonance

    # Not Used
    def calc_loss_vac_bel(self, preds, y, query_set=False):
        """
        Calculate the loss, evidence, vacuity, correct belief, and wrong belief
        Prediction is done on the basis of evidence
        :param preds: the NN predictions
        :param y: the groud truth labels
        :param query_set: whether the query set or support set of ask
        :return: loss, vacuity, wrong_belief_vector and cor_belief_vector
        """

        # Make evidence non negative (use softplus)
        #evidence = F.softplus(preds)

        # The prior parameters
        #alpha = evidence + 1
        evidence = preds - 1

        dirichlet_strength = torch.sum(preds, dim=1)
        dirichlet_strength = dirichlet_strength.reshape((-1, 1))

        # Belief
        belief = evidence / dirichlet_strength

        # Total belief
        sum_belief = torch.sum(belief, dim=1)

        # Vacuity
        vacuity = 1 - sum_belief

        # Dissonance
        dissonance = self.calculate_dissonance3(belief)

        # one hot vector for ground truth
        """gt = torch.eye(len(y))[y].to(self.device)
        gt = gt[:, :self.args.num_classes_per_set]"""

        wrong_belief_matrix = belief * (1 - y)

        wrong_belief_vector = torch.sum(wrong_belief_matrix, dim=1)
        cor_belief_vector = torch.sum(belief * y, dim=1)

        return vacuity, wrong_belief_vector, cor_belief_vector, dissonance

    # Not Used
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    # Not Used
    def max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a sentence pair classifier.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset.")
    return parser.parse_args()

def main():

    args = parse_args()
    model_path = args.model_path
    data_path = args.data_path
    
    print(f"Model Path: {model_path}")
    print(f"Data Path: {data_path}")


    empy_model = SentencePairClassifier()
    model = load_model(empy_model, model_path)
    model.to(device)
    data = load_sample_data(data_path)

    df_val = pd.DataFrame(data)

    val_set = D_CustomDataset(df_val, maxlen=128, with_labels=False)
    val_loader = DataLoader(val_set, batch_size=256, num_workers=0, shuffle=False, drop_last=True)

    probs, uncertainty, pred_E, acc, uncertainty_acc = test_pred(net=model, device=device,
                                                                        dataloader=val_loader,
                                                                        num_samples=len(val_set), with_labels=True,
                                                                        # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                                                                        result_file="")
    y_true = df_val['label'][:257000]
    y_index = df_val['indexes'][:257000]
    test_prob = [p[1] for p in probs]
    test_prob = test_prob[:257000]

    map10 = calculate_map10(test_prob, y_true, y_index)

    print(f"MAP@10: {map10}")
    print(f"Uncertainty: {uncertainty}")
    print(f"Probabilities: {probs}")

if __name__ == "__main__":
    main()
