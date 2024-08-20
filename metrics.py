from datasets import load_dataset
import torch
import torch.nn as nn
import pandas as pd
from torchmetrics import AveragePrecision

wiki = load_dataset('Tevatron/wikipedia-nq')

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from keras.utils import pad_sequences
from torch.utils.data import TensorDataset

from datasets import load_dataset
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import gc
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
    AdamW, get_linear_schedule_with_warmup, AutoConfig
from tqdm.auto import tqdm

import math

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(torch.__version__)


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

# train = wiki['train']  # 90 % of the original training data
val = wiki['dev']  # 10 % of the original training data
# Transform data into pandas dataframes
# df__train = pd.DataFrame(train)
df__val = pd.DataFrame(val)

train_query = []
train_doc = []
train_label = []

val_query = []
val_doc = []
val_label = []


class D_CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model="bert-base-uncased"):
        self.data = data  # pandas dataframe
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'query'])
        sent2 = str(self.data.loc[index, 'documents'])

        doc_list = []
        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        query_output = self.tokenizer(sent1, sent2,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects

        label = self.data.loc[index, 'label']

        q_token_ids = query_output['input_ids'].squeeze(0)  # tensor of token ids
        q_attn_masks = query_output['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values
        token_type_ids = query_output['token_type_ids'].squeeze(0)

        return q_token_ids, q_attn_masks, token_type_ids, label


v_q = []
v_d = []
v_l = []
for index in range(len(df__val)):
    if index % 2 == 0:
        sent1 = str(df__val.loc[index, "query"])
        sent2 = str(df__val.loc[index, 'positive_passages'][0]['text'])
        lb = 1
    else:
        sent1 = str(df__val.loc[index, "query"])
        sent2 = str(df__val.loc[index, 'negative_passages'][0]['text'])
        lb = 0

    v_q.append(sent1)
    v_d.append(sent2)
    v_l.append(lb)

df_v = pd.DataFrame(columns=["query", "documents", "label"])

df_v['query'] = v_q
df_v['documents'] = v_d
df_v['label'] = v_l

##
for i in range(len(df__val)):
    query = str(df__val.loc[i, "query"])
    pos_sent = df__val.loc[i, "positive_passages"]
    neg_sent = df__val.loc[i, "negative_passages"]
    for j in range(len(pos_sent)):
        if j == 3:
            break
        pos_doc = str(pos_sent[j]['text'])
        val_query.append(query)
        val_doc.append(pos_doc)
        val_label.append(1)
    if len(neg_sent) >= len(pos_sent):
        for j in range(len(pos_sent)):
            if j == 2:
                break
            neg_doc = str(neg_sent[j]['text'])
            val_query.append(query)
            val_doc.append(neg_doc)
            val_label.append(0)
    else:
        for j in range(len(neg_sent)):
            if j == 2:
                break
            neg_doc = str(neg_sent[j]['text'])
            val_query.append(query)
            val_doc.append(neg_doc)
            val_label.append(0)

"""for index in range(len(df__val)):
    if index % 2 == 0:
        sent1 = str(df__val.loc[index, "query"])
        sent2 = str(df__val.loc[index, 'positive_passages'][1]['text'])
        lb = 1
    else:
        sent1 = str(df__val.loc[index, "query"])
        sent2 = str(df__val.loc[index, 'negative_passages'][1]['text'])
        lb = 0

    val_query.append(sent1)
    val_doc.append(sent2)
    val_label.append(lb)"""

# df_train = pd.DataFrame(columns = ["query", "documents", "label"])
df_val = pd.DataFrame(columns=["query", "documents", "label"])

# df_train['query'] = train_query
# df_train['documents'] = train_doc
# df_train['label'] = train_label

df_val['query'] = val_query
df_val['documents'] = val_doc
df_val['label'] = val_label

t_short_index = []
for i in range(len(df_val)):
    if len(df_val["documents"][i].split(" ")) <= 5:
        t_short_index.append(i)

df_val = df_val[~df_val['documents'].isin(df_v['documents'])].reset_index(drop=True)
print(len(t_short_index))

# print("num_train : ",len(df_train))
print("num_val : ", len(df_val))
val_set = D_CustomDataset(df_val, maxlen=128, with_labels=False)
val_loader = DataLoader(val_set, batch_size=256, num_workers=0, shuffle=False, drop_last=True)


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
        self.dropout = nn.Dropout(p=0.1)
        self.cls_layer = nn.Linear(768, 2)

    def forward(self, q_ids, q_mask, token, p_label):
        output = self.bert_layer(q_ids.to(device), q_mask.to(device), token.to(device))
        pooled_output = output["pooler_output"]
        # dropout_rate = random.choice(self.rate)
        logits = self.cls_layer(self.dropout(pooled_output))

        """logit_list = []
        for dropout in self.dropout_layer:
            logit = self.cls_layer(dropout(pooled_output))
            logit_list.append(logit)

        logits = sum(logit_list) / len(self.rate)"""
        alpha = F.softplus(logits) + 1
        # logits = self.cls_layer(self.dropout(pooled_output))

        return logits, alpha

    def dir_prior_mult_likelihood_loss(self, gt, logits, alpha, current_epoch, p_label):
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
        #ece_loss = expected_calibration_error(p.detach().cpu().numpy(), p_label.detach().cpu().numpy())
        # print(ece_loss[0])
        """if self.args.fix_annealing_rate:
            annealing_rate = 1
            # print(annealing_rate)"""

        """alpha_new = (alpha - 1) * (1 - gt) + 1
        kl_err = self.args.kl_scaling_factor * annealing_rate * self.KL_flat_dirichlet(alpha_new)"""

        # return loss

        dirichlet_strength = torch.sum(alpha, dim=1)
        dirichlet_strength = dirichlet_strength.reshape((-1, 1))

        # Belief
        belief = (alpha - 1) / dirichlet_strength
        # print(ece_loss[0])
        inc_belief = belief * (1 - gt)
        inc_belief_error = (self.kl_scaling_factor) * annealing_rate * torch.mean(inc_belief, dim=1,
                                                                                                keepdim=True)

        loss = first_part_error + inc_belief_error

        return loss

    def get_output_embedding(self, input_ids, attn_mask):
        return self.bert_layer(input_ids, attn_mask)

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

    def calculate_dissonance(self, belief):

        dissonance = torch.zeros(belief.shape[0])
        for i in range(len(belief)):
            dissonance[i] = self.calculate_dissonance_from_belief(belief[i])
        return dissonance

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

    def calculate_dissonance2(self, belief):
        dissonance = torch.zeros(belief.shape[0])
        for i in range(len(belief)):
            dissonance[i] = self.calculate_dissonance_from_belief_vectorized(belief[i:i + 1, :])
            # break
        return dissonance

    def calculate_dissonance3(self, belief):
        # print("belief: ", belief.shape)
        dissonance = self.calculate_dissonance_from_belief_vectorized_again(belief)
        # break
        return dissonance

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
        # evidence = F.softplus(preds)

        # The prior parameters
        # alpha = evidence + 1
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

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def max_pooling(self, model_output, attention_mask):
        token_embeddings = model_output  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        token_embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
        return torch.max(token_embeddings, 1)[0]


from torchmetrics.classification import CalibrationError
from sklearn.metrics import brier_score_loss
bce = CalibrationError(n_bins=10, norm="max")
def test_pred(net, device, dataloader, num_samples, with_labels=True, result_file="results/output.txt"):
    net.eval()
    probs = []
    uncertainties = []
    predss = []
    true_labels = []
    ece_logits = []
    ece_label = []
    ece_loss_list = []
    correct = 0  # 새로 추가
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
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                probs += prob.tolist()
                uncertainties += uncertainty.tolist()
                predss += preds.tolist()
                true_labels += true_label.tolist()
                ece_logits += torch.sigmoid(logits).tolist()
                ece_label += true_label.tolist()
                correct += (true_label == preds).sum().cpu()
        b_out = bce(torch.tensor(ece_logits)[:, 1], torch.tensor(ece_label))
        brier_score = brier_score_loss(torch.tensor(true_labels), torch.tensor(ece_logits)[:, 1])

        # y_true = true_label # 새로 추가
        # correct = sum(1 for a, b in zip(y_true, predss) if a == b) # 새로 추가
        # acc = correct / len(predss) # 새로 추가
        u = np.array([i[0] for i in uncertainties])
        total_samples = len(u)

        for i in range(total_samples):
            if predss[i] == true_labels[i]:
                if u[i] <= 0.5:  # 모델이 정답을 맞추고 불확실성이 낮은 경우
                    unc_acc += 1
            else:
                if u[i] >= 0.5:  # 모델이 틀렸고 불확실성이 높은 경우
                    unc_acc += 1
            # 불확실성 정확도 계산
        unc_accuracy = unc_acc / total_samples
    return probs, uncertainties, predss, correct / num_samples, unc_accuracy, b_out, brier_score

li = [
    "44430000thr_0.5_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_1.6371_ep_2_acc_0.8826.pt",
    "3-8-30000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_2.5964_ep_3_acc_0.8804.pt"
]
"""li = ["3-8-10000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_0.7701_ep_1_acc_0.8454.pt",
      "3-8-20000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_0.7646_ep_1_acc_0.8636.pt",
      "3-8-30000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_0.7623_ep_1_acc_0.8684.pt",
      "3-8-40000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_0.7517_ep_1_acc_0.8511.pt",
      "3-8-50000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_0.7579_ep_1_acc_0.8505.pt",
      "3-8-10000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_1.6474_ep_2_acc_0.8806.pt",
      "3-8-20000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_1.6365_ep_2_acc_0.8836.pt",
      "3-8-30000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_1.6350_ep_2_acc_0.8778.pt",
      "3-8-40000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_1.6181_ep_2_acc_0.8726.pt",
      "3-8-50000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_1.6294_ep_2_acc_0.8792.pt",
      "3-8-10000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_2.6104_ep_3_acc_0.8867.pt",
      "3-8-20000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_2.5982_ep_3_acc_0.8829.pt",
      "3-8-30000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_2.5964_ep_3_acc_0.8804.pt",
      "3-8-40000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_2.5761_ep_3_acc_0.8810.pt",
      "3-8-50000thr_0.8_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_2.5908_ep_3_acc_0.8801.pt"]"""
def recall_at_k(true_labels, predictions, k):
    # Convert to numpy arrays for easy manipulation
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # Get the indices of the top-k predictions
    top_k_indices = np.argsort(predictions)[-k:]

    # Get the true labels of the top-k predictions
    top_k_true_labels = true_labels[top_k_indices]

    # Calculate the number of relevant documents in the top-k predictions
    relevant_in_top_k = np.sum(top_k_true_labels)

    # Calculate the total number of relevant documents
    total_relevant = np.sum(true_labels)

    # Compute Recall@k
    recall_at_k_value = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

    return recall_at_k_value

ac = []
uac = []
p = []
recall = []
maps = []
ece_list = []
briers = []
for i in li:
    print(i)
    print(type(i))
    path_to_model_enn = "models/"+i
    model = SentencePairClassifier()
    #model.load_state_dict(torch.load(path_to_model_enn), strict=False)
    checkpoint = torch.load(path_to_model_enn)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    # test_set = Not_BM25_CustomDataset(df_test.reset_index(drop=True), 128)
    # test_loader = DataLoader(test_set, batch_size=64, num_workers = 0, shuffle = False, drop_last = False)
    # true_label = [1 for i in range(1138)] + [0 for j in range(1139)] # 새로 추가
    path_to_output_file = '/content/drive/MyDrive/output_ENN.txt'

    print("Predicting on test data…")
    probs, uncertainty, pred_E, acc, uncertainty_acc, ece, brier = test_pred(net=model, device=device, dataloader=val_loader,
                                                                 num_samples=len(val_set), with_labels=True,
                                                                 # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                                                                 result_file=path_to_output_file)
    print()
    y_true = df_val['label']
    val_prob = [i[1] for i in probs]  # Assuming probs is a list of tuples
    average_precision = AveragePrecision(task="binary")
    m = average_precision(torch.tensor(val_prob), torch.tensor(y_true[:19712]))
    # acc_list_enn.append(acc)
    print(acc)  # 2277 개 데이터셋 정확도
    print(uncertainty_acc)
    print(m)
    print(ece)
    print(brier)
    ac.append(acc)
    uac.append(uncertainty_acc)
    maps.append(m)
    ece_list.append(ece)
    briers.append(brier)
    recall.append(recall_at_k(y_true, val_prob, len(y_true) // 2))

    Vanila_Enn = []
    for k in range(10):
        y_true = df_val['label'][:19712]

        test_prob = []
        for a in probs:
            test_prob.append(a[1])
        # list1을 기준으로 정렬된 인덱스

        if k != 0:
            # list1과 list2를 정렬된 인덱스를 사용하여 정렬

            indices_to_remove = [a for a, value in enumerate(np.array(uncertainty)) if value >= (k * 0.1)]
            # print(indices_to_remove)
            sorted_preds = [value for a, value in enumerate(test_prob) if a not in indices_to_remove]
            sorted_label = [value for a, value in enumerate(pred_E) if i not in indices_to_remove]
            sorted_target = [value for a, value in enumerate(y_true) if a not in indices_to_remove]

            average_precision = AveragePrecision(task="binary")
            print(len(sorted_preds))
            if len(sorted_preds) < 1:
                result = torch.tensor(0)
            else:
                result = average_precision(torch.tensor(sorted_preds), torch.tensor(sorted_target))
        else:
            average_precision = AveragePrecision(task="binary")
            result = average_precision(torch.tensor(test_prob), torch.tensor(y_true))
        Vanila_Enn.append(result.item())

    print("Acc")
    print(ac)
    print("UAcc")
    print(uac)
    print("ECE")
    print(ece_list)
    print("Brier")
    print(briers)
    print("recall")
    print(recall)
    print("Maps")
    print(maps)
    print("threshold maps")
    print(Vanila_Enn)
