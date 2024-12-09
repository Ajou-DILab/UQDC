from datasets import load_dataset
from tqdm.auto import tqdm
import torch.distributions as dist
from sklearn.metrics import roc_curve

maxlen = 128
bs = 16  # batch size
iters_to_accumulate = 1
lr = 2e-5  # learning rate
epochs = 5  # number of training epochs
sub_epochs = 2
num_active_sample = 10000
per_num_sample = 10000
sample_model = "models/CE10000thr_0.0_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_2e-05_train_acc_1.6377_ep_2_acc_0.8681.pt"
pth = ""
unc_threshold = 0.7
#model_name= "bert-base-uncased"

from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer
import nltk
from nltk import sent_tokenize
from lfqa_utils import *
#nltk.download("punkt")
import pandas as pd
wiki_nq = load_dataset("Tevatron/wikipedia-nq")

train = wiki_nq['train'][:58622]  # 90 % of the original training data
val = wiki_nq['dev']  # 10 % of the original training data
# Transform data into pandas dataframes
df__train = pd.DataFrame(train)
df__val = pd.DataFrame(val)

import torch
import torch.nn as nn
import torch.nn.functional as F


def expected_calibration_error(samples, true_labels, M=10):
    # uniform binning approach with M number of bins
    bin_boundaries = np.linspace(0, 1, M + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

   # keep confidences / predicted "probabilities" as they are
    confidences = np.max(samples, axis=1)
    # get predictions from confidences (positional in this case)
    predicted_label = np.argmax(samples, axis=1).astype(float)

    # get a boolean list of correct/false predictions
    accuracies = predicted_label==true_labels

    ece = np.zeros(1)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # determine if sample is in bin m (between bin lower & upper)
        in_bin = np.logical_and(confidences > bin_lower.item(), confidences <= bin_upper.item())
        # can calculate the empirical probability of a sample falling into bin m: (|Bm|/n)
        prop_in_bin = in_bin.astype(float).mean()

        if prop_in_bin.item() > 0:
            # get the accuracy of bin m: acc(Bm)
            accuracy_in_bin = accuracies[in_bin].astype(float).mean()
            # get the average confidence of bin m: conf(Bm)
            avg_confidence_in_bin = confidences[in_bin].mean()
            # calculate |acc(Bm) - conf(Bm)| * (|Bm|/n) for bin m and add to the total ECE
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

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

# qnli
"""wiki_nq = load_dataset("glue", "qnli")

import pandas as pd
train = wiki_nq['train']  # 90 % of the original training data
val = wiki_nq['validation']  # 10 % of the original training data
test =wiki_nq["test"]
# Transform data into pandas dataframes
df_train = pd.DataFrame(train)
df_val = pd.DataFrame(val)
df_test = pd.DataFrame(test)

df_train = df_train[:10000]
df_val = df_val[:1000]
"""
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


from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


from transformers import AutoTokenizer, AutoModel
import torch
from edl_function import *
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
    AdamW, get_linear_schedule_with_warmup, AutoConfig
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import torch.nn as nn
from torchmetrics.classification import CalibrationError
bce = CalibrationError(n_bins=10, norm="max")

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        loss = first_part_error
        #loss= inc_belief_error

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


def make_qa_retriever_model(model_name="google/bert_uncased_L-8_H-512_A-8", from_file=None, device="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name).to(device)
    # run bert_model on a dummy batch to get output dimension
    d_ids = torch.LongTensor(
        [[bert_model.config.bos_token_id if bert_model.config.bos_token_id is not None else 1]]
    ).to(device)
    d_mask = torch.LongTensor([[1]]).to(device)
    sent_dim = bert_model(d_ids, attention_mask=d_mask)[1].shape[-1]
    qa_embedder = RetrievalQAEmbedder(bert_model, sent_dim).to(device)
    if from_file is not None:
        param_dict = torch.load(from_file)  # has model weights, optimizer, and scheduler states
        qa_embedder.load_state_dict(param_dict["model"])
    return tokenizer, qa_embedder

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

import random
import os
import copy
import numpy as np

train_loss = []
train_accuracy = []
validation_loss = []
validation_accuracy = []

train_col_bel = []
train_wrong_bel = []
train_diss = []
train_vac = []

def test_pred(net, device, dataloader, num_samples, with_labels=True):
    net.eval()
    probs = []
    uncertainties = []
    predss = []
    ece_loss_list = []
    true_labels = []
    correct = 0 # 새로 추가
    with torch.no_grad():
        if with_labels:
            for q_ids, q_mask, q_token, label in tqdm(dataloader):
                q_ids, q_mask, q_token, true_label = q_ids.to(device), q_mask.to(device), q_token.to(device), label.to(device)
                logits, alpha = net(q_ids, q_mask, q_token, true_label)

                _, preds = torch.max(alpha, 1)


                predss += preds.tolist()
                correct += (true_label == preds).sum().cpu()
                p = torch.sigmoid(logits)
                probs += p.tolist()
                #print(b_out)
                true_labels += true_label.tolist()

        #y_true = true_label 
        #correct = sum(1 for a, b in zip(y_true, predss) if a == b) 
        #acc = correct / len(predss) 
    return predss, correct / num_samples, probs


def one_hot_embedding(labels, num_classes=2):
    y = torch.eye(num_classes)
    return y[labels]


def set_seed(seed):
    """ Set all seeds to make results reproducible """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def save_weights(model, file_path, epoch):
    
    with open(file_path, 'a') as file:
        
        file.write(str(epoch) + " epoch\n")
        file.write(str(model.cls_layer.weight.data))
        file.write('\n')  

def get_paraphrased_sentences(model, tokenizer, sentence, num_return_sequences=5, num_beams=5):
  # tokenize the text to be form of a list of token IDs
  inputs = tokenizer(sentence, truncation=True, padding="max_length", return_tensors="pt")
  inputs = inputs.to(device)
  # generate the paraphrased sentences
  outputs = model.generate(
    **inputs,
    num_beams=num_beams,
    num_return_sequences=num_return_sequences,
  )
  # decode the generated sentences using the tokenizer to get them back to text
  return tokenizer.batch_decode(outputs, skip_special_tokens=True)
def subset_train(net, criterion, optim, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate, ep):
    global pth
    running_loss = 0.0
    train_correct = 0
    all_loss = 0.0
    count = 0
    sub_batch = 8
    print("Predicting on test data...")
    chpt = torch.load(sample_model)
    net.load_state_dict(chpt["model_state_dict"])
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
        pred_CE, acc, probs = test_pred(net=net, device=device,
                                                              dataloader=train_subset_loader,
                                                              num_samples=len(active_train_set),
                                                              with_labels=True)
        # predss, correct / num_samples, p
        probs = torch.tensor(probs)
        pos_samples = subset_df[subset_df['label'] == 1]
        neg_samples = subset_df[subset_df['label'] == 0]

        pos_indices = pos_samples.index.tolist()
        neg_indices = neg_samples.index.tolist()

        selected_pos_indices = [i for i in pos_indices if probs[i, 1] >= unc_threshold]
        selected_neg_indices = [i for i in neg_indices if probs[i, 1] >= unc_threshold]

        if len(selected_pos_indices) < per_num_sample / 2:
            pos_indices_to_add = selected_pos_indices
            pos += len(pos_indices_to_add)
        else:
            pos_indices_to_add = random.sample(selected_pos_indices, per_num_sample // 2)
            pos += len(pos_indices_to_add)

        if len(selected_neg_indices) < per_num_sample / 2:
            neg_indices_to_add = selected_neg_indices
            neg += len(neg_indices_to_add)
        else:
            neg_indices_to_add = random.sample(selected_neg_indices, per_num_sample // 2)
            neg += len(neg_indices_to_add)

        accumulated_samples.extend(pos_indices_to_add + neg_indices_to_add)

        if neg >= (num_active_sample / 2):
            sub_df_train = subset_df.loc[pos_indices_to_add].copy()
        else:
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
                loss = criterion(logits.squeeze(-1), p_one_hot.float())
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
                    path_to_model = 'models/CE{}thr_{}_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
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
                    path_to_model = 'models/CE{}thr_{}_Div_highUnc_samples_sep5_Active_kl_dynamic_ADAMW_ENN_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(
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

            mean_loss += criterion(logits.squeeze(-1), p_one_hot.float())
            #mean_loss = torch.mean(loss)

            #p_predicted = (alpha.view(-1) >= 0.5).int()
            _, p_predicted = torch.max(alpha, 1)



            val_correct += (p_label == p_predicted).sum().cpu()
            count += 1

    return mean_loss / count, val_correct


def train_bert(net, criterion, optim, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):
    global num_active_sample
    global unc_threshold
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
chpt = torch.load(sample_model)
model.load_state_dict(chpt["model_state_dict"])
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

criterion = nn.BCEWithLogitsLoss()


for i in range(10):
    train_bert(model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, 1)
    unc_threshold += 0.1
    num_active_sample = 10000

print(train_loss)
print(validation_loss)

print(train_accuracy)
print(validation_accuracy)

print(train_col_bel)
print(train_wrong_bel)
print(train_vac)
print(train_diss)

import matplotlib.pyplot as plt
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
