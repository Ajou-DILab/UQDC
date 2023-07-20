from datasets import load_dataset
from tqdm.auto import tqdm

maxlen = 64
bs = 64  # batch size
iters_to_accumulate = 1
lr = 2e-5  # learning rate
epochs = 40  # number of training epochs
#model_name= "bert-base-uncased"
model_name = "huawei-noah/TinyBERT_General_4L_312D"


from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import nltk
from nltk import sent_tokenize
#nltk.download("punkt")
import pandas as pd
df_dat = pd.DataFrame(columns = ["query", "pos", "neg"])
"""
example = pd.read_csv("train_nq_dataset.csv")

df_dat = pd.DataFrame(columns = ["query", "pos", "neg"])

pos_passage = dict()
neg_passage = dict()

for i in range(len(example)):
    pos_list = []

    pos_list1 = example["pos_pas"][i][1:-1]
    pos_split = pos_list1.split("', '")
    for j in range(len(pos_split)):
        if j == 0:
            pos_list.append(pos_split[j][1:])
        else:
            pos_list.append(pos_split[j][:-1])
    pos_passage[i] = pos_list

    neg_list = []
    neg_list1 = example["neg_pas"][i][1:-1]
    neg_split = neg_list1.split("', '")
    for j in range(len(neg_split)):
        if j == 0:
            neg_list.append(neg_split[j][1:])
        else:
            neg_list.append(neg_split[j][:-1])
    neg_passage[i] = neg_list

df_dat["query"] = example["question"]
df_dat["pos"] = pos_passage.values()
df_dat['neg'] = neg_passage.values()
print(len(df_dat))

df__train = df_dat[:54000]  # 90 % of the original training data
df__val = df_dat[54000:56000]  # 10 % of the original training data
df__val = df__val.reset_index(drop = True)

# Transform data into pandas dataframes

train_query = []
train_doc = []
train_label = []

val_query = []
val_doc = []
val_label = []

#train_set
for index in range(len(df__train)):
    pos_length = len(df__train.loc[index, "pos"])
    neg_length = len(df__train.loc[index, "neg"])


    for j in range(pos_length):
        query = str(df__train.loc[index, "query"])
        passage = str(df__train.loc[index, "pos"][j])

        train_query.append(query)
        train_doc.append(passage)
        train_label.append(1)

    for j in range(neg_length):
        query = str(df__train.loc[index, "query"])
        passage = str(df__train.loc[index, "neg"][j])

        train_query.append(query)
        train_doc.append(passage)
        train_label.append(0)

#val_set

for index in range(len(df__val)):
    pos_length = len(df__val.loc[index, "pos"])

    neg_length = len(df__val.loc[index, "neg"])

    for j in range(pos_length):
        query = str(df__val.loc[index, "query"])
        passage = str(df__val.loc[index, "pos"][j])

        val_query.append(query)
        val_doc.append(passage)
        val_label.append(1)

    for j in range(neg_length):
        query = str(df__val.loc[index, "query"])
        passage = str(df__val.loc[index, "neg"][j])

        val_query.append(query)
        val_doc.append(passage)
        val_label.append(0)

df_train = pd.DataFrame(columns = ["query", "documents", "label"])
df_val = pd.DataFrame(columns = ["query", "documents", "label"])

df_train['query'] = train_query
df_train['documents'] = train_doc
df_train['label'] = train_label

df_val['query'] = val_query
df_val['documents'] = val_doc
df_val['label'] = val_label

t_short_index = []
for i in range(len(df_train)):
    text = sent_tokenize(df_train['documents'][i])
    for j in text:
        if len(j.split(" ")) <= 2:
            t_short_index.append(i)

v_short_index = []
for i in range(len(df_val)):
    text = sent_tokenize(df_val['documents'][i])
    for j in text:
        if len(j.split(" ")) <= 2:
            v_short_index.append(i)

index_over = []
for i in range(len(df_train)):
    if len(sent_tokenize(df_train["documents"][i])) > 7:
        index_over.append(i)
index_val = []
for i in range(len(df_val)):
    if len(sent_tokenize(df_val["documents"][i])) > 7:
        index_val.append(i)

df_train = df_train.drop(set(index_over + t_short_index), axis = 0)
df_train = df_train.reset_index(drop = True)

df_val = df_val.drop(set(index_val + v_short_index), axis = 0)
df_val = df_val.reset_index(drop = True)

print(len(df_train))
print(len(df_val))
"""

class D_CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model=model_name):
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


        sent3 = sent_tokenize(sent2)
        plus_len = 16 - len(sent3)
        sent3 += [self.tokenizer.pad_token] * plus_len

        doc_list = []
        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        query_output = self.tokenizer(sent1,
                               padding='max_length',  # Pad to max_length
                               truncation=True,  # Truncate to max_length
                               max_length=self.maxlen,
                               return_tensors='pt')  # Return torch.Tensor objects
        doc_output = self.tokenizer(sent3,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects


        label = self.data.loc[index, 'label']

        return query_output, doc_output, label

"""train_set = D_CustomDataset(df_train, maxlen=maxlen, with_labels=False)
train_loader = DataLoader(train_set, batch_size=bs, num_workers=0, shuffle=True, drop_last=True)

val_set = D_CustomDataset(df_val, maxlen=maxlen, with_labels=False)
val_loader = DataLoader(val_set, batch_size=bs, num_workers=0, shuffle=False, drop_last=True)"""
wiki_nq = load_dataset("Tevatron/wikipedia-nq")

import pandas as pd
train = wiki_nq['train']  # 90 % of the original training data
val = wiki_nq['dev']  # 10 % of the original training data
# Transform data into pandas dataframes
df__train = pd.DataFrame(train)
df__val = pd.DataFrame(val)

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


train_query = []
train_doc = []
train_label = []

val_query = []
val_doc = []
val_label = []

train_col_bel = []
train_wrong_bel = []
train_diss = []
train_vac = []

##
for i in range(len(df__train)):
    query = str(df__train.loc[i, "query"])
    pos_sent = df__train.loc[i, "positive_passages"]
    neg_sent = df__train.loc[i, "negative_passages"]
    for j in range(len(pos_sent)):
        if j == 3:
            break
        pos_doc = str(pos_sent[j]['text'])
        train_query.append(query)
        train_doc.append(pos_doc)
        train_label.append(1)
    if len(neg_sent) >= len(pos_sent):
        for j in range(len(pos_sent)):
            if j == 2:
                break
            neg_doc = str(neg_sent[j]['text'])
            train_query.append(query)
            train_doc.append(neg_doc)
            train_label.append(0)
    else:
        for j in range(len(neg_sent)):
            if j == 2:
                break
            neg_doc = str(neg_sent[j]['text'])
            train_query.append(query)
            train_doc.append(neg_doc)
            train_label.append(0)

"""for i in range(len(df__val)):
    query = str(df__val.loc[i, "query"])
    pos_sent = df__val.loc[i, "positive_passages"]
    neg_sent = df__val.loc[i, "negative_passages"]
    for j in range(len(pos_sent)):
        pos_doc = str(pos_sent[j]['text'])
        val_query.append(query)
        val_doc.append(pos_doc)
        val_label.append(1)
    if len(neg_sent) >= len(pos_sent):
        for j in range(len(pos_sent)):
            neg_doc = str(neg_sent[j]['text'])
            val_query.append(query)
            val_doc.append(neg_doc)
            val_label.append(0)
    else:
        for j in range(len(neg_sent)):
            neg_doc = str(neg_sent[j]['text'])
            val_query.append(query)
            val_doc.append(neg_doc)
            val_label.append(0)"""
##
print(len(train_doc))
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

#val_query = val_query[:len(val_doc)]
#val_label = val_label[:len(val_doc)]
"""for index in range(len(df__train)):
    if len(df__train.loc[index, 'positive_passages']) > 2 and len(df__train.loc[index, "negative_passages"]) > 2:
        sent1 = str(df__train.loc[index, 'query'])
        sent2 = str(df__train.loc[index, 'positive_passages'][0]['text'])
        sent3 = str(df__train.loc[index, "negative_passages"][0]['text'])
        sent4 = str(df__train.loc[index, 'positive_passages'][1]['text'])
        sent5 = str(df__train.loc[index, "negative_passages"][1]['text'])
        sent6 = str(df__train.loc[index, 'positive_passages'][2]['text'])
        sent7 = str(df__train.loc[index, "negative_passages"][2]['text'])

        train_query.append(sent1)
        train_doc.append(sent2)
        train_label.append(1)
        train_query.append(sent1)
        train_doc.append(sent3)
        train_label.append(0)

        train_query.append(sent1)
        train_doc.append(sent4)
        train_label.append(1)
        train_query.append(sent1)
        train_doc.append(sent5)
        train_label.append(0)

        train_query.append(sent1)
        train_doc.append(sent6)
        train_label.append(1)
        train_query.append(sent1)
        train_doc.append(sent7)
        train_label.append(0)
    else:
        sent1 = str(df__train.loc[index, 'query'])
        sent2 = str(df__train.loc[index, 'positive_passages'][0]['text'])
        sent3 = str(df__train.loc[index, "negative_passages"][0]['text'])

        train_query.append(sent1)
        train_doc.append(sent2)
        train_label.append(1)
        train_query.append(sent1)
        train_doc.append(sent3)
        train_label.append(0)


val_query = []
val_ans = []
val_doc = []
val_label = []

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
"""
print("process")
df_train = pd.DataFrame(columns = ["query", "documents", "label"])
df_val = pd.DataFrame(columns = ["query", "documents", "label"])

df_train['query'] = train_query
df_train['documents'] = train_doc
df_train['label'] = train_label

df_val['query'] = val_query
df_val['documents'] = val_doc
df_val['label'] = val_label


t_short_index = []
for i in range(len(df_train)):
    text = sent_tokenize(df_train['documents'][i])
    for j in text:
        if len(j.split(" ")) <= 2:
            t_short_index.append(i)
            break

v_short_index = []
for i in range(len(df_val)):
    text = sent_tokenize(df_val['documents'][i])
    for j in text:
        if len(j.split(" ")) <= 2:
            v_short_index.append(i)
            break

index_over = []
"""for i in range(len(df_train)):
    if len(sent_tokenize(df_train["documents"][i])) > 6:
        index_over.append(i)
index_val = []
for i in range(len(df_val)):
    if len(sent_tokenize(df_val["documents"][i])) > 6:
        index_val.append(i)"""
print(len(df_train))
df_train = df_train.drop(set(t_short_index), axis = 0)
print(len(df_train))
df_train = df_train.drop_duplicates(["documents"])
print(len(df_train))
df_train = df_train.reset_index(drop = True)

df_val = df_val.drop(set(v_short_index), axis = 0)
#df_val = df_val.drop_duplicates(["documents"])
df_val = df_val.reset_index(drop = True)

p = 0
n= 0
for i in df_train['label']:
    if i == 0:
        n += 1
    elif i == 1:
        p += 1
print(p)
print(n)
print(len(df_train))
print(len(df_val))

max_s = 0
for i in range(len(df_train)):
    sent = df_train["documents"][i]
    splt_sent = len(sent_tokenize(sent))
    if max_s < splt_sent:
        max_s = splt_sent
print(max_s)


train_set = D_CustomDataset(df_train, maxlen=maxlen, with_labels=False)
train_loader = DataLoader(train_set, batch_size=bs, num_workers=0, shuffle=True, drop_last=True)

val_set = D_CustomDataset(df_val, maxlen=maxlen, with_labels=False)
val_loader = DataLoader(val_set, batch_size=bs, num_workers=0, shuffle=False, drop_last=True)

from transformers import AutoTokenizer, AutoModel
import torch
from edl_function import *
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
    AdamW, get_linear_schedule_with_warmup, AutoConfig
import torch.optim as optim

model = AutoModel.from_pretrained(model_name)

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"


class SentencePairClassifier(nn.Module):

    def __init__(self, bert_model=model_name, freeze_bert=False):
        super(SentencePairClassifier, self).__init__()
        #  Instantiating BERT-based model object
        self.bert_layer = AutoModel.from_pretrained(bert_model)

        # Freeze bert layers and only train the classification layer weights
        if freeze_bert:
            for p in self.bert_layer.parameters():
                p.requires_grad = False


        # Classification layer
        #self.context_attn = nn.Linear(768, 1)
        #self.ffn2 = nn.Linear(256, 128)
        #self.ffn3 = nn.Linear(128, 64)
        #self.ffn4 = nn.Linear(64, 32)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.cls_layer = nn.Linear(312 * 4, 2)

    def forward(self, query, doc):
        doc_input = []
        doc_attn = []
        for i in range(16):

            input_ids = torch.stack([doc["input_ids"][:, i, :]], dim=0)
            attn_mask = torch.stack([doc["attention_mask"][:, i, :]], dim=0)
            doc_input.append(input_ids)
            doc_attn.append(attn_mask)

        # Feeding the inputs to the BERT-based model to obtain contextualized representations
        q_input_ids = query['input_ids'].squeeze(1).to(device)
        q_attn_mask = query['attention_mask'].squeeze(1).to(device)
        q_output = self.get_output_embedding(q_input_ids, q_attn_mask)
        pooled_q = self.mean_pooling(q_output[0], q_attn_mask)

        #context = pooled_q
        d_output_list = []
        d_attn_list = []
        for i in range(16):
            if doc_input[i][0][0][2] == 102:
                break
            d_output = self.get_output_embedding(doc_input[i][0].to(device), doc_attn[i][0].to(device))

            d_attn_list.append(doc_attn[i][0])
            pooled_d = self.mean_pooling(d_output[0], doc_attn[i][0].to(device))
            d_output_list.append(pooled_d)

            #context = self.update_context(context, pooled_d)

        if len(d_output_list) == 0:
            print(doc_input[i][0])
        all_sentence_cat = torch.stack(d_output_list, dim=1)
        #all_attn_cat = torch.cat((d_attn_list), dim=1)

        out = torch.max(all_sentence_cat, dim = 1)[0]


        #pooled_d = self.max_pooling(all_sentence_cat.to(device), all_attn_cat.to(device))

        concat_pair = torch.cat((pooled_q, out, torch.abs(pooled_q - out), pooled_q * out), dim=-1)
        #ffn_output = self.ffn(self.dropout(self.act(concat_pair)))
        """ffn_output2 = self.ffn2(self.dropout(self.act(ffn_output)))
        ffn_output3 = self.ffn3(self.dropout(self.act(ffn_output2)))
        ffn_output4 = self.ffn4(self.dropout(self.act(ffn_output3)))"""
        #ffn_output5 = self.ffn5(self.dropout(self.act(ffn_output4)))
        logits = self.dropout(self.cls_layer(concat_pair))


        return logits

    #
    """def update_context(self, context, sentence_embedding):
        attn_weights = self.context_attn(context)  # 문맥 벡터에 대한 어텐션 가중치 계산 [batch_size, 1]
        updated_context = context + attn_weights * sentence_embedding  # 어텐션 가중치를 곱하여 문맥 벡터 갱신
        return updated_context"""

    def dir_prior_mult_likelihood_loss(self, gt, alpha, current_epoch):
        """
        Calculate the loss based on the dirichlet prior and multinomial likelihoood
        :param gt: The ground truth (one hot vector)
        :param alpha: The prior parameters
        :param current_epoch: For the regularization parameter
        :return: loss
        """
        gt = gt.to(self.device)
        alpha = alpha.to(self.device)

        S = torch.sum(alpha, dim=1, keepdim=True)

        first_part_error = torch.sum(gt * (torch.log(S) - torch.log(alpha)), dim=1, keepdim=True)
        annealing_rate = torch.min(
            torch.tensor(1.0, dtype=torch.float32),
            torch.tensor(current_epoch / 10, dtype=torch.float32)
        )

        if self.args.fix_annealing_rate:
            annealing_rate = 1
            # print(annealing_rate)

        alpha_new = (alpha - 1) * (1 - gt) + 1
        kl_err = self.args.kl_scaling_factor * annealing_rate * self.KL_flat_dirichlet(alpha_new)

        # return loss

        dirichlet_strength = torch.sum(alpha, dim=1)
        dirichlet_strength = dirichlet_strength.reshape((-1, 1))

        # Belief
        belief = (alpha - 1) / dirichlet_strength

        inc_belief = belief * (1 - gt)
        inc_belief_error = self.args.kl_scaling_factor * annealing_rate * torch.mean(inc_belief, dim=1, keepdim=True)

        if self.args.use_kl_error:
            loss = first_part_error + kl_err
            # print("kl using")
        else:
            loss = first_part_error + inc_belief_error

        return loss

    def get_output_embedding(self, input_ids, attn_mask):
        return self.bert_layer(input_ids, attn_mask)

    def calculate_dissonance_from_belief(self, belief):
        num_classes = len(belief)
        Bal_mat = torch.zeros((num_classes, num_classes)).to(self.device)
        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                if belief[i] == 0 or belief[j] == 0:
                    Bal_mat[i, j] = 0
                else:
                    Bal_mat[i, j] = 1 - torch.abs(belief[i] - belief[j]) / (belief[i] + belief[j])
                Bal_mat[j, i] = Bal_mat[i, j]
        sum_belief = torch.sum(belief).to(self.device)
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
        evidence = F.softplus(preds)

        # The prior parameters
        alpha = evidence + 1

        dirichlet_strength = torch.sum(alpha, dim=1)
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
    # 모델의 가중치를 파일에 추가
    with open(file_path, 'a') as file:
        # 모델의 가중치를 문자열로 변환하여 파일에 작성
        file.write(str(epoch) + " epoch\n")
        file.write(str(model.cls_layer.weight.data))
        file.write('\n')  # 가중치 사이에 줄바꿈 추가


def evaluate_loss(net, dataloader, epoch, criterion):
    net.eval()

    mean_loss = 0
    count = 0
    val_correct = 0

    with torch.no_grad():
        for it, (p_input, d_input, p_label) in enumerate(tqdm(dataloader)):
            # q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)

            # Obtaining the logits from the model
            p_logits = net(p_input, d_input)

            p_one_hot = one_hot_embedding(p_label, 2)
            p_one_hot = p_one_hot.to(device)

            # mean_loss += criterion(logits.squeeze(-1), labels.float())
            mean_loss += criterion(p_logits, p_one_hot.float(), epoch, 2, 10, device)

            _, p_predicted = torch.max(p_logits, 1)

            p_label = p_label.to(device)

            val_correct += (p_label == p_predicted).sum().cpu()
            count += 1

    return mean_loss / count, val_correct


def train_bert(net, criterion, optim, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate):
    best_acc = -np.Inf
    best_ep = 1
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
            for p in net.ffn.parameters():
                p.requires_grad = True
            for p in net.ffn2.parameters():
                p.requires_grad = True

            for p in net.ffn3.parameters():
                p.requires_grad = True
            for p in net.ffn4.parameters():
                p.requires_grad = True
            for p in net.cls_layer.parameters():
                p.requires_grad = True
        else:
            for p in net.parameters():
                p.requires_grad = True"""

        avg_cor_bel = 0
        avg_wrong_bel = 0
        avg_diss = 0
        avg_vac = 0

        for it, (p_input, d_input, p_label) in enumerate(tqdm(train_loader)):

            # q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)

            p_one_hot = one_hot_embedding(p_label, 2)
            p_one_hot = p_one_hot.to(device)

            # Obtaining the logits from the model
            p_logits = net(p_input, d_input)
            # loss = criterion(logits.squeeze(-1), labels.float())
            # Computing loss
            loss = criterion(
                p_logits, p_one_hot.float(), ep, 2, 10, device
            )

            vacuity, wrong_bel, cor_bel, dissonance = net.calc_loss_vac_bel(p_logits, p_one_hot)
            avg_cor_bel += torch.mean(cor_bel)
            avg_wrong_bel += torch.mean(wrong_bel)
            avg_diss += torch.mean(dissonance)
            avg_vac += torch.mean(vacuity)
            # print(loss)

            _, p_predicted = torch.max(p_logits, 1)
            p_label = p_label.to(device)
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

                running_loss = 0.0

        val_loss, val_acc = evaluate_loss(net, val_loader, ep, criterion)  # Compute validation loss
        train_loss.append(all_loss / count)
        validation_loss.append(val_loss.item())
        print()
        val_acc = val_acc / (len(df_val))
        train_accuracy.append(train_correct / (len(df_train)))
        validation_accuracy.append(val_acc)

        train_col_bel.append((avg_cor_bel / count).detach().cpu().numpy())
        train_wrong_bel.append((avg_wrong_bel / count).detach().cpu().numpy())
        train_diss.append((avg_diss / count).detach().cpu().numpy())
        train_vac.append((avg_vac / count).detach().cpu().numpy())
        print("Epoch {} complete! Train ACC : {} Validation Loss : {} ACC : {}".format(ep + 1,
                                                                                       train_correct / (len(df_train)),
                                                                                       val_loss,
                                                                                       val_acc))

        if val_acc > best_acc:
            print("Best validation acc improved from {} to {}".format(best_acc, val_acc))
            print()
            path_to_model = 'models/888_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(lr,
                                                                                   train_correct / (len(df_train)),
                                                                                   best_ep,
                                                                                   val_acc)
            torch.save(net.state_dict(), path_to_model)
            print("The model has been saved in {}".format(path_to_model))
            best_acc = val_acc
            best_ep = ep + 1
        # Saving the model
        save_weights(model, "model_weight.txt", ep)

    del loss
    torch.cuda.empty_cache()

set_seed(42)

#model.load_state_dict(torch.load("models/444_lr_7e-07_train_acc_0.8050_ep_15_acc_0.7733.pt"), strict=False)
model.to(device)
#opti = AdamW(model.parameters(), lr=lr, eps=1e-8)
opti = torch.optim.Adam(model.parameters(), lr=lr)
#opti = torch.optim.SGD(model.parameters(), lr=lr, momentum= 0.9, nesterov= True, weight_decay = 1e-6)
# opti = AdamW(model.parameters(), lr = lr)
num_warmup_steps = 0  # The number of steps for the warmup phase.
num_training_steps = epochs * len(train_loader)  # The total number of training steps
t_total = (len(train_loader) // iters_to_accumulate) * epochs  # Necessary to take into account Gradient accumulation

lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=t_total)

criterion = edl_log_loss
train_bert(model, criterion, opti, lr, lr_scheduler, train_loader, val_loader, epochs, iters_to_accumulate)

print(train_loss)
print(validation_loss)

print(train_accuracy)
print(validation_accuracy)

import matplotlib.pyplot as plt
plt.subplots(constrained_layout = True)
plt.subplot(3, 1, 1)
plt.ylim(0.0, 1.0)
t1 = np.arange(1.0, epochs + 1, 1)
t2 = np.arange(1.0, epochs + 1, 1)
plt.xticks(t1)
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(t1, train_loss, label="Train")
plt.plot(t2, validation_loss, 'r-', label="Validation")
plt.legend()

plt.subplot(3, 1, 2)
plt.ylim(0.0, 1.0)
plt.xticks(t1)
plt.title("ACC Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(t1, train_accuracy, label="Train")
plt.plot(t2, validation_accuracy, 'r-', label="Validation")
plt.legend()

plt.subplot(3, 1, 3)
plt.ylim(0.0, 1.0)
plt.xticks(t1)
plt.title("ACC Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(t1, train_col_bel, label="cor_bel")
plt.plot(t2, train_wrong_bel, 'g-', label="wrong_bel")
plt.plot(t2, train_vac, 'r-', label="vacuity")
plt.plot(t2, train_diss, 'o-', label="dissonance")
plt.legend()
plt.show()
