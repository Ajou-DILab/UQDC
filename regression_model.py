from datasets import load_dataset
from tqdm.auto import tqdm

"""maxlen = 128
bs = 128  # batch size
iters_to_accumulate = 1
lr = 2e-5  # learning rate
epochs = 40  # number of training epochs"""
#model_name= "bert-base-uncased"


from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import nltk
from nltk import sent_tokenize
from lfqa_utils import *
#nltk.download("punkt")
import pandas as pd
df_dat = pd.DataFrame(columns = ["query", "pos", "neg"])

class D_CustomDataset(Dataset):

    def __init__(self, data, maxlen, with_labels=True, bert_model="google/bert_uncased_L-8_H-768_A-12"):
        self.data = data  # pandas dataframe
        # Initialize the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model)

        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Selecting sentence1 and sentence2 at the specified index in the data frame
        sent1 = str(self.data.loc[index, 'sentence1'])
        sent2 = str(self.data.loc[index, 'sentence2'])

        doc_list = []
        # Tokenize the pair of sentences to get token ids, attention masks and token type ids
        query_output = self.tokenizer(sent1,
                               padding='max_length',  # Pad to max_length
                               truncation=True,  # Truncate to max_length
                               max_length=self.maxlen,
                               return_tensors='pt')  # Return torch.Tensor objects
        doc_output = self.tokenizer(sent2,
                                      padding='max_length',  # Pad to max_length
                                      truncation=True,  # Truncate to max_length
                                      max_length=self.maxlen,
                                      return_tensors='pt')  # Return torch.Tensor objects


        label = self.data.loc[index, 'label']

        q_token_ids = query_output['input_ids'].squeeze(0)  # tensor of token ids
        q_attn_masks = query_output['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values

        d_token_ids = doc_output['input_ids'].squeeze(0)  # tensor of token ids
        d_attn_masks = doc_output['attention_mask'].squeeze(
            0)  # binary tensor with "0" for padded values and "1" for the other values

        return q_token_ids, q_attn_masks, d_token_ids, d_attn_masks, label/5

wiki_nq = load_dataset("glue", "stsb")

import pandas as pd
train = wiki_nq['train']  # 90 % of the original training data
val = wiki_nq['validation']  # 10 % of the original training data
test =wiki_nq["test"]
# Transform data into pandas dataframes
df__train = pd.DataFrame(train)
df__val = pd.DataFrame(val)
df__test = pd.DataFrame(test)

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

device = "cuda" if torch.cuda.is_available() else "cpu"

class RetrievalQAEmbedder(torch.nn.Module):
    def __init__(self, sent_encoder, dim):
        super(RetrievalQAEmbedder, self).__init__()
        self.sent_encoder = sent_encoder
        self.output_dim = 128
        self.project_q = torch.nn.Linear(dim, self.output_dim, bias=False)
        self.project_a = torch.nn.Linear(dim, self.output_dim, bias=False)

        self.edl_layer = DenseNormalGamma(128 * 2, 1)
        self.criterion = evidential_regression_loss(coeff=1e-2)
        #self.ce_loss = torch.nn.CrossEntropyLoss(reduction="mean")

    def embed_sentences_checkpointed(self, input_ids, attention_mask, checkpoint_batch_size=-1):
        # reproduces BERT forward pass with checkpointing
        if checkpoint_batch_size < 0 or input_ids.shape[0] < checkpoint_batch_size:
            return self.sent_encoder(input_ids, attention_mask=attention_mask)[1]
        else:
            # prepare implicit variables
            device = input_ids.device
            input_shape = input_ids.size()
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
            head_mask = [None] * self.sent_encoder.config.num_hidden_layers
            extended_attention_mask: torch.Tensor = self.sent_encoder.get_extended_attention_mask(
                attention_mask, input_shape, device
            )

            # define function for checkpointing
            def partial_encode(*inputs):
                encoder_outputs = self.sent_encoder.encoder(inputs[0], attention_mask=inputs[1], head_mask=head_mask,)
                sequence_output = encoder_outputs[0]
                pooled_output = self.sent_encoder.pooler(sequence_output)
                return pooled_output

            # run embedding layer on everything at once
            embedding_output = self.sent_encoder.embeddings(
                input_ids=input_ids, position_ids=None, token_type_ids=token_type_ids, inputs_embeds=None
            )
            # run encoding and pooling on one mini-batch at a time
            pooled_output_list = []
            for b in range(math.ceil(input_ids.shape[0] / checkpoint_batch_size)):
                b_embedding_output = embedding_output[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
                b_attention_mask = extended_attention_mask[b * checkpoint_batch_size : (b + 1) * checkpoint_batch_size]
                pooled_output = checkpoint.checkpoint(partial_encode, b_embedding_output, b_attention_mask)
                pooled_output_list.append(pooled_output)
            return torch.cat(pooled_output_list, dim=0)

    def embed_questions(self, q_ids, q_mask, checkpoint_batch_size=-1):
        q_reps = self.embed_sentences_checkpointed(q_ids, q_mask, checkpoint_batch_size)
        return self.project_q(q_reps)

    def embed_answers(self, a_ids, a_mask, checkpoint_batch_size=-1):
        a_reps = self.embed_sentences_checkpointed(a_ids, a_mask, checkpoint_batch_size)
        return self.project_a(a_reps)

    def forward(self, q_ids, q_mask, a_ids, a_mask, target, checkpoint_batch_size=-1):
        device = q_ids.device
        q_reps = self.embed_questions(q_ids, q_mask, checkpoint_batch_size)
        a_reps = self.embed_answers(a_ids, a_mask, checkpoint_batch_size)

        qa_reps = torch.cat((q_reps, a_reps), dim = -1)
        edl_output = self.edl_layer(qa_reps)
        loss_qa = self.criterion(torch.tensor(target, device = device), edl_output)

        return loss_qa


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



class ArgumentsQAR():
    def __init__(self):
        self.batch_size = 32
        self.max_length = 128
        self.checkpoint_batch_size = 32
        self.print_freq = 5
        self.pretrained_model_name = "google/bert_uncased_L-8_H-768_A-12"
        self.model_save_name = "eli5_retriever_model_l-8_h-768_b-512-512"
        self.learning_rate = 2e-6
        self.num_epochs = 10

qar_args = ArgumentsQAR()

# prepare torch Dataset objects

# load pre-trained BERT and make model
qar_tokenizer, qar_model = make_qa_retriever_model(
        model_name=qar_args.pretrained_model_name,
        from_file=None,
        device="cuda:0"
)

train_set = D_CustomDataset(df__train, maxlen = qar_args.max_length, with_labels=False)
train_loader = DataLoader(train_set, batch_size=qar_args.batch_size, num_workers=0, shuffle=True, drop_last=True)

val_set = D_CustomDataset(df__val, maxlen=qar_args.max_length, with_labels=False)
val_loader = DataLoader(val_set, batch_size=qar_args.batch_size, num_workers=0, shuffle=False, drop_last=True)

test_set = D_CustomDataset(df__test, maxlen=qar_args.max_length, with_labels=False)
test_loader = DataLoader(test_set, batch_size=qar_args.batch_size, num_workers=0, shuffle=False, drop_last=True)

model = qar_model

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
        for it, (p_input, p_attn, d_input, d_attn, p_label) in enumerate(tqdm(dataloader)):
            # q_ids, q_mask, a_ids, a_mask, labels = q_ids.to(device), q_mask.to(device), a_ids.to(device), a_mask.to(device), labels.to(device)

            # Obtaining the logits from the model
            p_input, p_attn, d_input, d_attn, p_label = p_input.to(device), p_attn.to(device), d_input.to(device),\
                                                        d_attn.to(device), p_label.to(device)
            loss = net(p_input, p_attn, d_input, d_attn, p_label)


            # mean_loss += criterion(logits.squeeze(-1), labels.float())
            mean_loss += loss

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

        for it, (p_input, p_attn, d_input, d_attn, p_label) in enumerate(tqdm(train_loader)):

            p_input, p_attn, d_input, d_attn, p_label = p_input.to(device), p_attn.to(device), d_input.to(device),\
                                                        d_attn.to(device), p_label.to(device)
            # Obtaining the logits from the model
            loss = net(p_input, p_attn, d_input, d_attn, p_label)

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
        validation_accuracy.append(val_acc)

        print("Epoch {} complete! Train ACC : {} Validation Loss : {} ACC : {}".format(ep + 1,
                                                                                       train_correct / (len(df__train)),
                                                                                       val_loss,
                                                                                       val_acc))

        if val_acc > best_acc:
            print("Best validation acc improved from {} to {}".format(best_acc, val_acc))
            print()
            path_to_model = 'models/888_lr_{}_train_acc_{:.4f}_ep_{}_acc_{:.4f}.pt'.format(lr,
                                                                                   train_correct / (len(df__train)),
                                                                                   best_ep,
                                                                                   val_acc)
            torch.save(net.state_dict(), path_to_model)
            print("The model has been saved in {}".format(path_to_model))
            best_acc = val_acc
            best_ep = ep + 1
        # Saving the model
        #save_weights(model, "model_weight.txt", ep)

    del loss
    torch.cuda.empty_cache()

set_seed(42)

#model.load_state_dict(torch.load("models/444_lr_7e-07_train_acc_0.8050_ep_15_acc_0.7733.pt"), strict=False)
model.to(device)
#opti = AdamW(model.parameters(), lr=lr, eps=1e-8)
opti = torch.optim.Adam(model.parameters(), lr=qar_args.learning_rate)
#opti = torch.optim.SGD(model.parameters(), lr=lr, momentum= 0.9, nesterov= True, weight_decay = 1e-6)
# opti = AdamW(model.parameters(), lr = lr)
num_warmup_steps = 0  # The number of steps for the warmup phase.
num_training_steps = qar_args.num_epochs * len(train_loader)  # The total number of training steps
t_total = (len(train_loader) // 1) * qar_args.num_epochs  # Necessary to take into account Gradient accumulation

lr_scheduler = get_linear_schedule_with_warmup(optimizer=opti, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=t_total)

criterion = edl_log_loss
train_bert(model, criterion, opti, qar_args.learning_rate, lr_scheduler, train_loader, val_loader, qar_args.num_epochs, 1)

print(train_loss)
print(validation_loss)

print(train_accuracy)
print(validation_accuracy)

import matplotlib.pyplot as plt

def plot_predictions(x_train, y_train, x_test, y_test, y_pred, n_stds=4, kk=0):
    x_test = x_test[:, 0]
    mu, v, alpha, beta, aleatoric, epistemic = y_pred
    mu = mu[:, 0]
    var = np.sqrt(beta / (v * (alpha - 1)))
    var = np.minimum(var, 1e3)[:, 0]  # for visualization

    plt.figure(figsize=(5, 3), dpi=200)
    plt.scatter(x_train, y_train, s=1., c='#463c3c', zorder=0, label="Train")
    plt.plot(x_test, y_test, 'r—', zorder=2, label="True")
    plt.plot(x_test, mu, color='#007cab', zorder=3, label="Pred")
    plt.plot([-4, -4], [-150, 150], 'k—', alpha=0.4, zorder=0)
    plt.plot([+4, +4], [-150, 150], 'k—', alpha=0.4, zorder=0)

    for k in np.linspace(0, n_stds, 4):
        plt.fill_between(
            x_test, (mu - k * var), (mu + k * var),
            alpha=0.3,
            edgecolor=None,
            facecolor='#00aeef',
            linewidth=0,
            zorder=1,
            label="Unc." if k == 0 else None)

    plt.gca().set_ylim(-150, 150)
    plt.gca().set_xlim(-7, 7)
    plt.legend(loc="upper left")
    plt.show()
