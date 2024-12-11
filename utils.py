import torch
import random
import numpy as np
import os
import pandas as pd

def calculate_map10(test_prob, y_true, y_index):
    """MAP@10 calculation."""
    average_precision_10 = RetrievalMAP(top_k=10)
    map_output = average_precision_10(torch.tensor(test_prob), torch.tensor(y_true), indexes=torch.tensor(y_index))
    return map_output

# Load model
def load_model(model, model_path):
    """Load the trained model."""
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model

# Load datasets
def load_sample_data(data_path):
    """Load sample data from DataFrame."""
    df = pd.read_csv(data_path)
    return df

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


def calculate_ece(accs, confs):
    bin_size = len(accs)
    ece = 0
    for i in range(bin_size):
        bin_error = np.abs(accs[i] - confs[i])
        ece += bin_error * (1 / bin_size)
    return ece

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

