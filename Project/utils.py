import torch
import pandas as pd

def load_model(model_path, input_dim):
    """Load the trained model."""
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_sample_data(data_path):
    """Load sample data from DataFrame."""
    df = pd.read_csv(data_path)
    return df

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
