dataset = load_dataset('glue', 'qnli')
print(dataset)


train = dataset['train']  # 90 % of the original training data
val = dataset['validation']  # 10 % of the original training data
# Transform data into pandas dataframes
df_train = pd.DataFrame(train)
df_val = pd.DataFrame(val)

print(df_train.isnull().sum())
print(df_val.isnull().sum())

class CustomDataset(Dataset):

    def __init__(self, data, maxlen):

        self.data = data  # pandas dataframe
        self.maxlen = maxlen
        self.tokenizer = AutoTokenizer.from_pretrained("yjernite/retribert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # Selecting sentence1 and sentence2 at the specified index in the data frame
        question = str(self.data.loc[index, 'question'])
        answer = str(self.data.loc[index, 'sentence'])
        label = self.data.loc[index, 'label']

        data = {'question' : question, 'answer' : answer, 'label' : label}

        return data
