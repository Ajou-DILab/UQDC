val = wiki["dev"]
df__val = pd.DataFrame(val)

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

        return q_token_ids, q_attn_masks, label

val_query = []
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

df_val = pd.DataFrame(columns = ["query", "documents", "label"])
df_val['query'] = val_query
df_val['documents'] = val_doc
df_val['label'] = val_label

t_short_index = []
for i in range(len(df_val)):
    if len(df_val["documents"][i].split(" ")) <= 5:
        t_short_index.append(i)

val_set = D_CustomDataset(df_val, maxlen = 128, with_labels=False)
val_loader = DataLoader(val_set, batch_size=64, num_workers=0, shuffle=False, drop_last=True)
#df_test = all_df_test[56345:]
#df_test = df_test.reset_index(drop=True)

#len(df_test)
path_to_model_enn = "/content/drive/MyDrive/checkpoints/calibration_model/kl_1_SGD_ENN_lr_2e-05_train_acc_0.8060_ep_46_acc_0.8687.pt"
model = ENN_SentencePairClassifier()
# model.load_state_dict(torch.load(path_to_model_ce), strict=False)
model.load_state_dict(torch.load(path_to_model_enn))
model.to(device)
#test_set = Not_BM25_CustomDataset(df_test.reset_index(drop=True), 128)
#test_loader = DataLoader(test_set, batch_size=64, num_workers = 0, shuffle = False, drop_last = False)
#true_label = [1 for i in range(1138)] + [0 for j in range(1139)] # 새로 추가
path_to_output_file = '/content/drive/MyDrive/output_ENN.txt'

def test_pred(net, device, dataloader, num_samples, with_labels=True,  result_file="results/output.txt"):
    net.eval()
    probs = []
    uncertainties = []
    predss = []
    correct = 0 # 새로 추가
    with torch.no_grad():
        if with_labels:
            for q_ids, q_mask, label in tqdm(dataloader):
                q_ids, q_mask, true_label = q_ids.to(device), q_mask.to(device), label.to(device)
                logits = net(q_ids, q_mask)
                logits = logits[0]
                alpha = F.relu(logits) + 1
                uncertainty = 2 / torch.sum(alpha, dim=1, keepdim=True)
                _, preds = torch.max(logits, 1)
                prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                probs += prob.tolist()
                uncertainties += uncertainty.tolist()
                predss += preds.tolist()
                _, p_predicted = torch.max(logits, 1)
                correct += (true_label == p_predicted).sum().cpu()

        #y_true = true_label # 새로 추가
        #correct = sum(1 for a, b in zip(y_true, predss) if a == b) # 새로 추가
        #acc = correct / len(predss) # 새로 추가
    return probs, uncertainties, predss, correct / num_samples

print("Predicting on test data...")
probs, uncertainty, pred_E, acc = test_pred(net=model, device=device, dataloader=val_loader, num_samples = len(val_set), with_labels=True,  # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                result_file=path_to_output_file)
print()
print("Predictions are available in : {}".format(path_to_output_file))
# acc_list_enn.append(acc)
print(acc) # 2277 개 데이터셋 정확도

import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
y_true = df_val['label']

val_prob=[]
for i in probs:
  val_prob.append(i[0])

y_true = np.array(y_true[:6464])
y_pred = np.array(val_prob)
prob_true, prob_pred = calibration_curve(y_true, y_pred, pos_label = 0, n_bins=10)

plt.title("Calibration Curve (KL-1.2_ACC-81.74-SGD)")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives")
plt.plot([0, 1], [0, 1], linestyle='--')
# plot model reliability
plt.plot(prob_true, prob_pred, marker='.')
plt.show()
##
