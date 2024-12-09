import torch
from torchmetrics.retrieval import RetrievalMAP
from utils import load_model, load_sample_data

def calculate_map10(test_prob, y_true, y_index):
    """MAP@10 calculation."""
    average_precision_10 = RetrievalMAP(top_k=10)
    map_output = average_precision_10(torch.tensor(test_prob), torch.tensor(y_true), indexes=torch.tensor(y_index))
    return map_output

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
                #prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
                prob = torch.sigmoid(logits)
                probs += prob.tolist()
                uncertainties += uncertainty.tolist()
                predss += preds.tolist()
                true_labels += true_label.tolist()
                ece_logits += torch.sigmoid(logits).tolist()
                ece_label += true_label.tolist()
                correct += (true_label == preds).sum().cpu()

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
    return probs, uncertainties, predss, correct / num_samples, unc_accuracy

def main():
    model_path = "models/AL_model.pt"
    data_path = "data/sample.csv"
    input_dim = 10  # 샘플 데이터의 피처 수

    # 모델 및 데이터 로드
    model = load_model(model_path, input_dim)
    data = load_sample_data(data_path)

    val_set = D_CustomDataset(df_val, maxlen=128, with_labels=False)
    val_loader = DataLoader(val_set, batch_size=256, num_workers=0, shuffle=False, drop_last=True)

    probs, uncertainty, pred_E, acc, uncertainty_acc = test_pred(net=model, device=device,
                                                                        dataloader=val_loader,
                                                                        num_samples=len(val_set), with_labels=True,
                                                                        # set the with_labels parameter to False if your want to get predictions on a dataset without labels
                                                                        result_file=path_to_output_file)
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
