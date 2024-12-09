import torch
from torchmetrics.retrieval import RetrievalMAP
from utils import load_model, load_sample_data

def calculate_map10(test_prob, y_true, y_index):
    """MAP@10 calculation."""
    average_precision_10 = RetrievalMAP(top_k=10)
    map_output = average_precision_10(torch.tensor(test_prob), torch.tensor(y_true), indexes=torch.tensor(y_index))
    return map_output

def main():
    model_path = "models/AL_model.pt"
    data_path = "data/sample.csv"
    input_dim = 10  # 샘플 데이터의 피처 수

    # 모델 및 데이터 로드
    model = load_model(model_path, input_dim)
    data = load_sample_data(data_path)

    val_set = D_CustomDataset(df_val, maxlen=128, with_labels=False)
    val_loader = DataLoader(val_set, batch_size=256, num_workers=0, shuffle=False, drop_last=True)

    probs, uncertainty, pred_E, acc, uncertainty_acc, brier = test_pred(net=model, device=device,
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
