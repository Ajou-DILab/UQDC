import torch
import random
import numpy as np
import os

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
