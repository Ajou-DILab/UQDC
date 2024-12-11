# Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

from torchmetrics.classification import CalibrationError

from datasets import load_dataset
from tqdm.auto import tqdm
import torch.distributions as dist
from sklearn.metrics import roc_curve

import pandas as pd

from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
    AdamW, get_linear_schedule_with_warmup, AutoConfig

import random
import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from ENN.ENN_model import SentencePairClassifier
from edl_function import *
from ENN.ENN_test import test_pred
from ENN.ENN_train import *
from ENN.ENN_eval import evaluate_loss

import arsparse
from utils import *


# Set the device to GPU if available, otherwise use CPU
device = "cuda:0" if torch.cuda.is_available() else "cpu"


set_seed(42)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate a sentence pair classifier.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pre-trained model.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset.")
    return parser.parse_args()

def main():

    args = parse_args()
    model_path = args.model_path
    data_path = args.data_path
    
    print(f"Model Path: {model_path}")
    print(f"Data Path: {data_path}")


    empy_model = SentencePairClassifier()
    model = load_model(empy_model, model_path)
    model.to(device)
    data = load_sample_data(data_path)

    df_val = pd.DataFrame(data)

    val_set = D_CustomDataset(df_val, maxlen=128, with_labels=False)
    val_loader = DataLoader(val_set, batch_size=256, num_workers=0, shuffle=False, drop_last=True)

    probs, uncertainty, pred_E, acc, uncertainty_acc = test_pred(net=model, device=device,
                                                                        dataloader=val_loader,
                                                                        num_samples=len(val_set), with_labels=True)
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
