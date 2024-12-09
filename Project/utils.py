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

