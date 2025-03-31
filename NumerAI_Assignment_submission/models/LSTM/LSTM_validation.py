import os
import json
import yaml
import torch
import argparse
import pandas as pd
import torch.nn as nn
from helper import validate_dl
from datetime import datetime
import matplotlib.pyplot as plt
from numerai_tools.scoring import numerai_corr
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()

feature_set = json.load(open(f"data/{configs['Dataset']['name']}/features.json"))["feature_sets"][f"{configs['Dataset']['set']}"]

# Load Validation dataset
validation_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/validation.parquet",columns = ["era", "data_type", "target"] + feature_set)

validation_dataset = pd.DataFrame(validation_dataset[validation_dataset["data_type"] == "validation"])
del validation_dataset["data_type"]

validation_dataset = pd.DataFrame(validation_dataset[validation_dataset["era"].isin(pd.Series(validation_dataset["era"]))])

# Eras are 1 week apart, but targets look 20 days (o 4 weeks/eras) into the future,
# so we need to "embargo" the first 4 eras following our last train era to avoid "data leakage"
if configs["Train"]["last_train_era"] == -1:
    # Load Train Dataset
    train_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/train.parquet",columns = ["era", "target"] + feature_set)
    # Reduce Dataset size
    train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"]))])
    configs["Train"]["last_train_era"] = int(train["era"].unique()[-1])

last_train_era = configs["Train"]["last_train_era"]

eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i+1 for i in range(4)]]
validation = pd.DataFrame(validation_dataset[~validation_dataset["era"].isin(eras_to_embargo)])



validation_loader_list = []
for era in sorted(validation['era'].unique()):
    era_df = pd.DataFrame(validation[validation['era'] == era])

    X = era_df[feature_set].values  # Features
    y = era_df['target'].values  # Target

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create dataset and loader for the entire validation set of this task
    dataset = TensorDataset(X_tensor, y_tensor)
    test_loader = DataLoader(
        dataset,
        batch_size=configs['Validation']['batch_size'],
        shuffle=False
    )

    # Append only the test loader as there's no separate train loader for validation
    validation_loader_list.append(test_loader)  # Using None for train_loader


class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=4, dropout=0.1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        # Linear layer to project input features to the LSTM model dimension (hidden_dim)
        self.feature_embedding = nn.Linear(input_dim, hidden_dim)
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.feature_embedding(x)
        x = x.unsqueeze(1)  # [batch_size, input_dim] -> [batch_size, 1, hidden_dim]
        x, (h_n, c_n) = self.lstm(x)  # x has shape [batch_size, seq_len, hidden_dim]
        x = x[:, -1, :]  # [batch_size, hidden_dim]
        x = torch.sigmoid(self.fc_out(x))  # Output between 0 and 1 for binary classification
        return x

model = LSTMModel(42,1)


# Load saved model
model.load_state_dict(torch.load(f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth"))

validate_dl(model=model, validation_loader_list=validation_loader_list, validation=validation,configs=configs, n_inner_steps=1)
