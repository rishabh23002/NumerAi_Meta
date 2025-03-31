import os
import copy
import json
import yaml
import torch
import argparse
import pandas as pd
import torch.nn as nn
from helper import *

# torch.autograd.set_detect_anomaly(True)

Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config_stacking.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()

feature_set = json.load(open(f"data/{configs['Dataset']['name']}/features.json"))["feature_sets"][f"{configs['Dataset']['set']}"]

# Load Train Dataset
train_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/train.parquet",columns = ["era", "target"] + feature_set)
# Reduce Dataset size
# train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"].unique()[::configs['Dataset'][configs['Dataset']['name']]['reduce_dataset_size']]))])
# train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"].unique()[-300:]))])
train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"]))])

# Save last train era information in config file
configs["Train"]["last_train_era"] = int(train["era"].unique()[-1])
# with open(Arguments.config, "w") as F: yaml.safe_dump(configs,F); F.close()


class Transformer_DL_Model(nn.Module):
    def __init__(self, input_dim, output_dim, nhead=4, num_layers=4, d_model=64, dim_feedforward=128, dropout=0.1):
        super(Transformer_DL_Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        # Linear layer to project input features to the transformer model dimension (d_model)
        self.feature_embedding = nn.Linear(input_dim, d_model)
        # Transformer encoder layer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        x = self.feature_embedding(x)
        x = x.unsqueeze(1)  # [batch_size, input_dim] -> [batch_size, 1, d_model]
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        x = x.squeeze(1)  # [batch_size, d_model]
        x = torch.sigmoid(self.fc_out(x))  # Output between 0 and 1 for binary classification
        return x.squeeze()

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
        return x.squeeze()


class Transformer_Meta_Model(nn.Module):
    def __init__(self,
                 Transformer_DL_checkpoints_path,
                 LSTM_DL_checkpoints_path,
                 input_dim, 
                 output_dim, 
                 nhead=4, 
                 num_layers=4, 
                 d_model=64, 
                 dim_feedforward=128, 
                 dropout=0.1):
        super(Transformer_Meta_Model, self).__init__()
        
        self.Transformer_DL = Transformer_DL_Model(input_dim-2,1)
        self.LSTM_DL = LSTMModel(input_dim-2,1)
        self.Transformer_DL.load_state_dict(torch.load(Transformer_DL_checkpoints_path,weights_only=True))
        self.LSTM_DL.load_state_dict(torch.load(LSTM_DL_checkpoints_path,weights_only=True))
        self.Transformer_DL.eval()
        self.LSTM_DL.eval()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        # Linear layer to project input features to the transformer model dimension (d_model)
        self.feature_embedding = nn.Linear(input_dim, d_model)
        # Transformer encoder layer with batch_first=True
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Output layer
        self.fc_out = nn.Linear(d_model, output_dim)
        
    def forward(self, x):
        with torch.no_grad():
            # Get outputs from Transformer_DL and LSTM_DL models
            transformer_out = self.Transformer_DL(x)
            lstm_out = self.LSTM_DL(x)
        
        # Concatenate original input with Transformer_DL and LSTM_DL outputs
        x = torch.cat([x, transformer_out.unsqueeze(1), lstm_out.unsqueeze(1)], dim=1)
        
        # Pass the concatenated input through the feature embedding layer
        x = self.feature_embedding(x)
        x = x.unsqueeze(1)  # [batch_size, input_dim] -> [batch_size, 1, d_model]
        x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
        x = x.squeeze(1)  # [batch_size, d_model]
        x = torch.sigmoid(self.fc_out(x))  # Output between 0 and 1 for binary classification
        return x.squeeze()

model = Transformer_Meta_Model(Transformer_DL_checkpoints_path='/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/Transformer_DL_MSE_60/Transformer.pth',
                               LSTM_DL_checkpoints_path='/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/LSTM_DL_MSE_60/LSTM.pth',
                               input_dim=44,
                               output_dim=1)

#model = LSTMModel(42,1)
#model = TransformerMetaModel(42,1)
# model = MLP(42,1)
#output = model(dummy_input)
#print(f"Output shape: {output.shape}")  # Should be [batch_size, output_dim]

# model = TransformerMetaModel(42,1)
# maml = MAML(model, inner_lr=0.001, outer_lr=0.0001)

print('Preparing Data')
maml = train_maml(net = model, data_loader_list=prepare_data_loader_list(configs),configs=configs, epochs=30,fast_adaptation_steps=5,inner_lr=1e-2,outer_lr=5e-3)

# train_maml(maml=maml,data_loader_list=data_loader_list,epochs=600)

# Check if directory exist
if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}"):
    os.makedirs(f"saved_models/{configs['Experiment_Name']}")

torch.save(maml.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth")


