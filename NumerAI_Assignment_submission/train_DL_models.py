import os
import json
import yaml
import torch
import argparse
import torch.nn as nn
import pandas as pd
from helper import *
from torch.nn.utils import weight_norm

Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()





# class TemporalBlock(nn.Module):
#     def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
#         super(TemporalBlock, self).__init__()
#         self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
#         self.dropout = nn.Dropout(dropout)
#         self.relu = nn.ReLU()
        
#         self.net = nn.Sequential(self.conv1, self.relu, self.dropout)

#     def forward(self, x):
#         return self.net(x)

# class TCNModel(nn.Module):
#     def __init__(self, input_dim, output_dim, num_channels, kernel_size=3, dropout=0.2):
#         super(TCNModel, self).__init__()
#         layers = []
#         num_levels = len(num_channels)
#         for i in range(num_levels):
#             dilation_size = 2 ** i
#             in_channels = input_dim if i == 0 else num_channels[i-1]
#             out_channels = num_channels[i]
#             layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
#         self.tcn = nn.Sequential(*layers)
#         self.fc = nn.Linear(num_channels[-1], output_dim)

#     def forward(self, x):
#         if x.dim() == 2:
#             x = x.unsqueeze(1)  # Add a sequence dimension if it doesn't exist
#         x = x.permute(0, 2, 1)  # [batch, channels, sequence]
#         x = self.tcn(x)
#         x = x.mean(dim=2)  # Global average pooling
#         return torch.sigmoid(self.fc(x)).squeeze()

# model = TCNModel(input_dim=42, output_dim=1, num_channels=[64, 64, 32])


class AutoencoderModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        super(AutoencoderModel, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        # Prediction layer
        self.fc = nn.Linear(latent_dim, output_dim)

    def forward(self, x):
        latent = self.encoder(x)  # Encode to latent space
        _ = self.decoder(latent)  # Decoder output (can ignore during prediction)
        x = self.fc(latent)  # Final output for prediction
        return torch.sigmoid(x).squeeze()

model = AutoencoderModel(input_dim=42, hidden_dim=64, latent_dim=32, output_dim=1)




print('Preparing Data')
model = train_dl(model=model,data_loader_list=prepare_data_loader_list(configs=configs),configs=configs,epochs=60)


# Check if directory exist
if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}"):
    os.makedirs(f"saved_models/{configs['Experiment_Name']}")

torch.save(model.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth")

