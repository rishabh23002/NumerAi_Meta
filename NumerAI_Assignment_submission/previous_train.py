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
Parse.add_argument("--config", type=str, default = 'Assignment1_config.yaml')
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


# class TransformerMetaModel(nn.Module):
#     def __init__(self, input_dim, output_dim, nhead=4, num_layers=4, d_model=64, dim_feedforward=128, dropout=0.1):
#         super(TransformerMetaModel, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.d_model = d_model
#         # Linear layer to project input features to the transformer model dimension (d_model)
#         self.feature_embedding = nn.Linear(input_dim, d_model)
#         # Transformer encoder layer with batch_first=True
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         # Output layer
#         self.fc_out = nn.Linear(d_model, output_dim)
        
#     def forward(self, x):
#         x = self.feature_embedding(x)
#         x = x.unsqueeze(1)  # [batch_size, input_dim] -> [batch_size, 1, d_model]
#         x = self.transformer_encoder(x)  # [batch_size, 1, d_model]
#         x = x.squeeze(1)  # [batch_size, d_model]
#         x = torch.sigmoid(self.fc_out(x))  # Output between 0 and 1 for binary classification
#         return x.squeeze()





# class LSTMModel(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.1):
#         super(LSTMModel, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim

#         # Linear layer to project input features to the LSTM model dimension (hidden_dim)
#         self.feature_embedding = nn.Linear(input_dim, hidden_dim)

#         # LSTM layer
#         self.lstm = nn.LSTM(
#             input_size=hidden_dim,
#             hidden_size=hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=dropout
#         )

#         # Output layer
#         self.fc_out = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # Project input features to LSTM model dimension
#         x = self.feature_embedding(x)

#         # Reshape input for LSTM (batch_first=True, so shape should be [batch_size, seq_len, hidden_dim])
#         x = x.unsqueeze(1)  # [batch_size, input_dim] -> [batch_size, 1, hidden_dim]

#         # LSTM forward pass
#         x, (h_n, c_n) = self.lstm(x)  # x has shape [batch_size, seq_len, hidden_dim]

#         # Get the output for the last time step (if sequence length = 1, we use the only time step)
#         x = x[:, -1, :]  # [batch_size, hidden_dim]

#         # Output layer to match target dimension
#         x = torch.sigmoid(self.fc_out(x))  # Output between 0 and 1 for binary classification
#         return x.squeeze()


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1 for binary classification
        return x.squeeze()


'''
# class MAML:
#     def __init__(self, model, inner_lr, outer_lr):
#         self.model = model
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)

#     def inner_update(self, train_loader):
#         # Create a temporary copy of the model for inner loop updates
#         adapted_model = copy.deepcopy(self.model)
#         inner_optimizer = torch.optim.Adam(adapted_model.parameters(), lr=self.inner_lr)

#         for X, y in train_loader:
#             X, y = X, y
#             predictions = adapted_model(X).squeeze()
#             loss = torch.nn.functional.mse_loss(predictions, y)
#             inner_optimizer.zero_grad()
#             loss.backward()
#             inner_optimizer.step()
        
#         return adapted_model

#     def outer_update(self, data_loader_list):
#         task_losses = []  # List to accumulate losses for each task

#         for task_idx, (train_loader, test_loader) in enumerate(data_loader_list):
#             # Get an adapted model for the task via inner loop updates
#             adapted_model = self.inner_update(train_loader)

#             # Initialize task loss as a tensor on the device
#             task_loss = torch.tensor(0.0,)

#             for X_test, y_test in test_loader:
#                 X_test, y_test = X_test, y_test
#                 predictions = adapted_model(X_test).squeeze()
#                 loss = torch.nn.functional.mse_loss(predictions, y_test)
#                 task_loss = task_loss + (loss / len(test_loader))  # Avoid in-place modifications

#             task_losses.append(task_loss)  # Store each task's final loss

#         # Combine all task losses for the outer update
#         total_loss_epoch = torch.stack(task_losses).sum()  # Sum the losses to keep gradients

#         # Perform meta-update (outer update)
#         self.meta_optimizer.zero_grad()
#         total_loss_epoch.backward()
#         self.meta_optimizer.step()

#         return total_loss_epoch.item()


# class MAML:
#     def __init__(self, model, inner_lr, outer_lr, n_inner_steps):
#         self.model = model
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.n_inner_steps = n_inner_steps
#         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
#         # self.inner_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.inner_lr)
#         self.training_distributions = {}  # Dictionary to store training distribution summaries

#     def inner_update(self, train_loader):
#         inner_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.inner_lr)
#         # Perform inner loop adaptation
#         for X, y in train_loader:
#             predictions = self.model(X).squeeze()
#             # loss = torch.nn.functional.binary_cross_entropy(predictions, y)
#             loss = torch.nn.functional.mse_loss(predictions, y)
#             inner_optimizer.zero_grad()
#             loss.backward()
#             inner_optimizer.step()
        
#         return self.model

#     def outer_update(self, data_loader_list):
#         task_losses = []

#         # for X_train, y_train, X_test, y_test, era in task_data:
#         for task_idx, (train_loader, test_loader) in enumerate(data_loader_list):
#             # Perform inner update (fine-tuning) on the task's training data
#             adapted_model = self.inner_update(train_loader)

#             total_loss_task = torch.tensor(0.0)
#             for X_test, y_test in test_loader:
#                 # Calculate the meta-loss on the task's test data
#                 predictions = adapted_model(X_test).squeeze()
#                 # loss = torch.nn.functional.binary_cross_entropy(predictions, y_test)
#                 loss = torch.nn.functional.mse_loss(predictions, y_test)
#                 total_loss_task += loss
#             task_losses.append(total_loss_task)
        
#         total_loss_epoch = torch.stack(task_losses).sum()  # Sum to retain gradients

#         # Perform meta-update (outer update)
#         self.meta_optimizer.zero_grad()
#         total_loss_epoch.backward()
#         self.meta_optimizer.step()

#         return total_loss_epoch.item()
'''


#model = LSTMModel(42,1)
#model = TransformerMetaModel(42,1)
model = MLP(42,1)
#output = model(dummy_input)
#print(f"Output shape: {output.shape}")  # Should be [batch_size, output_dim]

# model = TransformerMetaModel(42,1)
# maml = MAML(model, inner_lr=0.001, outer_lr=0.0001)

print('Preparing Data')
maml = train_maml(net = model, data_loader_list=prepare_data_loader_list(configs),configs=configs, epochs=60,fast_adaptation_steps=5,inner_lr=1e-2,outer_lr=5e-3)

# train_maml(maml=maml,data_loader_list=data_loader_list,epochs=600)

# Check if directory exist
if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}"):
    os.makedirs(f"saved_models/{configs['Experiment_Name']}")

torch.save(maml.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth")


