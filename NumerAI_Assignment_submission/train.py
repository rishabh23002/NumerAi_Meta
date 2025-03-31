import os
import copy
import json
import yaml
import torch
import argparse
import traceback
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.autograd import grad
from helper import *

# torch.autograd.set_detect_anomaly(True)

Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config_distribution.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()


def update_module(module, updates=None, memo=None):
    if memo is None:
        memo = {}
    if updates is not None:
        params = list(module.parameters())
        if not len(updates) == len(list(params)):
            msg = 'WARNING:update_module(): Parameters and updates have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(updates)) + ')'
            print(msg)
        for p, g in zip(params, updates):
            p.update = g

    # Update the params
    for param_key in module._parameters:
        p = module._parameters[param_key]
        if p in memo:
            module._parameters[param_key] = memo[p]
        else:
            if p is not None and hasattr(p, 'update') and p.update is not None:
                updated = p + p.update
                p.update = None
                memo[p] = updated
                module._parameters[param_key] = updated

    # Second, handle the buffers if necessary
    for buffer_key in module._buffers:
        buff = module._buffers[buffer_key]
        if buff in memo:
            module._buffers[buffer_key] = memo[buff]
        else:
            if buff is not None and hasattr(buff, 'update') and buff.update is not None:
                updated = buff + buff.update
                buff.update = None
                memo[buff] = updated
                module._buffers[buffer_key] = updated

    # Then, recurse for each submodule
    for module_key in module._modules:
        module._modules[module_key] = update_module(
            module._modules[module_key],
            updates=None,
            memo=memo,
        )

    if hasattr(module, 'flatten_parameters'):
        module._apply(lambda x: x)
    return module

def clone_module(module, memo=None):
    if memo is None:
        memo = {}

    if not isinstance(module, torch.nn.Module):
        return module
    clone = module.__new__(type(module))
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = clone._parameters.copy()
    clone._buffers = clone._buffers.copy()
    clone._modules = clone._modules.copy()

    # Second, re-write all parameters
    if hasattr(clone, '_parameters'):
        for param_key in module._parameters:
            if module._parameters[param_key] is not None:
                param = module._parameters[param_key]
                param_ptr = param.data_ptr
                if param_ptr in memo:
                    clone._parameters[param_key] = memo[param_ptr]
                else:
                    cloned = param.clone()
                    clone._parameters[param_key] = cloned
                    memo[param_ptr] = cloned

    # Third, handle the buffers if necessary
    if hasattr(clone, '_buffers'):
        for buffer_key in module._buffers:
            if clone._buffers[buffer_key] is not None and \
                    clone._buffers[buffer_key].requires_grad:
                buff = module._buffers[buffer_key]
                buff_ptr = buff.data_ptr
                if buff_ptr in memo:
                    clone._buffers[buffer_key] = memo[buff_ptr]
                else:
                    cloned = buff.clone()
                    clone._buffers[buffer_key] = cloned
                    memo[buff_ptr] = cloned

    # Then, recurse for each submodule
    if hasattr(clone, '_modules'):
        for module_key in clone._modules:
            clone._modules[module_key] = clone_module(
                module._modules[module_key],
                memo=memo,
            )

    if hasattr(clone, 'flatten_parameters'):
        clone = clone._apply(lambda x: x)
    return clone

class BaseLearner(nn.Module):
    def __init__(self, module=None):
        super(BaseLearner, self).__init__()
        self.module = module

    def __getattr__(self, attr):
        try:
            return super(BaseLearner, self).__getattr__(attr)
        except AttributeError:
            return getattr(self.__dict__['_modules']['module'], attr)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

def maml_update(model, lr, grads=None):
    if grads is not None:
        params = list(model.parameters())
        if not len(grads) == len(list(params)):
            msg = 'WARNING:maml_update(): Parameters and gradients have different length. ('
            msg += str(len(params)) + ' vs ' + str(len(grads)) + ')'
            print(msg)
        for p, g in zip(params, grads):
            if g is not None:
                p.update = - lr * g
    return update_module(model)

class MAML(BaseLearner):
    def __init__(self,
                 model,
                 lr,
                 first_order=False,
                 allow_unused=None,
                 allow_nograd=False):
        super(MAML, self).__init__()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.allow_nograd = allow_nograd
        if allow_unused is None:
            allow_unused = allow_nograd
        self.allow_unused = allow_unused
        self.train_loader_distributions = {}

    def forward(self, *args, **kwargs):                
        return self.module(*args, **kwargs)

    def adapt(self,
              loss,
              first_order=None,
              allow_unused=None,
              allow_nograd=None):

        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        second_order = not first_order

        if allow_nograd:
            # Compute relevant gradients
            diff_params = [p for p in self.module.parameters() if p.requires_grad]
            grad_params = grad(loss,
                               diff_params,
                               retain_graph=second_order,
                               create_graph=second_order,
                               allow_unused=allow_unused)
            gradients = []
            grad_counter = 0

            # Handles gradients for non-differentiable parameters
            for param in self.module.parameters():
                if param.requires_grad:
                    gradient = grad_params[grad_counter]
                    grad_counter += 1
                else:
                    gradient = None
                gradients.append(gradient)
        else:
            try:
                gradients = grad(loss,
                                 self.module.parameters(),
                                 retain_graph=second_order,
                                 create_graph=second_order,
                                 allow_unused=allow_unused)
            except RuntimeError:
                traceback.print_exc()
                print('learn2learn: Maybe try with allow_nograd=True and/or allow_unused=True ?')

        # Update the module
        self.module = maml_update(self.module, self.lr, gradients)

    def clone(self, first_order=None, allow_unused=None, allow_nograd=None):
        if first_order is None:
            first_order = self.first_order
        if allow_unused is None:
            allow_unused = self.allow_unused
        if allow_nograd is None:
            allow_nograd = self.allow_nograd
        return MAML(clone_module(self.module),
                    lr=self.lr,
                    first_order=first_order,
                    allow_unused=allow_unused,
                    allow_nograd=allow_nograd)
    
    def store_distribution(self, train_loader, num_batches=50):
        """
        Calculate and store the distribution (mean and std_dev) of features in train_loader.
        """
        # Initialize a dictionary to store aggregate statistics
        stats = {
            "mean": [],
            "std_dev": [],
        }
        
        # Accumulate feature data for calculating mean and std_dev
        feature_data = []
        for _ in range(num_batches):
            X_train, _ = next(iter(train_loader))
            feature_data.append(X_train.numpy())
        
        # Convert accumulated data to a single array
        feature_data = np.concatenate(feature_data, axis=0)
        
        # Calculate distribution statistics for each feature
        for i in range(feature_data.shape[1]):  # assuming feature data is (num_samples, num_features)
            feature_column = feature_data[:, i]
            stats["mean"].append(np.mean(feature_column))
            stats["std_dev"].append(np.std(feature_column))
        
        # Initialize lists to collect the data from 5 batches
        X_subset, y_subset = [], []

        # Collect 5 batches of data
        for i, (X_batch, y_batch) in enumerate(train_loader):
            if i == 6:  # Stop after 5 batches
                break
            X_subset.append(X_batch)
            y_subset.append(y_batch)

        # Concatenate the 5 batches into a single tensor for X and y
        X_subset = torch.cat(X_subset, dim=0)
        y_subset = torch.cat(y_subset, dim=0)

        # Create a new dataset and dataloader with the subset data
        subset_dataset = TensorDataset(X_subset, y_subset)
        subset_dataloader = DataLoader(subset_dataset, batch_size=train_loader.batch_size, shuffle=False)

        # Store the statistics in a global dictionary with loader_name as key
        loader_name = f"train_loader_{len(self.train_loader_distributions) + 1}"
        self.train_loader_distributions[loader_name] = [stats,subset_dataloader]
    
    def find_matching_train_loader(self,test_data):
        """
        Compare the distribution of test_loader with stored train_loader distributions
        and return the train_loader whose distribution matches the most.
        """
                
        test_stats = {
            "mean": [],
            "std_dev": [],
        }
        
        # Ensure data is a numpy array
        if isinstance(test_data, torch.Tensor):
            test_data = test_data.numpy()
        
        # Calculate distribution statistics for each feature (assuming data shape is [num_samples, num_features])
        for i in range(test_data.shape[1]):
            feature_column = test_data[:, i]
            test_stats["mean"].append(np.mean(feature_column))
            test_stats["std_dev"].append(np.std(feature_column))
        
        best_match = None
        best_similarity = float('inf')  # Start with a large value, as we want to minimize this
        
        # Compare test distribution with each stored train_loader distribution
        for loader_name, data in self.train_loader_distributions.items():
            train_stats,train_loader = data[0],data[1]
            # Calculate the difference in means and std_devs between test and train distributions
            mean_diff = np.linalg.norm(np.array(test_stats["mean"]) - np.array(train_stats["mean"]))
            std_dev_diff = np.linalg.norm(np.array(test_stats["std_dev"]) - np.array(train_stats["std_dev"]))
            
            # Calculate similarity score (smaller score indicates a better match)
            similarity_score = mean_diff + std_dev_diff
            if similarity_score < best_similarity:
                best_similarity = similarity_score
                best_match = train_loader
        
        return best_match










class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_layers=2, dropout=0.1):
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
        # Project input features to LSTM model dimension
        x = self.feature_embedding(x)

        # Reshape input for LSTM (batch_first=True, so shape should be [batch_size, seq_len, hidden_dim])
        x = x.unsqueeze(1)  # [batch_size, input_dim] -> [batch_size, 1, hidden_dim]

        # LSTM forward pass
        x, (h_n, c_n) = self.lstm(x)  # x has shape [batch_size, seq_len, hidden_dim]

        # Get the output for the last time step (if sequence length = 1, we use the only time step)
        x = x[:, -1, :]  # [batch_size, hidden_dim]

        # Output layer to match target dimension
        x = torch.sigmoid(self.fc_out(x))  # Output between 0 and 1 for binary classification
        return x.squeeze()







maml = MAML(LSTMModel(42,1),lr=1e-2)


print('Preparing Data')
maml = train_maml_D(maml = maml, data_loader_list=prepare_data_loader_list(configs),configs=configs, epochs=600,fast_adaptation_steps=5,inner_lr=1e-2,outer_lr=5e-3)

# train_maml(maml=maml,data_loader_list=data_loader_list,epochs=600)

# Check if directory exist
if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}"):
    os.makedirs(f"saved_models/{configs['Experiment_Name']}")

torch.save(maml.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth")


