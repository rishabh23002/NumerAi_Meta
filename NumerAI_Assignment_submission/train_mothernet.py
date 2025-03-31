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
from helper import *
from torch.autograd import grad
from torch.nn.utils import weight_norm

# torch.autograd.set_detect_anomaly(True)

Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config_mothernet.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()

# feature_set = json.load(open(f"data/{configs['Dataset']['name']}/features.json"))["feature_sets"][f"{configs['Dataset']['set']}"]

# # Load Train Dataset
# train_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/train.parquet",columns = ["era", "target"] + feature_set)
# # Reduce Dataset size
# # train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"].unique()[::configs['Dataset'][configs['Dataset']['name']]['reduce_dataset_size']]))])
# # train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"].unique()[-300:]))])
# train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"]))])

# # Save last train era information in config file
# configs["Train"]["last_train_era"] = int(train["era"].unique()[-1])
# with open(Arguments.config, "w") as F: yaml.safe_dump(configs,F); F.close()





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
                 data_loader_list=None,
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
        if not data_loader_list == None:
            count=0
            for train_loader, _ in data_loader_list:
                if count>250:
                    break
                count+=1
                self.store_distribution(train_loader)
        

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
        # Store the statistics in a global dictionary with loader_name as key
        loader_name = f"train_loader_{len(self.train_loader_distributions) + 1}"
        self.train_loader_distributions[loader_name] = [stats,train_loader]
    
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

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.net = nn.Sequential(self.conv1, self.relu, self.dropout)

    def forward(self, x):
        return self.net(x)

class TCNModel(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size=3, dropout=0.2):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add a sequence dimension if it doesn't exist
        x = x.permute(0, 2, 1)  # [batch, channels, sequence]
        x = self.tcn(x)
        x = x.mean(dim=2)  # Global average pooling
        return torch.sigmoid(self.fc(x)).squeeze()

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

class WeightGenerator(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(WeightGenerator, self).__init__()
        
        # Define the four linear layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        
        # Output layer to produce 5 weights
        self.fc_out = nn.Linear(hidden_size, 5)

    def forward(self, x):
        # Pass through the four linear layers with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        
        # Output 5 values
        weights = self.fc_out(x)
        
        # Apply softmax to ensure weights sum to 1 for ensemble usage
        weights = torch.tanh(weights)
        
        return weights


class EnsembleModel(nn.Module):
    def __init__(self,MotherNet_output,device, input_dim=42, output_dim=1):
        super(EnsembleModel, self).__init__()
        checkpoint_paths = [
            '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/MLP_DL_MSE_60/MLP.pth',
            '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/TCNModel_DL_MSE_60/TCNModel.pth',
            '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/LSTM_DL_MSE_60/LSTM.pth',
            '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/Transformer_DL_MSE_60/Transformer.pth',
            '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/AutoencoderModel_DL_MSE_60/AutoencoderModel.pth'
        ]
        
        # Load each model and set to evaluation mode
        self.models = nn.ModuleList([
            MLP(input_dim, output_dim),
            TCNModel(input_dim, output_dim, num_channels=[64, 64, 32]),
            LSTMModel(input_dim, output_dim),
            Transformer_DL_Model(input_dim, output_dim),
            AutoencoderModel(input_dim, hidden_dim=64, latent_dim=32, output_dim=output_dim)
        ])
        
        for i, model in enumerate(self.models):
            checkpoint = torch.load(checkpoint_paths[i],weights_only=True)
            model.load_state_dict(checkpoint)
            model.to(device)
            model.eval()  # Set model to evaluation mode
            
        # Define trainable weights for each model's output
        #self.weights = nn.Parameter(torch.ones(len(self.models)), requires_grad=True)

        self.weights = nn.Parameter(MotherNet_output.mean(dim=0), requires_grad=True)

    def forward(self, x):
        with torch.no_grad():
            # Collect outputs from each model
            outputs = [model(x) for model in self.models]
        
        # Stack outputs from each model along a new dimension (num_models, batch_size)
        outputs = torch.stack(outputs, dim=0)
        
        # Ensure self.weights has the same length as num_models
        weights = torch.softmax(self.weights, dim=0)  # Shape: [num_models]
        
        # Reshape weights to match the dimensions of outputs for broadcasting
        weighted_outputs = weights.view(-1, 1) * outputs
        
        # Return the weighted sum of model outputs
        return torch.sum(weighted_outputs, dim=0)


checkpoint_paths = [
    '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/MLP_DL_MSE_60/MLP.pth',
    '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/TCNModel_DL_MSE_60/TCNModel.pth',
    '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/LSTM_DL_MSE_60/LSTM.pth',
    '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/Transformer_DL_MSE_60/Transformer.pth',
    '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/AutoencoderModel_DL_MSE_60/AutoencoderModel.pth'
]


model = WeightGenerator(42)

print('Preparing Data')
model = train_mothernet(model=model,EnsembleModel=EnsembleModel,data_loader_list=prepare_data_loader_list(configs),configs=configs, epochs=600,lr=1e-2)
# Check if directory exist
if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}"):
    os.makedirs(f"saved_models/{configs['Experiment_Name']}")

torch.save(model.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth")


