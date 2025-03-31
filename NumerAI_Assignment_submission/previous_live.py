import json
import yaml
import torch
import argparse
import cloudpickle
import pandas as pd
from torch import nn
from torch import optim


Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Live_config.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()

feature_set = json.load(open(f"data/{configs['Dataset']['name']}/features.json"))["feature_sets"][f"{configs['Dataset']['set']}"]


live_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/live.parquet",columns = feature_set)


import traceback
from torch.autograd import grad

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

# class TransformerMetaModel(nn.Module):
#     def __init__(self, input_dim, output_dim, nhead=4, num_layers=2, d_model=64, dim_feedforward=128, dropout=0.1):
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
#         # Project input features to transformer model dimension
#         x = self.feature_embedding(x)

#         # Reshape for transformer (required shape with batch_first=True: [batch_size, seq_len, d_model])
#         x = x.unsqueeze(1)  # [batch_size, input_dim] -> [batch_size, 1, d_model]

#         # Transformer encoder
#         x = self.transformer_encoder(x)  # [batch_size, 1, d_model]

#         # Take the output for the "sequence" position (seq_len = 1 in this case)
#         x = x.squeeze(1)  # [batch_size, d_model]

#         # Output layer to match target dimension
#         x = torch.sigmoid(self.fc_out(x))  # Output between 0 and 1 for binary classification
#         return x

#     def distribution_similarity(self, test_features, train_features):
#         """
#         Calculate similarity between test and train distributions.
#         Here we use cosine similarity as an example.
#         """
#         test_mean = test_features.mean(dim=0)
#         train_mean = train_features.mean(dim=0)
#         similarity = torch.nn.functional.cosine_similarity(test_mean.unsqueeze(0), train_mean.unsqueeze(0))
#         return similarity.item()

# class MAML:
#     def __init__(self, model, inner_lr, outer_lr, n_inner_steps):
#         self.model = model
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.n_inner_steps = n_inner_steps
#         self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
#         self.training_distributions = {}  # Dictionary to store training distribution summaries

#     def inner_update(self, X, y):
#         inner_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr)
        
#         # Perform inner loop adaptation
#         for _ in range(self.n_inner_steps):
#             predictions = self.model(X).squeeze()
#             # loss = torch.nn.functional.binary_cross_entropy(predictions, y)
#             loss = torch.nn.functional.mse_loss(predictions, y)
#             inner_optimizer.zero_grad()
#             loss.backward()
#             inner_optimizer.step()
        
#         return self.model

#     def store_training_distribution(self, X_train, era):
#         """
#         Calculate and store the mean feature vector for a training distribution.
#         `X_train` is the training features for the specific distribution (or era).
#         """
#         mean_vector = X_train.mean(dim=0)
#         self.training_distributions[era] = mean_vector.detach().cpu()

#     def outer_update(self, task_data):
#         total_loss = 0

#         for X_train, y_train, X_test, y_test, era in task_data:
#             # Store distribution summary for each training set (based on "era")
#             self.store_training_distribution(X_train, era)

#             # Perform inner update (fine-tuning) on the task's training data
#             adapted_model = self.inner_update(X_train, y_train)

#             # Calculate the meta-loss on the task's test data
#             predictions = adapted_model(X_test).squeeze()
#             # loss = torch.nn.functional.binary_cross_entropy(predictions, y_test)
#             loss = torch.nn.functional.mse_loss(predictions, y_test)
#             total_loss += loss

#         # Perform meta-update (outer update)
#         self.meta_optimizer.zero_grad()
#         total_loss.backward()
#         self.meta_optimizer.step()

#         return total_loss.item()

#     def fine_tune_and_predict(self, test_features):
#         """
#         Find the closest training distribution to the test data, fine-tune the model on it,
#         and return predictions on the test data.
#         """
#         test_mean = test_features.mean(dim=0)
        
#         # Find the most similar training distribution
#         closest_era = None
#         max_similarity = -float('inf')
        
#         for era, train_mean in self.training_distributions.items():
#             similarity = torch.nn.functional.cosine_similarity(test_mean.unsqueeze(0), train_mean.unsqueeze(0))
#             if similarity > max_similarity:
#                 max_similarity = similarity
#                 closest_era = era

#         # Retrieve training data for the matched era
#         closest_train_data = self.training_distributions[closest_era]
#         adapted_model = self.inner_update(closest_train_data['X'], closest_train_data['y'])

#         # Make predictions on test data using the fine-tuned model
#         with torch.no_grad():
#             test_predictions = adapted_model(test_features)

#         return test_predictions




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
#         return x

# model = TransformerMetaModel(42,1)

# model = TransformerMetaModel(42,1)
# maml = MAML(model, inner_lr=0.001, outer_lr=0.0001, n_inner_steps=2)



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

# class MLP(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MLP, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1 for binary classification
#         return x.squeeze()

#model = LSTMModel(42,1)
#model = MLP(42,1)

#model = TransformerMetaModel(42,1)



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

maml = MAML(model, lr=1e-2)

# Load saved model
maml.load_state_dict(torch.load(f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth",weights_only=True))


# Define your prediction pipeline as a function
def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    live_predictions = maml(torch.tensor(live_features[feature_set].values, dtype=torch.float32)).squeeze()
    submission = pd.Series(live_predictions.detach().numpy(), index=live_features.index)
    return submission.to_frame("prediction")


# # Save the function with dill
# with open("hello_numerai.pkl", "wb") as f:
#     dill.dump(predict, f)

p = cloudpickle.dumps(predict)
with open(f"saved_models/{configs['Experiment_Name']}/live_{configs['Model']['name']}.pkl", "wb") as f:
    f.write(p)
