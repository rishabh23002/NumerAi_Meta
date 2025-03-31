import json
import yaml
import torch
import argparse
import cloudpickle
import pandas as pd
from torch import nn
from torch import optim
from torch.nn.utils import weight_norm



Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config_mothernet.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()

feature_set = json.load(open(f"data/{configs['Dataset']['name']}/features.json"))["feature_sets"][f"{configs['Dataset']['set']}"]


live_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/live.parquet",columns = feature_set)






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
    def __init__(self,MotherNet_output,a,b,c,d,e, input_dim=42, output_dim=1):
        super(EnsembleModel, self).__init__()
        # checkpoint_paths = [
        #     '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/MLP_DL_MSE_60/MLP.pth',
        #     '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/TCNModel_DL_MSE_60/TCNModel.pth',
        #     '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/LSTM_DL_MSE_60/LSTM.pth',
        #     '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/Transformer_DL_MSE_60/Transformer.pth',
        #     '/home/hiddensand/BARNEET_MT23028/ARCHIVES/Meta_Learning/NumerAI_Assignment/saved_models/AutoencoderModel_DL_MSE_60/AutoencoderModel.pth'
        # ]
        
        self.models = nn.ModuleList([a,b,c,d,e])
        
        
        # Load each model and set to evaluation mode
        # self.models = nn.ModuleList([
        #     MLP(input_dim, output_dim),
        #     TCNModel(input_dim, output_dim, num_channels=[64, 64, 32]),
        #     LSTMModel(input_dim, output_dim),
        #     Transformer_DL_Model(input_dim, output_dim),
        #     AutoencoderModel(input_dim, hidden_dim=64, latent_dim=32, output_dim=output_dim)
        # ])
        
        # for i, model in enumerate(self.models):
        #     checkpoint = torch.load(checkpoint_paths[i],weights_only=True)
        #     model.load_state_dict(checkpoint)
        #     model.to(device)
        #     model.eval()  # Set model to evaluation mode
            
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


a=MLP(42, 1)
b=TCNModel(42, 1, num_channels=[64, 64, 32])
c=LSTMModel(42, 1)
d=Transformer_DL_Model(42, 1)
e=AutoencoderModel(42, hidden_dim=64, latent_dim=32, output_dim=1)

a.load_state_dict(torch.load(checkpoint_paths[0],weights_only=True))
b.load_state_dict(torch.load(checkpoint_paths[1],weights_only=True))
c.load_state_dict(torch.load(checkpoint_paths[2],weights_only=True))
d.load_state_dict(torch.load(checkpoint_paths[3],weights_only=True))
e.load_state_dict(torch.load(checkpoint_paths[4],weights_only=True))

a.to(torch.device('cpu'))
b.to(torch.device('cpu'))
c.to(torch.device('cpu'))
d.to(torch.device('cpu'))
e.to(torch.device('cpu'))

a.eval()
b.eval()
c.eval()
d.eval()
e.eval()

model = WeightGenerator(42)

# Load saved model
model.load_state_dict(torch.load(f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth",weights_only=True))


# Define your prediction pipeline as a function
def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    weights = model(torch.tensor(live_features[feature_set].values, dtype=torch.float32))
    live_predictions = EnsembleModel(MotherNet_output=weights,a=a,b=b,c=c,d=d,e=e)(torch.tensor(live_features[feature_set].values, dtype=torch.float32))
    submission = pd.Series(live_predictions.detach().numpy(), index=live_features.index)
    return submission.to_frame("prediction")


# # Save the function with dill
# with open("hello_numerai.pkl", "wb") as f:
#     dill.dump(predict, f)

p = cloudpickle.dumps(predict)
with open(f"saved_models/{configs['Experiment_Name']}/live_{configs['Model']['name']}.pkl", "wb") as f:
    f.write(p)
