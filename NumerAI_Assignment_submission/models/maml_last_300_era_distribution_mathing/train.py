import os
import json
import yaml
import torch
import argparse
import pandas as pd
import lightgbm as lgb
from helper import *


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
train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"].unique()[-300:]))])

# Save last train era information in config file
configs["Train"]["last_train_era"] = int(train["era"].unique()[-1])
# with open(Arguments.config, "w") as F: yaml.safe_dump(configs,F); F.close()


# MetaModel class with distribution similarity method
class MetaModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MetaModel, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1 for binary classification
        return x

    def distribution_similarity(self, test_features, train_features):
        """
        Calculate similarity between test and train distributions.
        Here we use cosine similarity as an example.
        """
        test_mean = test_features.mean(dim=0)
        train_mean = train_features.mean(dim=0)
        similarity = torch.nn.functional.cosine_similarity(test_mean.unsqueeze(0), train_mean.unsqueeze(0))
        return similarity.item()

class MAML:
    def __init__(self, model, inner_lr, outer_lr, n_inner_steps):
        self.model = model
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.n_inner_steps = n_inner_steps
        self.meta_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.outer_lr)
        self.training_distributions = {}  # Dictionary to store training distribution summaries

    def inner_update(self, X, y):
        # Create a copy of the model for task-specific adaptation
        adapted_model = MetaModel(self.model.input_dim, self.model.output_dim)
        adapted_model.load_state_dict(self.model.state_dict())
        inner_optimizer = torch.optim.SGD(adapted_model.parameters(), lr=self.inner_lr)
        
        # Perform inner loop adaptation
        for _ in range(self.n_inner_steps):
            predictions = adapted_model(X).squeeze()
            loss = torch.nn.functional.binary_cross_entropy(predictions, y)
            inner_optimizer.zero_grad()
            loss.backward()
            inner_optimizer.step()
        
        return adapted_model

    def store_training_distribution(self, X_train, era):
        """
        Calculate and store the mean feature vector for a training distribution.
        `X_train` is the training features for the specific distribution (or era).
        """
        mean_vector = X_train.mean(dim=0)
        self.training_distributions[era] = mean_vector.detach().cpu()

    def outer_update(self, task_data):
        total_loss = 0

        for X_train, y_train, X_test, y_test, era in task_data:
            # Store distribution summary for each training set (based on "era")
            self.store_training_distribution(X_train, era)

            # Perform inner update (fine-tuning) on the task's training data
            adapted_model = self.inner_update(X_train, y_train)

            # Calculate the meta-loss on the task's test data
            predictions = adapted_model(X_test).squeeze()
            loss = torch.nn.functional.binary_cross_entropy(predictions, y_test)
            total_loss += loss

        # Perform meta-update (outer update)
        self.meta_optimizer.zero_grad()
        total_loss.backward()
        self.meta_optimizer.step()

        return total_loss.item()

    def fine_tune_and_predict(self, test_features):
        """
        Find the closest training distribution to the test data, fine-tune the model on it,
        and return predictions on the test data.
        """
        test_mean = test_features.mean(dim=0)
        
        # Find the most similar training distribution
        closest_era = None
        max_similarity = -float('inf')
        
        for era, train_mean in self.training_distributions.items():
            similarity = torch.nn.functional.cosine_similarity(test_mean.unsqueeze(0), train_mean.unsqueeze(0))
            if similarity > max_similarity:
                max_similarity = similarity
                closest_era = era

        # Retrieve training data for the matched era
        closest_train_data = self.training_distributions[closest_era]
        adapted_model = self.inner_update(closest_train_data['X'], closest_train_data['y'])

        # Make predictions on test data using the fine-tuned model
        with torch.no_grad():
            test_predictions = adapted_model(test_features)

        return test_predictions


model = MetaModel(42,1)
maml = MAML(model, inner_lr=0.01, outer_lr=0.001, n_inner_steps=1)

train_maml(maml=maml,data_loader_list=data_loader_list,epochs=20,n_inner_steps=1)

# Check if directory exist
if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}"):
    os.makedirs(f"saved_models/{configs['Experiment_Name']}")

torch.save(maml.model.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}.pth")


