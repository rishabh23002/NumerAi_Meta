import os
import time
import torch
import numpy as np
from torch import nn
from torch import optim
from IPython import display
from datetime import datetime
from matplotlib import pyplot as plt
from numerai_tools.scoring import numerai_corr
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split



#####################################################################################################
#####################################################################################################

import json
import yaml
import torch
import argparse
import pandas as pd


Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()

feature_set = json.load(open(f"data/{configs['Dataset']['name']}/features.json"))["feature_sets"][f"{configs['Dataset']['set']}"]

# Load Train Dataset
train_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/train.parquet",columns = ["era", "target"] + feature_set)

# Reduce Dataset size
train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"].unique()[::configs['Dataset'][configs['Dataset']['name']]['reduce_dataset_size']]))])

data_loader_list = []
for era in sorted(train['era'].unique()):
    era_df = pd.DataFrame(train[train['era']==era])

    X = era_df[feature_set].values  # Features
    y = era_df['target'].values  # Target 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, 
                              batch_size=configs['Train']['batch_size'], 
                              shuffle=configs['Train']['shuffle'],
                              drop_last=configs['Train']['drop_last'],
                              )
    test_loader = DataLoader(test_dataset, 
                             batch_size=configs['Train']['batch_size'], 
                             shuffle=False
                             )

    data_loader_list.append([train_loader,test_loader])


#####################################################################################################
######################################## Meta Model ################################################
#####################################################################################################
#####################################################################################################

# # Define a simple feedforward neural network
# class MetaModel(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(MetaModel, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.fc1 = nn.Linear(input_dim, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_dim)

#     def forward(self, x):
#         x = torch.relu(self.fc1(x))
#         x = torch.relu(self.fc2(x))
#         return torch.sigmoid(self.fc3(x))

# # Meta-learning wrapper for MAML
# class MAML:
#     def __init__(self, model, inner_lr, outer_lr, n_inner_steps):
#         self.model = model
#         self.inner_lr = inner_lr
#         self.outer_lr = outer_lr
#         self.n_inner_steps = n_inner_steps
#         self.optimizer = optim.Adam(self.model.parameters(), lr=self.outer_lr)

#     def inner_update(self, X, y):
#         # Copy the current model to a new inner model for task-specific adaptation
#         inner_model = MetaModel(self.model.input_dim, self.model.output_dim)
#         inner_model.load_state_dict(self.model.state_dict())

#         # Use an inner optimizer
#         inner_optimizer = optim.SGD(inner_model.parameters(), lr=self.inner_lr)
#         loss_fn = nn.MSELoss()

#         for step in range(self.n_inner_steps):
#             predictions = inner_model(X)
#             loss = loss_fn(predictions.squeeze(), y)
#             inner_optimizer.zero_grad()
#             loss.backward()
#             inner_optimizer.step()

#         # Return the adapted state dict
#         return inner_model.state_dict()

#     def outer_update(self, task_data):
#         total_loss = 0
#         loss_fn = nn.MSELoss()

#         # Accumulate gradients across tasks for meta-update
#         for (X_train, y_train, X_test, y_test) in task_data:
#             # Perform inner updates (task-specific adaptation)
#             adapted_params = self.inner_update(X_train, y_train)

#             # Temporarily load the adapted model weights into a copy of the meta-model
#             adapted_model = MetaModel(self.model.input_dim, self.model.output_dim)
#             adapted_model.load_state_dict(adapted_params)

#             # Compute the meta-loss using the meta-model (before adapting it)
#             predictions = adapted_model(X_test)
#             loss = loss_fn(predictions.squeeze(), y_test)
#             total_loss += loss

#         # Perform meta-learner update (outer update)
#         self.optimizer.zero_grad()
#         total_loss.backward()
#         self.optimizer.step()

#         return total_loss.item()


#####################################################################################################
######################################## Train Function #############################################
#####################################################################################################
#####################################################################################################

# Training function
def train_maml(maml, data_loader_list, epochs, n_inner_steps, inner_lr=0.01, outer_lr=0.001):
    for epoch in range(epochs):
        epoch_loss = 0

        # Iterate over each task's data (each task is a pair of train and test loaders)
        for task_idx, (train_loader, test_loader) in enumerate(data_loader_list):
            # Prepare task-specific data
            task_data = []

            # Gather all train data for inner loop adaptation
            X_train, y_train = next(iter(train_loader))
            X_train, y_train = X_train.to(torch.float32), y_train.to(torch.float32)

            # Gather all test data for outer loop meta-update
            X_test, y_test = next(iter(test_loader))
            X_test, y_test = X_test.to(torch.float32), y_test.to(torch.float32)

            # Append task data with an identifier for the task/era
            era = f"era_{task_idx}"  # Assign an identifier for each task (e.g., task index)
            task_data.append((X_train, y_train, X_test, y_test, era))

            # Perform outer update (meta-update) on the current task data
            task_loss = maml.outer_update(task_data)
            epoch_loss += task_loss  # Accumulate loss for the current epoch

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{epochs}, Meta Loss: {epoch_loss:.4f}")





#####################################################################################################
######################################## Validate Function ##########################################
#####################################################################################################
#####################################################################################################

# def validate_maml(maml, validation_loader_list, validation, configs, n_inner_steps):
#     total_correct = 0
#     total_samples = 0
#     all_predictions = []

#     for task_idx, val_test_loader in enumerate(validation_loader_list):
#         task_predictions = []
        
#         for X_test, y_test in val_test_loader:
#             X_test, y_test = X_test.to(torch.float32), y_test.to(torch.float32)

#             # Directly evaluate the meta-model without adaptation
#             predictions = maml.model(X_test).squeeze()
#             predicted_labels = predictions.round()  # Convert output to 0 or 1

#             # Collect predictions for out-of-sample analysis
#             task_predictions.extend(predictions.detach().cpu().numpy().ravel())

#             # Calculate correct predictions for accuracy
#             correct_predictions = (predicted_labels == y_test).sum().item()
#             total_correct += correct_predictions
#             total_samples += y_test.size(0)

#         all_predictions.extend(task_predictions)

#     # Calculate final accuracy
#     accuracy = total_correct / total_samples
#     print(f"Validation Accuracy: {accuracy:.4f}")

#     # Generate predictions against the out-of-sample validation features
#     validation["prediction"] = all_predictions

#     # Compute the per-era correlation between predictions and target values
#     per_era_corr = validation.groupby("era").apply(
#         lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
#     )

#     # Check if the directory exists; if not, create it
#     results_dir = f"Results/{configs['Experiment_Name']}"
#     if not os.path.isdir(results_dir):
#         os.makedirs(results_dir)

#     # Plot the per-era correlation
#     plt.figure(figsize=(12, 6))
#     plt.bar(per_era_corr.index, per_era_corr.values, label="Correlation")
#     plt.title("Validation CORR: Prediction vs Target", fontsize=14)
#     plt.xlabel("Era", fontsize=12)
#     plt.ylabel("Correlation Value", fontsize=12)
#     plt.xticks(rotation=90, fontsize=10)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.savefig(f"{results_dir}/Validation_CORR.png")

#     # Compute performance metrics
#     corr_mean = per_era_corr.mean()
#     corr_std = per_era_corr.std(ddof=0)
#     corr_sharpe = corr_mean / corr_std
#     corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()

#     # Create a dictionary to store the values for plotting
#     metrics = {
#         'Mean': corr_mean.item(),
#         'Std Dev': corr_std.item(),
#         'Sharpe Ratio': corr_sharpe.item(),
#         'Max Drawdown': corr_max_drawdown.item()
#     }

#     # Plot the performance metrics
#     plt.figure(figsize=(12, 6))
#     plt.bar(list(metrics.keys()), list(metrics.values()), color=['skyblue', 'orange', 'green', 'red'])
#     plt.title('Performance Metrics', fontsize=14)
#     plt.ylabel('Metric Values (Scaled)', fontsize=12)
#     plt.tight_layout()
#     plt.savefig(f"{results_dir}/Validation_performance_metrics.png")

#     # Save metrics to a file
#     with open(f"{results_dir}/board.txt", 'a') as file:
#         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         dash = "-" * 100
#         experiment_name = f" {configs['Experiment_Name']} "
#         date_and_time = f" Date and Time: {current_time} "
        
#         file.write(dash.center(120) + "\n")
#         file.write(dash.center(120) + "\n")
#         file.write(experiment_name.center(120) + "\n")
#         file.write(date_and_time.center(120) + "\n")
#         file.write(dash.center(120) + "\n")
#         file.write("\t\t\tMetrics:\n")
#         file.write(f"\t\t\tMean: {metrics['Mean']:.4f}" + "\t" * 7 + f"Std Dev: {metrics['Std Dev']:.4f}\n")
#         file.write(f"\t\t\tSharpe Ratio: {metrics['Sharpe Ratio']:.4f}" + "\t" * 5 + f"Max Drawdown: {metrics['Max Drawdown']:.4f}\n\n\n")



def validate_maml(maml, validation_loader_list, validation, configs, n_inner_steps):
    all_task_metrics = []  # To store metrics for each task
    results_dir = f"Results/{configs['Experiment_Name']}"
    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    for task_idx, (val_test_loader, era_name) in enumerate(zip(validation_loader_list, validation['era'].unique())):
        total_correct = 0
        total_samples = 0
        task_predictions = []

        for X_test, y_test in val_test_loader:
            X_test, y_test = X_test.to(torch.float32), y_test.to(torch.float32)

            # Directly evaluate the meta-model without adaptation
            predictions = maml.model(X_test).squeeze()
            predicted_labels = predictions.round()  # Convert output to 0 or 1

            # Collect predictions for analysis
            task_predictions.extend(predictions.detach().cpu().numpy().ravel())

            # Calculate correct predictions for accuracy
            correct_predictions = (predicted_labels == y_test).sum().item()
            total_correct += correct_predictions
            total_samples += y_test.size(0)

        # Calculate accuracy for this task
        accuracy = total_correct / total_samples
        print(f"Era {era_name} - Accuracy: {accuracy*100:.4f}")

        # Store predictions for this era in the validation DataFrame
        validation.loc[validation['era'] == era_name, 'prediction'] = task_predictions

        # Compute the correlation for this era
        era_data = validation[validation['era'] == era_name]
        per_era_corr = numerai_corr(era_data[['prediction']].dropna(), era_data['target'].dropna())
        print(f"Era {era_name} - Correlation: {per_era_corr.iloc[0]:.4f}")

        # Compute additional metrics
        corr_mean = per_era_corr.mean()
        corr_std = per_era_corr.std(ddof=0)
        corr_sharpe = corr_mean / corr_std if corr_std != 0 else float('nan')
        corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()

        # Log metrics for this task
        task_metrics = {
            'Era': era_name,
            'Accuracy': accuracy,
            'Mean Corr': corr_mean.item(),
            'Std Dev': corr_std.item(),
            'Sharpe Ratio': corr_sharpe,
            'Max Drawdown': corr_max_drawdown.item()
        }
        all_task_metrics.append(task_metrics)

        # # Plot per-era correlation for this task
        # plt.figure(figsize=(6, 4))
        # plt.bar([era_name], [per_era_corr[0]], color='skyblue', label="Correlation")
        # plt.title(f"Era {era_name} - Prediction vs Target Correlation", fontsize=12)
        # plt.ylabel("Correlation Value", fontsize=10)
        # plt.tight_layout()
        # plt.savefig(f"{results_dir}/Validation_CORR_Era_{era_name}.png")
        # plt.close()

    # Compile all metrics into a DataFrame for easier viewing
    metrics_df = pd.DataFrame(all_task_metrics)

    # Plot the metrics for each era
    metrics_df.set_index('Era', inplace=True)
    metrics_df[['Accuracy', 'Mean Corr', 'Sharpe Ratio']].plot(kind='bar', figsize=(12, 6), grid=True)
    plt.title('Per-Era Metrics')
    plt.ylabel('Metric Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/Per_Era_Metrics.png")

    # Save metrics to a file for logging
    with open(f"{results_dir}/board.txt", 'a') as file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dash = "-" * 100
        file.write(dash.center(120) + "\n")
        file.write(f"Validation Metrics for Experiment: {configs['Experiment_Name']} at {current_time}\n")
        file.write(dash.center(120) + "\n")
        for task_metrics in all_task_metrics:
            file.write(f"Era {task_metrics['Era']}: Accuracy = {task_metrics['Accuracy']*100:.4f}, "
                       f"Mean Corr = {task_metrics['Mean Corr']:.4f}, Std Dev = {task_metrics['Std Dev']:.4f}, "
                       f"Sharpe Ratio = {task_metrics['Sharpe Ratio']:.4f}, "
                       f"Max Drawdown = {task_metrics['Max Drawdown']:.4f}\n")
        file.write("\n\n")