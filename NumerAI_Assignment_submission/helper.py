import os
import torch
from tqdm import tqdm
import learn2learn as l2l
from datetime import datetime
from matplotlib import pyplot as plt
from numerai_tools.scoring import numerai_corr
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split


#####################################################################################################
#####################################################################################################

import json
import torch
import pandas as pd

def prepare_data_loader_list(configs):
    feature_set = json.load(open(f"data/{configs['Dataset']['name']}/features.json"))["feature_sets"][f"{configs['Dataset']['set']}"]
    train_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/train.parquet", columns=["era", "target"] + feature_set)
    # train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"].unique()[-30:]))])
    train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"]))])

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
    
    return data_loader_list


#####################################################################################################
######################################## Train Function #############################################
#####################################################################################################
#####################################################################################################


def train_maml_D(maml, data_loader_list,configs, epochs=100, fast_adaptation_steps=5, inner_lr=1e-2, outer_lr=5e-3):
    print('Training Started')
    # Check if CUDA is available and set device
    # device = torch.device(f"cuda:{configs['Train']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    
    optimizer = torch.optim.Adam(maml.parameters(), lr=outer_lr)
    loss_fn = torch.nn.MSELoss()  # Loss function
    # loss_fn = torch.nn.CrossEntropyLoss()

    # Check if directory exists
    if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}/graph"):
        os.makedirs(f"saved_models/{configs['Experiment_Name']}/graph")

    # Variables to store cumulative loss values and x-axis labels
    all_losses = []
    x_labels = []
    
    for epoch in tqdm(range(epochs)):
        total_adapt_loss = 0.0  # Accumulate total loss for each epoch

        # Iterate over each task in the data_loader_list
        for train_loader, test_loader in data_loader_list:
            # Clone the model for this specific task
            learner = maml.clone()
            # if epoch == 0:
            #     learner.store_distribution(train_loader)

            # Inner loop adaptation on train data
            for _ in range(fast_adaptation_steps):
                X_train, y_train = next(iter(train_loader))
                # X_train, y_train = X_train.to(device), y_train.to(device)
                train_preds = learner(X_train)
                train_loss = loss_fn(train_preds, y_train)
                learner.adapt(train_loss,allow_unused=True)  # Update the cloned model on task's train data

            # Evaluate adapted model on test data (meta-objective)
            X_test, y_test = next(iter(test_loader))
            # X_test, y_test = X_test.to(device), y_test.to(device)
            test_preds = learner(X_test)
            total_adapt_loss += loss_fn(test_preds, y_test)


        # Outer loop meta-update
        optimizer.zero_grad()
        total_adapt_loss.backward()  # Backward pass on accumulated loss
        optimizer.step()  # Update main model

        torch.cuda.empty_cache()
        avg_adapt_loss = total_adapt_loss / len(data_loader_list)
        # print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_adapt_loss:.5f}')
        all_losses.append(avg_adapt_loss.detach().cpu())
        x_labels.append(f"{epoch}")

        if (epoch%3==0):
            torch.save(maml.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}_{epoch}.pth")
            # Plot cumulative loss values for all epochs
            plt.figure(figsize=(12, 8))
            plt.plot(range(len(all_losses)), all_losses, label="Loss")
            plt.xticks(range(len(all_losses)), x_labels, rotation=90, fontsize=8)
            plt.xlabel("Epoch,Batch Index")
            plt.ylabel("Average Loss")
            plt.title("Loss Plot Over Epochs")
            plt.tight_layout()  # Adjust layout for readability
            plt.savefig(f"saved_models/{configs['Experiment_Name']}/graph/loss_plot_epoch_{epoch}.png")
            plt.close()  # Close the figure to avoid display overlap

    print("Training completed!")
    return maml


def train_mothernet(model,EnsembleModel, data_loader_list,configs, epochs=100, lr=1e-2):
    # Check if CUDA is available and set the device
    device = torch.device(f"cuda:{configs['Train']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)  # Move the model to the device

    # Check if directory exists
    if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}/graph"):
        os.makedirs(f"saved_models/{configs['Experiment_Name']}/graph")

    # Lists to gather all train and test data from each loader
    all_train_data = []
    all_test_data = []

    # Iterate over each task to collect the data
    for train_loader, test_loader in data_loader_list:
        # Collect all training data from train_loader
        for X_train, y_train in train_loader:
            all_train_data.append((X_train.to(torch.float32), y_train.to(torch.float32)))

        # Collect all test data from test_loader
        for X_test, y_test in test_loader:
            all_test_data.append((X_test.to(torch.float32), y_test.to(torch.float32)))

    # Convert lists of data tuples to a single tensor dataset
    train_dataset = TensorDataset(
        torch.cat([x for x, y in all_train_data]), 
        torch.cat([y for x, y in all_train_data])
    )
    test_dataset = TensorDataset(
        torch.cat([x for x, y in all_test_data]), 
        torch.cat([y for x, y in all_test_data])
    )

    # Create combined train_loader and test_loader
    combined_train_loader = DataLoader(
        train_dataset,
        batch_size=configs['Train']['batch_size'], 
        shuffle=configs['Train']['shuffle'],
        drop_last=configs['Train']['drop_last']
    )
    combined_test_loader = DataLoader(
        test_dataset, 
        batch_size=configs['Train']['batch_size'], 
        shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()  # Loss function
    # loss_fn = torch.nn.CrossEntropyLoss()

    # Variables to store cumulative loss values and x-axis labels
    all_losses = []
    x_labels = []

    # Define variables
    batch_fraction = 2  # Average loss every 1/20 of the data
    print('Training Started')
    # Training loop
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        total_batches = len(combined_train_loader)
        interval = max(1, total_batches // batch_fraction)  # Calculate interval based on fraction

        for batch_idx, (X, Y) in enumerate(combined_train_loader):
            # Move data to device
            X, Y = X.to(device), Y.to(device)

            weights = model(X)
            child = EnsembleModel(weights,device)
            predictions = child(X)
            loss = loss_fn(predictions, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Record averaged loss at specified interval
            if (batch_idx + 1) % interval == 0:
                averaged_loss = running_loss / interval
                all_losses.append(averaged_loss)
                x_labels.append(f"{epoch},{batch_idx + 1}")
                running_loss = 0.0  # Reset running loss

        if (epoch%5==0):
            torch.save(model.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}_{epoch}.pth")
            # Plot cumulative loss values for all epochs
            plt.figure(figsize=(12, 8))
            plt.plot(range(len(all_losses)), all_losses, label="Loss")
            plt.xticks(range(len(all_losses)), x_labels, rotation=90, fontsize=8)
            plt.xlabel("Epoch,Batch Index")
            plt.ylabel("Average Loss")
            plt.title("Loss Plot Over Epochs")
            plt.tight_layout()  # Adjust layout for readability
            plt.savefig(f"saved_models/{configs['Experiment_Name']}/graph/loss_plot_epoch_{epoch}.png")
            plt.close()  # Close the figure to avoid display overlap
    
    return model

    


def train_maml(net, data_loader_list,configs, epochs=100, fast_adaptation_steps=5, inner_lr=1e-2, outer_lr=5e-3):
    print('Training Started')
    # Check if CUDA is available and set device
    # device = torch.device(f"cuda:{configs['Train']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    net = net.to(device)  # Move network to device
    
    maml = l2l.algorithms.MAML(net, lr=inner_lr)

    optimizer = torch.optim.Adam(maml.parameters(), lr=outer_lr)
    #loss_fn = torch.nn.MSELoss()  # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Check if directory exists
    if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}/graph"):
        os.makedirs(f"saved_models/{configs['Experiment_Name']}/graph")

    # Variables to store cumulative loss values and x-axis labels
    all_losses = []
    x_labels = []
    
    for epoch in tqdm(range(epochs)):
        total_adapt_loss = 0.0  # Accumulate total loss for each epoch

        # Iterate over each task in the data_loader_list
        for train_loader, test_loader in data_loader_list:
            # Clone the model for this specific task
            learner = maml.clone()

            # Inner loop adaptation on train data
            for _ in range(fast_adaptation_steps):
                X_train, y_train = next(iter(train_loader))
                # X_train, y_train = X_train.to(device), y_train.to(device)
                train_preds = learner(X_train)
                train_loss = loss_fn(train_preds, y_train)
                learner.adapt(train_loss,allow_unused=True)  # Update the cloned model on task's train data

            # Evaluate adapted model on test data (meta-objective)
            X_test, y_test = next(iter(test_loader))
            # X_test, y_test = X_test.to(device), y_test.to(device)
            test_preds = learner(X_test)
            total_adapt_loss += loss_fn(test_preds, y_test)


        # Outer loop meta-update
        optimizer.zero_grad()
        total_adapt_loss.backward()  # Backward pass on accumulated loss
        optimizer.step()  # Update main model

        torch.cuda.empty_cache()
        avg_adapt_loss = total_adapt_loss / len(data_loader_list)
        # print(f'Epoch {epoch+1}/{epochs} - Loss: {avg_adapt_loss:.5f}')
        all_losses.append(avg_adapt_loss.detach().cpu())
        x_labels.append(f"{epoch}")

        if (epoch%3==0):
            torch.save(maml.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}_{epoch}.pth")
            # Plot cumulative loss values for all epochs
            plt.figure(figsize=(12, 8))
            plt.plot(range(len(all_losses)), all_losses, label="Loss")
            plt.xticks(range(len(all_losses)), x_labels, rotation=90, fontsize=8)
            plt.xlabel("Epoch,Batch Index")
            plt.ylabel("Average Loss")
            plt.title("Loss Plot Over Epochs")
            plt.tight_layout()  # Adjust layout for readability
            plt.savefig(f"saved_models/{configs['Experiment_Name']}/graph/loss_plot_epoch_{epoch}.png")
            plt.close()  # Close the figure to avoid display overlap

    print("Training completed!")
    return maml




# # Training function
# def train_maml(maml, data_loader_list, epochs):
#     epoch_losses = []
#     # Check if directory exist
#     if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}/graph"):
#         os.makedirs(f"saved_models/{configs['Experiment_Name']}/graph")
#     for epoch in range(epochs):
#         epoch_loss = 0

#         epoch_loss = maml.outer_update(data_loader_list)

#         # Store the epoch loss and update the plot
#         epoch_losses.append(epoch_loss)
#         if (epoch+1)%30 ==0:
#             torch.save(maml.model.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}_{epoch+1}.pth")
#             print(f"Epoch {epoch + 1}/{epochs}, Meta Loss: {epoch_loss:.4f}")
#             # Plot and save the loss graph after each epoch
#             plt.figure(figsize=(12, 8))
#             plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, linestyle='-', color='b', label="Meta Loss")
#             plt.xlabel("Epoch")
#             plt.ylabel("Meta Loss")
#             plt.title("Meta Loss Across Epochs")
#             plt.legend()
#             # plt.grid(True)
#             plt.savefig(f"saved_models/{configs['Experiment_Name']}/graph/meta_loss_plot_epoch_{epoch + 1}.png")
#             plt.close()


def train_dl(model, data_loader_list, epochs,configs, lr=0.001):
    # Check if CUDA is available and set the device
    device = torch.device(f"cuda:{configs['Train']['gpu_id']}" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)  # Move the model to the device

    # Check if directory exists
    if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}/graph"):
        os.makedirs(f"saved_models/{configs['Experiment_Name']}/graph")

    # Lists to gather all train and test data from each loader
    all_train_data = []
    all_test_data = []

    # Iterate over each task to collect the data
    for train_loader, test_loader in data_loader_list:
        # Collect all training data from train_loader
        for X_train, y_train in train_loader:
            all_train_data.append((X_train.to(torch.float32), y_train.to(torch.float32)))

        # Collect all test data from test_loader
        for X_test, y_test in test_loader:
            all_test_data.append((X_test.to(torch.float32), y_test.to(torch.float32)))

    # Convert lists of data tuples to a single tensor dataset
    train_dataset = TensorDataset(
        torch.cat([x for x, y in all_train_data]), 
        torch.cat([y for x, y in all_train_data])
    )
    test_dataset = TensorDataset(
        torch.cat([x for x, y in all_test_data]), 
        torch.cat([y for x, y in all_test_data])
    )

    # Create combined train_loader and test_loader
    combined_train_loader = DataLoader(
        train_dataset,
        batch_size=configs['Train']['batch_size'], 
        shuffle=configs['Train']['shuffle'],
        drop_last=configs['Train']['drop_last']
    )
    combined_test_loader = DataLoader(
        test_dataset, 
        batch_size=configs['Train']['batch_size'], 
        shuffle=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()  # Loss function
    # loss_fn = torch.nn.CrossEntropyLoss()

    # Variables to store cumulative loss values and x-axis labels
    all_losses = []
    x_labels = []

    # Define variables
    batch_fraction = 20  # Average loss every 1/20 of the data
    print('Training Started')
    # Training loop
    for epoch in tqdm(range(epochs)):
        running_loss = 0.0
        total_batches = len(combined_train_loader)
        interval = max(1, total_batches // batch_fraction)  # Calculate interval based on fraction

        for batch_idx, (X, Y) in enumerate(combined_train_loader):
            # Move data to device
            X, Y = X.to(device), Y.to(device)

            predictions = model(X).squeeze()
            loss = loss_fn(predictions, Y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            running_loss += loss.item()

            # Record averaged loss at specified interval
            if (batch_idx + 1) % interval == 0:
                averaged_loss = running_loss / interval
                all_losses.append(averaged_loss)
                x_labels.append(f"{epoch},{batch_idx + 1}")
                running_loss = 0.0  # Reset running loss

        if (epoch%5==0):
            torch.save(model.state_dict(),f"saved_models/{configs['Experiment_Name']}/{configs['Model']['name']}_{epoch}.pth")
            # Plot cumulative loss values for all epochs
            plt.figure(figsize=(12, 8))
            plt.plot(range(len(all_losses)), all_losses, label="Loss")
            plt.xticks(range(len(all_losses)), x_labels, rotation=90, fontsize=8)
            plt.xlabel("Epoch,Batch Index")
            plt.ylabel("Average Loss")
            plt.title("Loss Plot Over Epochs")
            plt.tight_layout()  # Adjust layout for readability
            plt.savefig(f"saved_models/{configs['Experiment_Name']}/graph/loss_plot_epoch_{epoch}.png")
            plt.close()  # Close the figure to avoid display overlap
    
    return model


# def train_dl(model, data_loader_list, epochs, lr=0.001):

#     # Check if directory exist
#     if not os.path.isdir(f"saved_models/{configs['Experiment_Name']}/graph"):
#         os.makedirs(f"saved_models/{configs['Experiment_Name']}/graph")

#     # Lists to gather all train and test data from each loader
#     all_train_data = []
#     all_test_data = []

#     # Iterate over each task to collect the data
#     for train_loader, test_loader in data_loader_list:
#         # Collect all training data from train_loader
#         for X_train, y_train in train_loader:
#             all_train_data.append((X_train.to(torch.float32), y_train.to(torch.float32)))

#         # Collect all test data from test_loader
#         for X_test, y_test in test_loader:
#             all_test_data.append((X_test.to(torch.float32), y_test.to(torch.float32)))

#     # Convert lists of data tuples to a single tensor dataset
#     train_dataset = TensorDataset(
#         torch.cat([x for x, y in all_train_data]), 
#         torch.cat([y for x, y in all_train_data])
#     )
#     test_dataset = TensorDataset(
#         torch.cat([x for x, y in all_test_data]), 
#         torch.cat([y for x, y in all_test_data])
#     )
#     # Create combined train_loader and test_loader
#     combined_train_loader = DataLoader(train_dataset,
#                                        batch_size=configs['Train']['batch_size'], 
#                                        shuffle=configs['Train']['shuffle'],
#                                        drop_last=configs['Train']['drop_last'],)
#     combined_test_loader = DataLoader(test_dataset, 
#                                       batch_size=configs['Train']['batch_size'], 
#                                       shuffle=False)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     # for epoch in range(epochs):
#     #     for X,Y in combined_train_loader:
#     #         predictions = model(X).squeeze()
#     #         loss = torch.nn.functional.mse_loss(predictions, Y)
#     #         optimizer.zero_grad()
#     #         loss.backward()
#     #         optimizer.step()

#     # Variables to store cumulative loss values and x-axis labels
#     all_losses = []
#     x_labels = []

#     # Define variables
#     batch_fraction = 20  # Average loss every 1/20 of the data

#     # Training loop
#     for epoch in range(epochs):
#         running_loss = 0.0
#         total_batches = len(combined_train_loader)
#         interval = max(1, total_batches // batch_fraction)  # Calculate interval based on fraction

#         for batch_idx, (X, Y) in tqdm(enumerate(combined_train_loader)):
#             predictions = model(X).squeeze()
#             loss = torch.nn.functional.mse_loss(predictions, Y)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # Accumulate loss
#             running_loss += loss.item()

#             # Record averaged loss at specified interval
#             if (batch_idx + 1) % interval == 0:
#                 averaged_loss = running_loss / interval
#                 all_losses.append(averaged_loss)
#                 x_labels.append(f"{epoch + 1},{batch_idx + 1}")
#                 running_loss = 0.0  # Reset running loss

#         # Plot cumulative loss values for all epochs
#         plt.figure(figsize=(12, 8))
#         plt.plot(range(len(all_losses)), all_losses, label="Loss")
#         plt.xticks(range(len(all_losses)), x_labels, rotation=90, fontsize=8)
#         plt.xlabel("Epoch,Batch Index")
#         plt.ylabel("Average Loss")
#         plt.title("Loss Plot Over Epochs")
#         plt.tight_layout()  # Adjust layout for readability
#         plt.savefig(f"saved_models/{configs['Experiment_Name']}/graph/loss_plot_epoch_{epoch + 1}.png")
#         plt.close()  # Close the figure to avoid display overlap
            
#####################################################################################################
######################################## Validate Function ##########################################
#####################################################################################################
#####################################################################################################

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
            predictions = maml(X_test).squeeze()
            predicted_labels = predictions.round()  # Convert output to 0 or 1

            # Collect predictions for analysis
            task_predictions.extend(predictions.detach().cpu().numpy().ravel())

            # Calculate correct predictions for accuracy
            correct_predictions = (predicted_labels == y_test).sum().item()
            total_correct += correct_predictions
            total_samples += y_test.size(0)

        # Calculate accuracy for this task
        accuracy = total_correct / total_samples

        # Store predictions for this era in the validation DataFrame
        validation.loc[validation['era'] == era_name, 'prediction'] = task_predictions

        # Compute the correlation for this era
        era_data = validation[validation['era'] == era_name]
        per_era_corr = numerai_corr(era_data[['prediction']].dropna(), era_data['target'].dropna())
        if int(era_name)%20 ==0:
            print(f"Era {era_name} - Accuracy: {accuracy*100:.4f}")
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
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    metrics_df = pd.DataFrame(all_task_metrics)

    # Plot the metrics for each era
    metrics_df.set_index('Era', inplace=True)
    metrics_df[['Accuracy', 'Mean Corr', 'Sharpe Ratio']].plot(kind='line', figsize=(12, 6), grid=True)
    plt.title('Per-Era Metrics')
    plt.ylabel('Metric Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/Per_Era_Metrics_{current_time}.png")

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



def validate_dl(model, validation_loader_list, validation, configs, n_inner_steps):
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
            predictions = model(X_test).squeeze()
            predicted_labels = predictions.round()  # Convert output to 0 or 1

            # Collect predictions for analysis
            task_predictions.extend(predictions.detach().cpu().numpy().ravel())

            # Calculate correct predictions for accuracy
            correct_predictions = (predicted_labels == y_test).sum().item()
            total_correct += correct_predictions
            total_samples += y_test.size(0)

        # Calculate accuracy for this task
        accuracy = total_correct / total_samples

        # Store predictions for this era in the validation DataFrame
        validation.loc[validation['era'] == era_name, 'prediction'] = task_predictions

        # Compute the correlation for this era
        era_data = validation[validation['era'] == era_name]
        per_era_corr = numerai_corr(era_data[['prediction']].dropna(), era_data['target'].dropna())
        if int(era_name)%20 ==0:
            print(f"Era {era_name} - Accuracy: {accuracy*100:.4f}")
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

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compile all metrics into a DataFrame for easier viewing
    metrics_df = pd.DataFrame(all_task_metrics)

    # Plot the metrics for each era
    metrics_df.set_index('Era', inplace=True)
    metrics_df[['Accuracy', 'Mean Corr', 'Sharpe Ratio']].plot(kind='line', figsize=(12, 6), grid=True)
    plt.title('Per-Era Metrics')
    plt.ylabel('Metric Values')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/Per_Era_Metrics_{current_time}.png")

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