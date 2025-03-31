import os
import copy
import json
import yaml
import torch
import argparse
import pandas as pd
import torch.nn as nn
from helper import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
from tabpfn import TabPFNClassifier
from numerai_tools.scoring import numerai_corr
from sklearn.preprocessing import KBinsDiscretizer


# torch.autograd.set_detect_anomaly(True)

Parse = argparse.ArgumentParser()
Parse.add_argument("--config", type=str, default = 'Assignment1_config.yaml')
Arguments = Parse.parse_args()

# Read Configuration file
with open(Arguments.config, "r") as F: configs = yaml.safe_load(F); F.close()

feature_set = json.load(open(f"data/{configs['Dataset']['name']}/features.json"))["feature_sets"][f"{configs['Dataset']['set']}"]
train_dataset = pd.read_parquet(f"data/{configs['Dataset']['name']}/train.parquet", columns=["era", "target"] + feature_set)
train = pd.DataFrame(train_dataset[train_dataset["era"].isin(pd.Series(train_dataset["era"].unique()[-3:]))])

X = train[feature_set].values
y  = train['target'].values

# print(y)


value_to_index = {0: 0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4}
index_to_value = {v: k for k, v in value_to_index.items()} 
y_new = [value_to_index[val] for val in y]
# print(y_new)

X_train, X_test, y_train, y_test = train_test_split(X, y_new, test_size=0.2, random_state=42)

clf = TabPFNClassifier(device='cpu')  # Use 'cuda' if available and desired
clf.fit(X_train, y_train,overwrite_warning=True)


y_eval = clf.predict(X_test)
y_from_y_new = [index_to_value[val] for val in y_eval]
y_test_new = [index_to_value[val] for val in y_test]

print('Corr', numerai_corr(pd.DataFrame(y_from_y_new),pd.Series(y_test_new)))