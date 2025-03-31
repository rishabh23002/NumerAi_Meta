import json
import yaml
import torch
import argparse
import pandas as pd

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
import numpy as np
import pandas as pd
import xgboost as xgb

class DeepIncrementalLearningModel:
    def __init__(self, base_params, n_layers=3, lookback_window=10):
        self.base_params = base_params
        self.n_layers = n_layers
        self.lookback_window = lookback_window
        self.models = []  # Stores models layer-wise

    def fit(self, data):
        """Train the model layer by layer using era-wise data."""
        for layer in range(self.n_layers):
            print(f"Training Layer {layer + 1}")
            layer_models = []

            # Process each era segment
            for era in range(0, len(data), self.lookback_window):
                # Prepare training data for the layer
                era_data = data.iloc[era:era + self.lookback_window].copy()  # Copy to avoid modifying the original data
                X_era = era_data.drop(columns=["target"]).values
                y_era = era_data["target"].values

                # Only use the predictions from the previous layer, if not the first layer
                if layer > 0:
                    # Generate predictions from the previous layer's models and add as a single feature
                    previous_layer_preds = np.mean([model.predict(X_era) for model in self.models[layer - 1]], axis=0)
                    X_era = np.column_stack((X_era, previous_layer_preds))

                # Train the model for the current layer and era
                model = xgb.XGBRegressor(**self.base_params)
                model.fit(X_era, y_era)
                layer_models.append(model)

            self.models.append(layer_models)

    def predict(self, X):
        """Predict using the hierarchical structure, only using predictions from the last layer."""
        for layer in range(self.n_layers):
            # Get predictions from each model in the current layer
            layer_preds = np.column_stack([model.predict(X) for model in self.models[layer]])

            # Average predictions across models in this layer to create a single prediction feature
            averaged_predictions = layer_preds.mean(axis=1).reshape(-1, 1)

            # For the next layer, use original features plus the current layer's averaged predictions
            X = np.column_stack((X[:, :42], averaged_predictions))

        # Final prediction is the averaged prediction from the last layer
        return averaged_predictions.squeeze()

# Example usage with the NumerAI data structure

# Sample dataset (replace with NumerAI data loader for actual data)
data = pd.DataFrame(np.random.rand(1000, 43), columns=[f'feature_{i}' for i in range(42)] + ['target'])

# Define XGBoost parameters
base_params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.01,
    'max_depth': 3,
    'n_estimators': 50
}

# Initialize and train the DIL model
dil_model = DeepIncrementalLearningModel(base_params=base_params, n_layers=3, lookback_window=10)
dil_model.fit(data)

# Predict on test data
X_test = data.drop(columns=['target']).values[:200]  # Replace with actual test data
predictions = dil_model.predict(X_test)
print("Predictions:", predictions)
