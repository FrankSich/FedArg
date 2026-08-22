import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class HospitalModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

def load_data(csv_path):
    """
    Load CSV, scale numeric features, one-hot encode categorical features,
    convert Outcome to tensor labels.
    """
    df = pd.read_csv(csv_path)

    # Fill missing vitals with 0
    for col in ["Pulse", "Resp", "Temp", "Sys", "Dia"]:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Numeric columns
    numeric_cols = ["Age", "Pulse", "Resp", "Temp", "Sys", "Dia"]
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Categorical columns
    cat_cols = ["Sponsor", "Region", "Procedures", "Medications", "Age_bin"]
    enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = enc.fit_transform(df[cat_cols])

    # Combine numeric + categorical
    X_numeric = df[numeric_cols].values
    X = torch.tensor(np.hstack([X_numeric, X_cat]), dtype=torch.float32)

    # Target column
    y_le = df["Outcome"].astype("category")
    y = torch.tensor(y_le.cat.codes.values, dtype=torch.long)

    # Save encoder info for federated learning
    encoders = {"Outcome_classes": y_le.cat.categories.tolist()}

    return X, y, encoders
