#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def load_dataset(columns_to_encode=None, flag=True):
    """
    Load the dataset and apply one-hot encoding to features (all columns or specific columns).
    Targets will first be one-hot encoded and then converted to categorical integer labels.

    Parameters:
        columns_to_encode (list or None): List of column names to be one-hot encoded.
                                          If None and `flag=True`, all columns are encoded.
        flag (bool): Whether to apply one-hot encoding to all columns.
                     If True, `columns_to_encode` will be ignored, and all columns will be processed.

    Returns:
        train_x, train_y, test_x, test_y (numpy.ndarray):
            Train features, train labels, test features, and test labels in NumPy array format.
    """
    # Load the dataset
    diabetes_data = fetch_ucirepo(id=296)

    # Extract features and targets
    features = diabetes_data.data.features
    targets = diabetes_data.data.targets

    # Apply one-hot encoding to features
    if flag or columns_to_encode is None:
        features_encoded = pd.get_dummies(features, drop_first=True)
    else:
        features_encoded = pd.get_dummies(features, columns=columns_to_encode, drop_first=True)

    # One-hot encode targets and convert to a single categorical variable
    targets_encoded = pd.get_dummies(targets, drop_first=False)
    targets_categorical = targets_encoded.idxmax(axis=1)  # Get the column name with the max value (One-Hot index)
    targets_categorical = targets_categorical.astype('category').cat.codes  # Convert to integer codes

    # Convert to NumPy arrays
    features_np = features_encoded.to_numpy(dtype=np.float32)
    targets_np = targets_categorical.to_numpy(dtype=np.float32)

    # Split the data
    train_x, test_x, train_y, test_y = train_test_split(
        features_np, targets_np, test_size=0.2, random_state=42
    )

    return train_x, train_y, test_x, test_y



def load():
    train_x, train_y, val_x, val_y = load_dataset()
    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    train_y = train_y.astype(np.int32)
    val_y = val_y.astype(np.int32)
    return train_x, train_y, val_x, val_y